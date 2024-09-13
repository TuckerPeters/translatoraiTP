# celery_worker.py

import os
from celery import Celery
import logging
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
from openai import OpenAI  # Ensure correct import
import nltk

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Initialize OpenAI client
client = OpenAI()  # Initializes using the OPENAI_API_KEY from environment variables

# Initialize Celery application
broker_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
celery = Celery('translatorai', broker=broker_url, backend=broker_url)

# Configure Celery (optional)
celery.conf.update(
    broker_url=broker_url,
    result_backend=broker_url,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# Download NLTK data locally and include in project
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


def chunk_text(text, max_tokens=3000, model="gpt-4o-mini"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@celery.task(bind=True)
def process_pdf(self, file_path, language):
    logging.debug(f"Processing PDF at {file_path} with language {language}")
    translation = ''
    summary = ''

    try:
        # Extract text from PDF
        text = extract_text(file_path)
        if not text:
            # If extraction fails, attempt OCR
            from pdf2image import convert_from_path
            from PIL import Image
            import pytesseract

            pages = convert_from_path(file_path, dpi=300)
            text = ""
            for page in pages:
                text += pytesseract.image_to_string(page)
            text = text.strip()

        if not text:
            return {'error': 'Failed to extract text from the PDF.'}

        logging.debug(f"Extracted text length: {len(text)} characters")

        # Chunk the text
        chunks = chunk_text(text)
        logging.debug(f"Total chunks created: {len(chunks)}")

        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logging.debug(f"Translating chunk {i+1}/{len(chunks)}")
            translation_messages = [
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {
                    "role": "user",
                    "content": f"Translate the following text from {language} to English:\n\n{chunk}"
                }
            ]
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=translation_messages
            )
            translated_chunk = completion.choices[0].message.content.strip()
            translated_chunks.append(translated_chunk)

        # Combine translated chunks
        translation = "\n\n".join(translated_chunks)
        logging.debug("Translation completed.")

        # Summarize the translation
        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {
                "role": "user",
                "content": f"Summarize the following English text in a few sentences:\n\n{translation}"
            }
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=summary_messages
        )
        summary = completion.choices[0].message.content.strip()
        logging.debug("Summarization completed.")

        return {
            'translation': translation,
            'summary': summary
        }

    except Exception as e:
        logging.error(f"Error in process_pdf task: {e}")
        return {'error': str(e)}
