import os
import openai
import logging
from celery import Celery

# Get Redis URL from environment variable
redis_url = os.getenv('REDIS_URL')

# Initialize Celery with Redis as the broker
celery = Celery('tasks', broker=redis_url)

# Configure Celery task settings
celery.conf.update(
    result_backend=os.getenv('RESULT_BACKEND')  # Add result backend if required
)

# Configure OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)

@celery.task(bind=True)  # bind=True will give us access to self, which contains task context
def process_pdf(self, file_path, language):
    try:
        # Import functions for extracting and chunking text
        from app import extract_text_from_pdf, chunk_text
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        if not text:
            return {"error": "Failed to extract text from PDF."}

        # Chunk text into smaller pieces
        chunks = chunk_text(text)

        translations = []
        summaries = []

        for chunk in chunks:
            # Translation using OpenAI
            translation_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": f"Translate the following text from {language} to English:\n\n{chunk}"}
                ]
            )
            translation = translation_response.choices[0].message['content'].strip()
            translations.append(translation)

            # Summarization using OpenAI
            summary_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": f"Summarize the following English text in a few sentences:\n\n{translation}"}
                ]
            )
            summary = summary_response.choices[0].message['content'].strip()
            summaries.append(summary)

        # Combine translations and summaries
        full_translation = "\n".join(translations)
        full_summary = "\n".join(summaries)

        return {"translation": full_translation, "summary": full_summary}

    except Exception as e:
        # Capture and log any error that occurs during task execution
        logging.error(f"Error processing PDF: {e}", exc_info=True)
        self.retry(exc=e, countdown=60, max_retries=3)  # Retry mechanism in case of failure
        return {"error": str(e)}

