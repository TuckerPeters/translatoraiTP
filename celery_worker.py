from celery import Celery
import os
import openai
import logging

# Get Redis URL from environment variable
redis_url = os.getenv('REDIS_URL')

# Initialize Celery with Redis as the broker
celery = Celery('tasks', broker=redis_url)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

@celery.task
def process_pdf(file_path, language):
    try:
        # Extract text from PDF
        from app import extract_text_from_pdf, chunk_text  # Import necessary functions
        text = extract_text_from_pdf(file_path)
        if not text:
            return {"error": "Failed to extract text from PDF."}

        # Chunk text
        chunks = chunk_text(text)

        translations = []
        summaries = []

        for chunk in chunks:
            # Translation
            translation_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": f"Translate the following text from {language} to English:\n\n{chunk}"}
                ]
            )
            translation = translation_response.choices[0].message.content.strip()
            translations.append(translation)

            # Summarization
            summary_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": f"Summarize the following English text in a few sentences:\n\n{translation}"}
                ]
            )
            summary = summary_response.choices[0].message.content.strip()
            summaries.append(summary)

        # Combine translations and summaries
        full_translation = "\n".join(translations)
        full_summary = "\n".join(summaries)

        return {"translation": full_translation, "summary": full_summary}

    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return {"error": str(e)}
