# app.py

from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify
import os
import logging
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import sent_tokenize
import io
from langdetect import detect, DetectorFactory
from datetime import datetime
from dotenv import load_dotenv
from celery_worker import process_pdf, celery  # Import the Celery task and Celery application
from celery.result import AsyncResult
from openai import OpenAI  # Import the OpenAI class

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Download NLTK data locally and include in project
# Pre-downloaded and placed in 'nltk_data/' directory
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Load environment variables from .env file (only in development)
if os.getenv('FLASK_ENV') == 'development':
    load_dotenv()

app = Flask(__name__)

# Set secret key from environment variable or default (use a secure key in production)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')

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

# Preset language options
preset_languages = ['Auto-Detect', 'Classical Chinese', 'French', 'Latin', 'Old English', 'Other']

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max 100MB upload size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_with_ocr(file_path):
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        import pytesseract

        pages = convert_from_path(file_path, dpi=300)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR extraction error: {e}")
        return None

def detect_language(text):
    try:
        language = detect(text)
        logging.debug(f"Detected language: {language}")
        return language
    except Exception as e:
        logging.error(f"Language detection error: {e}")
        return None

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.utcnow().year}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate_text', methods=['GET', 'POST'])
def translate_text():
    translation = ''
    summary = ''

    if request.method == 'POST':
        input_text = request.form.get('input_text', '').strip()
        selected_language = request.form.get('language', '').strip()
        custom_language = request.form.get('custom_language', '').strip()

        logging.debug(f"Received input_text: {input_text}")
        logging.debug(f"Selected language: {selected_language}")
        logging.debug(f"Custom language: {custom_language}")

        # Validate input
        if not input_text:
            flash("Please enter the text you want to translate.", "error")
            return redirect(url_for('translate_text'))

        # Determine the language
        if selected_language == 'Other' and custom_language:
            language = custom_language
        elif selected_language == 'Auto-Detect':
            language = detect_language(input_text)
            if not language:
                flash("Could not detect the language of the input text.", "error")
                return redirect(url_for('translate_text'))
        else:
            language = selected_language

        logging.debug(f"Determined language for translation: {language}")

        # Prepare the messages for translation
        translation_messages = [
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {
                "role": "user",
                "content": f"Translate the following text from {language} to English:\n\n{input_text}"
            }
        ]

        # Call OpenAI API for translation
        try:
            logging.debug("Sending translation request to OpenAI API.")
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=translation_messages
            )
            # Extract only the content
            translation = completion.choices[0].message.content.strip()
            logging.debug(f"Translation received: {translation}")
            flash("Translation completed successfully.", "success")
        except Exception as e:
            logging.error(f"Translation error: {e}")
            flash(f"An error occurred during translation: {e}", "error")
            return redirect(url_for('translate_text'))

        # Proceed to summarization only if translation was successful
        if translation:
            # Prepare the messages for summarization
            summary_messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {
                    "role": "user",
                    "content": f"Summarize the following English text in a few sentences:\n\n{translation}"
                }
            ]

            # Call OpenAI API for summarization
            try:
                logging.debug("Sending summarization request to OpenAI API.")
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=summary_messages
                )
                # Extract only the content
                summary = completion.choices[0].message.content.strip()
                logging.debug(f"Summary received: {summary}")
                flash("Summary generated successfully.", "success")
            except Exception as e:
                logging.error(f"Summarization error: {e}")
                flash(f"An error occurred during summarization: {e}", "error")
                return redirect(url_for('translate_text'))

    return render_template(
        'translate_text.html',
        preset_languages=preset_languages,
        translation=translation,
        summary=summary
    )

@app.route('/translate_pdf', methods=['GET', 'POST'])
def translate_pdf():
    translation = ''
    summary = ''
    task_id = None

    if request.method == 'POST':
        if 'input_file' not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for('translate_pdf'))

        file = request.files['input_file']

        if file.filename == '':
            flash("No file selected for uploading.", "error")
            return redirect(url_for('translate_pdf'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logging.debug(f"File saved to {file_path}")

            try:
                input_text = extract_text_from_pdf(file_path)
                if not input_text:
                    # Attempt OCR if text extraction fails
                    input_text = extract_text_with_ocr(file_path)
                    if not input_text:
                        flash("The uploaded PDF is empty or couldn't extract any text.", "error")
                        return redirect(url_for('translate_pdf'))
                logging.debug(f"Extracted text length: {len(input_text)} characters")
            except Exception as e:
                logging.error(f"PDF processing error: {e}")
                flash(f"An error occurred while processing the PDF file: {e}", "error")
                return redirect(url_for('translate_pdf'))

            # Determine the language
            selected_language = request.form.get('language', '').strip()
            custom_language = request.form.get('custom_language', '').strip()

            logging.debug(f"Selected language: {selected_language}")
            logging.debug(f"Custom language: {custom_language}")

            if selected_language == 'Other' and custom_language:
                language = custom_language
            elif selected_language == 'Auto-Detect':
                language = detect_language(input_text)
                if not language:
                    flash("Could not detect the language of the input text.", "error")
                    return redirect(url_for('translate_pdf'))
            else:
                language = selected_language

            logging.debug(f"Determined language for translation: {language}")

            # Enqueue the PDF processing task
            task = process_pdf.delay(file_path, language)
            task_id = task.id
            flash("Your PDF is being processed. Please check back shortly.", "info")

            return redirect(url_for('translate_pdf', task_id=task_id))

    # Check if a task_id is provided to display results
    task_id = request.args.get('task_id')
    if task_id:
        task = AsyncResult(task_id, app=celery)
        if task.state == 'SUCCESS':
            result = task.result
            if 'error' in result:
                flash(f"An error occurred: {result['error']}", "error")
            else:
                translation = result.get('translation', '')
                summary = result.get('summary', '')
                flash("PDF processing completed successfully.", "success")
        elif task.state == 'PENDING':
            flash("Your PDF is still being processed. Please check back shortly.", "info")
        else:
            flash(f"Task state: {task.state}", "warning")

    return render_template(
        'translate_pdf.html',
        preset_languages=preset_languages,
        translation=translation,
        summary=summary
    )

@app.route('/download')
def download_file():
    content = request.args.get('content', '')
    filename = request.args.get('filename', 'result.txt')

    # Create a text stream
    text_stream = io.StringIO(content)
    mem = io.BytesIO()
    mem.write(text_stream.getvalue().encode('utf-8'))
    mem.seek(0)
    text_stream.close()

    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )

@app.route('/task_status/<task_id>')
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'PENDING':
        response = {'state': task.state}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        # Something went wrong in the background job
        response = {
            'state': task.state,
            'result': str(task.info),  # This is the exception raised
        }
    return jsonify(response)

if __name__ == '__main__':
    # Get the port from environment variable or default to 5500
    port = int(os.environ.get('PORT', 5500))
    # Determine if the app is in development or production
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
