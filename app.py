from flask import Flask, render_template, request, send_file, redirect, url_for, flash
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
from openai import OpenAI  # Import the OpenAI class

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Download NLTK data (only needs to be done once)
# Modify to download only if not already downloaded
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)

# Configure NLTK to use the downloaded data
nltk.data.path.append(nltk_data_dir)

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
preset_languages = ['Classical Chinese', 'French', 'Latin', 'Old English', 'Other']

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_text(text, max_length=3000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

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
                model="gpt-4",
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
                    model="gpt-4",
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
                input_text = extract_text(file_path).strip()
                logging.debug(f"Extracted text length: {len(input_text)} characters")
                
                if not input_text:
                    flash("The uploaded PDF is empty or couldn't extract any text.", "error")
                    return redirect(url_for('translate_pdf'))
            except Exception as e:
                logging.error(f"PDF processing error: {e}")
                flash(f"An error occurred while processing the PDF file: {e}", "error")
                return redirect(url_for('translate_pdf'))
        else:
            flash("Allowed file types are PDF.", "error")
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
                model="gpt-4",
                messages=translation_messages
            )
            # Extract only the content
            translation = completion.choices[0].message.content.strip()
            logging.debug(f"Translation received: {translation}")
            flash("Translation completed successfully.", "success")
        except Exception as e:
            logging.error(f"Translation error: {e}")
            flash(f"An error occurred during translation: {e}", "error")
            return redirect(url_for('translate_pdf'))

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
                    model="gpt-4",
                    messages=summary_messages
                )
                # Extract only the content
                summary = completion.choices[0].message.content.strip()
                logging.debug(f"Summary received: {summary}")
                flash("Summary generated successfully.", "success")
            except Exception as e:
                logging.error(f"Summarization error: {e}")
                flash(f"An error occurred during summarization: {e}", "error")
                return redirect(url_for('translate_pdf'))

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

if __name__ == '__main__':
    # Get the port from environment variable or default to 5500
    port = int(os.environ.get('PORT', 5500))
    # Determine if the app is in development or production
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
