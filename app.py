from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
from pathlib import Path
from io import BytesIO
import base64

# --- ADDED FOR CHATBOT ---
import google.generativeai as genai
from dotenv import load_dotenv

# Force Python to find the .env file exactly where app.py is located
basedir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(basedir, '.env')
load_dotenv(env_path)
# -------------------------

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
IMAGE_SIZE = (224, 224)  # Target image size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# --- ADDED FOR CHATBOT ---
api_key = os.getenv("GEMINI_API_KEY")

# Let's print this to the terminal so we can see if it worked!
print(f"--- DEBUG: Looking for .env file at: {env_path} ---")
if api_key:
    print(f"--- DEBUG: API Key found! It starts with: {api_key[:5]}... ---")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful skin health assistant. Remind users that you are an AI, not a doctor."
    )
else:
    print("--- 🚨 CRITICAL ERROR: API Key is STILL None. Python cannot read the .env file! ---")
# -------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

    try:
        # Open and resize the image
        img = Image.open(file.stream)
        
        # Convert RGBA to RGB if necessary (for PNG files)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Calculate the crop box to maintain aspect ratio and fill 224x224
        width, height = img.size
        target_size = IMAGE_SIZE[0]  # 224
        
        if width > height:
            # Landscape - crop width
            new_width = height
            left = (width - new_width) // 2
            img = img.crop((left, 0, left + new_width, height))
        elif height > width:
            # Portrait - crop height
            new_height = width
            top = (height - new_height) // 2
            img = img.crop((0, top, width, top + new_height))
        
        # Resize to exactly 224x224
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to base64 without saving to disk
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f'data:image/jpeg;base64,{img_base64}'
        
        print(f'Base64 encoded image length: {len(img_base64)}')
        print(f'Data URL length: {len(data_url)}')

        # Return the base64 data URL (no file stored on disk)
        return jsonify({
            'success': True,
            'imageData': data_url
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- ADDED FOR CHATBOT ---
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message'}), 400
    
    # Check if the model actually loaded
    if 'model' not in globals():
        return jsonify({'error': "Python can't find the .env file or the key inside it."}), 500
        
    try:
        response = model.generate_content(user_input)
        return jsonify({'reply': response.text})
    except Exception as e:
        # This will send Google's EXACT error message to your screen
        return jsonify({'error': f"Google AI Error: {str(e)}"}), 500
# -------------------------


if __name__ == '__main__':
    app.run(debug=True)