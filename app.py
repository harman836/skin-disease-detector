from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import os
import threading
import time
from pathlib import Path
from io import BytesIO
import base64

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
IMAGE_SIZE = (224, 224)  # Target image size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global dictionary to store task progress
TASKS = {}


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


@app.route('/loading')
def loading():
    return render_template('loading.html')


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
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate the crop box to maintain aspect ratio and fill 224x224
        width, height = img.size
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

        # Encode as base64 to send back to the client (no file stored on disk)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f'data:image/jpeg;base64,{img_base64}'

        # Also save the file for the background processing task
        filename = secure_filename(file.filename)
        filename = os.path.splitext(filename)[0] + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            buffer.seek(0)
            f.write(buffer.read())

        print(f'Base64 encoded image length: {len(img_base64)}')
        print(f'Data URL length: {len(data_url)}')

        # Start analysis in background
        thread = threading.Thread(target=process_image, args=(filename,))
        thread.daemon = True
        thread.start()

        # Return the base64 data URL
        return jsonify({
            'success': True,
            'imageData': data_url,
            'filename': filename
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def download_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def process_image(filename):
    """
    Simulates the image analysis process.
    This is where the CNN model implementation will go later.
    """
    task_id = filename
    TASKS[task_id] = {'progress': 0, 'status': 'Starting...', 'completed': False}

    try:
        # Step 1: Preprocessing
        TASKS[task_id]['status'] = 'Preprocessing image...'
        time.sleep(1.0)
        TASKS[task_id]['progress'] = 20

        # Step 2: Model Loading
        TASKS[task_id]['status'] = 'Loading model...'
        time.sleep(1.0)
        TASKS[task_id]['progress'] = 40

        # Step 3: Inference
        TASKS[task_id]['status'] = 'Analyzing patterns...'
        time.sleep(1.5)
        TASKS[task_id]['progress'] = 70

        # Step 4: Post-processing
        TASKS[task_id]['status'] = 'Finalizing results...'
        time.sleep(0.5)
        TASKS[task_id]['progress'] = 90

        # Complete
        time.sleep(0.5)
        TASKS[task_id]['progress'] = 100
        TASKS[task_id]['status'] = 'Analysis complete!'
        TASKS[task_id]['completed'] = True

    except Exception as e:
        TASKS[task_id]['status'] = f'Error: {str(e)}'
        TASKS[task_id]['error'] = True


@app.route('/progress/<filename>')
def progress(filename):
    if filename in TASKS:
        return jsonify(TASKS[filename])
    return jsonify({'progress': 0, 'status': 'Waiting...', 'completed': False})


if __name__ == '__main__':
    app.run(debug=True)
