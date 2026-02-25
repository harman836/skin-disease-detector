from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
from pathlib import Path
from io import BytesIO
import base64

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
IMAGE_SIZE = (224, 224)  # Target image size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


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
        
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        # Change extension to .jpg for consistency
        filename = os.path.splitext(filename)[0] + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f'Base64 encoded image length: {len(img_base64)}')
        print(f'Data URL length: {len(data_url)}')

        # Start analysis in background
        thread = threading.Thread(target=process_image, args=(filename,))
        thread.daemon = True
        thread.start()

        # Return the base64 data URL (no file stored on disk)
        return jsonify({
            'success': True,
            'imageData': data_url
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def download_file(filename):
    # Secure the filename to prevent directory traversal attacks
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


import threading
import time

# Global dictionary to store task progress
# In a production app, use Redis or a database
TASKS = {}

def process_image(filename):
    """
    Simulates the image analysis process.
    This is where the CNN model implementation will go later.
    """
    task_id = filename
    TASKS[task_id] = {'progress': 0, 'status': 'Starting...', 'completed': False}
    
    try:
        # Step 1: Preprocessing
        # When CNN is implemented: Load image, resize, normalize
        TASKS[task_id]['status'] = 'Preprocessing image...'
        time.sleep(1.0) # Simulate work
        TASKS[task_id]['progress'] = 20
        
        # Step 2: Model Loading (if not cached)
        # When CNN is implemented: Ensure model is loaded to device
        TASKS[task_id]['status'] = 'Loading model...'
        time.sleep(1.0)
        TASKS[task_id]['progress'] = 40
        
        # Step 3: Inference
        # When CNN is implemented: model.predict(image)
        TASKS[task_id]['status'] = 'Analyzing patterns...'
        time.sleep(1.5)
        TASKS[task_id]['progress'] = 70
        
        # Step 4: Post-processing
        # When CNN is implemented: Format results, calculate confidence
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
