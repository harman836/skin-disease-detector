from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
from pathlib import Path
from io import BytesIO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
IMAGE_SIZE = (224, 224)  # Target image size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
        
        # Calculate the crop box to maintain aspect ratio and fill 614x614
        width, height = img.size
        target_size = IMAGE_SIZE[0]  # 614
        
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
        
        # Resize to exactly 614x614
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        # Change extension to .jpg for consistency
        filename = os.path.splitext(filename)[0] + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save as JPEG
        img.save(filepath, 'JPEG', quality=95)

        # Return the file path to display
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/uploads/{filename}'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def download_file(filename):
    # Secure the filename to prevent directory traversal attacks
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
