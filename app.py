from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import cv2
import numpy as np
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


def check_image_quality(img: Image.Image):
    """Check if image quality is sufficient for prediction. Returns (ok, message)."""
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)
    if brightness < 40:
        return False, "Image too dark - please use brighter lighting"
    if brightness > 215:
        return False, "Image too bright - reduce exposure"

    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    if focus_measure < 50:
        return False, "Image is blurry - please take a clearer photo"

    return True, "Image quality acceptable"


def prepare_image(file_stream):
    """
    Prepare skin image for YOLO prediction with EXIF correction, CLAHE contrast
    enhancement, and aspect-ratio-preserving letterbox resize to 224x224.
    Returns (data_url, jpeg_bytes, quality_warning_or_None).
    """
    target_size = IMAGE_SIZE[0]  # 224

    # Load and convert to RGB
    img = Image.open(file_stream).convert('RGB')

    # Fix orientation from phone cameras
    try:
        orientation_key = next(
            k for k, v in ExifTags.TAGS.items() if v == 'Orientation'
        )
        exif = img._getexif()
        if exif:
            orientation = exif.get(orientation_key, 1)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass

    # Quality check before processing
    quality_ok, quality_msg = check_image_quality(img)

    # CLAHE contrast enhancement for skin lesions
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(img_np)

    # Aspect-ratio-preserving resize with gray letterbox padding
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    letterboxed = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - img.size[0]) // 2
    paste_y = (target_size - img.size[1]) // 2
    letterboxed.paste(img, (paste_x, paste_y))

    # Encode as JPEG
    buffer = BytesIO()
    letterboxed.save(buffer, format='JPEG', quality=90)
    jpeg_bytes = buffer.getvalue()

    img_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{img_base64}'

    quality_warning = None if quality_ok else quality_msg
    return data_url, jpeg_bytes, quality_warning


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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        data_url, jpeg_bytes, quality_warning = prepare_image(file.stream)

        # Save for background processing
        filename = secure_filename(file.filename)
        filename = os.path.splitext(filename)[0] + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(jpeg_bytes)

        # Start analysis in background
        thread = threading.Thread(target=process_image, args=(filename,))
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'imageData': data_url,
            'filename': filename,
            'qualityWarning': quality_warning
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def download_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def process_image(filename):
    task_id = filename
    TASKS[task_id] = {'progress': 0, 'status': 'Starting...', 'completed': False}

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Step 1: Re-run preprocessing on the saved file to confirm pipeline
        TASKS[task_id].update({'status': 'Correcting orientation & enhancing contrast...', 'progress': 20})
        with open(filepath, 'rb') as f:
            _, _, _ = prepare_image(f)

        # Step 2: Load model (if available)
        TASKS[task_id].update({'status': 'Loading model...', 'progress': 40})
        model_path = os.path.join(os.path.dirname(__file__), 'runs', 'classify', 'train', 'weights', 'best.pt')
        model = None
        if os.path.exists(model_path):
            from ultralytics import YOLO
            model = YOLO(model_path)

        # Step 3: Run inference or simulate
        TASKS[task_id].update({'status': 'Analyzing skin patterns...', 'progress': 70})
        result_data = {}
        if model is not None:
            img = Image.open(filepath)
            results = model(img)
            probs = results[0].probs
            class_names = list(model.names.values())
            top_idx = probs.top1
            top3_indices = np.argsort(probs.data.numpy())[-3:][::-1]
            result_data = {
                'primary_diagnosis': class_names[top_idx],
                'confidence': round(probs.top1conf.item() * 100, 1),
                'top3': [
                    {'disease': class_names[i], 'probability': round(probs.data[i].item() * 100, 1)}
                    for i in top3_indices
                ]
            }

        TASKS[task_id].update({'status': 'Finalizing results...', 'progress': 90})
        time.sleep(0.3)

        TASKS[task_id].update({
            'progress': 100,
            'status': 'Analysis complete!',
            'completed': True,
            'result': result_data
        })

    except Exception as e:
        TASKS[task_id].update({'status': f'Error: {str(e)}', 'error': True})


@app.route('/progress/<filename>')
def progress(filename):
    if filename in TASKS:
        return jsonify(TASKS[filename])
    return jsonify({'progress': 0, 'status': 'Waiting...', 'completed': False})


if __name__ == '__main__':
    app.run(debug=True)
