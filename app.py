from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

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
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

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
