from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import cv2
import numpy as np
import os
import threading
import ast
from io import BytesIO
import base64
import onnxruntime as ort
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from ultralytics import YOLO

# --- ADDED FOR CHATBOT ---
from google import genai as google_genai
from google.genai import types as genai_types
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
_GEMINI_SYSTEM_PROMPT = (
    "You are a concise skin health assistant embedded in a skin disease detection app. "
    "Answer only questions related to the skin conditions shown in the analysis results: "
    "Melanoma, Basal Cell Carcinoma, Eczema, Psoriasis, and Tinea Ringworm. "
    "Keep replies short — 2-3 sentences max. Use plain language. "
    "Do not diagnose. If asked about unrelated topics, politely redirect to skin health. "
    "Always remind users you are an AI and not a substitute for a dermatologist."
)
if api_key:
    print(f"--- DEBUG: API Key found! It starts with: {api_key[:5]}... ---")
    _genai_client = google_genai.Client(api_key=api_key)
else:
    _genai_client = None
    print("--- CRITICAL ERROR: API Key is None. Python cannot read the .env file! ---")
# -------------------------
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ONNX model at startup
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8model.onnx')
_ort_session = None
_class_names = []

def _load_model():
    global _ort_session, _class_names
    if os.path.exists(_MODEL_PATH):
        _ort_session = ort.InferenceSession(_MODEL_PATH)
        meta = _ort_session.get_modelmeta().custom_metadata_map
        names_str = meta.get('names', '{}')
        names_dict = ast.literal_eval(names_str)
        _class_names = [names_dict[i] for i in sorted(names_dict.keys())]
        # Normalise typo in model metadata
        _class_names = ['Basal Cell Carcinoma' if n == 'Basic Cell Carcinoma' else n for n in _class_names]

_load_model()

# Load ResNet50 model at startup
_RESNET_PATH = os.path.join(os.path.dirname(__file__), 'best_resnet50_finetuned.pth')
_resnet_model = None

_resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _load_resnet_model():
    global _resnet_model
    if os.path.exists(_RESNET_PATH):
        try:
            net = tv_models.resnet50(weights=None)
            net.fc = torch.nn.Linear(2048, 5)
            state = torch.load(_RESNET_PATH, map_location='cpu', weights_only=True)
            net.load_state_dict(state)
            net.eval()
            _resnet_model = net
            print("--- ResNet50 model loaded successfully ---")
        except Exception as e:
            print(f"--- WARNING: Could not load ResNet50: {e} ---")
    else:
        print(f"--- WARNING: ResNet50 not found at {_RESNET_PATH} ---")

_load_resnet_model()

# Load best.pt model at startup
_BEST_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pt')
_best_model = None
_BEST_LABEL_MAP = {
    'bcc': 'Basal Cell Carcinoma',
    'eczema': 'Eczema',
    'melanoma': 'Melanoma',
    'psoriasis': 'Psoriasis',
    'tinea': 'Tinea Ringworm',
}

def _load_best_model():
    global _best_model
    if os.path.exists(_BEST_PATH):
        try:
            _best_model = YOLO(_BEST_PATH)
            print(f"--- best.pt loaded ({len(_best_model.names)} classes) ---")
        except Exception as e:
            print(f"--- WARNING: Could not load best.pt: {e} ---")
    else:
        print(f"--- WARNING: best.pt not found at {_BEST_PATH} ---")

_load_best_model()

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


def run_resnet50_inference(filepath):
    """Run ResNet50 inference on an image file. Returns (scores, top_class, confidence_pct)."""
    if _resnet_model is None:
        raise RuntimeError('ResNet50 model not loaded.')
    img = Image.open(filepath).convert('RGB')
    tensor = _resnet_transforms(img).unsqueeze(0)
    with torch.no_grad():
        logits = _resnet_model(tensor)[0]
        probs = torch.softmax(logits, dim=0).numpy()
    scores = {_class_names[i]: round(float(probs[i]), 4) for i in range(len(_class_names))}
    top_idx = int(np.argmax(probs))
    return scores, _class_names[top_idx], round(float(probs[top_idx]) * 100, 1)


def run_resnet50_inference_array(img_np):
    """Run ResNet50 inference on a 224x224 RGB numpy array."""
    if _resnet_model is None:
        raise RuntimeError('ResNet50 model not loaded.')
    img = Image.fromarray(img_np.astype(np.uint8))
    tensor = _resnet_transforms(img).unsqueeze(0)
    with torch.no_grad():
        logits = _resnet_model(tensor)[0]
        probs = torch.softmax(logits, dim=0).numpy()
    return probs


def run_best_inference(filepath):
    """Run best.pt inference. Returns scores dict keyed by display class names."""
    if _best_model is None:
        raise RuntimeError('best.pt model not loaded.')
    result = _best_model(filepath, verbose=False)[0]
    probs = result.probs.data.cpu().numpy()
    raw_names = _best_model.names
    scores = {
        _BEST_LABEL_MAP.get(raw_names[i], raw_names[i]): round(float(probs[i]), 4)
        for i in range(len(probs))
    }
    return scores


def run_onnx_inference(filepath):
    """Run YOLOv8 classification ONNX inference on an image file."""
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    img_np = img_np.transpose(2, 0, 1)                 # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0)            # Add batch dim: [1,3,224,224]

    input_name = _ort_session.get_inputs()[0].name
    outputs = _ort_session.run(None, {input_name: img_np})
    probs = outputs[0][0]  # Shape: [num_classes]

    # Check if already probabilities (sum close to 1 and all in [0,1])
    if np.sum(probs) > 0.99 and np.all(probs >= 0) and np.all(probs <= 1):
        # Already probabilities, just normalize
        probs = probs / np.sum(probs)
    else:
        # Apply softmax
        exp_probs = np.exp(probs - probs.max())
        probs = exp_probs / exp_probs.sum()

    scores = {_class_names[i]: round(float(probs[i]), 4) for i in range(len(_class_names))}
    top_idx = int(np.argmax(probs))
    return scores, _class_names[top_idx], round(float(probs[top_idx]) * 100, 1)


def process_image(filename):
    task_id = filename
    TASKS[task_id] = {'progress': 0, 'status': 'Starting...', 'completed': False}

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Step 1: Preprocessing confirmation
        TASKS[task_id].update({'status': 'Correcting orientation & enhancing contrast...', 'progress': 8})
        with open(filepath, 'rb') as f:
            _, _, _ = prepare_image(f)

        # Step 2: Load check
        TASKS[task_id].update({'status': 'Loading models...', 'progress': 15})

        # Step 3: Run both models
        TASKS[task_id].update({'status': 'Running YOLOv8 analysis...', 'progress': 25})
        yolo_scores = None
        if _ort_session is not None:
            yolo_scores, _, _ = run_onnx_inference(filepath)

        resnet_scores = None
        if _resnet_model is not None:
            TASKS[task_id].update({'status': 'Running ResNet50 analysis...', 'progress': 38})
            resnet_scores, _, _ = run_resnet50_inference(filepath)

        best_scores = None
        if _best_model is not None:
            TASKS[task_id].update({'status': 'Running best.pt analysis...', 'progress': 45})
            best_scores = run_best_inference(filepath)

        if yolo_scores is None and resnet_scores is None and best_scores is None:
            raise RuntimeError('No AI models loaded. Please install required packages (onnxruntime, torch, torchvision).')

        # Ensemble: equal weight across available models
        available = [s for s in [yolo_scores, resnet_scores, best_scores] if s is not None]
        if len(available) > 1:
            all_classes = list(available[0].keys())
            ensemble_scores = {
                cls: round(sum(s[cls] for s in available) / len(available), 4)
                for cls in all_classes
            }
        elif len(available) == 1:
            ensemble_scores = available[0]
        else:
            raise RuntimeError('No models available')

        # Primary result driven by ensemble
        primary_scores = ensemble_scores
        top_idx = max(primary_scores, key=primary_scores.get)
        confidence = round(ensemble_scores[top_idx] * 100, 1)
        sorted_scores = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [{'disease': name, 'probability': round(prob * 100, 1)} for name, prob in sorted_scores[:3]]

        # Step 4: Generate heatmaps while loading screen is still showing
        # YOLOv8 heatmap: overall progress 50–72%
        TASKS[task_id].update({'status': 'Generating YOLOv8 focus map...', 'progress': 50})
        def yolo_progress(row_pct):
            TASKS[task_id].update({'progress': 50 + round(row_pct * 0.22)})
        yolo_heatmap = generate_heatmap(filepath, model='yolo', progress_callback=yolo_progress)

        # ResNet50 heatmap: overall progress 72–94%
        resnet_heatmap = None
        if _resnet_model is not None:
            TASKS[task_id].update({'status': 'Generating ResNet50 focus map...', 'progress': 72})
            def resnet_progress(row_pct):
                TASKS[task_id].update({'progress': 72 + round(row_pct * 0.22)})
            resnet_heatmap = generate_heatmap(filepath, model='resnet', progress_callback=resnet_progress)

        result_data = {
            'primary_diagnosis': top_idx,
            'confidence': confidence,
            'top3': top3,
            'scores': ensemble_scores,
            'yolo_scores': yolo_scores,
            'resnet_scores': resnet_scores,
            'best_scores': best_scores,
            'ensemble_scores': ensemble_scores,
            'yolo_heatmap': yolo_heatmap,
            'resnet_heatmap': resnet_heatmap,
        }

        TASKS[task_id].update({'status': 'Almost done...', 'progress': 97})

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


def run_onnx_inference_array(img_np):
    """Run ONNX inference on a 224x224 RGB numpy array."""
    img_f = img_np.astype(np.float32) / 255.0
    img_f = img_f.transpose(2, 0, 1)
    img_f = np.expand_dims(img_f, axis=0)
    input_name = _ort_session.get_inputs()[0].name
    outputs = _ort_session.run(None, {input_name: img_f})
    probs = outputs[0][0]
    # Check if already probabilities
    if np.sum(probs) > 0.99 and np.all(probs >= 0) and np.all(probs <= 1):
        probs = probs / np.sum(probs)
    else:
        exp_probs = np.exp(probs - probs.max())
        probs = exp_probs / exp_probs.sum()
    return probs


def run_best_inference_array(img_np):
    """Run best.pt inference on a 224x224 RGB numpy array."""
    if _best_model is None:
        raise RuntimeError('best.pt model not loaded.')
    img = Image.fromarray(img_np.astype(np.uint8))
    result = _best_model(img, verbose=False)[0]
    return result.probs.data.cpu().numpy()


def generate_heatmap(filepath, model='yolo', progress_callback=None):
    """
    Occlusion sensitivity heatmap: occlude each patch in a 7x7 grid,
    measure confidence drop, then colormap and blend over the original image.
    Supports model='yolo', 'resnet', or 'best'.
    progress_callback(pct) is called after each row with a 0-100 value.
    """
    if model == 'resnet':
        infer_fn = run_resnet50_inference_array
    elif model == 'best':
        infer_fn = run_best_inference_array
    else:
        infer_fn = run_onnx_inference_array

    img = Image.open(filepath).convert('RGB').resize((224, 224))
    img_np = np.array(img)

    baseline_probs = infer_fn(img_np)
    top_idx = int(np.argmax(baseline_probs))
    baseline_conf = baseline_probs[top_idx]

    grid_n = 7
    patch = 224 // grid_n  # 32px per cell
    heatmap = np.zeros((grid_n, grid_n), dtype=np.float32)

    for i in range(grid_n):
        for j in range(grid_n):
            masked = img_np.copy()
            masked[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = 128
            probs = infer_fn(masked)
            heatmap[i, j] = max(0.0, float(baseline_conf - probs[top_idx]))
        if progress_callback:
            progress_callback(round((i + 1) / grid_n * 100))

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_large = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_large = np.clip(heatmap_large, 0, 1)

    heatmap_u8 = (heatmap_large * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (img_np * 0.55 + colored * 0.45).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


@app.route('/heatmap/<filename>')
def heatmap(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    model = request.args.get('model', 'yolo')
    if model == 'resnet' and _resnet_model is None:
        return jsonify({'error': 'ResNet50 model not loaded'}), 500
    if model == 'best' and _best_model is None:
        return jsonify({'error': 'best.pt model not loaded'}), 500
    if model not in ('resnet', 'best') and _ort_session is None:
        return jsonify({'error': 'YOLOv8 model not loaded'}), 500
    try:
        return jsonify({'heatmap': generate_heatmap(filepath, model=model)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- ADDED FOR CHATBOT ---
@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json
    user_input = payload.get('message')
    results    = payload.get('results')   # analysis scores dict
    image_data = payload.get('image')     # base64 data URL

    if not user_input:
        return jsonify({'error': 'No message'}), 400

    if _genai_client is None:
        return jsonify({'error': 'AI assistant unavailable — check API key.'}), 500

    try:
        parts = []

        # Attach the analysed image so Gemini can see it
        if image_data and ',' in image_data:
            img_b64 = image_data.split(',', 1)[1]
            img_bytes = base64.b64decode(img_b64)
            parts.append(genai_types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

        # Build a results context string
        context = ""
        if results and results.get('scores'):
            sorted_scores = sorted(results['scores'].items(), key=lambda x: x[1], reverse=True)
            top_name, top_prob = sorted_scores[0]
            breakdown = ", ".join(f"{k}: {round(v*100)}%" for k, v in sorted_scores)
            context = (
                f"The patient's skin analysis results are: "
                f"primary condition '{top_name}' at {round(top_prob*100)}% confidence. "
                f"Full breakdown — {breakdown}. "
                f"Use these results to answer the user's question below.\n\n"
            )

        parts.append(context + user_input)
        response = _genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parts,
            config=genai_types.GenerateContentConfig(system_instruction=_GEMINI_SYSTEM_PROMPT)
        )
        return jsonify({'reply': response.text})

    except Exception as e:
        return jsonify({'error': f"AI Error: {str(e)}"}), 500
# -------------------------


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug)