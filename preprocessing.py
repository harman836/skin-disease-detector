# Cell 1: Image preprocessing function for patient photos
import cv2
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
from ultralytics import YOLO
import io

def prepare_skin_image_for_yolo(image_input, target_size=224):
    """
    Prepare patient's skin rash/lesion image for YOLOv8 prediction
    
    Args:
        image_input: can be file path, PIL Image, or numpy array
        target_size: YOLOv8 input size (default 224)
    
    Returns:
        processed PIL Image ready for model prediction
    """
    
    # 1. Load image based on input type
    if isinstance(image_input, str):
        # It's a file path
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        # It's already a PIL Image
        img = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        # It's a numpy array (from camera/OpenCV)
        img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image input type")
    
    # 2. Fix orientation from phone cameras
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass  # No EXIF data or error
    
    # 3. Convert to numpy for processing
    img_np = np.array(img)
    
    # 4. Auto-enhance contrast for skin lesions
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 5. Convert back to PIL
    img = Image.fromarray(img_np)
    
    # 6. Resize maintaining aspect ratio (YOLO handles the rest)
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # 7. Create square image with padding
    new_img = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - img.size[0]) // 2
    paste_y = (target_size - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img

def check_image_quality(image_input):
    """
    Check if image quality is sufficient for prediction
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
    
    if img is None:
        return False, "Could not read image"
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Check brightness
    brightness = np.mean(gray)
    if brightness < 40:
        return False, "Image too dark - please use brighter lighting"
    elif brightness > 215:
        return False, "Image too bright - reduce exposure"
    
    # Check blurriness
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    if focus_measure < 50:
        return False, "Image is blurry - please take a clearer photo"
    
    return True, "Image quality acceptable"

def predict_skin_disease(model, image_input, class_names):
    """
    Predict skin disease from patient image
    """
    # Check image quality first
    quality_ok, quality_msg = check_image_quality(image_input)
    if not quality_ok:
        print(f"⚠️  Quality warning: {quality_msg}")
        # Continue anyway but with warning
    
    # Prepare image
    processed_img = prepare_skin_image_for_yolo(image_input)
    
    # Run prediction
    results = model(processed_img)
    
    # Extract results
    probs = results[0].probs
    top_class_idx = probs.top1
    confidence = probs.top1conf.item()
    predicted_disease = class_names[top_class_idx]
    
    # Get top 3 predictions
    top3_indices = np.argsort(probs.data.numpy())[-3:][::-1]
    top3_predictions = [
        (class_names[idx], probs.data[idx].item()) 
        for idx in top3_indices
    ]
    
    return {
        'primary_diagnosis': predicted_disease,
        'confidence': confidence,
        'top_3_predictions': top3_predictions,
        'all_probabilities': {
            class_names[i]: probs.data[i].item() 
            for i in range(len(class_names))
        },
        'processed_image': processed_img,
        'quality_warning': None if quality_ok else quality_msg
    }

# Cell 2: Load your trained model and test
print("Loading trained YOLOv8 model...")
model_path = "runs/classify/train/weights/best.pt"  # Update path if different
model = YOLO(model_path)
class_names = list(model.names.values())
print(f"Model loaded successfully! Classes: {class_names}")

# Cell 3: Function to predict from file path
def predict_from_file(image_path):
    """
    Predict skin disease from image file path
    """
    print(f"\n🔍 Analyzing: {image_path}")
    
    # Make prediction
    result = predict_skin_disease(model, image_path, class_names)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].set_title("Original Patient Image")
    axes[0].axis('off')
    
    # Processed image with prediction
    axes[1].imshow(result['processed_image'])
    title = f"Diagnosis: {result['primary_diagnosis']}\nConfidence: {result['confidence']:.1%}"
    if result['quality_warning']:
        title += f"\n⚠️ {result['quality_warning']}"
    axes[1].set_title(title, fontsize=11)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n📋 Primary Diagnosis: {result['primary_diagnosis']}")
    print(f"🎯 Confidence: {result['confidence']:.1%}")
    print("\n📊 Top 3 Possibilities:")
    for disease, prob in result['top_3_predictions']:
        print(f"   • {disease}: {prob:.1%}")
    
    return result

# Cell 4: Function to capture from camera (if using in webcam mode)
def predict_from_camera():
    """
    Capture image from camera and predict
    """
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image, ESC to exit")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Camera - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 32:  # SPACE pressed
            # Capture and predict
            result = predict_skin_disease(model, frame, class_names)
            
            # Display result
            cv2.imshow('Captured Image', frame)
            print(f"\n📋 Diagnosis: {result['primary_diagnosis']}")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            break
        elif key % 256 == 27:  # ESC pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return result

# Cell 5: Example usage - Test with a sample image
# Replace 'test_image.jpg' with your actual image path
test_image = 'test.jpg'  # Change this to your image path

if os.path.exists(test_image):
    result = predict_from_file(test_image)
else:
    print(f"Test image '{test_image}' not found. Please provide a valid image path.")
    print("\nTo test with your own image, run:")
    print('result = predict_from_file("path/to/your/skin_image.jpg")')

print("\n✅ Ready to predict! Use predict_from_file('image_path.jpg') for new images.")