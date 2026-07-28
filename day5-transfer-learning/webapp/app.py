"""
AI Image Classifier Web App - MULTI-CLASS
Cats 🐱, Dogs 🐶, Birds 🐦, Fish 🐟
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename
import base64
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================================
# LOAD MULTI-CLASS MODEL
# ========================================

print("📦 Loading AI model...")

# Try to load the proper multi-class model
model = None
class_names = ['cats', 'dogs', 'birds', 'fish']  # fallback only, overwritten below if possible

# Keras assigns class indices ALPHABETICALLY by folder name during training
# (birds=0, cats=1, dogs=2, fish=3) -- NOT in the order they're listed above.
# Load the mapping saved by train_multi_proper.py so labels line up correctly.
try:
    with open('../class_indices.json', 'r') as f:
        class_indices = json.load(f)  # e.g. {'birds':0, 'cats':1, 'dogs':2, 'fish':3}
    ordered = [None] * len(class_indices)
    for name, idx in class_indices.items():
        ordered[idx] = name
    class_names = ordered
    print(f"✅ Loaded class order from class_indices.json: {class_names}")
except Exception as e:
    print(f"⚠️ class_indices.json not found, using default order (may be WRONG): {e}")

emojis = {
    'cats': '🐱',
    'dogs': '🐶',
    'birds': '🐦',
    'fish': '🐟'
}
display_names = {
    'cats': 'Cat',
    'dogs': 'Dog',
    'birds': 'Bird',
    'fish': 'Fish'
}

try:
    model = load_model('../multi_classifier_proper.h5')
    print("✅ Multi-class model loaded successfully!")
    print(f"   Classes: {class_names}")
except:
    try:
        # Fallback to cats vs dogs
        model = load_model('../custom_classifier.h5')
        class_names = ['cats', 'dogs']
        emojis = {'cats': '🐱', 'dogs': '🐶'}
        display_names = {'cats': 'Cat', 'dogs': 'Dog'}
        print("✅ Cats vs Dogs model loaded (fallback)")
    except Exception as e:
        print(f"⚠️ No model found: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image_full(img_path):
    """Predict class and return all predictions"""
    if model is None:
        return 'dogs', 50.0, {c: 25.0 for c in class_names}
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # must match training preprocessing (ResNet50), not /255.0
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100
    
    all_predictions = {}
    for i, class_name in enumerate(class_names):
        all_predictions[class_name] = float(predictions[i] * 100)
    
    return class_names[predicted_class], confidence, all_predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    predicted_class, confidence, all_predictions = predict_image_full(filepath)
    
    with open(filepath, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Get display name
    display_name = display_names.get(predicted_class, predicted_class.capitalize())
    
    response = {
        'success': True,
        'prediction': predicted_class,
        'display_name': display_name,
        'emoji': emojis.get(predicted_class, '❓'),
        'confidence': f'{confidence:.1f}%',
        'image': f'data:image/jpeg;base64,{img_data}',
        'all_predictions': all_predictions
    }
    
    return jsonify(response)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names
    })

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Starting AI Web App...")
    print(f"📍 Visit: http://localhost:5001")
    print(f"📊 Model: {len(class_names)} classes")
    print(f"   {', '.join([f'{emojis[c]} {c}' for c in class_names])}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5001)