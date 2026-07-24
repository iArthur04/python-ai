"""
DAY 5: Transfer Learning
Why train from scratch when you can build on existing knowledge?
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import io

print("=" * 70)
print("🧠 TRANSFER LEARNING")
print("=" * 70)

print("""
💡 WHAT IS TRANSFER LEARNING?
-----------------------------
Instead of training a neural network from scratch (which takes
weeks and thousands of GPUs), we use a pre-trained model that
already knows how to recognize patterns.

Think of it like:
- 📚 Learning to read before learning to write
- 🎸 Learning guitar before learning bass
- 🧠 Learning patterns before learning specific tasks

Why it's powerful:
1. ⚡ 10x faster training
2. 🎯 100x less data needed
3. 🏆 State-of-the-art accuracy
4. 💰 Free (no GPU required!)

Common Pre-trained Models:
- ResNet50: 50 layers, 1.4M images, 1000 classes
- VGG16: 16 layers, 1.4M images, 1000 classes
- MobileNetV2: Lightweight for mobile, 1000 classes
- InceptionV3: 48 layers, Google's model
""")

# ========================================
# 1. LOAD A PRE-TRAINED MODEL
# ========================================

print("\n📦 1. LOADING PRE-TRAINED MODELS")

print("\nLoading ResNet50 (50 layers, trained on ImageNet)...")
resnet = ResNet50(weights='imagenet', include_top=True)
print("✅ ResNet50 loaded!")

print("\nLoading MobileNetV2 (lightweight, for mobile)...")
mobilenet = MobileNetV2(weights='imagenet', include_top=True)
print("✅ MobileNetV2 loaded!")

# ========================================
# 2. UNDERSTANDING THE MODEL ARCHITECTURE
# ========================================

print("\n🏗️ 2. MODEL ARCHITECTURE")

print(f"ResNet50 Input Shape: {resnet.input_shape}")
print(f"ResNet50 Output Shape: {resnet.output_shape}")
print(f"ResNet50 Total Parameters: {resnet.count_params():,}")

print(f"\nMobileNetV2 Input Shape: {mobilenet.input_shape}")
print(f"MobileNetV2 Output Shape: {mobilenet.output_shape}")
print(f"MobileNetV2 Total Parameters: {mobilenet.count_params():,}")

# ========================================
# 3. VISUALIZE THE MODEL
# ========================================

print("\n📊 3. VISUALIZING THE MODEL")

# Show first few layers
print("\nFirst 5 layers of ResNet50:")
for i, layer in enumerate(resnet.layers[:5]):
    # ✅ FIX: Handle InputLayer differently
    if hasattr(layer, 'output_shape'):
        shape = layer.output_shape
    else:
        shape = "Input layer"
    print(f"  Layer {i+1}: {layer.name} - {shape}")

print("\nLast 5 layers of ResNet50:")
for i, layer in enumerate(resnet.layers[-5:]):
    if hasattr(layer, 'output_shape'):
        shape = layer.output_shape
    else:
        shape = "Special layer"
    print(f"  Layer {i+1}: {layer.name} - {shape}")

# ========================================
# 4. WHAT THE MODEL KNOWS (1000 Classes)
# ========================================

print("\n📚 4. WHAT THE MODEL KNOWS")

# The model can recognize 1000 classes
print("ImageNet has 1000 classes including:")
print("  🐕 Dogs, 🐈 Cats, 🏠 Buildings, 🚗 Cars, 🌸 Flowers")
print("  👨 People, 🍕 Food, 🎵 Instruments, 🎮 Objects")

# Show a few examples
sample_classes = [
    'golden_retriever', 'tabby_cat', 'convertible', 'skyscraper',
    'pizza', 'guitar', 'teddy_bear', 'coffee_mug'
]

print("\nSome classes the model can recognize:")
for i, cls in enumerate(sample_classes, 1):
    print(f"  {i}. {cls}")

# ========================================
# 5. MAKE A PREDICTION WITH PRE-TRAINED MODEL
# ========================================

print("\n🔮 5. MAKING PREDICTIONS")

def load_and_prepare_image(img_url, target_size=(224, 224)):
    """Load an image from URL and prepare for prediction"""
    response = requests.get(img_url)
    img = Image.open(io.BytesIO(response.content))
    
    # Resize to model's expected input size
    img = img.resize(target_size)
    
    # Convert to array
    x = np.array(img)
    
    # ResNet expects (224, 224, 3)
    if len(x.shape) == 2:  # Grayscale image
        x = np.stack([x, x, x], axis=-1)
    
    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    
    # Preprocess for ResNet50
    x = preprocess_input(x)
    
    return x, img

def predict_image(model, img_url, model_name="ResNet50"):
    """Make a prediction using the model"""
    x, img = load_and_prepare_image(img_url)
    predictions = model.predict(x, verbose=0)
    
    # Decode predictions
    results = decode_predictions(predictions, top=5)[0]
    
    print(f"\n📸 {model_name} Predictions:")
    for i, (class_id, class_name, probability) in enumerate(results, 1):
        print(f"  {i}. {class_name}: {probability*100:.1f}%")
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Top Prediction: {results[0][1]} ({results[0][2]*100:.1f}%)")
    plt.axis('off')
    plt.show()
    
    return results

# Try some images (replace with your own URLs)
print("\nLet's test with some sample images!")
print("You can try: https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg")
print("or any image URL you like!")

# ========================================
# 6. COMPARE DIFFERENT MODELS
# ========================================

print("\n📊 6. COMPARING MODELS")

def compare_models(img_url):
    """Compare ResNet50 and MobileNetV2 predictions"""
    print(f"\n🔍 Comparing models on the same image:")
    
    # Load image once
    x, img = load_and_prepare_image(img_url)
    
    # ResNet50
    resnet_preds = resnet.predict(x, verbose=0)
    resnet_results = decode_predictions(resnet_preds, top=3)[0]
    
    # MobileNetV2
    mobilenet_preds = mobilenet.predict(x, verbose=0)
    mobilenet_results = decode_predictions(mobilenet_preds, top=3)[0]
    
    print("\n📊 Comparison:")
    print("  " + "-" * 50)
    print(f"  {'ResNet50':<20} {'MobileNetV2':<20}")
    print("  " + "-" * 50)
    
    for i in range(3):
        resnet_class = resnet_results[i][1]
        resnet_prob = resnet_results[i][2] * 100
        mobilenet_class = mobilenet_results[i][1]
        mobilenet_prob = mobilenet_results[i][2] * 100
        
        print(f"  {resnet_class:<20} {mobilenet_class:<20}")
        print(f"  {resnet_prob:.1f}%{' ' * 18} {mobilenet_prob:.1f}%")
        print("  " + "-" * 50)

# ========================================
# 7. UNDERSTANDING FEATURE EXTRACTION
# ========================================

print("\n🎨 7. FEATURE EXTRACTION")

def extract_features(model, img_url):
    """Extract features from the model (without the top layers)"""
    # Create model without top layers
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # Penultimate layer
    )
    
    # Load and prepare image
    x, img = load_and_prepare_image(img_url)
    
    # Extract features
    features = feature_extractor.predict(x, verbose=0)
    
    print(f"\n📊 Features extracted:")
    print(f"  Shape: {features.shape}")
    print(f"  Number of features: {features.shape[1]:,}")
    print(f"  Sample values: {features[0, :5].round(3)}")
    
    return features, img

# ========================================
# 8. WHY TRANSFER LEARNING WORKS
# ========================================

print("\n💡 8. WHY TRANSFER LEARNING WORKS")

print("""
🧠 The model has learned:
   Layer 1-3:  🟦 Edges and colors
   Layer 4-10: 🟧 Shapes and textures
   Layer 11-30: 🟨 Objects and patterns
   Layer 31-50: 🟩 Specific objects (dogs, cats, cars)

This knowledge transfers to YOUR task because:
   ✅ Edges are universal
   ✅ Shapes are universal
   ✅ Object patterns transfer well

You only need to:
   1. Use the pre-trained model as a feature extractor
   2. Add your own classification layer
   3. Train ONLY the new layer (fine-tuning)
""")

# ========================================
# 9. SAVE THE MODEL
# ========================================

print("\n💾 9. SAVING THE MODEL")

# Download and save a sample image for testing
try:
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog.jpg"
    x, img = load_and_prepare_image(sample_url)
    img.save('sample_image.jpg')
    print("✅ Sample image saved as 'sample_image.jpg'")
except:
    print("⚠️ Could not download sample image")

print("\n🎉 Transfer Learning Basics Complete!")
print("""
Next: We'll build a custom classifier using these pre-trained models!
""")