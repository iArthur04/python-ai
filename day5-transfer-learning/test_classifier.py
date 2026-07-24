# Create a test script
#cat > test_classifier.py << 'EOF'
"""
Test your custom classifier on new images
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import random

print("=" * 60)
print("🐱🐶 TEST YOUR CLASSIFIER")
print("=" * 60)

# Load the model
model = load_model('custom_classifier.h5')
print("✅ Model loaded!")

# Class names (must match training)
class_names = ['cats', 'dogs']

def predict_image(img_path):
    """Predict class of a single image"""
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Show result
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    color = 'green' if confidence > 80 else 'orange' if confidence > 50 else 'red'
    plt.title(f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.1f}%", 
              color=color, fontsize=14)
    plt.axis('off')
    plt.show()
    
    return class_names[predicted_class], confidence

def test_random_from_dataset(folder='dataset/validation'):
    """Test on random images from the validation set"""
    # Get all images
    images = []
    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append((os.path.join(class_path, img_file), class_name))
    
    if not images:
        print("❌ No images found in validation folder!")
        return
    
    # Pick 5 random images
    test_images = random.sample(images, min(5, len(images)))
    
    print(f"\n📸 Testing on {len(test_images)} random images...")
    correct = 0
    
    for img_path, true_label in test_images:
        predicted, confidence = predict_image(img_path)
        is_correct = predicted == true_label
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} True: {true_label}, Predicted: {predicted} ({confidence:.1f}%)")
    
    print(f"\n📊 Accuracy on random images: {correct/len(test_images)*100:.1f}%")

def test_custom_image():
    """Test on a custom image you provide"""
    print("\n📸 Test on your own image!")
    img_path = input("Enter image path (drag and drop the image here): ").strip()
    
    # Remove quotes if present
    img_path = img_path.strip("'\"")
    
    if not os.path.exists(img_path):
        print("❌ Image not found!")
        return
    
    predicted, confidence = predict_image(img_path)
    print(f"✅ Prediction: {predicted} ({confidence:.1f}% confidence)")

# ========================================
# Main Menu
# ========================================

while True:
    print("\n" + "=" * 40)
    print("🔮 What would you like to do?")
    print("1. Test on random images from validation set")
    print("2. Test on your own image")
    print("3. Test on a specific image path")
    print("4. Exit")
    print("=" * 40)
    
    choice = input("Choose (1-4): ")
    
    if choice == '1':
        test_random_from_dataset()
    elif choice == '2':
        test_custom_image()
    elif choice == '3':
        img_path = input("Enter image path: ").strip("'\"")
        if os.path.exists(img_path):
            predict_image(img_path)
        else:
            print("❌ Image not found!")
    elif choice == '4':
        print("👋 Goodbye!")
        break
    else:
        print("❌ Invalid choice!")
EOF

#python3 test_classifier.py