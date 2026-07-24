# Save this as test_image.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('custom_classifier.h5')

# Test on a new image
img_path = 'IMG_3914.jpg'  # Your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)
class_names = ['cats', 'dogs']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {predicted_class} ({confidence:.1f}%)")