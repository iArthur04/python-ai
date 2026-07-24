# Install webcam library
# pip3 install opencv-python

# cat > webcam_test.py << 'EOF'
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("📸 Starting webcam classifier...")
print("Press 'q' to quit")

model = load_model('custom_classifier.h5')
class_names = ['cats', 'dogs']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare frame for model
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    # Show prediction on frame
    label = f"{predicted_class}: {confidence:.1f}%"
    color = (0, 255, 0) if confidence > 70 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('AI Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
EOF

#python3 webcam_test.py