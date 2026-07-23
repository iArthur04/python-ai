"""
Simplified MNIST Classifier
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("✍️ MNIST DIGIT RECOGNIZER")
print("=" * 50)

# Load data
print("\n📊 Loading MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Build model
print("🧠 Building model...")
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
print("🏋️ Training...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📈 Test Accuracy: {test_acc*100:.2f}%")

# Save
model.save('mnist_simple.h5')
print("✅ Model saved!")

# Test on some images
predictions = model.predict(X_test[:10])
predicted = np.argmax(predictions, axis=1)

print("\n🔮 Sample Predictions:")
print(f"True labels: {y_test[:10]}")
print(f"Predicted:   {predicted}")

plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'{predicted[i]}', fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.savefig('simple_predictions.png', dpi=150)
plt.show()

print("\n🎉 Complete!")