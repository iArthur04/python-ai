"""
DAY 4: MNIST Handwritten Digit Recognition
Your first real AI application using TensorFlow/Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ FIXED: Import from tensorflow.keras directly
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 60)
print("✍️ MNIST HANDWRITTEN DIGIT RECOGNITION")
print("=" * 60)

# ========================================
# 1. LOAD THE DATA
# ========================================

print("\n📊 1. LOADING MNIST DATASET")

# ✅ FIXED: Use tensorflow.keras.datasets
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training images: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test images: {X_test.shape}")
print(f"Test labels: {y_test.shape}")
print(f"Image size: {X_train[0].shape}")
print(f"Pixel values: {X_train[0].min()} to {X_train[0].max()}")

# ========================================
# 2. EXPLORE THE DATA
# ========================================

print("\n🔍 2. EXPLORING THE DATA")

# Show some examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f'Digit: {y_train[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150)
print("✅ Sample images saved as 'mnist_samples.png'")
plt.show()

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution in training set:")
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} images")

# ========================================
# 3. PREPARE DATA FOR NEURAL NETWORK
# ========================================

print("\n🔧 3. PREPARING DATA")

# Normalize pixel values to 0-1 range
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Reshape for neural network (flatten 28x28 to 784)
X_train_flattened = X_train_normalized.reshape(X_train.shape[0], -1)
X_test_flattened = X_test_normalized.reshape(X_test.shape[0], -1)

print(f"Training data shape: {X_train_flattened.shape}")
print(f"Test data shape: {X_test_flattened.shape}")

# ========================================
# 4. BUILD THE NEURAL NETWORK
# ========================================

print("\n🧠 4. BUILDING THE NEURAL NETWORK")

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # Prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 digits (0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================================
# 5. TRAIN THE MODEL
# ========================================

print("\n🏋️ 5. TRAINING THE MODEL")

history = model.fit(
    X_train_flattened,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ========================================
# 6. EVALUATE THE MODEL
# ========================================

print("\n📈 6. EVALUATING THE MODEL")

test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("✅ Training history saved as 'training_history.png'")
plt.show()

# ========================================
# 7. CONFUSION MATRIX
# ========================================

print("\n📊 7. CONFUSION MATRIX")

y_pred = model.predict(X_test_flattened)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)
print("✅ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# ========================================
# 8. SAVE THE MODEL
# ========================================

print("\n💾 8. SAVING THE MODEL")

model.save('mnist_model.h5')
print("✅ Model saved as 'mnist_model.h5'")

# ========================================
# 9. MAKE PREDICTIONS ON CUSTOM IMAGES
# ========================================

print("\n🎯 9. MAKING PREDICTIONS")

# Find some test images
correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_pred_classes[i] == y_test[i]:
        correct_predictions.append(i)
    else:
        wrong_predictions.append(i)
    
    if len(correct_predictions) >= 5 and len(wrong_predictions) >= 5:
        break

# Show correct predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(correct_predictions[:5]):
    axes[i].imshow(X_test[idx], cmap='gray')
    axes[i].set_title(f'True: {y_test[idx]}\nPred: {y_pred_classes[idx]}', color='green')
    axes[i].axis('off')
plt.suptitle('✅ Correct Predictions', fontsize=16)
plt.tight_layout()
plt.savefig('correct_predictions.png', dpi=150)
plt.show()

# Show wrong predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(wrong_predictions[:5]):
    axes[i].imshow(X_test[idx], cmap='gray')
    axes[i].set_title(f'True: {y_test[idx]}\nPred: {y_pred_classes[idx]}', color='red')
    axes[i].axis('off')
plt.suptitle('❌ Wrong Predictions', fontsize=16)
plt.tight_layout()
plt.savefig('wrong_predictions.png', dpi=150)
plt.show()

print("\n🎉 MNIST Classifier Complete!")
print(f"Accuracy: {test_accuracy*100:.2f}%")