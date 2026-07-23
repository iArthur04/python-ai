"""
DAY 4: Visualizing Neural Network Predictions
Understanding what the neural network sees
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print("=" * 60)
print("🎨 VISUALIZING NEURAL NETWORK PREDICTIONS")
print("=" * 60)

# ========================================
# 1. LOAD THE SAVED MODEL
# ========================================

print("\n📂 1. LOADING SAVED MODEL")

try:
    model = keras.models.load_model('mnist_model.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ Model not found! Run mnist_classifier.py first.")
    exit()

# Load MNIST data
mnist = keras.datasets.mnist
(_, _), (X_test, y_test) = mnist.load_data()

# ========================================
# 2. PREDICT ON MULTIPLE IMAGES
# ========================================

print("\n🔮 2. PREDICTING ON 20 IMAGES")

n_images = 20
X_test_flattened = X_test[:n_images].reshape(n_images, -1) / 255.0
predictions = model.predict(X_test_flattened)
predicted_classes = np.argmax(predictions, axis=1)

# Show results
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.ravel()

for i in range(n_images):
    axes[i].imshow(X_test[i], cmap='gray')
    
    confidence = np.max(predictions[i]) * 100
    is_correct = predicted_classes[i] == y_test[i]
    
    color = 'green' if is_correct else 'red'
    axes[i].set_title(f'True: {y_test[i]}\nPred: {predicted_classes[i]}\nConf: {confidence:.1f}%', 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('predictions_grid.png', dpi=150)
print("✅ Prediction grid saved as 'predictions_grid.png'")
plt.show()

# ========================================
# 3. VISUALIZE CONFIDENCE LEVELS
# ========================================

print("\n📊 3. VISUALIZING CONFIDENCE LEVELS")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Show some predictions with their confidence bar chart
for i in range(10):
    ax1 = axes[0, i]
    ax2 = axes[1, i]
    
    # Show the image
    ax1.imshow(X_test[i], cmap='gray')
    ax1.set_title(f'True: {y_test[i]}', fontsize=10)
    ax1.axis('off')
    
    # Show confidence bar chart
    pred = predictions[i]
    classes = range(10)
    ax2.bar(classes, pred * 100, color='#6C63FF')
    ax2.set_xticks(classes)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Confidence %')
    ax2.set_title(f'Pred: {np.argmax(pred)}', fontsize=10)
    
    # Highlight the predicted class
    ax2.bar(np.argmax(pred), np.max(pred) * 100, color='#FF6B6B')
    
    # Remove x-axis labels for clarity
    ax2.set_xticklabels(classes, fontsize=8)

plt.tight_layout()
plt.savefig('confidence_bars.png', dpi=150)
print("✅ Confidence bars saved as 'confidence_bars.png'")
plt.show()

# ========================================
# 4. FIND THE CONFIDENT AND UNCERTAIN PREDICTIONS
# ========================================

print("\n🎯 4. FINDING CONFIDENT AND UNCERTAIN PREDICTIONS")

# Get confidences for all test images
n_test = 1000  # Use subset for speed
X_subset = X_test[:n_test].reshape(n_test, -1) / 255.0
y_subset = y_test[:n_test]
predictions_subset = model.predict(X_subset)
confidences = np.max(predictions_subset, axis=1)
pred_classes = np.argmax(predictions_subset, axis=1)

# Find most confident predictions
most_confident_idx = np.argsort(confidences)[-5:]
least_confident_idx = np.argsort(confidences)[:5]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Most confident
for i, idx in enumerate(most_confident_idx):
    axes[0, i].imshow(X_test[idx], cmap='gray')
    axes[0, i].set_title(f'Conf: {confidences[idx]*100:.1f}%\nPred: {pred_classes[idx]}', color='green')
    axes[0, i].axis('off')
axes[0, 0].set_ylabel('Most Confident', fontsize=12)

# Least confident
for i, idx in enumerate(least_confident_idx):
    axes[1, i].imshow(X_test[idx], cmap='gray')
    axes[1, i].set_title(f'Conf: {confidences[idx]*100:.1f}%\nPred: {pred_classes[idx]}', color='red')
    axes[1, i].axis('off')
axes[1, 0].set_ylabel('Least Confident', fontsize=12)

plt.tight_layout()
plt.savefig('confidence_extremes.png', dpi=150)
print("✅ Confidence extremes saved as 'confidence_extremes.png'")
plt.show()

# ========================================
# 5. ANALYZE MISTAKES
# ========================================

print("\n🔍 5. ANALYZING MISTAKES")

wrong_predictions = []
for i in range(n_test):
    if pred_classes[i] != y_subset[i]:
        wrong_predictions.append(i)

print(f"Total wrong predictions: {len(wrong_predictions)} out of {n_test}")
print(f"Error rate: {len(wrong_predictions)/n_test*100:.1f}%")

if len(wrong_predictions) > 0:
    print("\nFirst 10 mistakes:")
    for i in wrong_predictions[:10]:
        print(f"  Image {i}: True={y_subset[i]}, Pred={pred_classes[i]}, Conf={confidences[i]*100:.1f}%")

# Show mistake patterns
mistake_pairs = {}
for idx in wrong_predictions:
    pair = (y_subset[idx], pred_classes[idx])
    mistake_pairs[pair] = mistake_pairs.get(pair, 0) + 1

print("\nMost common mistakes:")
for pair, count in sorted(mistake_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {pair[0]} → {pair[1]}: {count} times")

print("\n🎉 Visualization complete!")