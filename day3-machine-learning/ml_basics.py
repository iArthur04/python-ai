"""
DAY 3: Machine Learning Basics
Your First Steps into AI!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("🤖 MACHINE LEARNING BASICS")
print("=" * 60)

print("""
What is Machine Learning?
------------------------
Machine Learning is teaching computers to learn from data
without being explicitly programmed.

Types of ML:
1. Supervised Learning: Learn from labeled examples
   - Regression: Predict numbers (house prices, exam scores)
   - Classification: Predict categories (spam/not spam)

2. Unsupervised Learning: Find patterns in unlabeled data
   - Clustering: Group similar items

3. Reinforcement Learning: Learn through trial and error
""")

# ========================================
# 1. UNDERSTANDING THE DATA
# ========================================

print("\n📊 1. UNDERSTANDING THE DATA")

# Create simple dataset
# X = hours studied, y = exam score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

print(f"Hours studied: {X.flatten()}")
print(f"Exam scores: {y}")

# ========================================
# 2. TRAIN/TEST SPLIT
# ========================================

print("\n🔀 2. TRAIN/TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")

# ========================================
# 3. CREATE AND TRAIN MODEL
# ========================================

print("\n🧠 3. TRAINING THE MODEL")

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model trained!")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Formula: Score = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")

# ========================================
# 4. MAKE PREDICTIONS
# ========================================

print("\n🔮 4. MAKING PREDICTIONS")

y_pred = model.predict(X_test)

print("Predictions vs Actual:")
for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
    print(f"  Sample {i+1}: Actual={actual:.0f}, Predicted={pred:.1f}")

# ========================================
# 5. EVALUATE THE MODEL
# ========================================

print("\n📈 5. EVALUATING THE MODEL")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f} (0-1, higher is better)")

# ========================================
# 6. VISUALIZE THE RESULTS
# ========================================

print("\n📊 6. VISUALIZING THE RESULTS")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Data and regression line
axes[0].scatter(X_train, y_train, color='blue', label='Training data', alpha=0.7)
axes[0].scatter(X_test, y_test, color='green', label='Test data', alpha=0.7)
axes[0].plot(X, model.predict(X), color='red', label='Regression line')
axes[0].set_xlabel('Hours Studied')
axes[0].set_ylabel('Exam Score')
axes[0].set_title('Linear Regression Model')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual
axes[1].scatter(y_test, y_pred, color='purple', alpha=0.7)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
axes[1].set_xlabel('Actual Scores')
axes[1].set_ylabel('Predicted Scores')
axes[1].set_title('Predictions vs Actual')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_basics.png', dpi=150)
print("✅ Chart saved as 'ml_basics.png'")
plt.show()

# ========================================
# 7. MAKE A REAL PREDICTION
# ========================================

print("\n🎯 7. MAKE A REAL PREDICTION")

hours = float(input("\nHow many hours did you study? "))
predicted_score = model.predict([[hours]])[0]
print(f"Predicted exam score: {predicted_score:.1f}%")

if predicted_score >= 70:
    print("✅ Great! You're on track for a good grade!")
elif predicted_score >= 50:
    print("👍 Keep studying! You can improve!")
else:
    print("💪 You need to study more! It's not too late!")