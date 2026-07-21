"""
DAY 3: Classification - Predict Student Success
Categorizing students as likely to succeed or not
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("🎓 STUDENT SUCCESS CLASSIFIER")
print("=" * 60)

# ========================================
# 1. CREATE DATASET
# ========================================

print("\n📊 1. CREATING DATASET")

np.random.seed(42)
n_students = 500

hours_studied = np.random.uniform(0, 15, n_students)
attendance = np.random.uniform(50, 100, n_students)
assignments_done = np.random.randint(0, 12, n_students)
sleep_hours = np.random.uniform(3, 9, n_students)

score = (hours_studied * 3 + attendance * 0.5 + assignments_done * 5 + sleep_hours * 2)
score = score + np.random.normal(0, 10, n_students)

threshold = 55
passed = (score > threshold).astype(int)

df = pd.DataFrame({
    'Hours_Studied': hours_studied.round(1),
    'Attendance': attendance.round(1),
    'Assignments_Done': assignments_done,
    'Sleep_Hours': sleep_hours.round(1),
    'Passed': passed
})

print(f"Dataset created: {len(df)} students")
print(df.head())

# ========================================
# 2. EXPLORE DATA
# ========================================

print("\n🔍 2. EXPLORING DATA")

print("Distribution of pass/fail:")
print(df['Passed'].value_counts())
print(f"Pass rate: {df['Passed'].mean()*100:.1f}%")

print("\nFeatures by outcome:")
for feature in ['Hours_Studied', 'Attendance', 'Assignments_Done', 'Sleep_Hours']:
    passed_avg = df[df['Passed'] == 1][feature].mean()
    failed_avg = df[df['Passed'] == 0][feature].mean()
    print(f"  {feature}:")
    print(f"    Passed: {passed_avg:.1f}")
    print(f"    Failed: {failed_avg:.1f}")

# ========================================
# 3. PREPARE DATA
# ========================================

print("\n🔧 3. PREPARING DATA")

X = df.drop('Passed', axis=1)
y = df['Passed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

# ========================================
# 4. TRAIN CLASSIFIER
# ========================================

print("\n🧠 4. TRAINING CLASSIFIER")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained!")

# ========================================
# 5. EVALUATE MODEL (FIXED — ALWAYS WORKS)
# ========================================

print("\n📈 5. EVALUATING MODEL")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.1f}%")

# ✅ Get unique classes in test set
unique_classes = np.unique(y_test)
print(f"\nClasses in test data: {unique_classes}")

# ✅ Always generate a valid classification report
print("\nClassification Report:")

if len(unique_classes) == 1:
    # Only one class in test data — show simplified but useful report
    print(f"⚠️ Only class '{unique_classes[0]}' appears in test data.")
    print(f"All predictions are class: {unique_classes[0]}")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("\n📋 Since only one class is present, a full classification report is not available.")
    print("💡 Tip: Try increasing dataset size or adjusting the threshold.")
else:
    # Normal case — multiple classes
    print(classification_report(y_test, y_pred, target_names=['Failed', 'Passed']))

# ✅ Confusion matrix — works for any case
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("               Predicted")
print("              Failed  Passed")

if cm.shape == (2, 2):
    print(f"Actual Failed   {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Passed   {cm[1,0]:3d}     {cm[1,1]:3d}")
else:
    print(f"Single-class matrix:\n{cm}")

# ========================================
# 6. FEATURE IMPORTANCE
# ========================================

print("\n💡 6. FEATURE IMPORTANCE")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)

print("Feature importance:")
for _, row in coefficients.iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: {direction} chance of passing")

# ========================================
# 7. PREDICT STUDENT SUCCESS
# ========================================

print("\n🎯 7. PREDICT STUDENT SUCCESS")

def predict_student(features):
    features_scaled = scaler.transform([list(features.values())])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction, probability

test_student = {
    'Hours_Studied': 8,
    'Attendance': 90,
    'Assignments_Done': 10,
    'Sleep_Hours': 7
}

prediction, probability = predict_student(test_student)
print(f"Student features:")
for key, value in test_student.items():
    print(f"  {key}: {value}")
print(f"\nPrediction: {'✅ Pass' if prediction == 1 else '❌ Fail'}")
print(f"Probability of passing: {probability*100:.1f}%")

# ========================================
# 8. VISUALIZE RESULTS
# ========================================

print("\n📊 8. VISUALIZING RESULTS")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, feature in enumerate(['Hours_Studied', 'Attendance']):
    ax = axes[0, i]
    passed = df[df['Passed'] == 1][feature]
    failed = df[df['Passed'] == 0][feature]
    
    ax.hist(passed, bins=20, alpha=0.5, label='Passed', color='green')
    ax.hist(failed, bins=20, alpha=0.5, label='Failed', color='red')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(f'{feature} Distribution')
    ax.legend()

im = axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Failed', 'Passed'])
axes[1, 0].set_yticklabels(['Failed', 'Passed'])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center')

axes[1, 1].barh(coefficients['Feature'], coefficients['Importance'])
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150)
print("✅ Chart saved as 'classification_results.png'")
plt.show()

print("\n🎉 Classification model complete!")