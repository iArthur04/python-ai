"""
DAY 3: Linear Regression - Predict House Prices
A real-world regression problem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 60)
print("🏠 HOUSE PRICE PREDICTION")
print("=" * 60)

# ========================================
# 1. CREATE REALISTIC DATASET
# ========================================

print("\n📊 1. CREATING DATASET")

np.random.seed(42)
n_houses = 200

# Features
square_feet = np.random.uniform(800, 3500, n_houses)
bedrooms = np.random.randint(1, 6, n_houses)
bathrooms = np.random.randint(1, 4, n_houses)
age = np.random.randint(0, 50, n_houses)
location_score = np.random.uniform(1, 10, n_houses)  # 1-10 rating

# Create price (with some noise)
base_price = square_feet * 150
bedroom_bonus = bedrooms * 5000
bathroom_bonus = bathrooms * 8000
location_bonus = location_score * 10000
age_penalty = age * -100

noise = np.random.normal(0, 20000, n_houses)
price = base_price + bedroom_bonus + bathroom_bonus + location_bonus + age_penalty + noise
price = np.maximum(price, 50000)  # Minimum price

# Create DataFrame
df = pd.DataFrame({
    'Square_Feet': square_feet.round(0),
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Location_Score': location_score.round(1),
    'Price': price.round(0)
})

print(f"Dataset created: {len(df)} houses")
print(df.head())

# ========================================
# 2. EXPLORE THE DATA
# ========================================

print("\n🔍 2. EXPLORING THE DATA")

print("Summary statistics:")
print(df.describe())

print(f"\nCorrelation with Price:")
correlations = df.corr()['Price'].sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'Price':
        print(f"  {feature}: {corr:.3f}")

# ========================================
# 3. PREPARE DATA FOR ML
# ========================================

print("\n🔧 3. PREPARING DATA")

# Features (X) and Target (y)
X = df.drop('Price', axis=1)
y = df['Price']

print(f"Features: {X.columns.tolist()}")
print(f"Target: Price")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} houses")
print(f"Test set: {len(X_test)} houses")

# ========================================
# 4. TRAIN THE MODEL
# ========================================

print("\n🧠 4. TRAINING THE MODEL")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")
print("\nFeature coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: ${coef:,.2f}")
print(f"Intercept: ${model.intercept_:,.2f}")

# ========================================
# 5. EVALUATE THE MODEL
# ========================================

print("\n📈 5. EVALUATING THE MODEL")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: ${mse:,.0f}")
print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"R² Score: {r2:.4f}")

print(f"\nModel performance:")
print(f"  Average house price: ${y.mean():,.0f}")
print(f"  Average prediction error: ${mae:,.0f}")
print(f"  Error percentage: {(mae/y.mean()*100):.1f}%")

# ========================================
# 6. PREDICT A HOUSE PRICE
# ========================================

print("\n🏠 6. PREDICT A HOUSE PRICE")

def predict_house_price(features):
    """Predict price for a house"""
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    return model.predict(features_df)[0]

# Example prediction
sample_house = {
    'Square_Feet': 2000,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Age': 10,
    'Location_Score': 7.5
}

predicted_price = predict_house_price(sample_house)
print(f"Sample house features:")
for key, value in sample_house.items():
    print(f"  {key}: {value}")
print(f"\nPredicted price: ${predicted_price:,.0f}")

# ========================================
# 7. VISUALIZE RESULTS
# ========================================

print("\n📊 7. VISUALIZING RESULTS")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Actual vs Predicted Prices')
axes[0, 0].legend()

# 2. Feature Importance
axes[0, 1].barh(X.columns, model.coef_)
axes[0, 1].set_xlabel('Coefficient ($)')
axes[0, 1].set_title('Feature Importance')

# 3. Square Feet vs Price
axes[1, 0].scatter(df['Square_Feet'], df['Price'], alpha=0.5, s=20)
axes[1, 0].set_xlabel('Square Feet')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title('Price vs Square Feet')

# 4. Age vs Price
axes[1, 1].scatter(df['Age'], df['Price'], alpha=0.5, s=20, color='green')
axes[1, 1].set_xlabel('Age (years)')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].set_title('Price vs Age')

plt.tight_layout()
plt.savefig('house_price_prediction.png', dpi=150)
print("✅ Chart saved as 'house_price_prediction.png'")
plt.show()

# ========================================
# 8. SAVE THE MODEL
# ========================================

print("\n💾 8. SAVING THE MODEL")

import joblib

# Save model
joblib.dump(model, 'house_price_model.pkl')
print("✅ Model saved as 'house_price_model.pkl'")

# Create prediction script
with open('predict_price.py', 'w') as f:
    f.write("""
import joblib
import pandas as pd

# Load model
model = joblib.load('house_price_model.pkl')

print("🏠 House Price Predictor")
print("=" * 40)

# Get user input
square_feet = float(input("Square feet: "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
age = int(input("Age (years): "))
location = float(input("Location score (1-10): "))

# Make prediction
features = pd.DataFrame([{
    'Square_Feet': square_feet,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Location_Score': location
}])

price = model.predict(features)[0]
print(f"\\nPredicted price: ${price:,.0f}")
""")
print("✅ Prediction script saved as 'predict_price.py'")