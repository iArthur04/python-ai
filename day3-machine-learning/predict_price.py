
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
print(f"\nPredicted price: ${price:,.0f}")
