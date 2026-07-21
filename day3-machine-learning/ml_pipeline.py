"""
DAY 3: Complete ML Pipeline
From data to prediction - everything in one place
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

print("=" * 60)
print("🚀 END-TO-END ML PIPELINE")
print("=" * 60)

class MLPipeline:
    """Complete machine learning pipeline"""
    
    def __init__(self, task='regression'):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def load_data(self, data_dict):
        """Load data from dictionary"""
        self.df = pd.DataFrame(data_dict)
        print(f"✅ Data loaded: {len(self.df)} samples, {len(self.df.columns)} columns")
        return self
    
    def create_features(self, target_col):
        """Prepare features and target"""
        self.target_col = target_col
        self.X = self.df.drop(target_col, axis=1)
        self.y = self.df[target_col]
        print(f"✅ Features: {self.X.shape[1]} features")
        print(f"✅ Target: {target_col}")
        return self
    
    def split_data(self, test_size=0.2):
        """Split into train/test sets"""
        # Use stratify for classification to keep class balance
        stratify = self.y if self.task == 'classification' else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=stratify
        )
        print(f"✅ Training set: {len(self.X_train)} samples")
        print(f"✅ Test set: {len(self.X_test)} samples")
        return self
    
    def scale_data(self):
        """Scale features"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("✅ Data scaled")
        return self
    
    def train_model(self, model_type='regression'):
        """Train the model"""
        if model_type == 'regression':
            self.model = LinearRegression()
        else:
            self.model = LogisticRegression(max_iter=1000)
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print(f"✅ {model_type} model trained!")
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        y_pred = self.model.predict(self.X_test_scaled)
        
        if self.task == 'regression':
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            self.results = {
                'MSE': mse,
                'R2': r2,
                'Predictions': y_pred
            }
            print(f"📊 MSE: {mse:.2f}")
            print(f"📊 R² Score: {r2:.4f}")
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results = {
                'Accuracy': accuracy,
                'Predictions': y_pred
            }
            print(f"📊 Accuracy: {accuracy*100:.1f}%")
            
            # Show classification report if multiple classes exist
            unique_classes = np.unique(self.y_test)
            if len(unique_classes) > 1:
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred))
        
        return self
    
    def predict(self, features):
        """Make a prediction"""
        features_scaled = self.scaler.transform([list(features.values())])
        prediction = self.model.predict(features_scaled)[0]
        return prediction
    
    def save_model(self, filename):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.X.columns.tolist(),
            'task': self.task,
            'results': self.results
        }, filename)
        print(f"✅ Model saved to {filename}")
        return self
    
    def load_model(self, filename):
        """Load model from file"""
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.task = data['task']
        self.results = data['results']
        print(f"✅ Model loaded from {filename}")
        return self
    
    def visualize(self):
        """Visualize results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if self.task == 'regression':
            # Actual vs Predicted
            axes[0].scatter(self.y_test, self.results['Predictions'], alpha=0.6)
            axes[0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--')
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title('Actual vs Predicted')
            
            # Residuals
            residuals = self.y_test - self.results['Predictions']
            axes[1].hist(residuals, bins=20)
            axes[1].set_xlabel('Residual')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Residual Distribution')
        else:
            # Classification visualization
            axes[0].bar(['0', '1'], [sum(self.y_test == 0), sum(self.y_test == 1)])
            axes[0].set_title('Actual Distribution')
            axes[1].bar(['0', '1'], 
                       [sum(self.results['Predictions'] == 0), 
                        sum(self.results['Predictions'] == 1)])
            axes[1].set_title('Predicted Distribution')
        
        plt.tight_layout()
        plt.savefig('pipeline_results.png', dpi=150)
        print("✅ Visualization saved as 'pipeline_results.png'")
        plt.show()
        return self

# ========================================
# 🔥 DEMO 1: REGRESSION PIPELINE
# ========================================

print("\n" + "=" * 60)
print("📊 DEMO 1: REGRESSION PIPELINE")
print("=" * 60)

# Create synthetic regression data
np.random.seed(42)
n_samples = 300

regression_data = {
    'Square_Feet': np.random.uniform(800, 3500, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Age': np.random.randint(0, 50, n_samples),
    'Location_Score': np.random.uniform(1, 10, n_samples)
}

# Create price with realistic relationships
regression_data['Price'] = (
    regression_data['Square_Feet'] * 150 +
    regression_data['Bedrooms'] * 5000 +
    regression_data['Location_Score'] * 10000 -
    regression_data['Age'] * 100 +
    np.random.normal(0, 20000, n_samples)
)
regression_data['Price'] = np.maximum(regression_data['Price'], 50000)

print("\n🚀 Running regression pipeline...")
reg_pipeline = MLPipeline(task='regression')
reg_pipeline.load_data(regression_data)
reg_pipeline.create_features('Price')
reg_pipeline.split_data(0.2)
reg_pipeline.scale_data()
reg_pipeline.train_model('regression')
reg_pipeline.evaluate()
reg_pipeline.visualize()

# Make a sample prediction
sample_house = {
    'Square_Feet': 2000,
    'Bedrooms': 3,
    'Age': 10,
    'Location_Score': 7.5
}
predicted_price = reg_pipeline.predict(sample_house)
print(f"\n🏠 Sample house prediction: ${predicted_price:,.0f}")

# Save model
reg_pipeline.save_model('regression_pipeline.pkl')

# ========================================
# 🔥 DEMO 2: CLASSIFICATION PIPELINE
# ========================================

print("\n" + "=" * 60)
print("📊 DEMO 2: CLASSIFICATION PIPELINE")
print("=" * 60)

# Create synthetic classification data
np.random.seed(42)
n_students = 500

classification_data = {
    'Hours_Studied': np.random.uniform(0, 15, n_students),
    'Attendance': np.random.uniform(50, 100, n_students),
    'Assignments_Done': np.random.randint(0, 12, n_students),
    'Sleep_Hours': np.random.uniform(3, 9, n_students)
}

# Create target (pass/fail)
score = (classification_data['Hours_Studied'] * 3 + 
         classification_data['Attendance'] * 0.5 + 
         classification_data['Assignments_Done'] * 5 + 
         classification_data['Sleep_Hours'] * 2)
score = score + np.random.normal(0, 10, n_students)
classification_data['Passed'] = (score > 55).astype(int)

print("\n🚀 Running classification pipeline...")
class_pipeline = MLPipeline(task='classification')
class_pipeline.load_data(classification_data)
class_pipeline.create_features('Passed')
class_pipeline.split_data(0.2)
class_pipeline.scale_data()
class_pipeline.train_model('classification')
class_pipeline.evaluate()
class_pipeline.visualize()

# Predict a student
sample_student = {
    'Hours_Studied': 8,
    'Attendance': 90,
    'Assignments_Done': 10,
    'Sleep_Hours': 7
}
pass_pred = class_pipeline.predict(sample_student)
prob = class_pipeline.model.predict_proba(class_pipeline.scaler.transform([list(sample_student.values())]))[0][1]
print(f"\n🎓 Student will {'PASS' if pass_pred == 1 else 'FAIL'}")
print(f"   Probability of passing: {prob*100:.1f}%")

# Save model
class_pipeline.save_model('classification_pipeline.pkl')

# ========================================
# ✅ COMPLETE!
# ========================================

print("\n" + "=" * 60)
print("🎉 COMPLETE ML PIPELINE DEMO FINISHED!")
print("=" * 60)
print("\n📁 Files created:")
print("  - regression_pipeline.pkl (saved model)")
print("  - classification_pipeline.pkl (saved model)")
print("  - pipeline_results.png (visualization)")
print("\n💡 Try loading a saved model:")
print("  pipeline = MLPipeline()")
print("  pipeline.load_model('regression_pipeline.pkl')")