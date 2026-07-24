"""
DAY 5: Custom Classifier Builder
Build your own classifier with your images!
"""

import os
import shutil
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

print("=" * 70)
print("🔧 CUSTOM CLASSIFIER BUILDER")
print("=" * 70)

print("""
📁 HOW TO USE:
1. Create folders:
   - dataset/train/cats/
   - dataset/train/dogs/
   - dataset/validation/cats/
   - dataset/validation/dogs/

2. Put your images in the folders

3. Run this script!

📷 Supported formats: JPG, JPEG, PNG

🎯 This will create a classifier for YOUR specific use case!
""")

class CustomClassifier:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.model = None
        self.history = None
        
    def prepare_data(self):
        """Prepare data generators"""
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        self.train_generator = train_datagen.flow_from_directory(
            f'{self.dataset_path}/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            f'{self.dataset_path}/validation',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        self.num_classes = len(self.train_generator.class_indices)
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"✅ Found {self.num_classes} classes: {self.class_names}")
        print(f"✅ Training: {self.train_generator.samples} images")
        print(f"✅ Validation: {self.val_generator.samples} images")
        
    def build_model(self):
        """Build classifier using MobileNetV2"""
        # Load pre-trained model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Add custom head
        self.model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✅ Model built!")
        self.model.summary()
        
    def train(self, epochs=20):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
            callbacks=callbacks,
            verbose=1
        )
        
    def evaluate(self):
        """Evaluate the model"""
        loss, accuracy = self.model.evaluate(self.val_generator)
        print(f"\n📊 Validation Accuracy: {accuracy*100:.2f}%")
        return accuracy
    
    def save(self, filename='custom_classifier.h5'):
        """Save the model"""
        self.model.save(filename)
        print(f"✅ Model saved as '{filename}'")
    
    def predict(self, img_path):
        """Predict class of an image"""
        from tensorflow.keras.preprocessing import image
        import matplotlib.pyplot as plt
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        print(f"\n🔮 Prediction: {self.class_names[predicted_class]}")
        print(f"📊 Confidence: {confidence:.1f}%")
        
        # Show the image
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.title(f"Prediction: {self.class_names[predicted_class]}")
        plt.axis('off')
        plt.show()
        
        return self.class_names[predicted_class], confidence
    
    def plot_history(self):
        """Plot training history"""
        if not self.history:
            print("⚠️ No training history yet!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history.history['accuracy'], label='Training')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['loss'], label='Training')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('custom_classifier_history.png', dpi=150)
        plt.show()

# ========================================
# Run the classifier
# ========================================

if __name__ == "__main__":
    try:
        classifier = CustomClassifier()
        classifier.prepare_data()
        classifier.build_model()
        classifier.train(epochs=10)
        classifier.evaluate()
        classifier.save()
        classifier.plot_history()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("""
💡 Make sure you have images in:
   dataset/train/cats/
   dataset/train/dogs/
   dataset/validation/cats/
   dataset/validation/gods/
""")