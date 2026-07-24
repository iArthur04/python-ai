"""
DAY 5: Custom Image Classifier
Using Transfer Learning to classify ANYTHING!
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shutil

print("=" * 70)
print("📸 CUSTOM IMAGE CLASSIFIER")
print("=" * 70)

# ========================================
# 1. CREATE SAMPLE DATASET
# ========================================

print("\n📊 1. CREATING SAMPLE DATASET")

# Since we don't have a real dataset, we'll create synthetic data
# In practice, you'd use real images from Kaggle or your own collection

def create_sample_dataset():
    """Create a synthetic dataset for demonstration"""
    print("Creating synthetic dataset...")
    
    # Create folder structure
    os.makedirs('dataset/train/cats', exist_ok=True)
    os.makedirs('dataset/train/dogs', exist_ok=True)
    os.makedirs('dataset/validation/cats', exist_ok=True)
    os.makedirs('dataset/validation/dogs', exist_ok=True)
    
    # Generate synthetic images (random noise)
    # In real use, you'd copy actual images
    n_train = 100  # Small dataset for demonstration
    n_val = 20
    
    for i in range(n_train):
        # Random noise as "images"
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        if i < n_train // 2:
            plt.imsave(f'dataset/train/cats/cat_{i}.jpg', img)
        else:
            plt.imsave(f'dataset/train/dogs/dog_{i}.jpg', img)
    
    for i in range(n_val):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        if i < n_val // 2:
            plt.imsave(f'dataset/validation/cats/cat_{i}.jpg', img)
        else:
            plt.imsave(f'dataset/validation/dogs/dog_{i}.jpg', img)
    
    print(f"✅ Dataset created: {n_train} training, {n_val} validation images")
    print("   (Using synthetic images - replace with real images for actual use)")

# ========================================
# 2. DATA AUGMENTATION
# ========================================

print("\n🔄 2. DATA AUGMENTATION")

def setup_data_generators():
    """Set up data generators with augmentation"""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data from directories
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

# ========================================
# 3. BUILD MODEL WITH TRANSFER LEARNING
# ========================================

print("\n🧠 3. BUILDING TRANSFER LEARNING MODEL")

def build_transfer_model(num_classes=1):
    """Build a model using ResNet50 as feature extractor"""
    
    # Load ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    print(f"Base model loaded: {base_model.name}")
    print(f"Trainable layers: {len(base_model.layers)}")
    
    # Add custom classification head
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# ========================================
# 4. TRAIN THE MODEL
# ========================================

print("\n🏋️ 4. TRAINING THE MODEL")

def train_model(model, train_generator, validation_generator):
    """Train the model"""
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ========================================
# 5. VISUALIZE RESULTS
# ========================================

print("\n📊 5. VISUALIZING RESULTS")

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_training_history.png', dpi=150)
    print("✅ Training history saved as 'transfer_training_history.png'")
    plt.show()

# ========================================
# 6. EVALUATE & PREDICT
# ========================================

print("\n🎯 6. EVALUATING & MAKING PREDICTIONS")

def evaluate_model(model, validation_generator):
    """Evaluate model performance"""
    loss, accuracy = model.evaluate(validation_generator)
    print(f"\n📊 Validation Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Validation Loss: {loss:.4f}")
    return accuracy

# ========================================
# 7. SAVE THE MODEL
# ========================================

print("\n💾 7. SAVING THE MODEL")

def save_model(model, filename='custom_classifier.h5'):
    """Save the trained model"""
    model.save(filename)
    print(f"✅ Model saved as '{filename}'")

# ========================================
# 8. MAKE PREDICTIONS ON NEW IMAGES
# ========================================

print("\n🔮 8. PREDICT ON NEW IMAGES")

def predict_new_image(model, img_path):
    """Predict class of a new image"""
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Display
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(f"Prediction: {'Cat' if prediction < 0.5 else 'Dog'} ({prediction*100:.1f}% confidence)")
    plt.axis('off')
    plt.show()
    
    return prediction

# ========================================
# 9. MAIN EXECUTION
# ========================================

print("\n🚀 9. RUNNING THE PIPELINE")

try:
    # Create dataset
    create_sample_dataset()
    
    # Setup generators
    train_gen, val_gen = setup_data_generators()
    
    # Build model
    model = build_transfer_model()
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Visualize
    plot_training_history(history)
    
    # Evaluate
    accuracy = evaluate_model(model, val_gen)
    
    # Save
    save_model(model)
    
    print("\n" + "=" * 70)
    print("🎉 TRANSFER LEARNING CLASSIFIER COMPLETE!")
    print("=" * 70)
    print("""
💡 NEXT STEPS:
1. Get real images (Kaggle, your own photos)
2. Organize them in dataset/train/ and dataset/validation/
3. Run this script again!
4. Your model will learn to recognize YOUR classes!
""")
    
except Exception as e:
    print(f"⚠️ Note: {e}")
    print("\n💡 This is a demonstration. To use with real images:")
    print("1. Download a dataset from Kaggle (e.g., cats vs dogs)")
    print("2. Organize images in dataset/train/class1/, dataset/train/class2/")
    print("3. Run this script again!")