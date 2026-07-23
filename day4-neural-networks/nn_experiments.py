"""
DAY 4: Neural Network Experiments
Testing different architectures and hyperparameters
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd
import seaborn as sns

print("=" * 70)
print("🧪 NEURAL NETWORK EXPERIMENTS")
print("=" * 70)

# ========================================
# 1. LOAD AND PREPARE DATA
# ========================================

print("\n📊 1. LOADING MNIST DATA")

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and flatten
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ========================================
# 2. DEFINE EXPERIMENTS
# ========================================

print("\n🧪 2. DEFINING EXPERIMENTS")

experiments = {
    # Experiment 1: Baseline (our original model)
    'baseline': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 2: Deeper network
    'deeper': {
        'layers': [256, 128, 64, 32],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 3: Wider network
    'wider': {
        'layers': [512, 256],
        'activation': 'relu',
        'dropout': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 4: Different activation
    'tanh': {
        'layers': [128, 64],
        'activation': 'tanh',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 5: Higher dropout (more regularization)
    'dropout_high': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.5,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 6: Different optimizer
    'sgd': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 7: Lower learning rate (slower learning)
    'lr_low': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 8: Higher learning rate (faster but risky)
    'lr_high': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 10
    },
    
    # Experiment 9: Smaller batch size
    'batch_small': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 10
    },
    
    # Experiment 10: Larger batch size
    'batch_large': {
        'layers': [128, 64],
        'activation': 'relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 10
    },

    # Experiment 11: My Design, My Architecture.
    'my_custom': {
        'layers': [256, 128, 64],
        'activation': 'relu',
        'dropout': 0.5,
        'optimizer': 'adam',
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 10
    },

    # Experiment 12: My Design, My Architecture.
    'my_custom1': {
        'layers': [256, 128, 64],
        'activation': 'tanh',
        'dropout': 0.3,
        'optimizer': 'sgd',
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10
        },

    # Experiment 13: My Design, My Architecture.
    'my_custom2': {
        'layers': [256, 128, 64],
        'activation': 'sigmoid',
        'dropout': 0.2,
        'optimizer': 'rmsprop',
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 10
   },

   # Experiment 14: Very deep network (dangerous!)
    'very_deep': {
        'layers': [256, 128, 64, 32, 16],
        'activation': 'relu',
        'dropout': 0.3,
        'optimizer': 'adam',
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 10
    },

    # Experiment 15: Very wide network (computationally expensive)
    'very_wide': {
        'layers': [1024, 512],
        'activation': 'relu',
        'dropout': 0.4,
        'optimizer': 'adam',
        'learning_rate': 0.0005,
        'batch_size': 64,
        'epochs': 10
    },

    # Experiment 16: Leaky ReLU (advanced activation)
    'leaky_relu': {
        'layers': [128, 64],
        'activation': 'leaky_relu',
        'dropout': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    }
}

# ========================================
# 3. RUN EXPERIMENTS
# ========================================

print("\n🏃 3. RUNNING EXPERIMENTS")

def build_model(config):
    """Build model from configuration"""
    model = Sequential()
    
    # Input layer
    model.add(layers.Dense(config['layers'][0], 
                          activation=config['activation'], 
                          input_shape=(784,)))
    model.add(layers.Dropout(config['dropout']))
    
    # Hidden layers
    for units in config['layers'][1:]:
        model.add(layers.Dense(units, activation=config['activation']))
        model.add(layers.Dropout(config['dropout']))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

def get_optimizer(name, learning_rate):
    """Get optimizer by name"""
    optimizers = {
        'adam': Adam,
        'sgd': SGD,
        'rmsprop': RMSprop
    }
    return optimizers.get(name, Adam)(learning_rate=learning_rate)

results = []

for exp_name, config in experiments.items():
    print(f"\n🧪 Running: {exp_name}")
    print(f"   Layers: {config['layers']}")
    print(f"   Activation: {config['activation']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Batch Size: {config['batch_size']}")
    
    # Build model
    model = build_model(config)
    
    # Compile
    optimizer = get_optimizer(config['optimizer'], config['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    
    # Time the training
    start_time = time.time()
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop]
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    epochs_completed = len(history.history['accuracy'])
    
    # Store results
    results.append({
        'Experiment': exp_name,
        'Layers': str(config['layers']),
        'Activation': config['activation'],
        'Dropout': config['dropout'],
        'Optimizer': config['optimizer'],
        'Learning Rate': config['learning_rate'],
        'Batch Size': config['batch_size'],
        'Test Accuracy': test_accuracy * 100,
        'Best Val Accuracy': best_val_acc * 100,
        'Final Train Accuracy': final_train_acc * 100,
        'Epochs': epochs_completed,
        'Training Time (s)': training_time
    })
    
    print(f"   ✅ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   ⏱️ Time: {training_time:.1f}s")

# ========================================
# 4. DISPLAY RESULTS
# ========================================

print("\n" + "=" * 70)
print("📊 4. RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df[['Experiment', 'Test Accuracy', 'Best Val Accuracy', 
                  'Epochs', 'Training Time (s)']].to_string(index=False))

# ========================================
# 5. FIND BEST PERFORMERS
# ========================================

print("\n🏆 5. BEST PERFORMING MODELS")

best_by_accuracy = results_df.sort_values('Test Accuracy', ascending=False)

print("\nTop 3 by Test Accuracy:")
for i, row in best_by_accuracy.head(3).iterrows():
    print(f"  {i+1}. {row['Experiment']}: {row['Test Accuracy']:.2f}%")

print("\nFastest Models (top 3):")
fastest = results_df.sort_values('Training Time (s)')
for i, row in fastest.head(3).iterrows():
    print(f"  {i+1}. {row['Experiment']}: {row['Training Time (s)']:.1f}s")

# ========================================
# 6. VISUALIZE RESULTS
# ========================================

print("\n📊 6. VISUALIZING RESULTS")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Test Accuracy Bar Chart
ax = axes[0, 0]
colors = ['#6C63FF' if x > 98 else '#FF6B6B' if x < 90 else '#FBBF24' 
          for x in results_df['Test Accuracy']]
bars = ax.bar(results_df['Experiment'], results_df['Test Accuracy'], color=colors)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy by Experiment')
ax.set_ylim(80, 100)
ax.axhline(y=results_df['Test Accuracy'].mean(), color='red', linestyle='--', 
           label=f"Mean: {results_df['Test Accuracy'].mean():.1f}%")
ax.legend()
ax.tick_params(axis='x', rotation=45)

# 2. Training Time vs Accuracy
ax = axes[0, 1]
scatter = ax.scatter(results_df['Training Time (s)'], 
                     results_df['Test Accuracy'],
                     s=100, c=results_df['Epochs'], cmap='viridis')
ax.set_xlabel('Training Time (seconds)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Accuracy vs Training Time')
plt.colorbar(scatter, ax=ax, label='Epochs')

# Add labels for each point
for i, row in results_df.iterrows():
    ax.annotate(row['Experiment'], 
                (row['Training Time (s)'], row['Test Accuracy']),
                fontsize=8, alpha=0.7)

# 3. Heatmap of Top Features
ax = axes[1, 0]
feature_cols = ['Test Accuracy', 'Best Val Accuracy', 'Final Train Accuracy', 'Epochs']
corr = results_df[feature_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')

# 4. Experiment Comparison (Accuracy vs Layers, Dropout)
ax = axes[1, 1]
results_df['Layer_Count'] = results_df['Layers'].apply(lambda x: len(eval(x)))
scatter = ax.scatter(results_df['Layer_Count'], 
                     results_df['Dropout'],
                     s=results_df['Test Accuracy']*2,
                     c=results_df['Test Accuracy'],
                     cmap='viridis')
ax.set_xlabel('Number of Layers')
ax.set_ylabel('Dropout Rate')
ax.set_title('Accuracy vs Architecture (Size = Accuracy)')
plt.colorbar(scatter, ax=ax, label='Test Accuracy (%)')

plt.tight_layout()
plt.savefig('experiment_results.png', dpi=150)
print("✅ Results visualization saved as 'experiment_results.png'")
plt.show()

# ========================================
# 7. DETAILED COMPARISON
# ========================================

print("\n📋 7. DETAILED COMPARISON")

# Group by activation function
print("\nBy Activation Function:")
activation_summary = results_df.groupby('Activation')['Test Accuracy'].agg(['mean', 'max', 'min'])
print(activation_summary.round(2))

# Group by optimizer
print("\nBy Optimizer:")
optimizer_summary = results_df.groupby('Optimizer')['Test Accuracy'].agg(['mean', 'max', 'min'])
print(optimizer_summary.round(2))

# Group by dropout rate
print("\nBy Dropout Rate:")
dropout_summary = results_df.groupby('Dropout')['Test Accuracy'].agg(['mean', 'max', 'min'])
print(dropout_summary.round(2))

# ========================================
# 8. BEST MODEL ANALYSIS
# ========================================

print("\n🏆 8. BEST MODEL ANALYSIS")

best_exp = results_df.loc[results_df['Test Accuracy'].idxmax()]
print(f"Best Experiment: {best_exp['Experiment']}")
print(f"Test Accuracy: {best_exp['Test Accuracy']:.2f}%")
print(f"Configuration:")
for key in ['Layers', 'Activation', 'Dropout', 'Optimizer', 'Learning Rate', 'Batch Size']:
    print(f"  {key}: {best_exp[key]}")

# ========================================
# 9. SAVE RESULTS
# ========================================

print("\n💾 9. SAVING RESULTS")

results_df.to_csv('experiment_results.csv', index=False)
print("✅ Results saved to 'experiment_results.csv'")

print("\n" + "=" * 70)
print("🎉 EXPERIMENTS COMPLETE!")
print("=" * 70)

print("""
💡 KEY INSIGHTS:
1. Deeper networks don't always perform better
2. Adam optimizer generally works best
3. Higher dropout prevents overfitting
4. Learning rate is crucial - too high or too low hurts performance
5. Batch size affects training speed and stability
""")