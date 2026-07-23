"""
DAY 4: Neural Network from Scratch
Building a neural network using only NumPy!
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("🧠 BUILDING A NEURAL NETWORK FROM SCRATCH")
print("=" * 60)

class NeuralNetwork:
    """
    A simple neural network with one hidden layer
    Built entirely with NumPy - no libraries!
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Store for backpropagation
        self.cache = {}
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward pass through the network"""
        # Hidden layer
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.sigmoid(self.cache['Z1'])
        
        # Output layer
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        return self.cache['A2']
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss
    
    def backward(self, X, y_true, learning_rate=0.01):
        """Backward pass - the learning happens here!"""
        m = X.shape[0]
        
        # Output layer gradient
        dZ2 = self.cache['A2'] - y_true
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.cache['A1'])
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights (gradient descent)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return self.compute_loss(y_true, self.cache['A2'])
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the neural network"""
        losses = []
        
        for i in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Backward pass
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
            
            if verbose and i % 100 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)

# ========================================
# DEMO: XOR Problem
# ========================================

print("\n🔢 1. XOR Problem")
print("-" * 40)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("XOR Truth Table:")
print("  Input  | Output")
print("  0 0    | 0")
print("  0 1    | 1")
print("  1 0    | 1")
print("  1 1    | 0")

# Create and train neural network
print("\n🧠 Training neural network on XOR...")
nn = NeuralNetwork(2, 4, 1)
losses = nn.train(X, y, epochs=5000, learning_rate=0.5, verbose=False)

# Test the network
print("\n🔮 Predictions:")
predictions = nn.predict(X)
for i, (inputs, pred) in enumerate(zip(X, predictions)):
    print(f"  {inputs[0]} XOR {inputs[1]} = {pred[0]}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - XOR Problem')
plt.grid(True, alpha=0.3)
plt.savefig('xor_training.png', dpi=150)
print("\n✅ Training visualization saved as 'xor_training.png'")
plt.show()

# ========================================
# DEMO: Simple Classification
# ========================================

print("\n\n📊 2. Simple Classification")
print("-" * 40)

# Create synthetic data
np.random.seed(42)
n_samples = 200

# Class 0: points clustered around (0, 0)
X0 = np.random.randn(n_samples // 2, 2) * 0.5
y0 = np.zeros((n_samples // 2, 1))

# Class 1: points clustered around (2, 2)
X1 = np.random.randn(n_samples // 2, 2) * 0.5 + 2
y1 = np.ones((n_samples // 2, 1))

X = np.vstack([X0, X1])
y = np.vstack([y0, y1])

# Shuffle data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Dataset: {len(X)} samples, 2 features")
print(f"Class 0: {sum(y == 0)[0]} samples")
print(f"Class 1: {sum(y == 1)[0]} samples")

# Train network
print("\n🧠 Training classifier...")
nn_classifier = NeuralNetwork(2, 8, 1)
losses = nn_classifier.train(X, y, epochs=2000, learning_rate=0.1, verbose=False)

# Evaluate
predictions = nn_classifier.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f"\n📊 Accuracy: {accuracy:.1f}%")

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    colors = ['red' if label == 0 else 'blue' for label in y.flatten()]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.savefig('decision_boundary.png', dpi=150)
    print("✅ Decision boundary saved as 'decision_boundary.png'")
    plt.show()

plot_decision_boundary(nn_classifier, X, y)

print("\n🎉 Neural network from scratch complete!")
print("You just built a neural network with NumPy! 🧠")