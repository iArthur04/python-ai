"""
DAY 2: NumPy Fundamentals
NumPy = Numerical Python - The foundation of AI/Data Science
"""

import numpy as np

print("=" * 60)
print("🔢 NUMERICAL PYTHON (NumPy)")
print("=" * 60)

# ========================================
# 1. CREATING ARRAYS
# ========================================

print("\n📊 1. CREATING ARRAYS")

# From a Python list
python_list = [1, 2, 3, 4, 5]
numpy_array = np.array(python_list)
print(f"Python list: {python_list}")
print(f"NumPy array: {numpy_array}")
print(f"Type: {type(numpy_array)}")

# Different ways to create arrays
print("\nDifferent ways to create arrays:")
zeros = np.zeros(5)          # Array of zeros
ones = np.ones(5)            # Array of ones
range_array = np.arange(10)  # Range from 0 to 9
linspace = np.linspace(0, 10, 5)  # 5 evenly spaced numbers from 0 to 10

print(f"Zeros: {zeros}")
print(f"Ones: {ones}")
print(f"Range: {range_array}")
print(f"Linspace: {linspace}")

# 2D Arrays (Matrices)
print("\n2D Arrays (Matrices):")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (rows, columns)

# ========================================
# 2. ARRAY OPERATIONS
# ========================================

print("\n🧮 2. ARRAY OPERATIONS")

arr = np.array([1, 2, 3, 4, 5])

# Basic operations
print(f"Array: {arr}")
print(f"Add 10: {arr + 10}")
print(f"Multiply by 2: {arr * 2}")
print(f"Square: {arr ** 2}")
print(f"Square root: {np.sqrt(arr)}")

# Statistical operations
print(f"\nStatistics:")
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Sum: {np.sum(arr)}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Standard deviation: {np.std(arr)}")

# ========================================
# 3. SLICING & INDEXING
# ========================================

print("\n✂️ 3. SLICING & INDEXING")

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Array: {arr}")

# Indexing
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")

# Slicing (similar to Python lists)
print(f"Elements 2-5: {arr[2:6]}")
print(f"Every other element: {arr[::2]}")
print(f"Reversed: {arr[::-1]}")

# ========================================
# 4. BROADCASTING
# ========================================

print("\n📡 4. BROADCASTING")
print("NumPy's superpower - operations on different sized arrays!")

# 1D + scalar
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Array + 10: {arr + 10}")

# 2D + 1D
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([10, 20, 30])
print(f"\nMatrix:\n{matrix}")
print(f"Row to add: {row}")
print(f"Matrix + Row:\n{matrix + row}")

# ========================================
# 5. WHY NUMPY IS FAST
# ========================================

print("\n⚡ 5. WHY NUMPY IS FAST")

import time

# Compare Python list vs NumPy array
size = 1_000_000
python_list = list(range(size))
numpy_array = np.arange(size)

# Python list operation
start = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start

# NumPy array operation
start = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster! 🚀")

# ========================================
# 6. YOUR FIRST AI-RELATED OPERATION
# ========================================

print("\n🤖 6. YOUR FIRST AI-RELATED OPERATION")

# Create random data (like AI model inputs)
data = np.random.randn(100, 5)  # 100 samples, 5 features
print(f"Random data shape: {data.shape}")

# Normalize the data (common AI preprocessing)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized = (data - mean) / std

print(f"\nOriginal data stats:")
print(f"Mean: {mean}")
print(f"Std: {std}")

print(f"\nNormalized data stats:")
print(f"Mean: {np.mean(normalized, axis=0)}")
print(f"Std: {np.std(normalized, axis=0)}")