import numpy as np
import matplotlib.pyplot as plt
import os

def create_synthetic_image(shape='circle', color=(255, 0, 0)):
    """Create a simple synthetic image"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    if shape == 'circle':
        for i in range(224):
            for j in range(224):
                if (i-112)**2 + (j-112)**2 < 50**2:
                    img[i, j] = color
    elif shape == 'square':
        img[60:164, 60:164] = color
    return img

for animal in ['cats', 'dogs']:
    os.makedirs(f'dataset/train/{animal}', exist_ok=True)
    os.makedirs(f'dataset/validation/{animal}', exist_ok=True)

for i in range(100):
    plt.imsave(f'dataset/train/cats/cat_{i}.jpg', create_synthetic_image('circle', (255, 100, 100)))
    plt.imsave(f'dataset/train/dogs/dog_{i}.jpg', create_synthetic_image('square', (100, 100, 255)))

for i in range(20):
    plt.imsave(f'dataset/validation/cats/cat_val_{i}.jpg', create_synthetic_image('circle', (255, 150, 150)))
    plt.imsave(f'dataset/validation/dogs/dog_val_{i}.jpg', create_synthetic_image('square', (150, 150, 255)))

print("✅ Synthetic dataset created!")
