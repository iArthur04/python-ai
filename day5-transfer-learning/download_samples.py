# Create the download script
cat > download_samples.py << 'EOF'
import requests
import os
from PIL import Image
from io import BytesIO
import random

print("📸 Downloading sample images...")

def download_image(url, path):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.resize((224, 224))
            img.save(path)
            print(f"  ✅ {path}")
            return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# Create directories
for animal in ['cats', 'dogs']:
    os.makedirs(f'dataset/train/{animal}', exist_ok=True)
    os.makedirs(f'dataset/validation/{animal}', exist_ok=True)

# URLs for sample images (using placeholder services)
cat_urls = [
    'https://cataas.com/cat?width=224&height=224',
    'https://cataas.com/cat?width=225&height=225',
    'https://cataas.com/cat?width=226&height=226',
    'https://cataas.com/cat?width=227&height=227',
    'https://cataas.com/cat?width=228&height=228',
    'https://cataas.com/cat?width=229&height=229',
    'https://cataas.com/cat?width=230&height=230',
]

dog_urls = [
    'https://placedog.net/224/224',
    'https://placedog.net/225/225',
    'https://placedog.net/226/226',
    'https://placedog.net/227/227',
    'https://placedog.net/228/228',
    'https://placedog.net/229/229',
    'https://placedog.net/230/230',
]

print("\n🐱 Downloading cat images...")
for i, url in enumerate(cat_urls):
    download_image(url, f'dataset/train/cats/cat_{i}.jpg')

print("\n🐶 Downloading dog images...")
for i, url in enumerate(dog_urls):
    download_image(url, f'dataset/train/dogs/dog_{i}.jpg')

# Validation images
print("\n📁 Downloading validation images...")
for i, url in enumerate(cat_urls[:3]):
    download_image(url, f'dataset/validation/cats/cat_val_{i}.jpg')

for i, url in enumerate(dog_urls[:3]):
    download_image(url, f'dataset/validation/dogs/dog_val_{i}.jpg')

print("\n✅ All images downloaded!")
print(f"   Training cats: {len(os.listdir('dataset/train/cats'))}")
print(f"   Training dogs: {len(os.listdir('dataset/train/dogs'))}")
print(f"   Validation cats: {len(os.listdir('dataset/validation/cats'))}")
print(f"   Validation dogs: {len(os.listdir('dataset/validation/dogs'))}")
EOF

# Run the script
# python3 download_samples.py