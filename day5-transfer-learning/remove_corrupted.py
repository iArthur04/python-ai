# Navigate to your dataset
#cd ~/Documents/python-ai/day5-transfer-learning

# Create a script to remove corrupted images
#cat > remove_corrupted.py << 'EOF'
import os
from PIL import Image

def check_and_remove_corrupted(folder_path):
    """Check all images in a folder and remove corrupted ones"""
    corrupted = []
    valid = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            filepath = os.path.join(folder_path, filename)
            try:
                # Try to open the image
                with Image.open(filepath) as img:
                    img.verify()  # Verify it's a valid image
                valid += 1
            except Exception as e:
                # Image is corrupted
                print(f"❌ Removing corrupted image: {filename}")
                os.remove(filepath)
                corrupted.append(filename)
    
    return corrupted, valid

# Check all folders
folders = [
    'dataset/train/cats',
    'dataset/train/dogs',
    'dataset/validation/cats',
    'dataset/validation/dogs'
]

total_corrupted = 0
total_valid = 0

for folder in folders:
    if os.path.exists(folder):
        corrupted, valid = check_and_remove_corrupted(folder)
        total_corrupted += len(corrupted)
        total_valid += valid
        print(f"📁 {folder}: {valid} valid, {len(corrupted)} corrupted removed")

print(f"\n✅ Done! Removed {total_corrupted} corrupted images")
print(f"✅ {total_valid} valid images remaining")
EOF

#python3 remove_corrupted.py