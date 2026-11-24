
from pathlib import Path
from PIL import Image

folder = Path("dataset/normal_skin")
files = list(folder.glob("*.jpg"))

valid = 0
corrupted = []

print("Total JPG files found:", len(files))

for file in files:
    try:
        with Image.open(file) as img:
            img.verify()
        valid += 1
    except Exception:
        corrupted.append(file.name)

print("\nVerification completed!")
print("Valid images:", valid)
print("Corrupted images:", len(corrupted))

if corrupted:
    print("\nCorrupted image names:")
    for name in corrupted:
        print(name)
