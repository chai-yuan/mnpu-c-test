"""
Download Fruits-360 dataset via curl and extract 5 common fruits
as 100x100 images into data/ directory.

Selected 5 fruits: Apple, Banana, Grape, Orange, Strawberry
"""
import os
import shutil
import random
import zipfile
import subprocess
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
EXTRACT_DIR = os.path.join(CACHE_DIR, "extracted")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/moltean/fruits"
ZIP_PATH = os.path.join(CACHE_DIR, "fruits.zip")

# Exact class names in fruits-360_100x100/fruits-360/Training/
CLASS_NAMES = [
    "Apple Red 1",
    "Banana 1",
    "Grape White 1",
    "Orange 1",
    "Strawberry 1",
]

SAFE_NAMES = ["apple", "banana", "grape", "orange", "strawberry"]

TARGET_SIZE = (100, 100)
TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 80
TEST_PER_CLASS = 80

# Zip uses prefix: fruits-360_100x100/fruits-360/
ZIP_PREFIX = "fruits-360_100x100/fruits-360/"


def download_fruits360():
    """Download Fruits-360 zip via curl (no auth needed)."""
    if os.path.exists(ZIP_PATH):
        print(f"Using cached: {ZIP_PATH}")
        return ZIP_PATH

    print("Downloading Fruits-360 from Kaggle API...")
    cmd = ["curl", "-L", "-o", ZIP_PATH, DATASET_URL]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to: {ZIP_PATH}")
    return ZIP_PATH


def extract_class_files(zf, class_name, dest_dir):
    """Extract only files for a given fruit class from the zip."""
    members = []
    for folder in ["Training", "Test"]:
        prefix = f"{ZIP_PREFIX}{folder}/{class_name}/"
        members += [m for m in zf.namelist()
                    if m.startswith(prefix)
                    and m.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not members:
        return None

    class_dir = os.path.join(dest_dir, class_name.replace(" ", "_"))
    os.makedirs(class_dir, exist_ok=True)
    extracted = 0
    for m in members:
        dst = os.path.join(class_dir, os.path.basename(m))
        if not os.path.exists(dst):
            with zf.open(m) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    print(f"  Extracted {extracted} files")
    return class_dir


def process_and_save(src_dir, safe_name, num_train, num_val, num_test):
    """Resize images (if needed) and save as numpy arrays."""
    images = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(src_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                # The 100x100 version is already 100x100, but resize for safety
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE, Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                images.append(arr)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    print(f"  Loaded {len(images)} images")
    random.shuffle(images)

    total_needed = num_train + num_val + num_test
    if len(images) < total_needed:
        print(f"  WARNING: only {len(images)} images, need {total_needed}")
        total_needed = len(images)
        num_train = int(total_needed * 0.7)
        num_val = int(total_needed * 0.15)
        num_test = total_needed - num_train - num_val

    train = images[:num_train]
    val = images[num_train:num_train + num_val]
    test = images[num_train + num_val:num_train + num_val + num_test]

    np.save(os.path.join(DATA_DIR, f"{safe_name}_train.npy"), np.array(train, dtype=np.uint8))
    np.save(os.path.join(DATA_DIR, f"{safe_name}_val.npy"), np.array(val, dtype=np.uint8))
    np.save(os.path.join(DATA_DIR, f"{safe_name}_test.npy"), np.array(test, dtype=np.uint8))

    print(f"  Saved: train={len(train)}, val={len(val)}, test={len(test)}")
    return len(train), len(val), len(test)


def main():
    # Download zip
    try:
        zip_path = download_fruits360()
    except subprocess.CalledProcessError as e:
        print(f"curl download failed: {e}")
        print(f"\nPlease download manually:")
        print(f"  curl -L -o {ZIP_PATH} '{DATASET_URL}'")
        print("Then re-run this script.")
        return

    # Extract only needed files
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print("Extracting only needed fruit classes from zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for class_name, safe in zip(CLASS_NAMES, SAFE_NAMES):
            print(f"\nProcessing {class_name} -> {safe}...")
            class_dir = extract_class_files(zf, class_name, EXTRACT_DIR)
            if class_dir is None:
                print(f"  ERROR: Could not find class '{class_name}' in zip")
                continue
            process_and_save(class_dir, safe, TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS)

    # Clean up extracted files
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    # Write metadata
    with open(os.path.join(DATA_DIR, "labels.txt"), "w") as f:
        for i, name in enumerate(SAFE_NAMES):
            f.write(f"{i} {name}\n")

    print(f"\nDone! Data saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
