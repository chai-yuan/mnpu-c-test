"""
Download ESC-50 dataset from GitHub and extract 5 common environmental
sound classes, converting each audio clip to a mel-spectrogram saved
as .npy in data/ directory.

Selected 5 classes: dog, rain, crying_baby, clock_tick, crackling_fire

Normalization: saves RAW mel-dB values (approx [-80, 0]).
The training script applies a fixed normalisation (x + 80) / 80 → [0, 1].
"""
import os
import sys
import json
import shutil
import zipfile
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ZIP_PATH = os.path.join(CACHE_DIR, "esc50.zip")
EXTRACT_DIR = os.path.join(CACHE_DIR, "esc50_extracted")

# 5 common environmental sound classes
CLASS_NAMES = ["dog", "rain", "crying_baby", "clock_tick", "crackling_fire"]

# Use more training data: 30/5/5 per class (40 total per class)
TRAIN_PER_CLASS = 30
VAL_PER_CLASS = 5
TEST_PER_CLASS = 5

# Mel-spectrogram parameters
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
N_TIME_FRAMES = 64  # resize to fixed number of time frames

# Fixed normalisation parameters (mel-dB roughly in [-80, 0])
NORM_OFFSET = 80.0   # subtract this (i.e. x + 80)
NORM_SCALE = 80.0    # divide by this → [0, 1]


def download_esc50():
    """Download ESC-50 zip via curl."""
    if os.path.exists(ZIP_PATH):
        print(f"Using cached: {ZIP_PATH}")
        return ZIP_PATH

    print("Downloading ESC-50 from GitHub...")
    cmd = ["curl", "-L", "-o", ZIP_PATH, DATASET_URL]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to: {ZIP_PATH}")
    return ZIP_PATH


def process_audio_to_melspec(audio_path):
    """Convert an audio file to a mel-spectrogram (n_mels x n_time_frames)."""
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa is required. Install with: pip install librosa")
        sys.exit(1)

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Ensure consistent length (5 seconds)
    target_len = SAMPLE_RATE * 5
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # Convert to log scale (dB) – values roughly in [-80, 0]
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize time axis to fixed N_TIME_FRAMES
    import cv2
    mel_spec_resized = cv2.resize(mel_spec_db, (N_TIME_FRAMES, N_MELS),
                                   interpolation=cv2.INTER_LINEAR)

    return mel_spec_resized.astype(np.float32)


def download_and_process():
    """Main download and processing routine."""
    # Download zip
    try:
        zip_path = download_esc50()
    except subprocess.CalledProcessError as e:
        print(f"curl download failed: {e}")
        print(f"\nPlease download manually:")
        print(f"  curl -L -o {ZIP_PATH} '{DATASET_URL}'")
        print("Then re-run this script.")
        return

    # Extract
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print("Extracting ESC-50 zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    # Read metadata to map filenames to categories
    import csv
    meta_path = os.path.join(EXTRACT_DIR, "ESC-50-master", "meta", "esc50.csv")
    audio_dir = os.path.join(EXTRACT_DIR, "ESC-50-master", "audio")

    # Build filename -> category mapping
    filename_to_category = {}
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename_to_category[row["filename"]] = row["category"]

    # Process each class
    for class_name in CLASS_NAMES:
        print(f"\nProcessing class: {class_name}...")

        # Find all audio files for this class
        class_files = []
        for fname, cat in filename_to_category.items():
            if cat == class_name:
                class_files.append(os.path.join(audio_dir, fname))

        print(f"  Found {len(class_files)} audio files")

        # Convert to mel-spectrograms
        spectrograms = []
        for fpath in sorted(class_files):
            try:
                mel = process_audio_to_melspec(fpath)
                spectrograms.append(mel)
            except Exception as e:
                print(f"  Skipping {os.path.basename(fpath)}: {e}")

        print(f"  Processed {len(spectrograms)} spectrograms")

        # Shuffle deterministically (same seed → reproducible splits)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(spectrograms))
        spectrograms = [spectrograms[i] for i in indices]

        train = spectrograms[:TRAIN_PER_CLASS]
        val = spectrograms[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
        test = spectrograms[TRAIN_PER_CLASS + VAL_PER_CLASS:
                            TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS]

        # Save RAW mel-dB values (float32)
        np.save(os.path.join(DATA_DIR, f"{class_name}_train.npy"),
                np.array(train, dtype=np.float32))
        np.save(os.path.join(DATA_DIR, f"{class_name}_val.npy"),
                np.array(val, dtype=np.float32))
        np.save(os.path.join(DATA_DIR, f"{class_name}_test.npy"),
                np.array(test, dtype=np.float32))

        print(f"  Saved: train={len(train)}, val={len(val)}, test={len(test)}")

    # Clean up extracted files
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    # Write normalization config so training & preprocessing stay in sync
    norm_cfg = {
        "offset": NORM_OFFSET,
        "scale": NORM_SCALE,
        "n_mels": N_MELS,
        "n_time_frames": N_TIME_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }
    with open(os.path.join(DATA_DIR, "norm_cfg.json"), "w") as f:
        json.dump(norm_cfg, f, indent=2)

    # Write metadata
    with open(os.path.join(DATA_DIR, "labels.txt"), "w") as f:
        for i, name in enumerate(CLASS_NAMES):
            display_name = name.replace("_", " ").title()
            f.write(f"{i} {display_name}\n")

    print(f"\nDone! Data saved to {DATA_DIR}")


if __name__ == "__main__":
    download_and_process()
