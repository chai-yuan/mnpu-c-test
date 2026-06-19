"""
Download Speech Commands v0.02 dataset and extract "on", "off" commands
plus an "unknown" class assembled from other words + background noise.

Converts each 1‑second audio clip to a 32×32 mel‑spectrogram saved as .npy.

Classes:  on (0),  off (1),  unknown (2)

* "on" / "off"    – from the official word folders
* "unknown"       – mixed:  ~60% other speech commands,
                     ~40% background‑noise random 1‑s slices

Normalisation:  saves RAW mel‑dB values (approx [‑80, 0]).
Training / preprocessing both use  (x + 80) / 80  → [0, 1].
"""
import os
import sys
import json
import shutil
import tarfile
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TGZ_PATH = os.path.join(CACHE_DIR, "speech_commands_v0.02.tar.gz")
EXTRACT_DIR = os.path.join(CACHE_DIR, "extracted")

# ---------------------------------------------------------------------------
#  Classes & split sizes
# ---------------------------------------------------------------------------
CLASS_NAMES = ["on", "off", "unknown"]

# How many samples per class (training is generous because data is plentiful)
TRAIN_PER_CLASS = {"on": 800, "off": 800, "unknown": 1600}
VAL_PER_CLASS   = {"on": 150, "off": 150, "unknown": 300}
TEST_PER_CLASS  = {"on": 150, "off": 150, "unknown": 300}

# Words used to build the "unknown" class (exclude "on" and "off")
UNKNOWN_WORDS = [
    "yes", "no", "up", "down", "left", "right",
    "stop", "go", "zero", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house",
    "marvin", "sheila", "tree", "wow",
]
# Reserve these for test variety (not in unknown training)
UNKNOWN_TEST_EXTRA = ["backward", "forward", "follow", "learn", "visual"]

# ---------------------------------------------------------------------------
#  Mel‑spectrogram parameters  (compact – 32×32 for fast inference)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000      # Speech Commands native rate
CLIP_SECONDS = 1.0       # exactly 1 s
N_MELS = 32
N_FFT = 512
HOP_LENGTH = 256
N_TIME_FRAMES = 32

NORM_OFFSET = 80.0
NORM_SCALE  = 80.0


# ===================================================================
def download_dataset():
    if os.path.exists(TGZ_PATH):
        print(f"Using cached: {TGZ_PATH}")
        return TGZ_PATH

    print("Downloading Speech Commands v0.02 (~1.5 GB, this may take a while)...")
    cmd = ["curl", "-L", "--progress-bar", "-o", TGZ_PATH, DATASET_URL]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to: {TGZ_PATH}")
    return TGZ_PATH


def process_audio_to_melspec(audio_data, sr):
    """Convert raw audio samples → 32×32 mel‑dB spectrogram."""
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa is required.  pip install librosa")
        sys.exit(1)

    # Ensure exactly CLIP_SECONDS
    target_len = int(SAMPLE_RATE * CLIP_SECONDS)
    if len(audio_data) < target_len:
        audio_data = np.pad(audio_data, (0, target_len - len(audio_data)))
    else:
        audio_data = audio_data[:target_len]

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data.astype(np.float64),
                                       orig_sr=sr, target_sr=SAMPLE_RATE)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data.astype(np.float32), sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize time axis → N_TIME_FRAMES
    import cv2
    mel_spec_resized = cv2.resize(
        mel_spec_db, (N_TIME_FRAMES, N_MELS),
        interpolation=cv2.INTER_LINEAR)

    return mel_spec_resized.astype(np.float32)


def load_wav_spectrograms(wav_dir, max_count=None):
    """Load all WAVs in a directory → list of (32,32) float32 arrays."""
    specs = []
    fnames = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")])
    if max_count:
        fnames = fnames[:max_count]

    for fn in fnames:
        fpath = os.path.join(wav_dir, fn)
        try:
            import soundfile as sf
            y, sr = sf.read(fpath)
        except ImportError:
            import librosa
            y, sr = librosa.load(fpath, sr=None, mono=True)
        except Exception as e:
            print(f"  Skipping {fn}: {e}")
            continue

        try:
            mel = process_audio_to_melspec(y, sr)
            specs.append(mel)
        except Exception as e:
            print(f"  Skipping {fn}: {e}")

    return specs


def load_noise_spectrograms(noise_dir, count):
    """Extract random 1‑s slices from background noise files."""
    specs = []
    noise_files = sorted([f for f in os.listdir(noise_dir)
                           if f.lower().endswith(".wav")])

    while len(specs) < count:
        fn = np.random.choice(noise_files)
        fpath = os.path.join(noise_dir, fn)
        try:
            import soundfile as sf
            y, sr = sf.read(fpath)
        except ImportError:
            import librosa
            y, sr = librosa.load(fpath, sr=None, mono=True)
        except Exception:
            continue

        if sr != SAMPLE_RATE:
            try:
                import librosa
                y = librosa.resample(y.astype(np.float64),
                                      orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception:
                continue

        clip_len = int(SAMPLE_RATE * CLIP_SECONDS)
        if len(y) < clip_len:
            continue

        # Random start
        start = np.random.randint(0, len(y) - clip_len)
        segment = y[start:start + clip_len]

        try:
            mel = process_audio_to_melspec(segment, SAMPLE_RATE)
            specs.append(mel)
        except Exception:
            continue

    return specs


def save_class(name, train_specs, val_specs, test_specs):
    """Shuffle & save train/val/test .npy for a class."""
    np.save(os.path.join(DATA_DIR, f"{name}_train.npy"),
            np.array(train_specs, dtype=np.float32))
    np.save(os.path.join(DATA_DIR, f"{name}_val.npy"),
            np.array(val_specs, dtype=np.float32))
    np.save(os.path.join(DATA_DIR, f"{name}_test.npy"),
            np.array(test_specs, dtype=np.float32))
    print(f"  Saved: train={len(train_specs)}, val={len(val_specs)}, "
          f"test={len(test_specs)}")


# ===================================================================
def main():
    # 1. Download & extract
    try:
        tgz_path = download_dataset()
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"Download manually and place at: {TGZ_PATH}")
        return

    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print("Extracting tar.gz ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(EXTRACT_DIR)
    print("Extraction done.")

    rng = np.random.RandomState(42)  # reproducible splits

    # ------------------------------------------------------------------
    # 2. Process "on"
    # ------------------------------------------------------------------
    print("\n=== Processing class: on ===")
    on_specs = load_wav_spectrograms(os.path.join(EXTRACT_DIR, "on"))
    print(f"  Loaded {len(on_specs)} spectrograms")
    idx = rng.permutation(len(on_specs))
    on_specs = [on_specs[i] for i in idx]
    n = TRAIN_PER_CLASS["on"];  v = VAL_PER_CLASS["on"];  t = TEST_PER_CLASS["on"]
    save_class("on", on_specs[:n], on_specs[n:n+v], on_specs[n+v:n+v+t])

    # ------------------------------------------------------------------
    # 3. Process "off"
    # ------------------------------------------------------------------
    print("\n=== Processing class: off ===")
    off_specs = load_wav_spectrograms(os.path.join(EXTRACT_DIR, "off"))
    print(f"  Loaded {len(off_specs)} spectrograms")
    idx = rng.permutation(len(off_specs))
    off_specs = [off_specs[i] for i in idx]
    n = TRAIN_PER_CLASS["off"];  v = VAL_PER_CLASS["off"];  t = TEST_PER_CLASS["off"]
    save_class("off", off_specs[:n], off_specs[n:n+v], off_specs[n+v:n+v+t])

    # ------------------------------------------------------------------
    # 4. Build "unknown" class  (speech words + background noise)
    # ------------------------------------------------------------------
    print("\n=== Building unknown class ===")
    unknown_specs = []

    # 4a. Other speech commands
    for word in UNKNOWN_WORDS:
        wdir = os.path.join(EXTRACT_DIR, word)
        if not os.path.isdir(wdir):
            continue
        specs = load_wav_spectrograms(wdir)
        unknown_specs.extend(specs)
        print(f"  {word}: {len(specs)}")

    # 4b. Background noise slices
    noise_dir = os.path.join(EXTRACT_DIR, "_background_noise_")
    if os.path.isdir(noise_dir):
        noise_count = max(int(len(unknown_specs) * 0.4), 200)
        noise_specs = load_noise_spectrograms(noise_dir, noise_count)
        unknown_specs.extend(noise_specs)
        print(f"  background_noise: {len(noise_specs)}")

    # Also add a few utterances from reserved test-extra words (both
    # for variety in unknown train and to make test more realistic)
    for word in UNKNOWN_TEST_EXTRA:
        wdir = os.path.join(EXTRACT_DIR, word)
        if not os.path.isdir(wdir):
            continue
        specs = load_wav_spectrograms(wdir)
        unknown_specs.extend(specs)
        print(f"  {word} (extra): {len(specs)}")

    print(f"  Total unknown spectrograms: {len(unknown_specs)}")

    idx = rng.permutation(len(unknown_specs))
    unknown_specs = [unknown_specs[i] for i in idx]
    n = TRAIN_PER_CLASS["unknown"]
    v = VAL_PER_CLASS["unknown"]
    t = TEST_PER_CLASS["unknown"]
    save_class("unknown", unknown_specs[:n],
               unknown_specs[n:n+v], unknown_specs[n+v:n+v+t])

    # ------------------------------------------------------------------
    # 5. Clean up & write config
    # ------------------------------------------------------------------
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    norm_cfg = {
        "offset": NORM_OFFSET,
        "scale": NORM_SCALE,
        "n_mels": N_MELS,
        "n_time_frames": N_TIME_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "clip_seconds": CLIP_SECONDS,
    }
    with open(os.path.join(DATA_DIR, "norm_cfg.json"), "w") as f:
        json.dump(norm_cfg, f, indent=2)

    with open(os.path.join(DATA_DIR, "labels.txt"), "w") as f:
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i} {name}\n")

    print(f"\nDone!  Data saved to {DATA_DIR}")
    print(f"Total files on disk: {len(os.listdir(DATA_DIR))}")


if __name__ == "__main__":
    main()
