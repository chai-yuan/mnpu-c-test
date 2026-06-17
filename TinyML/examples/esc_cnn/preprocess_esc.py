"""
Preprocess a single audio file into INT8 .bin format ready for C inference.

Usage:
    python preprocess_esc.py <audio_path> [output_path]

The audio is converted to a 64x64 mel-spectrogram, normalised with the
same fixed formula as training: (mel_db + 80) / 80, then quantised to INT8.

Output: raw INT8 bytes in HWC layout (64x64x1 = 4096 bytes)

The output .bin file can be used directly in main.c as:
    static const int8_t my_audio_data[] = {
    #include "my_audio.bin.h"
    };
"""
import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ---------------------------------------------------------------------------
# Load normalisation config  (written by download_esc.py)
# ---------------------------------------------------------------------------
def _load_norm_cfg():
    cfg_path = os.path.join(DATA_DIR, "norm_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    # Fallback defaults (must match download_esc.py & train_esc.py)
    return {
        "offset": 80.0,
        "scale": 80.0,
        "n_mels": 64,
        "n_time_frames": 64,
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 512,
    }


NORM_CFG = _load_norm_cfg()


def get_quant_params():
    """Read INT8 quantisation params from the trained TFLite model."""
    try:
        import tensorflow as tf
        tflite_path = os.path.join(OUT_DIR, "esc_cnn_int8.tflite")
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        scale, zero_point = idet['quantization']
        return scale, zero_point
    except Exception:
        print("WARNING: Could not read TFLite model, using default quantisation.")
        return 0.003921568859368563, -128


def compute_mel_spectrogram(audio_path):
    """Load audio and compute 64x64 mel-dB spectrogram."""
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa is required. Install with: pip install librosa")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required. Install with: pip install opencv-python")
        sys.exit(1)

    cfg = NORM_CFG

    # Load audio
    y, sr = librosa.load(audio_path, sr=cfg["sample_rate"], mono=True)

    # Ensure consistent length
    target_len = cfg["sample_rate"] * 5
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"], n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize time axis to fixed size
    mel_spec_resized = cv2.resize(
        mel_spec_db,
        (cfg["n_time_frames"], cfg["n_mels"]),
        interpolation=cv2.INTER_LINEAR)

    return mel_spec_resized.astype(np.float32)


def preprocess(audio_path, output_path=None):
    """Convert an audio file to INT8 .bin for C inference."""
    print(f"Processing: {audio_path}")

    # Compute mel-dB spectrogram
    mel_db = compute_mel_spectrogram(audio_path)

    # Fixed normalisation: (x + offset) / scale → [0, 1]
    # MUST match the normalisation in train_esc.py
    mel_norm = np.clip(
        (mel_db + NORM_CFG["offset"]) / NORM_CFG["scale"], 0.0, 1.0)

    # Add channel dimension: (64, 64) -> (64, 64, 1)
    mel_norm = mel_norm[..., np.newaxis].astype(np.float32)

    print(f"Spectrogram shape: {mel_norm.shape}")
    print(f"Value range: [{mel_norm.min():.4f}, {mel_norm.max():.4f}]")

    # Quantise to INT8
    scale, zero_point = get_quant_params()
    arr_int8 = np.clip(
        np.round(mel_norm / scale + zero_point), -128, 127).astype(np.int8)

    print(f"Input scale: {scale}, zero_point: {zero_point}")
    print(f"INT8 range: [{arr_int8.min()}, {arr_int8.max()}]")

    # Determine output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUT_DIR, f"{base}.bin")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write raw binary
    with open(output_path, "wb") as f:
        f.write(arr_int8.tobytes())

    # Also write a .bin.h header for easy inclusion in C
    hdr_path = output_path + ".h"
    with open(hdr_path, "w") as f:
        f.write(f"// Preprocessed from: {audio_path}\n")
        f.write(f"// Shape: {arr_int8.shape}, size: {arr_int8.size}\n")
        xs = [f"0x{b:02x}" for b in arr_int8.tobytes()]
        for k in range(0, len(xs), 16):
            f.write("    " + ", ".join(xs[k:k + 16]) + ",\n")

    print(f"Saved: {output_path} ({arr_int8.nbytes} bytes)")
    print(f"Header: {hdr_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_esc.py <audio_path> [output_path]")
        print("Example: python preprocess_esc.py dog_bark.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    preprocess(audio_path, output_path)
