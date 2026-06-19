"""
Preprocess a single audio file (WAV) into INT8 .bin for C inference.

Usage:
    python preprocess_speech.py <audio_path> [output_path]

The audio is converted to a 32×32 mel‑spectrogram, normalised with the
fixed formula  (x + 80) / 80, quantised to INT8, and saved as raw binary.

Output:  32×32×1 = 1024 bytes  (INT8, HWC layout).

The .bin file can be included in C via the generated .bin.h header:
    static const int8_t my_audio_data[] = {
    #include "my_audio.bin.h"
    };
"""
import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(SCRIPT_DIR, "out")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def _load_norm_cfg():
    cfg_path = os.path.join(DATA_DIR, "norm_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {
        "offset": 80.0, "scale": 80.0,
        "n_mels": 32, "n_time_frames": 32,
        "sample_rate": 16000, "n_fft": 512, "hop_length": 256,
        "clip_seconds": 1.0,
    }


NORM_CFG = _load_norm_cfg()


def get_quant_params():
    try:
        import tensorflow as tf
        tflite_path = os.path.join(OUT_DIR, "speech_cmd_int8.tflite")
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        return idet['quantization']  # (scale, zero_point)
    except Exception:
        print("WARNING: cannot read TFLite model – using default quant.")
        return 0.003921568859368563, -128


def compute_mel_spectrogram(audio_path):
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa required.  pip install librosa")
        sys.exit(1)
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required.  pip install opencv-python")
        sys.exit(1)

    cfg = NORM_CFG
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    target_len = int(cfg["sample_rate"] * cfg.get("clip_seconds", 1.0))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    if sr != cfg["sample_rate"]:
        y = librosa.resample(y.astype(np.float64),
                              orig_sr=sr, target_sr=cfg["sample_rate"])

    mel_spec = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"])

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_resized = cv2.resize(
        mel_db, (cfg["n_time_frames"], cfg["n_mels"]),
        interpolation=cv2.INTER_LINEAR)

    return mel_resized.astype(np.float32)


def preprocess(audio_path, output_path=None):
    print(f"Processing: {audio_path}")

    mel_db = compute_mel_spectrogram(audio_path)

    # Fixed normalisation (MUST match train_speech.py)
    mel_norm = np.clip(
        (mel_db + NORM_CFG["offset"]) / NORM_CFG["scale"], 0.0, 1.0)
    mel_norm = mel_norm[..., np.newaxis].astype(np.float32)

    print(f"Spectrogram shape: {mel_norm.shape}  "
          f"range: [{mel_norm.min():.4f}, {mel_norm.max():.4f}]")

    scale, zero_point = get_quant_params()
    arr_int8 = np.clip(
        np.round(mel_norm / scale + zero_point), -128, 127).astype(np.int8)

    print(f"Quant: scale={scale:.6f}  zp={zero_point}  "
          f"INT8 range [{arr_int8.min()}, {arr_int8.max()}]")

    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUT_DIR, f"{base}.bin")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(arr_int8.tobytes())

    hdr_path = output_path + ".h"
    with open(hdr_path, "w") as f:
        f.write(f"// Preprocessed from: {audio_path}\n")
        f.write(f"// Shape: {arr_int8.shape}, size: {arr_int8.size}\n")
        xs = [f"0x{b:02x}" for b in arr_int8.tobytes()]
        for k in range(0, len(xs), 16):
            f.write("    " + ", ".join(xs[k:k + 16]) + ",\n")

    print(f"Saved: {output_path}  ({arr_int8.nbytes} bytes)")
    print(f"Header: {hdr_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_speech.py <audio_path> [output_path]")
        print("Example: python preprocess_speech.py my_on.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    preprocess(audio_path, output_path)
