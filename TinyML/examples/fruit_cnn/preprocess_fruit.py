"""
Preprocess a single fruit image into INT8 .bin format ready for C inference.

Usage:
    python preprocess_fruit.py <image_path> [output_path]

Output: raw INT8 bytes in HWC layout (100x100x3 = 30000 bytes)

The output .bin file can be used directly in main.c as:
    static const int8_t my_fruit_data[] = {
    #include "my_fruit.bin.h"
    };
"""
import os
import sys
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out")

TARGET_SIZE = (100, 100)


def get_quant_params():
    """Read quantization params from the trained TFLite model."""
    try:
        import tensorflow as tf
        tflite_path = os.path.join(OUT_DIR, "fruit_cnn_int8.tflite")
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        scale, zero_point = idet['quantization']
        return scale, zero_point
    except Exception:
        # Fallback: typical INT8 quantization for [0,1] range
        print("WARNING: Could not read TFLite model, using default quantization.")
        return 0.003921568859368563, -128


def preprocess(image_path, output_path=None):
    """Convert an image to INT8 .bin for C inference."""
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

    # Quantize to INT8
    scale, zero_point = get_quant_params()
    arr_int8 = np.clip(np.round(arr / scale + zero_point), -128, 127).astype(np.int8)

    print(f"Image shape: {arr_int8.shape}")
    print(f"Input scale: {scale}, zero_point: {zero_point}")
    print(f"INT8 range: [{arr_int8.min()}, {arr_int8.max()}]")

    # Determine output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUT_DIR, f"{base}.bin")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write raw binary
    with open(output_path, "wb") as f:
        f.write(arr_int8.tobytes())

    # Also write a .bin.h header for easy inclusion in C
    hdr_path = output_path + ".h"
    with open(hdr_path, "w") as f:
        f.write(f"// Preprocessed from: {image_path}\n")
        f.write(f"// Shape: {arr_int8.shape}, size: {arr_int8.size}\n")
        xs = [f"0x{b:02x}" for b in arr_int8.tobytes()]
        for k in range(0, len(xs), 16):
            f.write("    " + ", ".join(xs[k:k + 16]) + ",\n")

    print(f"Saved: {output_path} ({arr_int8.nbytes} bytes)")
    print(f"Header: {hdr_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_fruit.py <image_path> [output_path]")
        print("Example: python preprocess_fruit.py my_apple.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    preprocess(image_path, output_path)
