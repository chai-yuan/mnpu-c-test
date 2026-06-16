"""
Train a simple MLP on MNIST, apply full INT8 quantization,
export to .tflite, convert to .tmdl with the TinyML compiler,
and dump sample test images as raw INT8 binaries.
"""
import os
import sys
import random
import struct
import subprocess
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TINYML_ROOT = os.path.join(EXAMPLE_DIR, "../..")
OUT_DIR = os.path.join(EXAMPLE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Allow importing the TinyML compiler package
sys.path.insert(0, TINYML_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from compiler.main import main as compiler_main

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 64
learning_rate = 1e-3
epochs = 6
num_test_bins = 10   # Number of quantized test images to export as .bin

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def create_model(batch_size):
    return tf.keras.models.Sequential([
        tf.keras.Input(batch_shape=(batch_size, 28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model = create_model(batch_size=None)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model.fit(train_images, train_labels,
          batch_size=batch_size, epochs=epochs,
          validation_split=0.1)

# ---------------------------------------------------------------------------
# INT8 Quantization & TFLite export
# ---------------------------------------------------------------------------
export_model = create_model(batch_size=1)
export_model.set_weights(model.get_weights())

def representative_data_gen():
    for i in range(100):
        data = train_images[i:i + 1].astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
tflite_path = os.path.join(OUT_DIR, "mnist_mlp_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"\nTFLite model exported to: {tflite_path}")

# ---------------------------------------------------------------------------
# Print input / output metadata & conversion recipe
# ---------------------------------------------------------------------------
interpreter_tmp = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter_tmp.allocate_tensors()
_in = interpreter_tmp.get_input_details()[0]
_out = interpreter_tmp.get_output_details()[0]

print("\n" + "=" * 56)
print("  Model I/O Information")
print("=" * 56)
print(f"  Input  shape : {_in['shape'].tolist()}")
print(f"  Input  dtype : {_in['dtype'].__name__}")
print(f"  Output shape : {_out['shape'].tolist()}")
print(f"  Output dtype : {_out['dtype'].__name__}")

_in_scale, _in_zp = _in['quantization']
print(f"\n  Input quantization:")
print(f"    scale      = {_in_scale}")
print(f"    zero_point = {_in_zp}")

print("\n  How to convert a 0-255 grayscale image to model input:")
print("    1. Normalize to [0.0, 1.0] :  pixel_f = gray / 255.0")
print("    2. Quantize to INT8        :  int8_val = round(pixel_f / scale) + zero_point")
print("    3. Clip to [-128, 127]     :  int8_val = clip(int8_val, -128, 127)")
print("=" * 56)

# ---------------------------------------------------------------------------
# Convert TFLite -> TMDL with TinyML compiler
# ---------------------------------------------------------------------------
tmdl_path = os.path.join(OUT_DIR, "mnist_mlp_int8.tmdl")
header_path = os.path.join(OUT_DIR, "mnist_mlp_int8.bin.h")

# Call compiler programmatically (suppress sys.argv side effects)
old_argv = sys.argv
sys.argv = [
    "compiler",
    tflite_path,
    tmdl_path,
    "--out-deq", "0",
    "-H", header_path,
]
try:
    compiler_main()
except SystemExit:
    pass
sys.argv = old_argv

print(f"TMDL model exported to: {tmdl_path}")
print(f"C header exported to: {header_path}")

# ---------------------------------------------------------------------------
# Helper: write a unified C header for all test images
# ---------------------------------------------------------------------------
def write_test_images_header(image_entries, out_path):
    """
    image_entries: list of (name: str, int8_data: bytes, label: int)
    """
    lines = [
        "/* Auto-generated test images header – do not edit. */",
        "#ifndef TEST_IMAGES_H",
        "#define TEST_IMAGES_H",
        "#include <stdint.h>",
        f"#define TEST_IMAGE_COUNT {len(image_entries)}",
        "",
    ]

    for name, data, label in image_entries:
        var = name.replace('.', '_')
        size = len(data)
        lines.append(f"static const int8_t {var}_data[{size}] = {{")
        hex_vals = [f"0x{b:02x}" for b in data]
        for k in range(0, len(hex_vals), 16):
            lines.append("    " + ", ".join(hex_vals[k:k + 16]) + ",")
        lines.append("};")
        lines.append(f"#define {var}_size {size}")
        lines.append(f"#define {var}_label {label}")
        lines.append("")

    lines.append("typedef struct {")
    lines.append("    const int8_t *data;")
    lines.append("    int size;")
    lines.append("    int label;")
    lines.append("    const char *name;")
    lines.append("} test_image_t;")
    lines.append("")
    lines.append("static const test_image_t test_images[TEST_IMAGE_COUNT] = {")
    for name, data, label in image_entries:
        var = name.replace('.', '_')
        lines.append(f'    {{{var}_data, {var}_size, {var}_label, "{name}"}},')
    lines.append("};")
    lines.append("")
    lines.append("#endif /* TEST_IMAGES_H */")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Exported test images header to {out_path}")


# ---------------------------------------------------------------------------
# Evaluation & Test Data Export
# ---------------------------------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

correct = 0
total = len(test_images)
bin_indices = set(random.sample(range(total), num_test_bins))

image_entries = []

for i in range(total):
    img_float = np.expand_dims(test_images[i], axis=0).astype(np.float32)

    if input_scale != 0:
        img_int8 = img_float / input_scale + input_zero_point
        img_int8 = np.clip(np.round(img_int8), -128, 127).astype(np.int8)
    else:
        img_int8 = img_float.astype(np.int8)

    interpreter.set_tensor(input_details['index'], img_int8)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    pred = np.argmax(output_data[0])
    if pred == test_labels[i]:
        correct += 1

    if i in bin_indices:
        bin_filename = os.path.join(OUT_DIR, f"test_img_{i}_label_{test_labels[i]}.bin")
        with open(bin_filename, "wb") as f:
            f.write(img_int8.tobytes())
        print(f"Exported test image to {bin_filename}")

        name = f"test_img_{i}_label_{test_labels[i]}.bin"
        image_entries.append((name, img_int8.tobytes(), int(test_labels[i])))

if image_entries:
    write_test_images_header(image_entries, os.path.join(OUT_DIR, "test_images.bin.h"))

acc = 100. * correct / total
print(f"\nTest accuracy: {correct}/{total} ({acc:.2f}%)")
