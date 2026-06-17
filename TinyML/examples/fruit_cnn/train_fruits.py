"""
Train a CNN on 5-class 100x100 fruit images and export INT8 TMDL model.

Architecture (stride=2 conv for downsampling, no MaxPool):

  Input(100, 100, 3)
  Conv2D(16, 3x3, s2, same, relu)  -> 50x50x16
  Conv2D(32, 3x3, s2, same, relu)  -> 25x25x32
  Conv2D(64, 3x3, s2, same, relu)  -> 13x13x64
  GlobalAveragePooling2D()          -> 64
  Dense(5, softmax)                 -> 5
"""
import os
import sys
import random
import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TINYML_ROOT = os.path.join(EXAMPLE_DIR, "../..")
OUT_DIR = os.path.join(EXAMPLE_DIR, "out")
DATA_DIR = os.path.join(EXAMPLE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, TINYML_ROOT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from compiler.main import main as compiler_main

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 20
NUM_TEST_BINS = 10

SAFE_NAMES = ["apple", "banana", "grape", "orange", "strawberry"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    for label, name in enumerate(SAFE_NAMES):
        train = np.load(os.path.join(DATA_DIR, f"{name}_train.npy"))
        val = np.load(os.path.join(DATA_DIR, f"{name}_val.npy"))
        test = np.load(os.path.join(DATA_DIR, f"{name}_test.npy"))

        x_train_list.append(train)
        y_train_list.append(np.full(len(train), label, dtype=np.int32))
        x_val_list.append(val)
        y_val_list.append(np.full(len(val), label, dtype=np.int32))
        x_test_list.append(test)
        y_test_list.append(np.full(len(test), label, dtype=np.int32))

    x_train = np.concatenate(x_train_list).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_list)
    x_val = np.concatenate(x_val_list).astype(np.float32) / 255.0
    y_val = np.concatenate(y_val_list)
    x_test = np.concatenate(x_test_list).astype(np.float32) / 255.0
    y_test = np.concatenate(y_test_list)

    # Shuffle
    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_data()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build():
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=(100, 100, 3)),
        tf.keras.layers.Conv2D(16, 3, 2, 'same', activation='relu'),
        tf.keras.layers.Conv2D(32, 3, 2, 'same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, 2, 'same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

model = build()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)

model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          verbose=2)

# ---------------------------------------------------------------------------
# INT8 quantize & export
# ---------------------------------------------------------------------------
export = build()
export.set_weights(model.get_weights())

def rep_gen():
    for i in range(min(200, len(x_train))):
        yield [x_train[i:i + 1]]

conv = tf.lite.TFLiteConverter.from_keras_model(export)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_buf = conv.convert()

tflite_path = os.path.join(OUT_DIR, "fruit_cnn_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_buf)

# ---------------------------------------------------------------------------
# Compile TFLite -> TMDL
# ---------------------------------------------------------------------------
tmdl_path = os.path.join(OUT_DIR, "fruit_cnn_int8.tmdl")
hdr_path = os.path.join(OUT_DIR, "fruit_cnn_int8.bin.h")

old = sys.argv
sys.argv = ["compiler", tflite_path, tmdl_path,
            "--out-deq", "0", "-H", hdr_path]
try:
    compiler_main()
except SystemExit:
    pass
sys.argv = old

# ---------------------------------------------------------------------------
# Evaluate & export test images
# ---------------------------------------------------------------------------
interp = tf.lite.Interpreter(model_content=tflite_buf)
interp.allocate_tensors()
idet = interp.get_input_details()[0]
odet = interp.get_output_details()[0]
iscale, izp = idet['quantization']

correct = 0
total = len(x_test)
picks = set(random.sample(range(total), min(NUM_TEST_BINS, total)))
entries = []

for i in range(total):
    f32 = x_test[i:i + 1]
    i8 = np.clip(np.round(f32 / iscale + izp), -128, 127).astype(np.int8)
    interp.set_tensor(idet['index'], i8)
    interp.invoke()
    out = interp.get_tensor(odet['index'])
    if np.argmax(out) == y_test[i]:
        correct += 1
    if i in picks:
        name = f"test_img_{i}_label_{SAFE_NAMES[y_test[i]]}.bin"
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(i8.tobytes())
        entries.append((name, i8.tobytes(), int(y_test[i])))

# Write combined test_images.bin.h
label_names_str = ", ".join('"' + n + '"' for n in SAFE_NAMES)
lines = [
    "/* Auto-generated test images header - do not edit. */",
    "#ifndef TEST_IMAGES_H", "#define TEST_IMAGES_H",
    "#include <stdint.h>",
    f"#define TEST_IMAGE_COUNT {len(entries)}",
    f"static const char *TEST_LABEL_NAMES[] = {{{label_names_str}}};",
    "",
]
for n, d, l in entries:
    v = n.replace('.', '_').replace('-', '_')
    lines.append(f"static const int8_t {v}_data[{len(d)}] = {{")
    xs = [f"0x{b:02x}" for b in d]
    for k in range(0, len(xs), 16):
        lines.append("    " + ", ".join(xs[k:k + 16]) + ",")
    lines.extend(["};", f"#define {v}_size {len(d)}", f"#define {v}_label {l}", ""])
lines += [
    "typedef struct { const int8_t *data; int size; int label; const char *name; } test_image_t;",
    "", "static const test_image_t test_images[TEST_IMAGE_COUNT] = {",
]
for n, d, l in entries:
    v = n.replace('.', '_').replace('-', '_')
    lines.append(
        f'    {{{v}_data, {v}_size, {v}_label, "{n}"}},')
lines += ["};", "", "#endif"]
with open(os.path.join(OUT_DIR, "test_images.bin.h"), "w") as f:
    f.write("\n".join(lines))

print(f"\nTest accuracy (TFLite INT8): {correct}/{total} ({100. * correct / total:.2f}%)")
