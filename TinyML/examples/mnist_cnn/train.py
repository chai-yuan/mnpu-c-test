"""
Train a tiny CNN on MNIST and export INT8 TMDL model.

Architecture (stride=2 conv for downsampling, no MaxPool):

  Input(28, 28, 1)
  Conv2D(16, 3x3, s1, same, relu)  -> 28x28x16
  Conv2D(16, 3x3, s2, same, relu)  -> 14x14x16
  Conv2D(32, 3x3, s2, same, relu)  -> 7x7x32
  GlobalAveragePooling2D()          -> 32
  Dense(10, softmax)                -> 10
"""
import os, sys, random
import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TINYML_ROOT = os.path.join(EXAMPLE_DIR, "../..")
OUT_DIR = os.path.join(EXAMPLE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, TINYML_ROOT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from compiler.main import main as compiler_main

# ---------------------------------------------------------------------------
batch_size = 64
epochs = 12
num_test_bins = 10

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.array(y_train, dtype=np.int32)
y_test  = np.array(y_test,  dtype=np.int32)
x_train = np.expand_dims(x_train, -1).astype(np.float32) / 255.0
x_test  = np.expand_dims(x_test,  -1).astype(np.float32) / 255.0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build():
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, 1, 'same', activation='relu'),
        tf.keras.layers.Conv2D(16, 3, 2, 'same', activation='relu'),
        tf.keras.layers.Conv2D(32, 3, 2, 'same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

model = build()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
          validation_split=0.1)

# ---------------------------------------------------------------------------
# INT8 quantize & export
# ---------------------------------------------------------------------------
export = build()
export.set_weights(model.get_weights())

def rep_gen():
    for i in range(200):
        yield [x_train[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(export)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_buf = conv.convert()

tflite_path = os.path.join(OUT_DIR, "mnist_cnn_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_buf)

# ---------------------------------------------------------------------------
# Compile TFLite -> TMDL
# ---------------------------------------------------------------------------
tmdl_path = os.path.join(OUT_DIR, "mnist_cnn_int8.tmdl")
hdr_path  = os.path.join(OUT_DIR, "mnist_cnn_int8.bin.h")

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
total   = len(x_test)
picks   = set(random.sample(range(total), num_test_bins))
entries = []
for i in range(total):
    f32 = x_test[i:i+1]
    i8  = np.clip(np.round(f32 / iscale + izp), -128, 127).astype(np.int8)
    interp.set_tensor(idet['index'], i8)
    interp.invoke()
    out = interp.get_tensor(odet['index'])
    if np.argmax(out) == y_test[i]: correct += 1
    if i in picks:
        name = f"test_img_{i}_label_{y_test[i]}.bin"
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(i8.tobytes())
        entries.append((name, i8.tobytes(), int(y_test[i])))

# Write combined test_images.bin.h
lines = [
    "/* Auto-generated test images header – do not edit. */",
    "#ifndef TEST_IMAGES_H", "#define TEST_IMAGES_H",
    "#include <stdint.h>", f"#define TEST_IMAGE_COUNT {len(entries)}", "",
]
for n, d, l in entries:
    v = n.replace('.', '_')
    lines.append(f"static const int8_t {v}_data[{len(d)}] = {{")
    xs = [f"0x{b:02x}" for b in d]
    for k in range(0, len(xs), 16):
        lines.append("    " + ", ".join(xs[k:k+16]) + ",")
    lines.extend(["};", f"#define {v}_size {len(d)}", f"#define {v}_label {l}", ""])
lines += [
    "typedef struct { const int8_t *data; int size; int label; const char *name; } test_image_t;",
    "", "static const test_image_t test_images[TEST_IMAGE_COUNT] = {",
]
for n, d, l in entries:
    lines.append(f'    {{{n.replace(".","_")}_data, {n.replace(".","_")}_size, {n.replace(".","_")}_label, "{n}"}},')
lines += ["};", "", "#endif"]
with open(os.path.join(OUT_DIR, "test_images.bin.h"), "w") as f:
    f.write("\n".join(lines))

print(f"\nTest accuracy: {correct}/{total} ({100.*correct/total:.2f}%)")
