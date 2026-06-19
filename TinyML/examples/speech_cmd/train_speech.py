"""
Train a tiny CNN for Speech Commands (on / off / unknown) and export TMDL.

Architecture  (compact – designed for fast embedded inference):

  Input(32, 32, 1)
  Conv2D(8,  3×3, s1, same, relu)  → 32×32×8
  Conv2D(12, 3×3, s2, same, relu)  → 16×16×12
  Conv2D(20, 3×3, s2, same, relu)  → 8×8×20
  GlobalAveragePooling2D()          → 20
  Dense(3, softmax)                 → 3

Total model parameters ≈ 6 k  (very lightweight).

Classes:  0 = on   1 = off   2 = unknown
"""
import os
import sys
import json
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
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 60
NUM_TEST_BINS = 12  # export 4 per class for C inference

CLASS_NAMES = ["on", "off", "unknown"]

NORM_OFFSET = 80.0
NORM_SCALE  = 80.0


def normalise(x):
    return np.clip((x + NORM_OFFSET) / NORM_SCALE, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Spectrogram augmentation
# ---------------------------------------------------------------------------
def augment_spectrogram(spec):
    h, w = spec.shape
    aug = spec.copy()

    # Time masking
    if random.random() < 0.5:
        t = random.randint(1, max(1, w // 8))
        t0 = random.randint(0, w - t)
        aug[:, t0:t0 + t] = aug.min()

    # Frequency masking
    if random.random() < 0.4:
        f = random.randint(1, max(1, h // 10))
        f0 = random.randint(0, h - f)
        aug[f0:f0 + f, :] = aug.min()

    # Gain perturbation
    if random.random() < 0.5:
        gain = np.random.uniform(0.7, 1.3)
        aug = aug * gain

    return aug


def augment_batch(x_batch):
    out = x_batch.copy()
    for i in range(len(out)):
        out[i, :, :, 0] = augment_spectrogram(out[i, :, :, 0])
    return out


def data_generator(x, y, batch_size, augment=True):
    n = len(x)
    while True:
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            bx = x[batch_idx].copy()
            by = y[batch_idx]
            if augment:
                bx = augment_batch(bx)
            yield bx, by


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    x_train_list, y_train_list = [], []
    x_val_list,   y_val_list   = [], []
    x_test_list,  y_test_list  = [], []

    for label, name in enumerate(CLASS_NAMES):
        train = np.load(os.path.join(DATA_DIR, f"{name}_train.npy"))
        val   = np.load(os.path.join(DATA_DIR, f"{name}_val.npy"))
        test  = np.load(os.path.join(DATA_DIR, f"{name}_test.npy"))

        train = normalise(train)[..., np.newaxis]
        val   = normalise(val)[..., np.newaxis]
        test  = normalise(test)[..., np.newaxis]

        x_train_list.append(train)
        y_train_list.append(np.full(len(train), label, dtype=np.int32))
        x_val_list.append(val)
        y_val_list.append(np.full(len(val), label, dtype=np.int32))
        x_test_list.append(test)
        y_test_list.append(np.full(len(test), label, dtype=np.int32))

    x_train = np.concatenate(x_train_list).astype(np.float32)
    y_train = np.concatenate(y_train_list)
    x_val   = np.concatenate(x_val_list).astype(np.float32)
    y_val   = np.concatenate(y_val_list)
    x_test  = np.concatenate(x_test_list).astype(np.float32)
    y_test  = np.concatenate(y_test_list)

    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    print(f"Train: {len(x_train)}  Val: {len(x_val)}  Test: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1:]}  "
          f"range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    # Show per-class distribution
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: train={np.sum(y_train==i)}  "
              f"val={np.sum(y_val==i)}  test={np.sum(y_test==i)}")
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_data()

# ---------------------------------------------------------------------------
# Model  (tiny – ≈ 6 k params)
# ---------------------------------------------------------------------------
def build():
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=(32, 32, 1)),

        # Block 1:  32×32×1 → 32×32×8
        tf.keras.layers.Conv2D(8, 3, 1, 'same', activation='relu',
                               kernel_initializer='he_normal'),
        # Block 2:  32×32 → 16×16
        tf.keras.layers.Conv2D(12, 3, 2, 'same', activation='relu',
                               kernel_initializer='he_normal'),
        # Block 3:  16×16 → 8×8
        tf.keras.layers.Conv2D(20, 3, 2, 'same', activation='relu',
                               kernel_initializer='he_normal'),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])


model = build()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
steps_per_epoch = len(x_train) // BATCH_SIZE

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
]

print("\n--- Training with augmentation ---")
train_gen = data_generator(x_train, y_train, BATCH_SIZE, augment=True)
model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2)

# Fine-tune without augmentation
print("\n--- Fine-tuning ---")
for layer in model.layers[:2]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=min(25, EPOCHS // 2),
    validation_data=(x_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ],
    verbose=2)
for layer in model.layers:
    layer.trainable = True

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

tflite_path = os.path.join(OUT_DIR, "speech_cmd_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_buf)

# ---------------------------------------------------------------------------
# Compile TFLite → TMDL
# ---------------------------------------------------------------------------
tmdl_path = os.path.join(OUT_DIR, "speech_cmd_int8.tmdl")
hdr_path  = os.path.join(OUT_DIR, "speech_cmd_int8.bin.h")

old = sys.argv
sys.argv = ["compiler", tflite_path, tmdl_path,
            "--out-deq", "0", "-H", hdr_path]
try:
    compiler_main()
except SystemExit:
    pass
sys.argv = old

# ---------------------------------------------------------------------------
# Evaluate & export test samples (balanced: 4 per class)
# ---------------------------------------------------------------------------
interp = tf.lite.Interpreter(model_content=tflite_buf)
interp.allocate_tensors()
idet = interp.get_input_details()[0]
odet = interp.get_output_details()[0]
iscale, izp = idet['quantization']

print(f"\nINT8 quant: input_scale={iscale:.6f}  input_zp={izp}")

correct = 0
total = len(x_test)

# Pick balanced test bins: 4 per class
picks = set()
per_class = {}
for i in range(total):
    lbl = int(y_test[i])
    per_class.setdefault(lbl, []).append(i)
for lbl, idxs in per_class.items():
    picks.update(random.sample(idxs, min(NUM_TEST_BINS // NUM_CLASSES, len(idxs))))

entries = []
for i in range(total):
    f32 = x_test[i:i + 1]
    i8  = np.clip(np.round(f32 / iscale + izp), -128, 127).astype(np.int8)
    interp.set_tensor(idet['index'], i8)
    interp.invoke()
    out = interp.get_tensor(odet['index'])
    if np.argmax(out) == y_test[i]:
        correct += 1
    if i in picks:
        name = f"test_audio_{i}_label_{CLASS_NAMES[y_test[i]]}.bin"
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(i8.tobytes())
        entries.append((name, i8.tobytes(), int(y_test[i])))

# Write combined test_audios.bin.h
label_names_str = ", ".join('"' + n + '"' for n in CLASS_NAMES)
lines = [
    "/* Auto-generated test audio header – do not edit. */",
    "#ifndef TEST_AUDIOS_H", "#define TEST_AUDIOS_H",
    "#include <stdint.h>",
    f"#define TEST_AUDIO_COUNT {len(entries)}",
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
    "typedef struct { const int8_t *data; int size; int label; const char *name; } test_audio_t;",
    "", "static const test_audio_t test_audios[TEST_AUDIO_COUNT] = {",
]
for n, d, l in entries:
    v = n.replace('.', '_').replace('-', '_')
    lines.append(
        f'    {{{v}_data, {v}_size, {v}_label, "{n}"}},')
lines += ["};", "", "#endif"]
with open(os.path.join(OUT_DIR, "test_audios.bin.h"), "w") as f:
    f.write("\n".join(lines))

acc = 100. * correct / total
print(f"\nTest accuracy (TFLite INT8): {correct}/{total} ({acc:.2f}%)")

float_preds = model.predict(x_test, verbose=0)
float_correct = np.sum(np.argmax(float_preds, axis=1) == y_test)
print(f"Test accuracy (float32):   {float_correct}/{total} "
      f"({100. * float_correct / total:.2f}%)")

print(f"\nOutput files:")
print(f"  Model:       {tmdl_path}")
print(f"  C header:    {hdr_path}")
print(f"  Test header: {os.path.join(OUT_DIR, 'test_audios.bin.h')}")
