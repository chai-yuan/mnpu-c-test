"""
Train a simple MLP on MNIST, apply full INT8 quantization, 
and export to a .tflite file along with sample inputs for C inference.
"""
import os
import random
import struct
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# -----------------------------------------------------------------------------
# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 6
out_dir = 'out'
model_tflite_name = 'mnist_full_int8.tflite'
num_test_bins = 10  # Number of quantized test images to export as .bin

# -----------------------------------------------------------------------------
# Data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# -----------------------------------------------------------------------------
# Model, optimizer, loss
def create_model(batch_size):
    return tf.keras.models.Sequential([
        tf.keras.Input(batch_shape=(batch_size, 28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model = create_model(batch_size=None)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------------------------------------------------------
# Training
os.makedirs(out_dir, exist_ok=True)
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# -----------------------------------------------------------------------------
# INT8 Quantization & Export
export_model = create_model(batch_size=1)
export_model.set_weights(model.get_weights())

def representative_data_gen():
    for i in range(100):
        data = train_images[i:i+1].astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
tflite_path = os.path.join(out_dir, model_tflite_name)
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"\nModel exported to: {tflite_path}")

# -----------------------------------------------------------------------------
# Evaluation & Test Data Export
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

correct = 0
total = len(test_images)
bin_indices = set(random.sample(range(total), num_test_bins))

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
        bin_filename = os.path.join(out_dir, f"test_img_{i}_label_{test_labels[i]}.bin")
        with open(bin_filename, "wb") as f:
            f.write(img_int8.tobytes())
        print(f"Exported test image to {bin_filename}")

acc = 100. * correct / total
print(f"\nTest ---- accuracy: {correct}/{total} ({acc:.2f}%)")