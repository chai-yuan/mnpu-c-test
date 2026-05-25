"""
Export TFLite INT8 model to a custom binary format for C inference.

Binary layout (512-byte header + per-layer data):
  Header: magic(4) | version(4) | input_dim(4) | num_layers(4) |
          layer_out_features[16](64) | input_zp(4) |
          output_zp(4) | reserved(424)
  Each layer:
    requant_multiplier[out_c]: int32
    requant_shift[out_c]: int32
    out_zero_point: int32
    bias[out_c]: int32
    weight[out_c * in_c]: int8
"""
import os
import struct
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

HEADER_SIZE = 512
MAX_LAYERS  = 16
MAGIC       = 0x4D4C4938  # "MLI8"

# -----------------------------------------------------------------------------
# Quantization helpers (same as manual_inference.py)
# -----------------------------------------------------------------------------
def _get_quant_params(details, tensor_idx):
    """Extract quantization parameters: returns (scales, zero_points)."""
    detail = details[tensor_idx]
    qp = detail.get('quantization_parameters', None)
    if qp is not None and qp['scales'].size > 0:
        scales = qp['scales'].copy()
        zps = qp['zero_points'].copy()
        return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.int32)
    scale, zp = detail['quantization']
    return np.array(scale, dtype=np.float32), np.array(zp, dtype=np.int32)


def _quantize_multiplier(double_multiplier):
    """TFLite QuantizeMultiplier: float → (Q0.31 int32 multiplier, int32 shift)."""
    if double_multiplier == 0.0:
        return np.int32(0), np.int32(0)
    m, shift = math.frexp(double_multiplier)
    q_fixed = int(round(m * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        shift = 0
        q_fixed = 0
    return np.int32(q_fixed), np.int32(shift)


def _compute_int_multipliers(s_in, s_w, s_out):
    """Pre-compute per-channel (q_mult, shift) for requantization."""
    s_in_val  = float(np.squeeze(s_in))
    s_out_val = float(np.squeeze(s_out))
    eff_scale = (s_in_val * s_w.astype(np.float64)) / s_out_val
    q_mults = np.zeros(s_w.size, dtype=np.int32)
    shifts  = np.zeros(s_w.size, dtype=np.int32)
    for i in range(s_w.size):
        q_mults[i], shifts[i] = _quantize_multiplier(eff_scale[i])
    return q_mults, shifts

# -----------------------------------------------------------------------------
# Main export function
# -----------------------------------------------------------------------------
def export_tflite_to_bin(tflite_path, bin_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    ops = interpreter._get_ops_details()

    fc_ops = [op for op in ops if op['op_name'] == 'FULLY_CONNECTED']
    assert len(fc_ops) == 2, f"Expected 2 FC layers, got {len(fc_ops)}"

    # --- Extract per-layer data ---
    layers_raw = []
    for fc_op in fc_ops:
        in_idx, w_idx, b_idx = fc_op['inputs']
        out_idx = fc_op['outputs'][0]

        in_scale, in_zp  = _get_quant_params(details, in_idx)
        w                = interpreter.get_tensor(w_idx)
        b                = interpreter.get_tensor(b_idx)
        w_scale, w_zp    = _get_quant_params(details, w_idx)
        out_scale, out_zp = _get_quant_params(details, out_idx)

        q_mults, shifts = _compute_int_multipliers(in_scale, w_scale, out_scale)

        # Verify weight zero-point is 0 (TFLite INT8 symmetric quantization)
        assert np.all(w_zp == 0), f"Expected weight zp=0, got {w_zp}"

        layers_raw.append({
            'in_features':   int(w.shape[1]),
            'out_features':  int(w.shape[0]),
            'weight':        w.astype(np.int8),
            'bias':          b.astype(np.int32),
            'out_zp':        int(np.squeeze(out_zp)),
            'q_mults':       q_mults.astype(np.int32),
            'shifts':        shifts.astype(np.int32),
        })

    # --- Global input / output params ---
    first_in_idx   = fc_ops[0]['inputs'][0]
    last_out_idx   = fc_ops[-1]['outputs'][0]
    inp_scale, inp_zp = _get_quant_params(details, first_in_idx)
    out_scale, out_zp = _get_quant_params(details, last_out_idx)

    input_zp     = int(np.squeeze(inp_zp))
    output_zp    = int(np.squeeze(out_zp))

    # --- Write binary ---
    with open(bin_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))                     # magic
        f.write(struct.pack('<i', 1))                         # version
        f.write(struct.pack('<i', layers_raw[0]['in_features']))  # input_dim
        f.write(struct.pack('<i', len(layers_raw)))           # num_layers
        for layer in layers_raw:
            f.write(struct.pack('<i', layer['out_features']))
        for _ in range(MAX_LAYERS - len(layers_raw)):
            f.write(struct.pack('<i', 0))
        f.write(struct.pack('<i', input_zp))
        f.write(struct.pack('<i', output_zp))
        # Padding
        header_used = 4 + 4 + 4 + 4 + MAX_LAYERS * 4 + 4 + 4
        f.write(b'\x00' * (HEADER_SIZE - header_used))

        # Layer data
        for layer in layers_raw:
            out_c, in_c = layer['out_features'], layer['in_features']
            f.write(layer['q_mults'].tobytes())                        # int32[out_c]
            f.write(layer['shifts'].tobytes())                         # int32[out_c]
            f.write(struct.pack('<i', layer['out_zp']))                # int32
            f.write(layer['bias'].tobytes())                           # int32[out_c]
            f.write(layer['weight'].tobytes())                         # int8[out_c * in_c]

    file_size = os.path.getsize(bin_path)
    print(f"✅ Model exported to: {bin_path} ({file_size:,} bytes)")
    print(f"   Architecture: {layers_raw[0]['in_features']} → {' → '.join(str(l['out_features']) for l in layers_raw)}")
    print(f"   Input:  zp={input_zp}")
    for i, l in enumerate(layers_raw):
        print(f"   Layer {i}: out={l['out_features']}, out_zp={l['out_zp']}")
    print(f"   Output: zp={output_zp}")

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export TFLite INT8 model to binary for C inference')
    parser.add_argument('--tflite', type=str, default='out/mnist_full_int8.tflite', help='input .tflite file')
    parser.add_argument('--output', type=str, default='out/model.bin', help='output .bin file')
    args = parser.parse_args()
    export_tflite_to_bin(args.tflite, args.output)
