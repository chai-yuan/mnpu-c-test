"""
Export trained MLP to a binary file for C inference.
Supports version 1 (basic weights) and version 2 (with input normalization params).
"""
import os
import struct
import argparse
import torch
import numpy as np
from model import MLP, ModelArgs

# -----------------------------------------------------------------------------
# Export helpers
def serialize_fp32(file, tensor):
    """Write one fp32 tensor to a file open in 'wb' mode."""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

HEADER_SIZE = 512
MAX_LAYERS = 16

def model_export(model, filepath, version=3):
    """
    Export model weights to .bin file.
    Header layout (fixed 512 bytes):
        Offset  Size    Field
        0       4       magic:      uint32 = 0x4d4c5046  ("MLPF")
        4       4       version:    int32
        8       4       input_dim:  int32
        12      4       output_dim: int32
        16      4       num_layers: int32
        20      64      layer_out_features[16]: int32[16] (0 for unused slots)
        84      4       input_mean: float32
        88      4       input_std:  float32
        92      420     reserved:   bytes (padding to 512 bytes)
        512     ...     weights:
                        for each linear layer:
                            weight (out_features * in_features floats)
                            bias   (out_features floats)
    """
    linear_layers = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) > MAX_LAYERS:
        raise ValueError(f"Too many layers: {len(linear_layers)} > {MAX_LAYERS}")

    # Hard-coded MNIST normalization values (used in training)
    input_mean = 0.1307
    input_std  = 0.3081

    with open(filepath, 'wb') as f:
        # 1) magic
        f.write(struct.pack('I', 0x4d4c5046))
        # 2) version
        f.write(struct.pack('i', version))
        # 3) dimensions
        f.write(struct.pack('i', model.args.input_dim))
        f.write(struct.pack('i', model.args.output_dim))
        # 4) number of linear layers
        f.write(struct.pack('i', len(linear_layers)))
        # 5) per-layer out_features (fixed 16 slots)
        for layer in linear_layers:
            f.write(struct.pack('i', layer.out_features))
        for _ in range(MAX_LAYERS - len(linear_layers)):
            f.write(struct.pack('i', 0))
        # 6) input normalization
        f.write(struct.pack('f', input_mean))
        f.write(struct.pack('f', input_std))
        # 7) padding to 512 bytes
        header_used = 4 + 4 + 4 + 4 + 4 + MAX_LAYERS * 4 + 4 + 4  # = 92
        padding = HEADER_SIZE - header_used
        f.write(b'\x00' * padding)
        # 8) weights and biases
        for layer in linear_layers:
            weight = layer.weight  # shape: [out, in]
            bias = layer.bias      # shape: [out]
            serialize_fp32(f, weight)
            serialize_fp32(f, bias)

    print(f"exported model to {filepath} (version {version})")

# -----------------------------------------------------------------------------
# Load functions
def load_checkpoint(checkpoint_path):
    """Load model from a training checkpoint (.pt)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = MLP(args)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

# -----------------------------------------------------------------------------
# CLI entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MLP to binary format')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint.pt')
    parser.add_argument('--output', type=str, default='model.bin', help='output binary file')
    parser.add_argument('--version', type=int, default=3, help='export version (3 = fixed 512-byte header)')
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model_export(model, args.output, version=args.version)
