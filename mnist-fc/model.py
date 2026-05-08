"""
Simple fully-connected network for MNIST digit recognition.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

@dataclass
class ModelArgs:
    input_dim: int = 784       # 28x28 flattened
    hidden_dims: List[int] = None  # e.g. [128, 64]
    output_dim: int = 10
    dropout: float = 0.0

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        if args.hidden_dims is None:
            args.hidden_dims = [128, 64]
        # Build layers
        layers = []
        in_dim = args.input_dim
        for h_dim in args.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if args.dropout > 0:
                layers.append(nn.Dropout(args.dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, args.output_dim))
        self.layers = nn.Sequential(*layers)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.layers(x)
