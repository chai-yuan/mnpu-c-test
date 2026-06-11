# TinyML Compiler - Convert TFLite INT8 models to TMDL format.
#
# This package provides a clean, extensible compiler for converting
# TensorFlow Lite quantized (INT8) models into the TinyMaix TMDL format
# suitable for microcontrollers.
#
# Usage:
#     python -m compiler <tflite_path> <output_path> --in-dims H,W,C ...
#
# Or programmatically:
#     from compiler import compile_model
#     compile_model("model.tflite", "model.tmdl", in_dims=(28,28,1))

from .main import main as _main

__version__ = "1.0.0"
