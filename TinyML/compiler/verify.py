#!/usr/bin/env python3
"""Compare the old and new compiler outputs byte‑for‑byte."""

import sys
import os
import tempfile
import hashlib
import types

# ── Add the workspace root so we can import both packages ─────────────
_ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _ws = /workspace/TinyML, and compiler/ is its child
if _ws not in sys.path:
    sys.path.insert(0, _ws)

# ── 1. New compiler ──────────────────────────────────────────────────
from compiler.tflite_reader import read_tflite as new_read
from compiler.packer import pack_tmdl as new_pack


# ── 2. Original packer ───────────────────────────────────────────────
# The original tflite_reader.py has stale keras imports on Keras 3.x.
# Load its source and strip them out.
_rdr_path = os.path.join(_ws, "TinyMaix/tools/tflite_reader.py")
_rdr_src = open(_rdr_path).read()
# Strip unused keras imports that are incompatible with Keras 3.x
_lines_to_skip = [
    "from keras.datasets import mnist",
    "from tensorflow.python.keras.backend import set_session",
    "from tensorflow.python.keras.models import load_model",
    "from tensorflow.keras.models import Model, load_model, Sequential",
    "from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, "
    "Softmax, Activation, BatchNormalization, Flatten, Dropout, DepthwiseConv2D",
    "from tensorflow.keras.layers import MaxPool2D, AvgPool2D, "
    "AveragePooling2D, GlobalAveragePooling2D,ZeroPadding2D,Input,Embedding,PReLU",
    "from keras.callbacks import ModelCheckpoint",
    "from keras.callbacks import TensorBoard",
    "from keras.preprocessing.image import ImageDataGenerator",
    "import keras.backend as K",
    "import time",
]
for _line in _lines_to_skip:
    _rdr_src = _rdr_src.replace(_line + "\n", "")

_mod = types.ModuleType("tflite_reader_patched")
exec(compile(_rdr_src, "tflite_reader_patched.py", "exec"), _mod.__dict__)
old_read = _mod.read_tflite

# Also load the original packer (tflite2tmdl.py)
_pkr_path = os.path.join(_ws, "TinyMaix/tools/tflite2tmdl.py")
_pkr_src = open(_pkr_path).read()
# Fix relative import → reference our already-patched reader
_pkr_src = _pkr_src.replace(
    "from .tflite_reader import read_tflite", "")
_mod2 = types.ModuleType("tflite2tmdl_patched")
exec(compile(_pkr_src, "tflite2tmdl_patched.py", "exec"), _mod2.__dict__)
old_pack = _mod2.pack_tmdl
_mod2.read_tflite = old_read  # patch the import reference


def compare(tflite_path, out_deq=1, in_dims=(28, 28, 1), out_dims=(10,)):
    """Run both compilers and compare their outputs byte‑for‑byte."""
    print(f"  Model : {tflite_path}")
    print(f"  In    : {in_dims}  Out : {out_dims}")

    # ── New compiler ─────────────────────────────────────────────────
    layers_new = new_read(tflite_path)
    fd_new, tmp_new = tempfile.mkstemp(suffix=".tmdl")
    os.close(fd_new)
    new_pack(layers_new, tmp_new, in_dims, out_dims, out_deq=out_deq)

    # ── Old compiler ─────────────────────────────────────────────────
    layers_old = old_read(tflite_path)
    fd_old, tmp_old = tempfile.mkstemp(suffix=".tmdl")
    os.close(fd_old)
    # TM_MDL_INT8 = 0
    # NOTE: write_c_header=True to avoid UnboundLocalError in the original
    try:
        old_pack(layers_old, tmp_old, 0, out_deq,
                 list(in_dims), list(out_dims), "<", write_c_header=True)
    except Exception as e:
        print(f"  Old compiler crashed: {e}")
        # File was already written before the crash – continue comparison

    # ── Compare ──────────────────────────────────────────────────────
    with open(tmp_new, "rb") as f:
        d_new = f.read()
    with open(tmp_old, "rb") as f:
        d_old = f.read()

    os.unlink(tmp_new)
    os.unlink(tmp_old)

    if d_new == d_old:
        h = hashlib.sha256(d_new).hexdigest()[:16]
        print(f"  \033[32m✓ IDENTICAL\033[0m  ({len(d_new)} B, sha256={h}…)")
        return True
    else:
        print(f"  \033[31m✗ MISMATCH\033[0m  new={len(d_new)} B  old={len(d_old)} B")
        limit = min(len(d_new), len(d_old))
        for i in range(limit):
            if d_new[i] != d_old[i]:
                print(f"  First diff at byte {i}: "
                      f"new=0x{d_new[i]:02x}  old=0x{d_old[i]:02x}")
                # Print context around the diff
                ctx_s = max(0, i - 16)
                print(f"  Context: new[{ctx_s}:{i+16}] = "
                      f"{d_new[ctx_s:i+16].hex()}")
                print(f"           old[{ctx_s}:{i+16}] = "
                      f"{d_old[ctx_s:i+16].hex()}")
                break
        else:
            print(f"  Files differ only in length "
                  f"(new={len(d_new)}, old={len(d_old)})")
        return False


def main():
    base = os.path.join(_ws, "TinyMaix/tools/tflite")
    user_model = os.path.join(os.path.dirname(_ws), "mnist-int8/out/mnist_full_int8.tflite")

    tests = [
        (os.path.join(base, "mnist_dw_q.tflite"),     (28, 28, 1), (10,)),
        (os.path.join(base, "mnist_valid_q.tflite"),  (28, 28, 1), (10,)),
        (os.path.join(base, "mnist_resnet_q.tflite"), (28, 28, 1), (10,)),
    ]

    # + user's model if it exists
    if os.path.exists(user_model):
        tests.append((user_model, (28, 28, 1), (10,)))

    all_ok = True
    for path, in_d, out_d in tests:
        print(f"\n{'=' * 56}")
        ok = compare(path, in_dims=in_d, out_dims=out_d)
        all_ok = all_ok and ok

    print(f"\n{'=' * 56}")
    if all_ok:
        print("  \033[32mALL TESTS PASSED ✓\033[0m")
    else:
        print("  \033[31mSOME TESTS FAILED ✗\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
