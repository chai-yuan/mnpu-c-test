"""Small utility helpers shared across the compiler."""

import numpy as np

from .config import UNIT_SIZE


def align8(x: int) -> int:
    """Round *x* up to the next multiple of 8 (required for TMDL alignment)."""
    return (x + 7) // 8 * 8


def shape_to_dims(shape) -> list:
    """Convert a tensor shape list (with optional batch dim) into the
    4‑element *dims* vector used by TMDL.

    The batch dimension (i.e. ``shape[0]``) is stripped first, matching
    the original ``tflite2tmdl`` logic.

    Examples
    --------
    * ``[1, 28, 28, 1]`` → ``[3, 28, 28, 1]``
    * ``[1, 10]``        → ``[1, 1, 1, 10]``
    """
    # Strip batch dim if present (same as original: shape[1:])
    spatial = shape[1:] if len(shape) > 1 and shape[0] == 1 else list(shape)
    ndim = len(spatial)
    dims = [ndim] + [1] * (3 - ndim)
    dims.extend(spatial)
    return dims


def cal_buf_size(layers: list, out_deq: int, log=print) -> tuple:
    """Compute ping‑pong buffer size and ADD keep‑buffer size.

    Returns ``(pingpong_buf_size, keep_buf_size)`` in bytes.
    """
    buf_sizes = []
    keep_sizes = [0]

    for layer in layers:
        in_vol = np.prod(layer["in_shape"])
        out_vol = np.prod(layer["out_shape"])
        name = layer["name"]

        if layer["is_output"] and out_deq:
            buf_size = (
                align8(in_vol * UNIT_SIZE)
                + align8(out_vol * UNIT_SIZE)
                + align8(out_vol * 4)
            )
            if layer["is_keep"]:
                raise ValueError("keep flag is not supported on output layer")
        elif name == "SOFTMAX":
            buf_size = align8(in_vol * UNIT_SIZE) + align8(out_vol * 4)
            if layer["is_keep"]:
                raise ValueError("keep flag is not supported on SOFTMAX layer")
        else:
            buf_size = align8(in_vol * UNIT_SIZE) + align8(out_vol * UNIT_SIZE)
            if layer["is_keep"]:
                keep_sizes.append(align8(out_vol * UNIT_SIZE))

        if name == "RESHAPE":
            buf_size -= align8(in_vol * UNIT_SIZE)  # in‑place

        buf_sizes.append(buf_size)

    total_buf = max(buf_sizes)
    total_keep = max(keep_sizes)
    log(f"  Ping‑pong buf : {total_buf} Byte")
    log(f"  ADD keep buf  : {total_keep} Byte")
    log(f"  Total         : {total_buf + total_keep} Byte")
    return total_buf, total_keep
