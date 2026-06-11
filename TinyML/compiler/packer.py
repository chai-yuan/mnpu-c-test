"""TMDL binary packer – assemble layer metadata and weights into the
TinyMaix model format.

All binary layout details are encapsulated here so that the CLI layer
only deals with argument parsing.

Only INT8 quantised models are supported (``mdl_type == TM_MDL_INT8``).
"""

import struct
import numpy as np

from .config import (
    TM_MDL_INT8,
    UNIT_SIZE,
    BUNIT_SIZE,
    LAYER_NAME_TO_TYPE,
    MDLBINHEAD_SIZE,
    LAYERHEAD_SIZE,
    TM_PAD_SAME,
    TM_PAD_VALID,
)
from .utils import align8, shape_to_dims, cal_buf_size


# ---------------------------------------------------------------------------
# Top‑level entry point
# ---------------------------------------------------------------------------

def pack_tmdl(layers: list,
              output_path: str,
              in_dims: tuple,
              out_dims: tuple,
              out_deq: int = 1,
              big_endian: bool = False,
              c_header_path: str = None,
              log=print):
    """Pack *layers* into a ``.tmdl`` file and write it to *output_path*.

    Parameters
    ----------
    layers : list[dict]
        Layer descriptors as returned by :func:`tflite_reader.read_tflite`.
    output_path : str or Path
        Destination ``.tmdl`` file.
    in_dims : tuple[int]
        Input dimensions excluding batch, e.g. ``(28, 28, 1)``.
    out_dims : tuple[int]
        Output dimensions excluding batch, e.g. ``(10,)``.
    out_deq : int
        1 to enable output dequantisation, 0 otherwise.
    big_endian : bool
        If True, write multi‑byte integers in big‑endian order.
    c_header_path : str or None
        If given, also write a C header file (``.h``) with the model
        as a ``const uint8_t`` array, ready for ``#include``.
    log : callable
        Logger function (default ``print``).
    """
    endian = ">" if big_endian else "<"

    # ── validate model type ──────────────────────────────────────────
    if not layers[0].get("quant"):
        raise RuntimeError(
            "Model is not quantised. This compiler only supports INT8 models."
        )

    log("=" * 56)
    log("  PACKING TMDL MODEL")
    log("=" * 56)
    log(f"  Model type      : INT8")
    log(f"  Output dequant  : {'yes' if out_deq else 'no'}")
    log(f"  Endian          : {'big' if big_endian else 'little'}")
    log(f"  Input dims      : {in_dims}")
    log(f"  Output dims     : {out_dims}")
    log(f"  Layer count     : {len(layers)}")

    # ── pre‑compute buffer sizes ─────────────────────────────────────
    buf_size, keep_size = cal_buf_size(layers, out_deq, log=log)
    total_buf = buf_size + keep_size
    log(f"  Total RAM buf   : {total_buf} Byte")

    # ── dims vectors (4× uint16) ─────────────────────────────────────
    in_dims_vec = shape_to_dims(list(in_dims))
    out_dims_vec = shape_to_dims(list(out_dims))
    log(f"  In dims vector  : {in_dims_vec}")
    log(f"  Out dims vector : {out_dims_vec}")

    # ── allocate offsets helper ───────────────────────────────────────
    ctx = _BufferCtx(buf_size, UNIT_SIZE)

    with open(output_path, "wb") as f:
        # ── model header ─────────────────────────────────────────────
        header = _pack_model_header(
            endian, out_deq, len(layers), total_buf,
            in_dims_vec, out_dims_vec, log
        )
        f.write(header)
        model_size = len(header)

        # ── layers ───────────────────────────────────────────────────
        layer_sizes = []
        for idx, layer in enumerate(layers):
            log(f"\n  ── Layer {idx}: {layer['name']} "
                f"{'(KEEP)' if layer['is_keep'] else ''}")

            lh, lbody, layer_size = _pack_one_layer(
                layer, idx, ctx, endian, buf_size, out_deq, log)
            f.write(lh + lbody)
            model_size += layer_size
            layer_sizes.append(layer_size)

        log(f"\n  ── Packing complete ──")
        log(f"  Model size : {model_size} B  ({model_size / 1024:.1f} KB)")
        log(f"  RAM buffer : {total_buf} B  ({total_buf / 1024:.1f} KB)")

    # ── optional C header ──────────────────────────────────────────
    if c_header_path is not None:
        _write_c_header(output_path, c_header_path, buf_size, layer_sizes, log)

    return model_size, total_buf, layer_sizes


# ---------------------------------------------------------------------------
# Buffer offset context (emulates the original ping‑pong scheme)
# ---------------------------------------------------------------------------

class _BufferCtx:
    """Tracks ping‑pong buffer offsets across layers."""

    def __init__(self, buf_size: int, unit_size: int):
        self.buf_size = buf_size
        self.unit_size = unit_size
        self.out_oft = 0
        self.out_size = 0
        self.out_oft_virt = 0

    def set_initial_out_size(self, in_dims_vec):
        self.out_size = int(np.prod(in_dims_vec[1:])) * self.unit_size

    def compute_offsets(self, layer: dict, out_deq: int):
        """Update and return ``(in_oft, out_oft)`` for *layer*."""
        in_size = self.out_size

        out_vol = int(np.prod(layer["out_shape"]))
        name = layer["name"]

        if layer["is_output"] and out_deq:
            self.out_size = (align8(out_vol * self.unit_size)
                             + align8(out_vol * 4))
        elif name == "SOFTMAX":
            self.out_size = align8(out_vol * 4)
        else:
            self.out_size = align8(out_vol * self.unit_size)

        in_oft = self.out_oft

        if name == "RESHAPE":
            self.out_oft = in_oft
        else:
            if self.out_oft != self.buf_size:
                if not layer["is_keep"]:
                    self.out_oft = (0 if self.out_oft != 0
                                    else self.buf_size - self.out_size)
                else:
                    self.out_oft_virt = (0 if self.out_oft != 0
                                         else self.buf_size - self.out_size)
                    self.out_oft = self.buf_size
            else:
                self.out_oft = (0 if self.out_oft_virt != 0
                                else self.buf_size - self.out_size)

        return in_oft, self.out_oft


# ---------------------------------------------------------------------------
# Model header (64 bytes)
# ---------------------------------------------------------------------------

def _pack_model_header(endian, out_deq, layer_cnt, total_buf,
                       in_dims_vec, out_dims_vec, log) -> bytes:
    """Build the 64‑byte TMDL model header."""
    hdr = b"MAIX"
    hdr += struct.pack(endian + "B",  TM_MDL_INT8)
    hdr += struct.pack(endian + "B",  out_deq)
    hdr += struct.pack(endian + "H",  1)               # input_cnt
    hdr += struct.pack(endian + "H",  1)               # output_cnt
    hdr += struct.pack(endian + "H",  layer_cnt)
    hdr += struct.pack(endian + "I",  total_buf)
    hdr += struct.pack(endian + "I",  0)               # sub_size
    hdr += struct.pack(endian + "4H", *in_dims_vec)
    hdr += struct.pack(endian + "4H", *out_dims_vec)
    hdr += bytes(28)  # pad to 64 bytes

    log(f"  Header size     : {len(hdr)} B")
    assert len(hdr) == MDLBINHEAD_SIZE, \
        f"Header is {len(hdr)} B, expected {MDLBINHEAD_SIZE}"
    return hdr


# ---------------------------------------------------------------------------
# Layer header (48 bytes)
# ---------------------------------------------------------------------------

def _pack_layer_header(endian, layer, layer_type, in_oft, out_oft,
                       in_dims_vec, out_dims_vec) -> bytearray:
    """Build the 48‑byte per‑layer header (layer_size is zero initially)."""
    lh = bytearray()
    lh += struct.pack(endian + "H",  layer_type)
    lh += struct.pack(endian + "H",  layer["is_output"])
    lh += struct.pack(endian + "I",  0)               # layer_size placeholder
    lh += struct.pack(endian + "I",  in_oft)
    lh += struct.pack(endian + "I",  out_oft)
    lh += struct.pack(endian + "4H", *in_dims_vec)
    lh += struct.pack(endian + "4H", *out_dims_vec)
    lh += struct.pack(endian + "f",  layer["i_scale"])
    lh += struct.pack(endian + "i",  int(layer["i_zeropoint"]))
    lh += struct.pack(endian + "f",  layer["o_scale"])
    lh += struct.pack(endian + "i",  int(layer["o_zeropoint"]))

    assert len(lh) == LAYERHEAD_SIZE, \
        f"Layer header is {len(lh)} B, expected {LAYERHEAD_SIZE}"
    return lh


# ---------------------------------------------------------------------------
# Pack one layer → (header, body, total_size)
# ---------------------------------------------------------------------------

def _pack_one_layer(layer, idx, ctx, endian, buf_size, out_deq, log):
    """Pack a single layer, returning ``(layer_head, layer_body, total_size)``."""
    name = layer["name"]

    layer_type = LAYER_NAME_TO_TYPE[name]
    in_dims_vec = shape_to_dims(layer["in_shape"])
    out_dims_vec = shape_to_dims(layer["out_shape"])

    log(f"       shape   : in={in_dims_vec}  out={out_dims_vec}")

    # ── buffer offsets ───────────────────────────────────────────────
    in_oft, out_oft = ctx.compute_offsets(layer, out_deq)
    log(f"       offsets : in={in_oft}  out={out_oft}")

    # ── layer header ─────────────────────────────────────────────────
    lh = _pack_layer_header(endian, layer, layer_type,
                            in_oft, out_oft, in_dims_vec, out_dims_vec)

    # ── layer body ───────────────────────────────────────────────────
    lbody = _pack_layer_body(layer, name, endian, buf_size, log)

    # ── fill actual layer size ───────────────────────────────────────
    layer_size = len(lh) + len(lbody)
    lh[4:8] = struct.pack(endian + "I", layer_size)
    log(f"       size    : {layer_size} B  (hdr={len(lh)}  body={len(lbody)})")

    return bytes(lh), lbody, layer_size


# ---------------------------------------------------------------------------
# Layer body packers (dispatched by layer name)
# ---------------------------------------------------------------------------

def _pack_layer_body(layer, name, endian, buf_size, log) -> bytes:
    if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        return _pack_conv_body(layer, endian, log)
    elif name == "MEAN":
        return _pack_gap_body(layer, log)
    elif name == "FULLY_CONNECTED":
        return _pack_fc_body(layer, endian, log)
    elif name == "SOFTMAX":
        return _pack_softmax_body()
    elif name == "RESHAPE":
        return _pack_reshape_body()
    elif name == "ADD":
        return _pack_add_body(layer, endian, buf_size, log)
    else:
        raise RuntimeError(f"Unknown layer type for packing: {name}")


# ──────────────────────────────── CONV ────────────────────────────────

def _pack_conv_body(layer, endian, log) -> bytes:
    """Pack body for CONV_2D and DEPTHWISE_CONV_2D."""
    is_dwconv = (layer["name"] == "DEPTHWISE_CONV_2D")

    w = layer["weight"].transpose(0, 3, 1, 2).flatten()  # co,ci,h,w → flat
    b = layer.get("bias", np.zeros((layer["out_shape"][-1],))).copy()
    kw = layer["weight"].shape[2]
    kh = layer["weight"].shape[1]

    # Fuse input zero‑point into bias
    mzp = int(layer["i_zeropoint"])
    maxk = kh * kw
    mi_c = layer["in_shape"][-1] if not is_dwconv else 1
    mo_c = layer["out_shape"][-1]
    tmp = np.array([np.sum(w[c * mi_c * maxk:(c + 1) * mi_c * maxk])
                    for c in range(mo_c)], dtype=np.int64)
    b = b.astype(np.int64) + (-mzp * tmp)
    b = b.astype(np.int32)

    body = b""
    body += struct.pack("B", kw)
    body += struct.pack("B", kh)
    body += struct.pack("B", layer["stride_w"])
    body += struct.pack("B", layer["stride_h"])
    body += struct.pack("B", layer["dilation_w_factor"])
    body += struct.pack("B", layer["dilation_h_factor"])
    body += struct.pack(endian + "H",
                        layer.get("fused_activation_function", 0))

    # ── padding ──────────────────────────────────────────────────────
    kernel_ext_w = layer["dilation_w_factor"] * (kw - 1) + 1
    kernel_ext_h = layer["dilation_h_factor"] * (kh - 1) + 1

    pad_type = layer.get("padding", 0)  # 0=same, 1=valid, 2=fused

    if pad_type == TM_PAD_SAME:  # 0
        wpad = (kernel_ext_w
                + (layer["in_shape"][2] - 1) // layer["stride_w"] * layer["stride_w"]
                - layer["in_shape"][2])
        hpad = (kernel_ext_h
                + (layer["in_shape"][1] - 1) // layer["stride_h"] * layer["stride_h"]
                - layer["in_shape"][1])
        assert wpad >= 0 and hpad >= 0, \
            f"Negative padding: wpad={wpad}, hpad={hpad}"
        pad_t, pad_b = hpad // 2, hpad - hpad // 2
        pad_l, pad_r = wpad // 2, wpad - wpad // 2
        log(f"       pad     : same (T,B,L,R) = {pad_t},{pad_b},{pad_l},{pad_r}")
        body += struct.pack("BBBB", pad_t, pad_b, pad_l, pad_r)

    elif pad_type == TM_PAD_VALID:  # 1
        log("       pad     : valid")
        body += struct.pack(endian + "I", 0)

    elif pad_type == 2:  # fused PAD
        lp = layer["pad"]
        log(f"       pad     : fused (T,B,L,R) = {lp[0]},{lp[1]},{lp[2]},{lp[3]}")
        body += struct.pack("BBBB", lp[0], lp[1], lp[2], lp[3])

    else:
        raise RuntimeError(f"Unsupported padding type: {pad_type}")

    # depth_multiplier
    body += struct.pack(endian + "I",
                        0 if not is_dwconv else layer["depth_multiplier"])
    body += struct.pack(endian + "I", 0)  # pad

    # ── offsets and data ─────────────────────────────────────────────
    ws_oft = LAYERHEAD_SIZE + len(body) + 12  # 3× uint32 offsets
    ws_size = align8(mo_c * 4)
    w_oft = ws_oft + ws_size
    w_size = align8(w.size * UNIT_SIZE)
    b_oft = w_oft + w_size
    b_size = align8(b.size * BUNIT_SIZE)

    body += struct.pack(endian + "I", ws_oft)
    body += struct.pack(endian + "I", w_oft)
    body += struct.pack(endian + "I", b_oft)
    assert len(body) % 8 == 0, f"Body length {len(body)} not 8‑aligned"

    # ── weight scales (per output channel, float32) ──────────────────
    ws = layer["w_scale"]
    body += struct.pack(endian + f"{ws.size}f", *ws)
    if ws_size != ws.size * 4:
        body += bytes(ws_size - ws.size * 4)
    assert len(body) % 8 == 0, f"After ws: {len(body)}"

    # ── weights (int8) ──────────────────────────────────────────────
    body += w.astype(np.int8).tobytes()
    if w_size != w.size * UNIT_SIZE:
        body += bytes(w_size - w.size * UNIT_SIZE)
    assert len(body) % 8 == 0, f"After w: {len(body)}"

    # ── bias (int32) ─────────────────────────────────────────────────
    body += struct.pack(endian + f"{b.size}i", *b)
    if b_size != b.size * BUNIT_SIZE:
        body += bytes(b_size - b.size * BUNIT_SIZE)
    assert len(body) % 8 == 0, f"After b: {len(body)}"

    log(f"       w/scale : {ws.size} ch, weight={list(w.shape[-2:])}→{w.size}")
    return body


# ──────────────────────────────── GAP ─────────────────────────────────

def _pack_gap_body(layer, log) -> bytes:
    r"""Pack MEAN / Global‑Average‑Pooling body.

    Only ``reduce_idx == [0, 1]`` (height + width) is supported.
    """
    if list(layer.get("reduce_idx", [0, 1])) != [0, 1]:
        raise RuntimeError("Only Global‑Average‑Pooling (reduce h,w) is supported")
    log("       reduce  : [0,1] (GAP)")
    return b""


# ──────────────────────────────── FC ──────────────────────────────────

def _pack_fc_body(layer, endian, log) -> bytes:
    """Pack body for FULLY_CONNECTED."""
    mi_c = layer["in_shape"][-1]
    mo_c = layer["out_shape"][-1]

    w = layer["weight"].flatten()
    b = layer.get("bias", np.zeros((mo_c,))).copy().astype(np.int64)
    # Fuse input zero‑point into bias
    mzp = int(layer["i_zeropoint"])
    tmp = np.array([np.sum(w[c * mi_c:(c + 1) * mi_c])
                    for c in range(mo_c)], dtype=np.int64)
    b = b + (-mzp * tmp)
    b = b.astype(np.int32)

    body = b""
    ws_oft = LAYERHEAD_SIZE + len(body) + 16  # 4× uint32
    ws_size = align8(mo_c * 4)
    w_oft = ws_oft + ws_size
    w_size = align8(w.size * UNIT_SIZE)
    b_oft = w_oft + w_size
    b_size = align8(b.size * BUNIT_SIZE)

    body += struct.pack(endian + "I", ws_oft)
    body += struct.pack(endian + "I", w_oft)
    body += struct.pack(endian + "I", b_oft)
    body += struct.pack(endian + "I", 0)  # reserve (align)
    assert len(body) % 8 == 0

    # ws (float32 per output channel)
    ws = layer["w_scale"]
    body += struct.pack(endian + f"{ws.size}f", *ws)
    if ws_size != ws.size * 4:
        body += bytes(ws_size - ws.size * 4)
    assert len(body) % 8 == 0

    # w (int8)
    body += w.astype(np.int8).tobytes()
    if w_size != w.size * UNIT_SIZE:
        body += bytes(w_size - w.size * UNIT_SIZE)
    assert len(body) % 8 == 0

    # b (int32)
    body += struct.pack(endian + f"{b.size}i", *b)
    if b_size != b.size * BUNIT_SIZE:
        body += bytes(b_size - b.size * BUNIT_SIZE)
    assert len(body) % 8 == 0

    log(f"       w/scale : {ws.size} ch, weight={mi_c}×{mo_c} → {w.size}")
    return body


# ──────────────────────────── SOFTMAX / RESHAPE ────────────────────────

def _pack_softmax_body() -> bytes:
    return b""


def _pack_reshape_body() -> bytes:
    return b""


# ──────────────────────────────── ADD ─────────────────────────────────

def _pack_add_body(layer, endian, buf_size, log) -> bytes:
    """Pack body for ADD (element‑wise)."""
    if layer.get("fused_activation_function"):
        raise RuntimeError("ADD with fused activation is not supported")

    body = b""
    body += struct.pack(endian + "i", buf_size)         # input1 buf offset
    body += struct.pack(endian + "f", layer["i_scale1"])
    body += struct.pack(endian + "i", int(layer["i_zeropoint1"]))
    body += struct.pack(endian + "i", 0)                # pad
    assert len(body) % 8 == 0
    return body


# ---------------------------------------------------------------------------
# C header writer
# ---------------------------------------------------------------------------

def _write_c_header(tmdl_path, header_path, buf_size, layer_sizes, log):
    """Generate a ``.h`` file with the TMDL binary as a C byte array.

    The header defines ``MDL_BUF_LEN``, ``LBUF_LEN``, and
    ``const uint8_t mdl_data[]`` so that the model can be embedded
    directly in C firmware.
    """
    with open(tmdl_path, "rb") as f:
        data = f.read()

    lbuf_len = MDLBINHEAD_SIZE + max(layer_sizes)

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated by TinyML compiler – do not edit. */\n")
        f.write("#ifndef TINYML_MODEL_H\n")
        f.write("#define TINYML_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/** RAM needed for ping‑pong buffers (bytes). */\n")
        f.write(f"#define MDL_BUF_LEN ({buf_size})\n")
        f.write(f"/** RAM needed for single‑layer mode (bytes). */\n")
        f.write(f"#define LBUF_LEN ({lbuf_len})\n\n")
        f.write(f"/** Model binary ({len(data)} bytes). */\n")
        f.write(f"static const uint8_t mdl_data[{len(data)}] = {{\n")

        for i in range(len(data)):
            f.write(f"0x{data[i]:02x}, ")
            if i % 16 == 15:
                f.write("\n")

        f.write("\n};\n\n")
        f.write("#endif /* TINYML_MODEL_H */\n")

    log(f"  C header      : {header_path}")
