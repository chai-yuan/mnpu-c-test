"""TFLite model reader – parse a .tflite file into an internal layer list.

This module decodes the FlatBuffer binary and uses the TensorFlow Lite
interpreter to extract weights, biases, shapes and quantisation parameters.
Only the first sub‑graph is processed.

Only INT8 quantised models are supported.
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


# ---------------------------------------------------------------------------
# FlatBuffer → dict helpers (adapted from TF Lite visualize.py)
# ---------------------------------------------------------------------------

def _builtin_code_to_name(code: int) -> str:
    """Map a TFLite ``BuiltinOperator`` enum value to its string name."""
    for name, value in schema_fb.BuiltinOperator.__dict__.items():
        if value == code:
            return name
    return None


def _camel_to_snake(name: str) -> str:
    """Convert ``CamelCase`` to ``snake_case``."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _flatbuffer_to_dict(fb, preserve_as_numpy: bool):
    """Recursively convert a FlatBuffer object into plain Python dict / list."""
    if isinstance(fb, (int, float, str)):
        return fb
    if isinstance(fb, np.ndarray):
        return fb if preserve_as_numpy else fb.tolist()
    if hasattr(fb, "__dict__"):
        result = {}
        for attr_name in dir(fb):
            attr = getattr(fb, attr_name)
            if not callable(attr) and not attr_name.startswith("_"):
                snake_name = _camel_to_snake(attr_name)
                preserve = (attr_name == "buffers") or preserve_as_numpy
                result[snake_name] = _flatbuffer_to_dict(attr, preserve)
        return result
    if hasattr(fb, "__len__"):
        return [_flatbuffer_to_dict(entry, preserve_as_numpy) for entry in fb]
    return fb


def _create_dict_from_flatbuffer(buffer_data: bytes) -> dict:
    """Parse raw ``.tflite`` bytes into a nested dict."""
    model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return _flatbuffer_to_dict(model, preserve_as_numpy=False)


# ---------------------------------------------------------------------------
# Helpers: reading common tensor attributes
# ---------------------------------------------------------------------------

def _read_tensor_quant(tensor_info: dict, prefix: str, layer: dict):
    """Fill ``prefix+'_scale'`` and ``prefix+'_zeropoint'`` on *layer*."""
    q = tensor_info.get("quantization", {})
    if q.get("scale") is not None:
        layer[f"{prefix}_scale"] = float(q["scale"][0])
        layer[f"{prefix}_zeropoint"] = int(q["zero_point"][0])
    else:
        layer[f"{prefix}_scale"] = 1.0
        layer[f"{prefix}_zeropoint"] = 0


def _read_weight_bias(layer, w_idx, b_idx, tensors, interpreter, log,
                      has_bias: bool = True):
    """Read weight and optionally bias from interpreter, storing in *layer*."""
    weight = interpreter.get_tensor(w_idx)
    layer["weight"] = weight
    log(f"         weight  : {tensors[w_idx]['name']}  shape={list(weight.shape)}")

    q_w = tensors[w_idx].get("quantization", {})
    if q_w.get("scale") is not None:
        layer["w_scale"] = np.array(q_w["scale"])
        layer["w_zeropoint"] = np.array(q_w["zero_point"])
    else:
        layer["w_scale"] = 1.0
        layer["w_zeropoint"] = 0

    if has_bias and b_idx >= 0:
        bias = interpreter.get_tensor(b_idx)
        layer["bias"] = bias
        log(f"         bias    : {tensors[b_idx]['name']}  shape={list(bias.shape)}")


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def get_model_shapes(tflite_path: str) -> tuple:
    """Quickly extract input / output shapes (no TF interpreter needed).

    Returns ``(in_shape, out_shape)`` where each is a tuple of ints
    with the batch dimension stripped.
    """
    with open(tflite_path, "rb") as f:
        model_buffer = f.read()

    data = _create_dict_from_flatbuffer(model_buffer)
    subg = data["subgraphs"][0]
    tensors = subg["tensors"]
    input_idxs = subg["inputs"]
    output_idxs = subg["outputs"]

    def _strip_batch(shape):
        if shape and shape[0] == 1:
            return tuple(shape[1:])
        return tuple(shape)

    in_shape = _strip_batch(list(tensors[input_idxs[0]]["shape"]))
    out_shape = _strip_batch(list(tensors[output_idxs[0]]["shape"]))
    return in_shape, out_shape


def read_tflite(tflite_path: str, log=print) -> list:
    """Read a TFLite model and return a list of layer descriptor dicts.

    Returns
    -------
    list[dict]
        Each dict describes one layer.  Common keys:

        * ``name``         – op name (``"CONV_2D"``, ``"FULLY_CONNECTED"``, …)
        * ``is_output``    – 1 if output tensor, else 0
        * ``is_keep``      – 1 if ADD keep‑buf is needed
        * ``in_shape``     – input tensor shape (list)
        * ``out_shape``    – output tensor shape (list)
        * ``i_scale``, ``i_zeropoint`` – input quantisation params
        * ``o_scale``, ``o_zeropoint`` – output quantisation params
        * ``quant``        – 1 if the model is quantised
        * layer‑specific: ``weight``, ``bias``, ``w_scale``, etc.
    """
    with open(tflite_path, "rb") as f:
        model_buffer = f.read()

    # TensorFlow interpreter (needed to extract weight / bias values)
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()

    # FlatBuffer metadata
    data = _create_dict_from_flatbuffer(model_buffer)
    op_codes = data["operator_codes"]
    subg = data["subgraphs"][0]
    tensors = subg["tensors"]
    output_idxs = subg["outputs"]

    # Decode tensor names from bytes to UTF‑8
    for t in tensors:
        t["name"] = bytearray(t["name"]).decode("utf-8")

    layers = []
    last_pad = None

    for idx, op in enumerate(subg["operators"]):
        layer = {}
        op_code = op_codes[op["opcode_index"]]["builtin_code"]
        layer_name = _builtin_code_to_name(op_code)
        layer["name"] = layer_name
        layer["is_keep"] = 0

        log(f"\n  [{idx}] {layer_name}")

        # ── builtin options ──────────────────────────────────────────
        opts = op.get("builtin_options")
        if opts is not None:
            log(f"       options : {opts}")
            layer.update(opts)

        # ── input / output tensor indices ────────────────────────────
        in_idxs = op["inputs"]
        out_idxs = op["outputs"]

        if len(out_idxs) > 1:
            raise RuntimeError("Multi‑output ops are not supported yet")

        layer["is_output"] = 1 if out_idxs[0] in output_idxs else 0
        if layer["is_output"]:
            log("       ** OUTPUT **")

        # ── input tensor ─────────────────────────────────────────────
        input_idx = in_idxs[0]
        if last_pad is not None:
            shape = list(tensors[input_idx]["shape"])
            shape[1] -= last_pad[0] + last_pad[1]
            shape[2] -= last_pad[2] + last_pad[3]
            layer["in_shape"] = shape
        else:
            layer["in_shape"] = list(tensors[input_idx]["shape"])

        layer["in_name"] = tensors[input_idx]["name"]
        log(f"       in      : {layer['in_name']}  shape={layer['in_shape']}")
        _read_tensor_quant(tensors[input_idx], "i", layer)
        layer["quant"] = 1 if tensors[input_idx]["quantization"].get("scale") else 0

        # ── output tensor ────────────────────────────────────────────
        output_idx = out_idxs[0]
        layer["out_shape"] = list(tensors[output_idx]["shape"])
        layer["out_name"] = tensors[output_idx]["name"]
        log(f"       out     : {layer['out_name']}  shape={layer['out_shape']}")
        _read_tensor_quant(tensors[output_idx], "o", layer)

        # ── layer‑specific parameters ────────────────────────────────
        if layer_name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            _read_weight_bias(layer, in_idxs[1], in_idxs[2],
                              tensors, interpreter, log)
            if last_pad is not None:
                layer["padding"] = 2
                layer["pad"] = list(last_pad)

        elif layer_name == "MEAN":
            data_idx = in_idxs[1]
            log(f"       data    : {tensors[data_idx]['name']}")
            reduce_idx = interpreter.get_tensor(data_idx)
            layer["reduce_idx"] = (reduce_idx - 1).tolist()

        elif layer_name == "FULLY_CONNECTED":
            _read_weight_bias(layer, in_idxs[1], in_idxs[2],
                              tensors, interpreter, log)

        elif layer_name in ("SOFTMAX", "RESHAPE"):
            log("       (no extra params)")

        elif layer_name == "PAD":
            # Fuse PAD into the following CONV_2D / DWCONV_2D (VALID only)
            log("       fusing PAD with next CONV / DWCONV …")
            next_op = subg["operators"][idx + 1]
            next_code = op_codes[next_op["opcode_index"]]["builtin_code"]
            next_name = _builtin_code_to_name(next_code)

            if next_name not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                raise RuntimeError(
                    "PAD must be followed by CONV_2D or DEPTHWISE_CONV_2D")
            if next_op["builtin_options"].get("padding") != 1:  # not VALID
                raise RuntimeError(
                    "PAD fusion only supported when next op uses VALID padding")

            pad_idx = in_idxs[1]
            log(f"       pad     : {tensors[pad_idx]['name']}")
            pad = interpreter.get_tensor(pad_idx)
            assert pad[0, 0] == 0 and pad[0, 1] == 0
            assert pad[3, 0] == 0 and pad[3, 1] == 0
            last_pad = [int(pad[1][0]), int(pad[1][1]),
                        int(pad[2][0]), int(pad[2][1])]
            continue  # PAD is not a real layer in TMDL

        elif layer_name == "ADD":
            if len(in_idxs) > 1:
                in1_idx = in_idxs[1]
                layer["in_shape1"] = list(tensors[in1_idx]["shape"])
                layer["in_name1"] = tensors[in1_idx]["name"]
                log(f"       input1  : {layer['in_name1']}  shape={layer['in_shape1']}")
                # Store as i_scale1/i_zeropoint1 (not via _read_tensor_quant
                # which would produce i_scale / i_zeropoint).
                q1 = tensors[in1_idx].get("quantization", {})
                if q1.get("scale") is not None:
                    layer["i_scale1"] = float(q1["scale"][0])
                    layer["i_zeropoint1"] = int(q1["zero_point"][0])
                else:
                    layer["i_scale1"] = 1.0
                    layer["i_zeropoint1"] = 0

                # The input from the "farther" producer must be kept
                in_name0 = tensors[in_idxs[0]]["name"]
                in_name1 = tensors[in1_idx]["name"]
                src0 = src1 = -1
                for i, prev in enumerate(layers):
                    if prev["out_name"] == in_name0:
                        src0 = i
                    if prev["out_name"] == in_name1:
                        src1 = i
                if src0 < 0 or src1 < 0:
                    raise RuntimeError(
                        "ADD input producer not found in previous layers")
                if abs(idx - src0) > abs(idx - src1):
                    layers[src0]["is_keep"] = 1
                else:
                    layers[src1]["is_keep"] = 1
            log("       (ADD no extra params)")

        elif layer_name in ("SHAPE", "STRIDED_SLICE", "PACK"):
            log(f"       ignoring unsupported op: {layer_name}")
            continue

        elif layer_name == "QUANTIZE":
            raise RuntimeError(
                "QUANTIZE op found – the model appears to be unquantised. "
                "This compiler only supports INT8 quantised models.")

        else:
            raise RuntimeError(f"Unsupported layer type: {layer_name}")

        layers.append(layer)
        last_pad = None

    log(f"\n  Total layers: {len(layers)}")
    return layers
