"""Constants and configuration for the TinyML TFLite-to-TMDL compiler.

This module centralises all magic numbers, op-code constants, and
binary‑format sizes so that other modules can stay focused on logic.
"""

# ---------------------------------------------------------------------------
# Model data types (the compiler only supports INT8)
# ---------------------------------------------------------------------------
TM_MDL_INT8 = 0

UNIT_SIZE = 1       # bytes per weight element (INT8)
BUNIT_SIZE = 4      # bytes per bias element  (INT32)

# ---------------------------------------------------------------------------
# TMDL layer type identifiers (as stored in the binary header)
# ---------------------------------------------------------------------------
TML_CONV2D = 0
TML_GAP = 1
TML_FC = 2
TML_SOFTMAX = 3
TML_RESHAPE = 4
TML_DWCONV2D = 5
TML_ADD = 6

# ---------------------------------------------------------------------------
# Padding / activation enums
# ---------------------------------------------------------------------------
TM_PAD_SAME  = 0  # TFLite enum: SAME=0, VALID=1
TM_PAD_VALID = 1

TM_ACT_NONE = 0
TM_ACT_RELU = 1

# ---------------------------------------------------------------------------
# Binary format sizes (bytes)
# ---------------------------------------------------------------------------
MDLBINHEAD_SIZE = 64   # total model header
LAYERHEAD_SIZE = 48    # per‑layer header

# ---------------------------------------------------------------------------
# Dispatch table: TFLite op‑code name → TMDL layer type
# ---------------------------------------------------------------------------
LAYER_NAME_TO_TYPE = {
    "CONV_2D":           TML_CONV2D,
    "MEAN":              TML_GAP,
    "FULLY_CONNECTED":   TML_FC,
    "SOFTMAX":           TML_SOFTMAX,
    "RESHAPE":           TML_RESHAPE,
    "DEPTHWISE_CONV_2D": TML_DWCONV2D,
    "ADD":               TML_ADD,
}
