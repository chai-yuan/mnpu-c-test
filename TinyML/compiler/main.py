#!/usr/bin/env python3
"""TinyML TFLite‑to‑TMDL compiler – command‑line interface.

Converts TensorFlow Lite INT8 quantised models into the TinyMaix TMDL
binary format suitable for microcontrollers.

Usage examples
--------------

.. code-block:: bash

    # Simplest – auto‑detect dims from the model
    python -m compiler model.tflite model.tmdl

    # Override dims manually (rarely needed)
    python -m compiler model.tflite model.tmdl \\
        --in-dims 28,28,1 --out-dims 10

    # Without output dequantisation
    python -m compiler model.tflite model.tmdl --out-deq 0

    # Big‑endian target
    python -m compiler model.tflite model.tmdl --big-endian
"""

import argparse
import sys
import textwrap

from .tflite_reader import read_tflite, get_model_shapes
from .packer import pack_tmdl


def _parse_dims(s: str) -> tuple:
    """Parse comma‑separated dimensions like ``"28,28,1"`` → ``(28, 28, 1)``."""
    parts = [x.strip() for x in s.split(",")]
    if not all(p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError(
            f"invalid dims '{s}' – expected comma‑separated integers, "
            f"e.g. 28,28,1")
    return tuple(int(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="compiler",
        description="Convert TFLite INT8 models to TinyMaix TMDL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s mnist.tflite mnist.tmdl
              %(prog)s model.tflite model.tmdl --out-deq 0
              %(prog)s model.tflite model.tmdl --in-dims 28,28,1 --out-dims 10
        """),
    )

    parser.add_argument(
        "tflite",
        metavar="TFLITE",
        help="Path to the input .tflite file (INT8 quantised).",
    )
    parser.add_argument(
        "output",
        metavar="TMDL",
        help="Path for the output .tmdl file.",
    )
    parser.add_argument(
        "--in-dims",
        type=_parse_dims,
        default=None,
        metavar="H,W,C",
        help="Input dimensions excluding batch (auto‑detected if omitted), "
             "e.g. 28,28,1",
    )
    parser.add_argument(
        "--out-dims",
        type=_parse_dims,
        default=None,
        metavar="N",
        help="Output dimensions excluding batch (auto‑detected if omitted), "
             "e.g. 10",
    )
    parser.add_argument(
        "--out-deq",
        type=int,
        choices=(0, 1),
        default=1,
        metavar="{0,1}",
        help="Enable (1) / disable (0) output dequantisation (default: 1).",
    )
    parser.add_argument(
        "--big-endian",
        action="store_true",
        help="Emit multi‑byte numbers in big‑endian order (default: little).",
    )
    parser.add_argument(
        "--c-header", "-H",
        dest="c_header",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Also export as a C header (.h).  ``-H`` alone auto‑names it "
             "next to the .tmdl; ``-H path.h`` writes to the given path.",
    )

    return parser


def main(args=None):
    """Run the compiler from *args* (list of strings) or ``sys.argv``."""
    parser = build_parser()
    ns = parser.parse_args(args)

    # ── 0. Auto‑detect dims from model if not given ───────────────────
    model_in, model_out = get_model_shapes(ns.tflite)

    if ns.in_dims is None:
        ns.in_dims = model_in
    if ns.out_dims is None:
        ns.out_dims = model_out

    print("=" * 56)
    print("  TinyML TFLite → TMDL Compiler  [INT8 only]")
    print("=" * 56)
    print(f"  Source        : {ns.tflite}")
    print(f"  Target        : {ns.output}")
    print(f"  Input  dims   : {ns.in_dims}"
          f"{'  (auto)' if ns.in_dims == model_in else ''}")
    print(f"  Output dims   : {ns.out_dims}"
          f"{'  (auto)' if ns.out_dims == model_out else ''}")
    print(f"  Out dequant   : {ns.out_deq}")
    print(f"  Endian        : {'big' if ns.big_endian else 'little'}")
    print()

    # ── resolve C header path ─────────────────────────────────────────
    c_header = None
    if ns.c_header is True:
        # -H alone → auto‑name next to .tmdl
        base = ns.output.rsplit(".", 1)[0]
        c_header = base + ".bin.h"
    elif isinstance(ns.c_header, str):
        c_header = ns.c_header

    # ── 1. Parse TFLite ──────────────────────────────────────────────
    print("─" * 56)
    print("  STEP 1 – Parsing TFLite model")
    print("─" * 56)
    layers = read_tflite(ns.tflite)

    if not layers:
        print("ERROR: No layers found in the model!")
        sys.exit(1)

    # ── 2. Verify INT8 quantisation ──────────────────────────────────
    if not layers[0].get("quant"):
        print()
        print("ERROR: Model is not quantised!")
        print("This compiler only supports INT8 quantised TFLite models.")
        print("Please quantise your model before conversion.")
        sys.exit(1)

    print(f"\n  ✓ Parsed {len(layers)} layers (quantised model)")

    # ── 3. Pack TMDL ─────────────────────────────────────────────────
    print()
    print("─" * 56)
    print("  STEP 2 – Packing TMDL binary")
    print("─" * 56)
    model_size, total_buf, layer_sizes = pack_tmdl(
        layers,
        output_path=ns.output,
        in_dims=ns.in_dims,
        out_dims=ns.out_dims,
        out_deq=ns.out_deq,
        big_endian=ns.big_endian,
        c_header_path=c_header,
    )

    # ── 4. Summary ───────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("  COMPILATION FINISHED")
    print("=" * 56)
    print(f"  Output        : {ns.output}")
    if c_header:
        print(f"  C header      : {c_header}")
    print(f"  Flash (model) : {model_size / 1024:.1f} KB  ({model_size} B)")
    print(f"  RAM  (buffer) : {total_buf / 1024:.1f} KB  ({total_buf} B)")
    print(f"  Layers        : {len(layers)}")
    print()


if __name__ == "__main__":
    main()
