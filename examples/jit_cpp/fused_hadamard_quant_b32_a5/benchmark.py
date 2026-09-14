"""Fused block-32 Hadamard + MXFP4 quantize on Ascend A5, against the unfused
pair and against a device-to-device copy.

The harness is in `benchmark_fused_common`, next door with the full-row
example; this names the widths to sweep.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fused_hadamard_quant_a5"))

from benchmark_fused_common import main  # noqa: E402
from jit_util_fused_b32_a5 import build_and_load  # noqa: E402

# K=32 is a row of 0.5M elements at M=16384, where the fused arm is on the
# dispatch floor rather than on bytes; kept because that is worth seeing.
SHAPES = (32, 1024, 4096, 16384)

if __name__ == "__main__":
    sys.exit(main(build_and_load, SHAPES, description=__doc__))
