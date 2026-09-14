"""Fused full-row Hadamard + MXFP4 quantize on Ascend A5, against the unfused
pair and against a device-to-device copy.

The harness is in `benchmark_fused_common`; this names the widths to sweep.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark_fused_common import main  # noqa: E402
from jit_util_fused_a5 import build_and_load  # noqa: E402

# At M=16384 the unfused intermediates are 2*M*k bytes. Below K=4096 that fits
# the 128 MiB L2, so the unfused arms partly read from cache and the ladder
# understates fusing -- 2.1x at K=1024 against 4.1x at K=4096. Kept in the sweep
# because the effect is worth seeing, not because those rows are the headline.
SHAPES = (1024, 4096, 8192, 16384)

if __name__ == "__main__":
    sys.exit(main(build_and_load, SHAPES, description=__doc__))
