"""Build and load the block-32 fused Hadamard + MXFP4 quantize kernel.

Deliberately thin: the kernel's own launcher dispatches on K, so this only has
to name the source, the entry points and the widths, and let
``jit_util_fused_common`` do the rest. That module lives with the full-row
example next door, the same way ``layernorm`` reaches into ``fast_hadamard``;
see there for the build and the wrapper, and for the note on the unnormalised
Sylvester factor.
"""

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_COMMON = HERE.parent / "fused_hadamard_quant_a5"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from jit_util_fused_common import (  # noqa
    MX_BLOCK,
    VECTOR_CORES,
    KernelSpec,
    current_stream_ptr,
)
from jit_util_fused_common import build_and_load as _build_and_load  # noqa
from jit_util_fused_common import compile_kernel as _compile_kernel  # noqa

SOURCE = HERE / "fused_hadamard_quant_b32_a5.cpp"
BUILDDIR = HERE / "build"

# The rotation is always 32 wide, so it puts no power-of-two constraint on the
# row: any width the quantizer supports works, 4096 included. It does not make
# every multiple of 32 legal -- RowsFor still needs Rows*K to be a whole
# 1024-element grain, which is why 11008 is absent. Must match SUPPORTED_K in
# the kernel.
SUPPORTED_K = (
    32,
    64,
    96,
    128,
    192,
    256,
    512,
    768,
    896,
    1024,
    1152,
    1280,
    1408,
    1536,
    1664,
    1792,
    2048,
    2560,
    2816,
    3072,
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
    14336,
    16384,
)

SPEC = KernelSpec(
    source=SOURCE,
    lib_stem="fused_b32",
    launch_symbol="call_hadamard_mxfp4_b32",
    rows_symbol="hadamard_mxfp4_b32_rows_for",
    supported_k=SUPPORTED_K,
    width_rule=(
        "Widths must be a multiple of 32 whose Rows*K is a whole "
        "1024-element grain, and have an instantiation."
    ),
)

compile_kernel = functools.partial(_compile_kernel, SPEC)
build_and_load = functools.partial(_build_and_load, SPEC)
