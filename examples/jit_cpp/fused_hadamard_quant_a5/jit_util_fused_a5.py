"""Build and load the full-row fused Hadamard + MXFP4 quantize kernel.

Deliberately thin: the kernel's own launcher dispatches on K, so this only has
to name the source, the entry points and the widths, and let
``jit_util_fused_common`` do the rest. See there for the build and the wrapper,
and for the note on the unnormalised Sylvester factor.
"""

import functools
from pathlib import Path

from jit_util_fused_common import (  # noqa: F401
    MX_BLOCK,
    VECTOR_CORES,
    KernelSpec,
    current_stream_ptr,
)
from jit_util_fused_common import build_and_load as _build_and_load
from jit_util_fused_common import compile_kernel as _compile_kernel

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fused_hadamard_quant_a5.cpp"
BUILDDIR = HERE / "build"

# The rotation is order K, so K must be a power of two. The width is not
# otherwise capped: the kernel does the transform as window-local work plus
# cross-window stages, and neither phase holds more than one 256-element window
# in registers. Must match SUPPORTED_K in the kernel.
SUPPORTED_K = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

SPEC = KernelSpec(
    source=SOURCE,
    lib_stem="fused_full",
    launch_symbol="call_hadamard_mxfp4_full",
    rows_symbol="hadamard_mxfp4_full_rows_for",
    supported_k=SUPPORTED_K,
    width_rule="The rotation is row wide, so K must be a power of two.",
)

compile_kernel = functools.partial(_compile_kernel, SPEC)
build_and_load = functools.partial(_build_and_load, SPEC)
