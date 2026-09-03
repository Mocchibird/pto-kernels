"""Build and load the block-32 fused Hadamard + MXFP4 quantize kernel.

Deliberately thin: the kernel's own launcher dispatches on K, so this only has to
compile one .so and hand back a callable. Modelled on mxfp4_quant's jit helper,
with the entry points renamed and the width list narrowed to those the rotation
supports.
"""

import ctypes
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = Path(__file__).resolve().parent
BUILDDIR = HERE / "build"
_LIB_NAME = "fused_full.so"
SOURCE = HERE / "fused_hadamard_quant_a5.cpp"

MX_BLOCK = 32
VECTOR_CORES = 64  # vector cores on an A5
# The rotation is order K, so K must be a power of two. The width is not
# otherwise capped: the kernel does the transform as window-local work plus
# cross-window stages, and neither phase holds more than one 256-element window
# in registers. Must match SUPPORTED_K in the kernel.
SUPPORTED_K = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def _flags(home):
    return (
        f"-xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE "
        f"-std=c++17 -O2 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
        f"-mllvm -cce-aicore-stack-size=0x8000 "
        f"-mllvm -cce-aicore-function-stack-size=0x8000 "
        f"-mllvm -cce-aicore-addr-transform "
        f"-mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end "
        f"-I{home}/aarch64-linux/include -I{home}/include"
    ).split()


def compile_kernel(verbose=True, extra_defs=()):
    """Compile the fused kernel to a .so. One .so serves every supported K.

    extra_defs are extra -D tokens for a tuning or A/B variant. They go into the
    .so NAME as well as the command line, so a variant can never be served from
    the default build's cache -- silently timing the wrong binary is the failure
    this guards.
    """
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise RuntimeError("source a CANN set_env.sh first: ASCEND_HOME_PATH is unset")
    BUILDDIR.mkdir(parents=True, exist_ok=True)
    tag = "".join("_" + d.lstrip("-D").replace("=", "") for d in sorted(extra_defs))
    # Reuse an .so newer than its source. These kernels unroll to hundreds of
    # tile instructions and a rebuild can outlast the task queue's 600 s cap, so
    # recompiling per call is not merely wasteful.
    cached = BUILDDIR / _LIB_NAME.replace(".so", f"{tag}.so")
    if cached.exists() and cached.stat().st_mtime > SOURCE.stat().st_mtime:
        if verbose:
            print("reusing", cached)
        return cached
    obj = BUILDDIR / f"fused_b32{tag}.o"
    lib = cached
    for step in (
        [
            f"{home}/bin/bisheng",
            *_flags(home),
            *extra_defs,
            "-c",
            str(SOURCE),
            "-o",
            str(obj),
        ],
        [
            f"{home}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{lib.name}",
            str(obj),
            "-o",
            str(lib),
        ],
    ):
        if verbose:
            print("compile:", " ".join(step[:3]), "...")
        subprocess.run(step, check=True)
    return lib


def current_stream_ptr():
    return ctypes.c_void_p(torch.npu.current_stream().npu_stream)


def row_quantum(k):
    """Rows per tile. The kernel pads a partial tile, so any batch is legal."""
    return 1


# The butterfly is the UNNORMALISED Sylvester matrix, so its output is sqrt(32)
# larger than an orthogonal block Hadamard's. That factor is deliberate and left
# to the caller: MXFP4's E8M0 scale is a power of two and sqrt(32) is not, so the
# scale cannot absorb it and the nibbles genuinely differ. Scale x by
# 1/sqrt(32) on the way in if orthogonal semantics are wanted.


def build_and_load(k=256, verbose=True, extra_defs=()):
    """Return `fused(x) -> (nibbles, scales)` for row width `k`.

    Allocates its outputs, mirroring `torch_npu.npu_dynamic_mx_quant`, so the two
    are comparable on the same call path.

    extra_defs reaches the compiler, so the reduced builds the benchmark's ladder
    needs come from this one source: FUSED_ROTATE_ONLY leaves the butterfly
    alone, FUSED_NO_ROTATE leaves the quantizer alone.
    """
    if k not in SUPPORTED_K:
        raise ValueError(
            f"K={k} has no instantiation; supported: {sorted(SUPPORTED_K)}. "
            "Widths must be a multiple of 32 with an instantiation."
        )
    lib = ctypes.CDLL(str(compile_kernel(verbose=verbose, extra_defs=extra_defs)))
    launch = lib.call_hadamard_mxfp4_full
    launch.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    launch.restype = None
    rows_for = lib.hadamard_mxfp4_full_rows_for
    rows_for.argtypes = [ctypes.c_uint32]
    rows_for.restype = ctypes.c_uint32

    def fused(x, out=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"expected bfloat16, got {x.dtype}")
        if x.shape[-1] != k:
            raise ValueError(f"expected last dim {k}, got {tuple(x.shape)}")
        if not x.is_contiguous():
            raise ValueError("expected a contiguous tensor; call .contiguous()")
        batch = x.numel() // k
        if out is None:
            q = torch.empty((batch, k // 2), dtype=torch.uint8, device=x.device)
            s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8, device=x.device)
        else:
            q, s = out
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
            batch,
            k,
        )
        return q, s

    fused.rows_for = lambda: rows_for(k)
    fused.k = k
    return fused
