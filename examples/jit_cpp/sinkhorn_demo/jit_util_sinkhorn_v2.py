"""
JIT wrapper for the v2 doubly-stochastic Sinkhorn kernel (K=4 only).

Mirrors `jit_util_sinkhorn.py` but points at `kernel_sinkhon_v2.cpp` and
its `call_sinkhorn_ds_kernel(..., N, K, repeat, eps)` C ABI.

`sinkhorn_normalize_v2` is the user-facing drop-in (allocates output,
reshapes).  `launch_v2_raw` is the lean launch path for benchmarking:
cached block_dim + stream, no allocation, no reshape, takes pre-built
ctypes pointers.
"""
import ctypes
import os
import subprocess
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent


def _resolve_pto_include_dir() -> str:
    env = os.environ.get("PTO_LIB_PATH")
    if env:
        return f"{env}/include"
    vendored = _HERE.parents[2] / "build" / "_deps" / "libpto_isa_headers-src" / "include"
    if (vendored / "pto" / "pto-inst.hpp").is_file():
        return str(vendored)
    raise RuntimeError(
        "Could not find PTO-ISA 9.0.0+ headers.  Either set PTO_LIB_PATH=/path/to/pto-isa, "
        "or run `cmake -B build` in the pto-kernels repo root to populate the FetchContent mirror."
    )


_PTO_INCLUDE_DIR = _resolve_pto_include_dir()


_KERNEL_ARGTYPES = [
    ctypes.c_uint32,  # cube_core_num
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_uint32,  # N
    ctypes.c_uint32,  # K  (always 4 for this kernel)
    ctypes.c_uint32,  # repeat
    ctypes.c_float,   # eps
]


def _compile_kernel() -> ctypes.CDLL:
    src = _HERE / "kernel_sinkhorn_v2.cpp"
    so  = _HERE / "outputs" / "kernel_sinkhorn_v2.so"
    so.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "bisheng",
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE",
        "-O2", "-std=c++17", "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220-vec",
        "-isystem", _PTO_INCLUDE_DIR,
        str(src), "-o", str(so),
    ], check=True)
    lib = ctypes.CDLL(str(so))
    lib.call_sinkhorn_ds_kernel.argtypes = _KERNEL_ARGTYPES
    lib.call_sinkhorn_ds_kernel.restype  = None
    return lib


# Module-global cache populated on first call.
#
# Stream caching gotcha (learned the hard way on 2026-04-24):
#   Caching only the raw pointer (_as_parameter_) and skipping any Python-
#   level stream interaction on hot calls causes PyTorch's caching allocator
#   to reuse the output tensor's memory prematurely — the kernel's writes
#   land in memory the allocator has already handed back out, producing
#   stray NaNs in later outputs.
#
#   The fix: keep the Stream *object* alive, and on every launch touch
#   `_stream_obj.npu_stream` (~0.6us, plain attribute access on a Python
#   int).  That touch carries whatever bookkeeping side effect the allocator
#   relies on; the native pointer it returns is identical to the cached
#   `_as_parameter_.value`, so we can pass it straight to ctypes.
#   Previously we called `torch.npu.current_stream()` per launch (~27us) —
#   46× more expensive for the same correctness guarantee.
_lib = None
_kernel = None
_block_dim = None
_stream_obj = None  # cached Stream object — keep alive for native-handle validity


def _ensure_ready() -> None:
    global _lib, _kernel, _block_dim, _stream_obj
    if _lib is not None:
        return
    _lib = _compile_kernel()
    _kernel = _lib.call_sinkhorn_ds_kernel
    _block_dim = torch.npu.get_device_properties("npu:0").cube_core_num
    _stream_obj = torch.npu.current_stream()


def _current_stream_ptr():
    # Attribute access on the cached Stream object; returns an int equal to
    # _as_parameter_.value.  ctypes auto-converts int → c_void_p.
    return _stream_obj.npu_stream


def launch_v2_raw(in_ptr: ctypes.c_void_p, out_ptr: ctypes.c_void_p,
                  n_matrices: int, repeat: int, eps: float) -> None:
    """Low-overhead launch path.  Takes pre-built ctypes pointers so no
    `ctypes.c_void_p(tensor.data_ptr())` cost on the hot path."""
    _ensure_ready()
    _kernel(_block_dim, _current_stream_ptr(), in_ptr, out_ptr,
            n_matrices, 4, repeat, eps)


def _run_kernel(x: torch.Tensor, out: torch.Tensor, repeat: int, eps: float) -> None:
    _ensure_ready()
    _kernel(
        _block_dim,
        _current_stream_ptr(),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        x.numel() // (4 * 4),     # N (number of 4x4 matrices)
        4,                          # K (fixed)
        repeat,
        float(eps),
    )


def sinkhorn_normalize_v2(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Forward-only drop-in for the v2 K=4 sinkhorn kernel.

    Accepts `(..., 4, 4)` fp16 NPU tensor; returns a same-shape tensor.
    """
    assert x.dtype == torch.float16, "v2 kernel requires fp16"
    assert x.shape[-2:] == (4, 4),   "v2 kernel supports K=4 only"
    x_flat   = x.reshape(-1, 4, 4).contiguous()
    out_flat = torch.empty_like(x_flat)
    _run_kernel(x_flat, out_flat, repeat, eps)
    # Kernel is launched via ctypes outside PyTorch's stream tracking, so
    # callers comparing `out_flat` against a pure-PyTorch reference on the
    # same stream would race the kernel's writes.  Explicit sync mirrors
    # what `jit_util_sinkhorn.py` does for the v1 kernel.
    torch.npu.synchronize()
    return out_flat.reshape_as(x)
