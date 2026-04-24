"""
Minimal PTO demo: doubly-stochastic Sinkhorn normalization (fp16, K=4).

Mirrors DeepSeek TileKernels `sinkhorn_normalize_ref` and passes their
`test_sinkhorn_comprehensive` under autograd.

The demo uses ``TCOLEXPANDDIV``, which is a PTO-ISA 9.0.0+ op absent from
stock CANN 8.5.0 headers.  We resolve the header root in this order:

1. ``$PTO_LIB_PATH`` env var (set by the team Docker via
   ``ENV PTO_LIB_PATH=/sources/pto-isa``).
2. The pto-kernels CMake ``FetchContent`` mirror at
   ``<repo>/build/_deps/libpto_isa_headers-src`` (populated by any
   ``cmake -B build`` in the repo root).
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


# ---- DeepSeek reference (ground truth + backward op) -----------------------
def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Exact copy of ``sinkhorn_normalize_ref`` from deepseek-ai/TileKernels."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


# ---- JIT-compile the kernel once on first use ------------------------------
_KERNEL_ARGTYPES = [
    ctypes.c_uint32,  # cube_core_num
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_uint32,  # num_matrices
    ctypes.c_uint32,  # repeat
    ctypes.c_float,   # eps
]


def _compile_kernel() -> ctypes.CDLL:
    src = _HERE / "kernel_sinkhorn.cpp"
    so  = _HERE / "outputs" / "kernel_sinkhorn.so"
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
    lib.call_sinkhorn.argtypes = _KERNEL_ARGTYPES
    lib.call_sinkhorn.restype  = None
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
    _kernel = _lib.call_sinkhorn
    _block_dim = torch.npu.get_device_properties("npu:0").cube_core_num
    _stream_obj = torch.npu.current_stream()


def _current_stream_ptr():
    # Attribute access on the cached Stream object; returns an int equal to
    # _as_parameter_.value.  ctypes auto-converts int → c_void_p.
    return _stream_obj.npu_stream


def launch_v1_raw(in_ptr: ctypes.c_void_p, out_ptr: ctypes.c_void_p,
                  n_matrices: int, repeat: int, eps: float) -> None:
    """Low-overhead launch path.  Takes pre-built ctypes pointers so no
    `ctypes.c_void_p(tensor.data_ptr())` cost on the hot path."""
    _ensure_ready()
    _kernel(_block_dim, _current_stream_ptr(), in_ptr, out_ptr,
            n_matrices, repeat, eps)


def _run_kernel(x: torch.Tensor, out: torch.Tensor, repeat: int, eps: float) -> None:
    _ensure_ready()
    _kernel(
        _block_dim,
        _current_stream_ptr(),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        x.numel() // (4 * 4),     # number of 4×4 matrices
        repeat,
        float(eps),
    )


# ---- Autograd wrapper ------------------------------------------------------
# Forward runs the PTO kernel; backward re-runs the reference under autograd.
# Correct-but-slow backward is fine for a demo — replace with a hand-written
# gradient if this graduates into a production path.
class _SinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
        assert x.dtype == torch.float16,   "demo requires fp16"
        assert x.shape[-2:] == (4, 4),     "demo supports K=4 only"
        x_flat   = x.reshape(-1, 4, 4).contiguous()
        out_flat = torch.empty_like(x_flat)
        _run_kernel(x_flat, out_flat, repeat, eps)
        torch.npu.synchronize()
        ctx.save_for_backward(x.detach())
        ctx.repeat, ctx.eps = repeat, eps
        return out_flat.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            x_ = x.clone().requires_grad_()
            (grad_in,) = torch.autograd.grad(
                sinkhorn_normalize_ref(x_, ctx.repeat, ctx.eps), x_, grad_out,
            )
        return grad_in, None, None


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in for DeepSeek's ``sinkhorn_normalize``. Accepts ``(..., 4, 4)`` fp16 NPU tensor."""
    return _SinkhornFn.apply(x, repeat, eps)