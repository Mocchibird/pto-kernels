"""Build and load machinery shared by the two fused Hadamard + MXFP4 kernels.

The two examples differ in the rotation they apply -- one over the whole row,
one over independent 32-element blocks -- and therefore in which row widths
have an instantiation. Everything on either side of that is the same: the
bisheng command line, the .so cache, the ctypes signatures, and the wrapper
that allocates the outputs.

Import the example's own thin module (``jit_util_fused_a5`` or
``jit_util_fused_b32_a5``), not this one. Each of those names its source, its
entry points and its width list, and binds them to the functions here.

The butterfly is the UNNORMALISED Sylvester matrix, so its output is sqrt(32)
larger than an orthogonal block Hadamard's. That factor is deliberate and left
to the caller: MXFP4's E8M0 scale is a power of two and sqrt(32) is not, so the
scale cannot absorb it and the nibbles genuinely differ. Scale x by
1/sqrt(32) on the way in if orthogonal semantics are wanted.
"""

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch_npu  # noqa

MX_BLOCK = 32
VECTOR_CORES = 64  # vector cores on an A5


@dataclass(frozen=True)
class KernelSpec:
    """Everything that differs between the two fused kernels.

    ``width_rule`` is appended to the error a bad ``k`` raises, because the two
    kernels reject a width for different reasons and a caller needs to know
    which one it hit.
    """

    source: Path
    lib_stem: str
    launch_symbol: str
    rows_symbol: str
    supported_k: Tuple[int, ...]
    width_rule: str

    @property
    def build_dir(self) -> Path:
        return self.source.parent / "build"


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


def compile_kernel(spec, verbose=True, extra_defs=()):
    """Compile the fused kernel to a .so. One .so serves every supported K.

    extra_defs are extra -D tokens for a tuning or A/B variant. They go into the
    .so NAME as well as the command line, so a variant can never be served from
    the default build's cache -- silently timing the wrong binary is the failure
    this guards.
    """
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise RuntimeError("source a CANN set_env.sh first: ASCEND_HOME_PATH is unset")
    build_dir = spec.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join("_" + d.lstrip("-D").replace("=", "") for d in sorted(extra_defs))
    # Reuse an .so newer than its source. These kernels unroll to hundreds of
    # tile instructions and a rebuild can outlast the task queue's 600 s cap, so
    # recompiling per call is not merely wasteful.
    cached = build_dir / f"{spec.lib_stem}{tag}.so"
    if cached.exists() and cached.stat().st_mtime > spec.source.stat().st_mtime:
        if verbose:
            print("reusing", cached)
        return cached
    obj = build_dir / f"{spec.lib_stem}{tag}.o"
    lib = cached
    for step in (
        [
            f"{home}/bin/bisheng",
            *_flags(home),
            *extra_defs,
            "-c",
            str(spec.source),
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


def build_and_load(spec, k=256, verbose=True, extra_defs=()):
    """Return `fused(x) -> (nibbles, scales)` for row width `k`.

    Allocates its outputs, mirroring `torch_npu.npu_dynamic_mx_quant`, so the two
    are comparable on the same call path.

    extra_defs reaches the compiler, so the reduced builds the benchmark's ladder
    needs come from this one source: FUSED_ROTATE_ONLY leaves the butterfly
    alone, FUSED_NO_ROTATE leaves the quantizer alone.
    """
    if k not in spec.supported_k:
        raise ValueError(
            f"K={k} has no instantiation; supported: {sorted(spec.supported_k)}. "
            + spec.width_rule
        )
    lib = ctypes.CDLL(str(compile_kernel(spec, verbose=verbose, extra_defs=extra_defs)))
    launch = getattr(lib, spec.launch_symbol)
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
    rows_for = getattr(lib, spec.rows_symbol)
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
