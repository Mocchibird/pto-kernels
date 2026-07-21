"""JIT loader for the register-resident fast_hadamard_a5 kernel.

Mirrors examples/jit_cpp/fast_hadamard/standard/jit_util_hadamard.py. The kernel
is in-place fp16 over (batch, N) with N == 128 (see README for scope).
"""

import ctypes

from jit_util_common import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_fast_hadamard_a5",
        [
            ctypes.c_uint32,   # block_dim
            ctypes.c_void_p,   # stream
            ctypes.c_void_p,   # x (in-place, fp16)
            ctypes.c_uint32,   # batch (number of length-N rows)
        ],
    )

    def hadamard_func(x, batch, block_dim=resolved_block_dim, stream_ptr=None):
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            batch,
        )

    hadamard_func.block_dim = resolved_block_dim
    return hadamard_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
):
    # NOTE: N is a compile-time macro (single-register butterfly). This wrapper
    # compiles with the kernel's built-in default (HAD_N=128). To build another
    # supported N, pass -DHAD_N/-DHAD_LOG2N via the standalone sim_test/run.sh,
    # which is also the authoritative on-device correctness check.
    return jit_compile_with_loader(
        src_path,
        load_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
    )
