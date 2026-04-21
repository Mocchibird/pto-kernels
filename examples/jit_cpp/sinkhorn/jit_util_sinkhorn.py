# pylint: disable=wrong-import-position
import ctypes
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from jit_util_common import (  # noqa: E402
    BLOCK_DIM,
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)

UINT32_MAX = (1 << 32) - 1
MAX_DIM = 256

SINKHORN_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # matrix_in
    ctypes.c_void_p,  # matrix_out
    ctypes.c_void_p,  # mu1_out
    ctypes.c_void_p,  # mu2_out
    ctypes.c_uint32,  # N
    ctypes.c_uint32,  # K
    ctypes.c_uint32,  # L
    ctypes.c_uint32,  # order
    ctypes.c_float,   # lr
    ctypes.c_float,   # eps
    ctypes.c_float,   # invK
    ctypes.c_float,   # invL
    ctypes.c_float,   # invK1
    ctypes.c_float,   # invL1
]


def _validate_sinkhorn_io(matrix_in, matrix_out, mu1_out, mu2_out, K, L):
    if matrix_in.dim() != 3:
        raise ValueError("matrix_in must be a 3D tensor (N, K, L).")
    N = matrix_in.shape[0]
    if matrix_in.shape[1] != K or matrix_in.shape[2] != L:
        raise ValueError(f"matrix_in must have shape (N, {K}, {L}).")
    if matrix_out.shape != matrix_in.shape:
        raise ValueError("matrix_out must have the same shape as matrix_in.")
    if mu1_out.shape != (N, L):
        raise ValueError(f"mu1_out must have shape ({N}, {L}).")
    if mu2_out.shape != (N, K):
        raise ValueError(f"mu2_out must have shape ({N}, {K}).")
    for name, t in [
        ("matrix_in", matrix_in),
        ("matrix_out", matrix_out),
        ("mu1_out", mu1_out),
        ("mu2_out", mu2_out),
    ]:
        if t.dtype != torch.float16:
            raise TypeError(f"{name} must use torch.float16.")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if not (
        matrix_in.device
        == matrix_out.device
        == mu1_out.device
        == mu2_out.device
    ):
        raise ValueError("All tensors must be on the same device.")
    if K > MAX_DIM or L > MAX_DIM:
        raise ValueError(f"K and L must be <= {MAX_DIM}.")
    if K == 0 or L == 0:
        raise ValueError("K and L must be positive.")


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_sinkhorn_kernel",
        SINKHORN_ARGTYPES,
    )

    def sinkhorn_func(
        matrix_in,
        matrix_out,
        mu1_out,
        mu2_out,
        *,
        order=10,
        lr=0.5,
        eps=1e-3,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        N, K, L = matrix_in.shape
        _validate_sinkhorn_io(matrix_in, matrix_out, mu1_out, mu2_out, K, L)

        # Precompute float inverses on host (int-to-float cast is forbidden
        # inside aicore functions on 910B2).
        inv_k = 1.0 / K
        inv_l = 1.0 / L
        inv_k1 = 1.0 / (K - 1) if K > 1 else 1.0
        inv_l1 = 1.0 / (L - 1) if L > 1 else 1.0

        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(matrix_in),
            torch_to_ctypes(matrix_out),
            torch_to_ctypes(mu1_out),
            torch_to_ctypes(mu2_out),
            N,
            K,
            L,
            order,
            float(lr),
            float(eps),
            float(inv_k),
            float(inv_l),
            float(inv_k1),
            float(inv_l1),
        )

    sinkhorn_func.block_dim = resolved_block_dim
    return sinkhorn_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    if so_dir is None:
        so_dir = THIS_DIR / "outputs" / "so"
    return jit_compile_with_loader(
        src_path,
        load_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
        block_dim=block_dim,
    )
