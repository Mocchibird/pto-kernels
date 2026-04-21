from pathlib import Path

import pytest
import torch
import torch_npu  # noqa

from jit_util_sinkhorn import jit_compile

DTYPE = torch.float16
KERNEL_CPP = Path(__file__).resolve().parent / "kernel_sinkhorn.cpp"

TEST_SHAPES = [
    (1, 16, 16),
    (1, 32, 32),
    (1, 64, 64),
    (1, 128, 128),
    (1, 256, 256),
    (2, 64, 64),
    (4, 32, 64),
    (4, 64, 32),
    (8, 128, 128),
    (1, 16, 256),
    (1, 256, 16),
]
TEST_ORDERS = [1, 5, 10]
TEST_SEEDS = [0, 42]
TEST_CASES = [
    (N, K, L, order, seed)
    for N, K, L in TEST_SHAPES
    for order in TEST_ORDERS
    for seed in TEST_SEEDS
]


def sinkhorn_ref(matrix_in, order=10, lr=0.5, eps=1e-3):
    """Pure-PyTorch reference matching the NPU kernel algorithm."""
    cm = matrix_in.float()
    N, K, L = cm.shape
    mu1 = torch.ones(N, 1, L, device=cm.device)
    mu2 = torch.ones(N, K, 1, device=cm.device)

    invK = 1.0 / K
    invL = 1.0 / L
    invK1 = 1.0 / (K - 1) if K > 1 else 1.0
    invL1 = 1.0 / (L - 1) if L > 1 else 1.0

    def compute_stds(cm, mu1, mu2):
        x = cm / mu1 / mu2
        row_sum = x.sum(dim=2)
        row_sqsum = (x * x).sum(dim=2)
        row_var = (row_sqsum - row_sum * row_sum * invL) * invL1
        row_std = row_var.clamp(min=0).sqrt()

        col_sum = x.sum(dim=1)
        col_sqsum = (x * x).sum(dim=1)
        col_var = (col_sqsum - col_sum * col_sum * invK) * invK1
        col_std = col_var.clamp(min=0).sqrt()
        return row_std, col_std

    # Initial target
    row_std, col_std = compute_stds(cm, mu1, mu2)
    tgt = (
        torch.min(
            row_std.min(dim=1, keepdim=True).values,
            col_std.min(dim=1, keepdim=True).values,
        )
        + eps
    )

    # Sinkhorn iterations
    for _ in range(order):
        row_std, col_std = compute_stds(cm, mu1, mu2)
        mu2 = mu2 * (row_std.unsqueeze(2) / tgt.unsqueeze(2)).pow(lr)
        mu1 = mu1 * (col_std.unsqueeze(1) / tgt.unsqueeze(1)).pow(lr)

    out = cm / mu1 / mu2
    return (
        out.to(matrix_in.dtype),
        mu1.squeeze(1).to(matrix_in.dtype),
        mu2.squeeze(2).to(matrix_in.dtype),
    )


@pytest.fixture(scope="session")
def sinkhorn_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=True, device=npu_device)


@pytest.mark.parametrize("N,K,L,order,seed", TEST_CASES)
def test_sinkhorn_matches_reference(sinkhorn_kernel, npu_device, N, K, L, order, seed):
    torch.manual_seed(seed)
    # Use positive values (sinkhorn assumes positive cost matrices)
    matrix_in = torch.rand(N, K, L, device=npu_device, dtype=DTYPE) + 0.1

    matrix_out = torch.empty_like(matrix_in)
    mu1_out = torch.empty(N, L, device=npu_device, dtype=DTYPE)
    mu2_out = torch.empty(N, K, device=npu_device, dtype=DTYPE)

    lr, eps = 0.5, 1e-3
    sinkhorn_kernel(
        matrix_in, matrix_out, mu1_out, mu2_out, order=order, lr=lr, eps=eps
    )
    torch.npu.synchronize()

    ref_out, ref_mu1, ref_mu2 = sinkhorn_ref(
        matrix_in.cpu(), order=order, lr=lr, eps=eps
    )

    # fp16 accumulation loses precision; use relaxed tolerances
    torch.testing.assert_close(matrix_out.cpu(), ref_out, rtol=5e-2, atol=1e-2)
    torch.testing.assert_close(mu1_out.cpu(), ref_mu1, rtol=5e-2, atol=1e-2)
    torch.testing.assert_close(mu2_out.cpu(), ref_mu2, rtol=5e-2, atol=1e-2)
