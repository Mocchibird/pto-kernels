"""
Test for the minimal PTO sinkhorn demo.

Mirrors deepseek-ai/TileKernels tests/mhc/test_sinkhorn.py. Runs forward
and autograd backward against the reference.
"""
import pytest
import torch
import torch_npu  # noqa: F401

from jit_util_sinkhorn import sinkhorn_normalize, sinkhorn_normalize_ref
from jit_util_sinkhorn_v2 import sinkhorn_normalize_v2


def _generate(n0: int, n1: int, mhc: int, device: str):
    return {
        "comb_res_mix": torch.randn((n0, n1, mhc, mhc), dtype=torch.float16, device=device),
        "out_grad":     torch.randn((n0, n1, mhc, mhc), dtype=torch.float16, device=device),
        "repeat": 10,
        "eps":    1e-6,
    }


def _tester(impl, test_data):
    x = test_data["comb_res_mix"].clone().requires_grad_()
    out = impl(x, test_data["repeat"], test_data["eps"])
    torch.autograd.backward([out], [test_data["out_grad"]])
    return out, x.grad


@pytest.fixture(scope="session")
def device(npu_device):
    return npu_device


@pytest.mark.parametrize("n0",  [1, 2])
@pytest.mark.parametrize("n1",  [1, 1024, 4096])
@pytest.mark.parametrize("mhc", [4])
def test_sinkhorn_comprehensive(device, n0, n1, mhc):
    torch.manual_seed(0)
    test_data = _generate(n0=n0, n1=n1, mhc=mhc, device=device)

    out_pto, grad_pto = _tester(sinkhorn_normalize, test_data)
    out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    torch.testing.assert_close(out_pto, out_ref, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(grad_pto, grad_ref, rtol=1e-2, atol=1e-5)

@pytest.mark.parametrize("n0",  [1, 2])
@pytest.mark.parametrize("n1",  [1, 1024, 4096])
@pytest.mark.parametrize("mhc", [4])
def test_sinkhorn_v2_forward(device, n0, n1, mhc):
    torch.manual_seed(0)
    test_data = _generate(n0=n0, n1=n1, mhc=mhc, device=device)
    x = test_data["comb_res_mix"]

    out_v2  = sinkhorn_normalize_v2(x, test_data["repeat"], test_data["eps"])
    out_ref = sinkhorn_normalize_ref(x, test_data["repeat"], test_data["eps"])

    torch.testing.assert_close(out_v2, out_ref, rtol=1e-2, atol=1e-5)