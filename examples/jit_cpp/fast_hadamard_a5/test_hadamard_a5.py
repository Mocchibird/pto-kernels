"""Reference + (optional) on-device test for fast_hadamard_a5.

The numpy checks (transform math) always run. The on-device check runs only when
torch + torch_npu are importable and a compiled .so is available; the
authoritative on-device numeric proof is sim_test/run.sh (camodel).
"""

import os

import numpy as np
import pytest

N = 128
LOG2N = N.bit_length() - 1


def cg_concat_halves(x):
    """Constant-geometry FWHT with concat-halves recombine — exactly what the
    kernel computes (sums to first half, diffs to second half each stage)."""
    x = x.astype(np.float64).copy()
    for _ in range(LOG2N):
        e, o = x[0::2], x[1::2]
        x = np.concatenate([e + o, e - o])
    return x


def sylvester(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def test_concat_halves_is_natural_sylvester():
    """The recombine used by the kernel == natural Sylvester order (no bitrev)."""
    H = sylvester(N)
    rng = np.random.default_rng(0)
    for _ in range(16):
        x = rng.standard_normal(N)
        assert np.allclose(cg_concat_halves(x), H @ x)


def test_interleave_recombine_is_not_a_hadamard():
    """Guard against the seductive-but-wrong interleave recombine."""
    def cg_interleave(x):
        x = x.astype(np.float64).copy()
        for _ in range(LOG2N):
            e, o = x[0::2], x[1::2]
            s, d = e + o, e - o
            y = np.empty_like(x)
            y[0::2], y[1::2] = s, d
            x = y
        return x

    H = sylvester(N)
    x = np.random.default_rng(1).standard_normal(N)
    got = cg_interleave(x)
    # not a permutation of the true WHT (magnitudes differ)
    assert not np.allclose(np.sort(np.abs(got)), np.sort(np.abs(H @ x)))


@pytest.mark.skipif(
    "PTO_RUN_DEVICE" not in os.environ,
    reason="set PTO_RUN_DEVICE=1 with torch_npu + built .so to run on device",
)
def test_on_device_matches_reference():
    import torch  # noqa
    import torch_npu  # noqa
    from jit_util_hadamard_a5 import jit_compile

    here = os.path.dirname(__file__)
    fn = jit_compile(os.path.join(here, "fast_hadamard_a5.cpp"))

    batch = 256
    inv = 1.0 / np.sqrt(N)
    H = sylvester(N)
    x_np = np.random.default_rng(2).standard_normal((batch, N)).astype(np.float16)
    gold = (x_np.astype(np.float64) @ H.T * inv).astype(np.float16)

    x = torch.from_numpy(x_np).npu()
    fn(x, batch)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float64)

    max_diff = np.abs(out - gold.astype(np.float64)).max()
    assert max_diff < 0.1, f"max_diff={max_diff}"
