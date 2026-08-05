# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness for the A5 MXFP4 quantization kernel, over K and batch.

The gate is **bit-exact** against ``mxfp4_ref``, not a tolerance: scale bytes and
E2M1 nibbles are integers, so any mismatch is a bug. That is stricter than the
repo skill's numeric thresholds, deliberately — see PLAN.md 4.8.

Per ``.skills/testing-pto-kernels``: real-device runs repeat
(``PTO_DEVICE_REPEATS``, default 5) because a three-pass pipeline is exactly where
a missing ``set_flag``/``wait_flag`` shows up nondeterministically, and each
synchronize is bounded (``PTO_SYNC_TIMEOUT_S``) because a sync bug deadlocks
rather than mismatching.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import mxfp4_ref as ref  # noqa: E402
from jit_util_mxfp4_a5 import (  # noqa: E402
    K,
    MX_BLOCK,
    SUPPORTED_K,
    build_and_load,
    compile_kernel,
    kernel_rows_for,
    load_lib,
    rows_for,
)

# The skill ships the shared helpers; use them rather than hand-rolling. If the
# checkout lacks them, say so loudly instead of silently diverging.
_REFERENCE = (
    Path(__file__).resolve().parents[3] / ".skills/testing-pto-kernels/reference"
)
if _REFERENCE.is_dir():
    sys.path.insert(0, str(_REFERENCE))
    import pto_demo_utils as demo  # noqa: E402
else:  # pragma: no cover - only on a partial checkout
    demo = None


def repeats() -> int:
    return demo.device_repeats() if demo else 5


def sync() -> None:
    if demo:
        demo.synchronize_device()
    else:
        torch.npu.synchronize()


def block_dim() -> int:
    """Derive from the device rather than hardcoding 64: A5 SKUs differ."""
    if demo:
        try:
            return demo.vector_core_count("npu:0")
        except Exception:  # pragma: no cover - fall back on query failure
            pass
    return 64


def make_bf16(batch, k, seed):
    """A bf16 tensor and the exact bit patterns the reference will consume."""
    rng = np.random.default_rng(seed)
    x32 = rng.standard_normal((batch, k)).astype(np.float32)
    bits = ref.f32_to_bf16_bits(x32)  # round once, host side
    exact = ref.bf16_bits_to_f32(bits)  # what the device will actually see
    x = torch.from_numpy(exact).to(torch.bfloat16).npu()
    return x, bits


def run_and_compare(kernel, k, batch, seed, label):
    x, bits = make_bf16(batch, k, seed)
    want_q, want_s = ref.quantize(bits)
    # repeat: a sync bug is nondeterministic, so one clean pass proves little
    for attempt in range(repeats()):
        q, s = kernel(x)
        sync()
        got_q = q.cpu().numpy()
        got_s = s.cpu().numpy()
        assert np.array_equal(got_s, want_s), (
            f"{label}: scale bytes differ on attempt {attempt} "
            f"({int((got_s != want_s).sum())} of {want_s.size})"
        )
        assert np.array_equal(got_q, want_q), (
            f"{label}: nibbles differ on attempt {attempt} "
            f"({int((got_q != want_q).sum())} of {want_q.size} bytes)"
        )
    return got_q, got_s


@pytest.fixture(scope="module")
def quant_default():
    return build_and_load(block_dim=block_dim(), verbose=False)


# 64 is one tile; 128 two tiles; 1000/4097 are non-multiples that exercise the
# padding wrapper; 65536 is more logical work than physical cores (skill: shape
# coverage), at K=4096 that is 32768 tiles over 64 cores.
@pytest.mark.parametrize("batch", [64, 128, 1000, 4097, 65536])
def test_matches_reference(quant_default, batch):
    run_and_compare(quant_default, K, batch, batch, f"batch={batch}")


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_matches_reference_at_row_width(k):
    kernel = build_and_load(block_dim=block_dim(), k=k, verbose=False)
    run_and_compare(kernel, k, 4 * rows_for(k), k, f"k={k}")


def test_nibble_order_is_pinned(quant_default):
    """One block of known codes, all bytes asserted exactly. No auto-fitting.

    Device-measured: e0=1.0, e1=2.0 with scale byte 127 gives first byte 0x42,
    i.e. element 2j occupies the LOW nibble.
    """
    x32 = np.zeros((rows_for(K), K), dtype=np.float32)
    x32[0, 0], x32[0, 1], x32[0, 31] = 1.0, 2.0, 6.0
    bits = ref.f32_to_bf16_bits(x32)
    x = torch.from_numpy(ref.bf16_bits_to_f32(bits)).to(torch.bfloat16).npu()
    q, s = quant_default(x)
    sync()
    assert int(s.cpu().numpy()[0, 0]) == 127, "scale byte for amax=6.0 must be 127"
    assert (
        int(q.cpu().numpy()[0, 0]) == 0x42
    ), "nibble order changed: element 0 must be the low nibble"


ADVERSARIAL = {
    "clamp_window": [2.0**-15, 2.0**-14, 2.0**-13],
    "subnormal_amax": [2.0**-20, 2.0**-24],
    "clip_to_six_band": [6.5, 7.0, 7.9],
    "e2m1_midpoints": [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    "all_zero": [0.0],
    "huge_outlier": [1024.0],
    "near_bf16_max": [3.0e38],
    "signed_zero": [-0.0],
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_blocks(quant_default, name):
    """Random N(0,1) reaches none of these, which is why they are enumerated."""
    values = ADVERSARIAL[name]
    rows = rows_for(K)
    x32 = np.zeros((rows, K), dtype=np.float32)
    for i, v in enumerate(values):
        blk = (i % (K // MX_BLOCK)) * MX_BLOCK
        x32[0, blk : blk + MX_BLOCK] = v
        x32[0, blk] = v  # amax position varies within the block
        if len(values) > 1:
            x32[0, blk + 1] = v / 2.0
    bits = ref.f32_to_bf16_bits(x32)
    x = torch.from_numpy(ref.bf16_bits_to_f32(bits)).to(torch.bfloat16).npu()
    want_q, want_s = ref.quantize(bits)
    q, s = quant_default(x)
    sync()
    assert np.array_equal(s.cpu().numpy(), want_s), f"{name}: scale bytes differ"
    assert np.array_equal(q.cpu().numpy(), want_q), f"{name}: nibbles differ"


def test_spec_cross_check():
    """The bit-chain reference and the float64 spec formula must agree, or a bias
    constant is wrong in a way the bit chain would reproduce faithfully."""
    _, bits = make_bf16(256, K, 7)
    _, chain = ref.quantize(bits)
    spec = ref.scale_bytes_from_spec(bits)
    assert np.array_equal(chain, spec), "bit-chain and spec scale bytes disagree"


def test_output_is_nontrivial(quant_default):
    """Catches the silent-no-op arch-flag failure: a kernel compiled for the wrong
    architecture returns success having written nothing.

    Two different inputs must give two different outputs. That needs no
    caller-supplied buffers, so it does not depend on their GM alignment.
    """
    outs = []
    for seed in (11, 12):
        x, _ = make_bf16(rows_for(K), K, seed)
        q, s = quant_default(x)
        sync()
        outs.append((q.cpu().numpy().copy(), s.cpu().numpy().copy()))
    assert not np.array_equal(
        outs[0][0], outs[1][0]
    ), "q identical for different inputs: kernel did not run"
    assert not np.array_equal(
        outs[0][1], outs[1][1]
    ), "scale identical for different inputs: kernel did not run"


def test_quantization_quality(quant_default):
    """RMSE relative to output magnitude, and R-squared — the skill's requirement
    for outlier-heavy kernels. Max-error alone only reports whichever value landed
    worst in a 16-level grid."""
    x, bits = make_bf16(1024, K, 13)
    q, s = quant_default(x)
    sync()
    original = ref.bf16_bits_to_f32(bits)
    recon = ref.dequantize(q.cpu().numpy(), s.cpu().numpy())
    rmse_rel, r2 = ref.quality(original, recon)
    print(f"\n  MXFP4 quality: rmse/rms={rmse_rel:.4f}  R^2={r2:.4f}")
    # MXFP4 keeps 3 magnitude bits over a 32-element block; on N(0,1) that is
    # ~0.1 relative RMSE. These bounds catch a broken kernel, not a subtle one.
    assert rmse_rel < 0.25, f"relative RMSE {rmse_rel:.4f} too high for MXFP4"
    assert r2 > 0.9, f"R^2 {r2:.4f} too low for MXFP4"


@pytest.mark.parametrize("bad_k", [0, 32, 96, 8192])
def test_unsupported_row_width_is_rejected(bad_k):
    # The dispatching launcher's default case is a silent no-op, so an unvalidated
    # k returns an untouched output buffer rather than failing.
    with pytest.raises(ValueError):
        build_and_load(k=bad_k, verbose=False)
    with pytest.raises(ValueError):
        load_lib(compile_kernel(verbose=False), k=bad_k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wrong_dtype_is_rejected(quant_default, dtype):
    x = torch.zeros((rows_for(K), K), dtype=dtype).npu()
    with pytest.raises(AssertionError, match="bfloat16"):
        quant_default(x)


def test_non_contiguous_is_rejected(quant_default):
    wide = torch.zeros((rows_for(K), 2 * K), dtype=torch.bfloat16).npu()
    view = wide[:, :K]
    assert not view.is_contiguous(), "test needs a genuinely strided view"
    with pytest.raises(AssertionError, match="contiguous"):
        quant_default(view)
    quant_default(view.contiguous())  # same data, accepted


def test_rows_for_matches_kernel():
    """rows_for() is stated in Python (the padding wrapper needs it before any .so
    exists) and again as RowsFor<K> in the kernel. Pin them together."""
    query = kernel_rows_for(compile_kernel(verbose=False))
    mismatched = {
        k: (rows_for(k), query(k)) for k in SUPPORTED_K if rows_for(k) != query(k)
    }
    assert not mismatched, f"host/kernel tiling disagree: {mismatched}"
