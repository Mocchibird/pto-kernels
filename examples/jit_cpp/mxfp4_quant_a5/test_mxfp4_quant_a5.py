# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness for mxfp4_quant_a5, bit-exact against torch_npu.npu_dynamic_mx_quant.

Device runs repeat (PTO_DEVICE_REPEATS, default 5). If the vendor op is absent the
comparisons skip, which is not the same as passing.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from jit_util_mxfp4_a5 import (  # noqa: E402
    K,
    MX_BLOCK,
    SUPPORTED_K,
    build_and_load,
    compile_kernel,
    kernel_rows_for,
    load_lib,
    row_quantum,
    rows_for,
)

# The skill ships the shared helpers; use them rather than hand-rolling.
_REFERENCE = (
    Path(__file__).resolve().parents[3] / ".skills/testing-pto-kernels/reference"
)
if _REFERENCE.is_dir():
    sys.path.insert(0, str(_REFERENCE))
    import pto_demo_utils as demo  # noqa: E402
else:  # pragma: no cover - only on a partial checkout
    demo = None

VENDOR_DST_TYPE = 296  # torch_npu.float4_e2m1fn_x2
# E2M1 magnitude grid by 3-bit field, used only by the quality report
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def repeats() -> int:
    """Floored at 1: a repeat count of 0 would make every comparison loop below
    iterate zero times, so 25 tests would pass having asserted nothing."""
    return max(1, int(demo.device_repeats() if demo else 5))


def sync() -> None:
    (demo.synchronize_device if demo else torch.npu.synchronize)()


def block_dim() -> int:
    """Query the device; A5 SKUs differ in vector core count."""
    try:
        return demo.vector_core_count("npu:0") if demo else 64
    except Exception:  # pragma: no cover - query failure
        return 64


def no_vendor(why: str):
    """Losing the reference must FAIL, not skip.

    The vendor op is the entire correctness gate: without it 25 of these tests
    assert nothing. Skipping left a green suite that proved nothing, which is the
    failure mode GitHub's AI-code-review checklist names explicitly. Set
    PTO_ALLOW_NO_VENDOR=1 to downgrade to a skip on a machine that genuinely
    lacks the operator -- deliberately, and visibly in the run command.
    """
    if os.environ.get("PTO_ALLOW_NO_VENDOR") == "1":
        pytest.skip(f"{why} (PTO_ALLOW_NO_VENDOR=1)")
    pytest.fail(f"{why}: no reference to compare against, so this proves nothing")


def vendor_quantize(x):
    """(q, scale) from the CANN operator, with scale reshaped to match ours."""
    fn = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    if fn is None:
        no_vendor("torch_npu.npu_dynamic_mx_quant is missing")
    try:
        q, s = fn(x, dst_type=VENDOR_DST_TYPE)
    except Exception as exc:  # pragma: no cover - op signature drift
        no_vendor(f"vendor op rejected the call: {type(exc).__name__}: {exc}")
    sync()
    batch, k = x.shape
    return (
        q.cpu().numpy().reshape(batch, k // 2),
        s.cpu().numpy().reshape(batch, k // MX_BLOCK),
    )


def make_bf16(batch, k, seed):
    """Random bf16, rounded once on the host so both sides see the same values."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, k, generator=gen, dtype=torch.float32)
    return x.to(torch.bfloat16).npu()


def run_and_compare(kernel, x, label):
    want_q, want_s = vendor_quantize(x)
    # repeat: a sync bug is nondeterministic, so one clean pass proves little
    for attempt in range(repeats()):
        q, s = kernel(x)
        sync()
        got_q, got_s = q.cpu().numpy(), s.cpu().numpy()
        for what, got, want in (("scale", got_s, want_s), ("nibble", got_q, want_q)):
            assert np.array_equal(got, want), (
                f"{label}: {what} differs from the vendor on attempt {attempt} "
                f"({int((got != want).sum())} of {want.size})"
            )


@pytest.fixture(scope="module")
def quant_default():
    return build_and_load(block_dim=block_dim(), verbose=False)


# 1000 and 4097 do not fill a whole number of tiles, so they exercise the kernel's
# partial-tile tail; 65536 is more logical work than physical cores (skill: shape
# coverage). None of these reaches the host padding path -- row_quantum is 1 at
# this K, so nothing is ever padded. See test_host_padding_path for that.
@pytest.mark.parametrize("batch", [64, 128, 1000, 4097, 65536])
def test_matches_vendor(quant_default, batch):
    run_and_compare(quant_default, make_bf16(batch, K, batch), f"batch={batch}")


@pytest.mark.parametrize("k", [128, 256, 512])
def test_host_padding_path(k):
    """A batch the wrapper must round up, which nothing else here reaches.

    The kernel's tail takes whole rows, but the scale store has a 32-byte DMA
    floor, so widths below 1024 still need the host to pad the batch. That path
    allocates a zero tensor, copies into it and synchronizes -- and it is the one
    place ordering against torch's copy matters, so it needs its own coverage.
    """
    quantum = row_quantum(k)
    assert quantum > 1, f"k={k} never pads; this test needs a narrower width"
    batch = 2 * rows_for(k) + quantum - 1
    assert batch % quantum, "a multiple would skip the padding path entirely"
    kernel = build_and_load(block_dim=block_dim(), k=k, verbose=False)
    run_and_compare(kernel, make_bf16(batch, k, batch), f"k={k} padded")


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_matches_vendor_at_row_width(k):
    kernel = build_and_load(block_dim=block_dim(), k=k, verbose=False)
    run_and_compare(kernel, make_bf16(4 * rows_for(k), k, k), f"k={k}")


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_partial_last_tile(k):
    """A batch that does NOT fill its last tile, so the kernel's tail runs.

    Every other shape test uses a whole number of tiles, which would leave the
    partial-tile path unexercised: it would look correct because it never ran.
    The batch is a multiple of row_quantum so the wrapper does not pad, making
    this the kernel's tail rather than the host's zero-fill.
    """
    rows, quantum = rows_for(k), row_quantum(k)
    batch = 3 * rows + quantum
    assert batch % rows, f"k={k}: batch {batch} fills whole tiles, no tail"
    assert batch % quantum == 0, f"k={k}: would pad, hiding the kernel tail"
    kernel = build_and_load(block_dim=block_dim(), k=k, verbose=False)
    run_and_compare(kernel, make_bf16(batch, k, batch), f"k={k} tail")


def test_row_quantum_is_the_dma_floor():
    """Pinned: the batch multiple the wrapper needs, and that it beats the tile."""
    expected = {128: 8, 256: 4, 512: 2, 1024: 1, 2048: 1, 4096: 1}
    assert {k: row_quantum(k) for k in SUPPORTED_K} == expected
    # the whole point of the tail: the quantum is no longer the tile height
    assert all(row_quantum(k) < rows_for(k) for k in SUPPORTED_K if rows_for(k) > 1)


def test_nibble_order_is_pinned(quant_default):
    """One block of known codes, asserted exactly. No auto-fitting."""
    x = torch.zeros((rows_for(K), K), dtype=torch.bfloat16)
    x[0, 0], x[0, 1], x[0, 31] = 1.0, 2.0, 6.0
    q, s = quant_default(x.npu())
    sync()
    assert int(s.cpu().numpy()[0, 0]) == 127, "amax=6.0 must give scale byte 127"
    assert int(q.cpu().numpy()[0, 0]) == 0x42, "element 0 must be the low nibble"


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
    x = torch.zeros((rows_for(K), K), dtype=torch.bfloat16)
    for i, v in enumerate(values):
        blk = (i % (K // MX_BLOCK)) * MX_BLOCK
        x[0, blk : blk + MX_BLOCK] = v
        if len(values) > 1:
            x[0, blk + 1] = v / 2.0
    run_and_compare(quant_default, x.npu(), name)


def test_output_is_nontrivial(quant_default):
    """Two different inputs must give two different outputs."""
    outs = []
    for seed in (11, 12):
        q, s = quant_default(make_bf16(rows_for(K), K, seed))
        sync()
        outs.append((q.cpu().numpy().copy(), s.cpu().numpy().copy()))
    for what, a, b in (
        ("q", outs[0][0], outs[1][0]),
        ("scale", outs[0][1], outs[1][1]),
    ):
        assert not np.array_equal(a, b), f"{what} same for both inputs: did not run"


def test_quantization_quality(quant_default):
    """Relative RMSE and R-squared, not max error alone."""
    x = make_bf16(1024, K, 13)
    q, s = quant_default(x)
    sync()
    packed, rows, nblk = q.cpu().numpy(), x.shape[0], K // MX_BLOCK
    codes = np.empty((rows, K), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 0x0F, packed >> 4
    mag = E2M1_GRID[codes & 0x07]
    signed = np.where((codes & 0x08) != 0, -mag, mag)
    scale = np.exp2(s.cpu().numpy().astype(np.float64) - 127.0)
    recon = (signed.reshape((rows, nblk, MX_BLOCK)) * scale[:, :, None]).reshape(
        rows, K
    )
    original = x.float().cpu().numpy().astype(np.float64)
    mse = float(np.mean((recon - original) ** 2))
    rmse_rel = mse**0.5 / (float(np.sqrt(np.mean(original**2))) or 1.0)
    r2 = 1.0 - mse / float(np.var(original))
    print(f"\n  MXFP4 quality: rmse/rms={rmse_rel:.4f}  R^2={r2:.4f}")
    # MXFP4 keeps 3 magnitude bits over a 32-element block, so ~0.1 on N(0,1).
    # These bounds catch a broken kernel, not a subtle one.
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
    """rows_for() is stated in Python (the padding wrapper needs it before any .so"""
    query = kernel_rows_for(compile_kernel(verbose=False))
    mismatched = {
        k: (rows_for(k), query(k)) for k in SUPPORTED_K if rows_for(k) != query(k)
    }
    assert not mismatched, f"host/kernel tiling disagree: {mismatched}"
