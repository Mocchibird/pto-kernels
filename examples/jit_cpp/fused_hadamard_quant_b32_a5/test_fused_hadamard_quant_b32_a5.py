"""Does the fused kernel rotate and quantize correctly?

The fused kernel cannot be bit-exact against a torch reference: it rotates in
bf16 with a specific operand order, and no torch expression reproduces that
tree. So correctness is established in three ways instead, from strongest to
weakest:

1. **The scale bytes** must match a reference that rotates in fp32 and quantizes
   with `torch_npu`. A scale is a power of two derived from a block maximum, so
   bf16 rounding inside the butterfly almost never moves it -- if scales
   disagree, the rotation is wrong, not merely rounded differently.
2. **The dequantized values** must track the fp32-rotated reference to within
   MXFP4's own resolution. This catches a correct-looking permutation, which a
   relative-error check on the packed bytes would not.
3. **The output must be non-trivial.** A kernel that writes nothing, or writes
   the input back, is the characteristic silent failure on this hardware, and it
   would otherwise pass a loose tolerance.
"""

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from jit_util_fused_b32_a5 import (
    MX_BLOCK,
    SUPPORTED_K,
    VECTOR_CORES,
    build_and_load,
)

VENDOR_DST_TYPE = 296  # E2M1, matching mxfp4_quant_a5's tests


def hadamard_matrix(n):
    """Natural-order Sylvester +/-1 matrix, unnormalised -- the convention
    fast_hadamard_a5 and its tests use."""
    m = np.array([[1.0]], dtype=np.float64)
    while m.shape[0] < n:
        m = np.block([[m, m], [m, -m]])
    return m


def reference(x, k):
    """Rotate each 32-block in fp32 on the host, then quantize with the vendor op.

    Block-diagonal, matching the kernel: a row of k is k/32 independent
    rotations. fp32 deliberately, so the reference does not depend on the
    kernel's bf16 arithmetic.
    """
    h = torch.from_numpy(hadamard_matrix(MX_BLOCK)).to(torch.float32)
    flat = x.float().cpu().reshape(-1, MX_BLOCK) @ h
    rot = flat.reshape(x.shape[0], k).to(torch.bfloat16).npu()
    q, s = torch_npu.npu_dynamic_mx_quant(rot, dst_type=VENDOR_DST_TYPE)
    return rot, q, s.reshape(s.shape[0], -1)[:, : k // MX_BLOCK]


E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def dequant(q, s, k):
    """Unpack E2M1 nibbles and apply the E8M0 scale, on the host in fp32."""
    q = q.cpu()
    lo, hi = q & 0x0F, (q >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    mag = E2M1_LEVELS[(codes & 0x07).long()]
    vals = torch.where(codes >= 8, -mag, mag)
    exp = s.cpu().to(torch.int32) - 127
    scale = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
    return vals.reshape(-1, k // MX_BLOCK, MX_BLOCK) * scale.unsqueeze(-1)


@pytest.fixture(scope="module")
def seeded():
    torch.manual_seed(20260818)
    torch.npu.set_device(0)


# A spread rather than all 26 -- each width is a separate .so compile, so the
# full set costs minutes. But the spread has to cover both UNROLL CLASSES, which
# the original five did not.
#
# The butterfly's unroll width is UnrollFor<windows_per_tile, 8>, so a width whose
# window count is not a multiple of 8 unrolls by 4 instead. Those two paths are
# different code. Every one of the original five (64, 256, 1024, 4096, 14336)
# unrolls by 8, so the unroll-by-4 path was never executed -- and it was broken:
# the sweep was instantiated with 8 slots regardless, so iterations overlapped by
# four windows and the last ran 1024 elements past the tile. K=96 failed this
# file's own thresholds and nothing noticed.
#
# Which class a width lands in is NOT a property of the width: it is
# rows_for(k) * k / 256, and rows_for depends on TILE_ELEMS. Raising the tile from
# 16384 to 24576 moved 96, 192, 768 and 2816 from unroll-4 to unroll-8 and left
# the matrix single-class -- caught by the test below, which is why the class
# membership is now DERIVED from the .so instead of being listed here.
#
# What the list has to guarantee is coverage: at the shipped tile some width here
# must land in each class. 896 and 3584 are the unroll-4 members at 24576 (both
# 84 windows), chosen at opposite ends of the range; the rest are unroll-8.
WIDTHS = (32, 64, 96, 192, 256, 768, 896, 1024, 2816, 3584, 4096, 14336, 16384)


@pytest.mark.parametrize("k", WIDTHS)
def test_matches_reference(seeded, k):
    """Scales exact, dequantized values within MXFP4 resolution."""
    batch = 64
    fused = build_and_load(k=k, verbose=False)
    x = torch.randn(batch, k, dtype=torch.bfloat16, device="npu")
    q, s = fused(x)
    torch.npu.synchronize()
    _, q_ref, s_ref = reference(x, k)

    scale_match = (s.cpu() == s_ref.cpu()).float().mean().item()
    assert scale_match > 0.98, (
        f"K={k}: only {scale_match:.3f} of scale bytes match the fp32-rotated "
        "reference -- that is a wrong rotation, not bf16 rounding"
    )

    got = dequant(q, s, k).reshape(batch, k)
    want = dequant(q_ref, s_ref, k).reshape(batch, k)
    denom = want.abs().mean().clamp_min(1e-6)
    rel = (got - want).abs().mean() / denom
    assert rel < 0.05, f"K={k}: dequantized mean rel error {rel:.4f} vs reference"


# Deep enough that the pipeline machinery runs, at both unroll classes and at
# both ends of the rows range: 896 is unroll-4 with 24 rows per tile, 14336 is
# unroll-8 with a single row.
DEEP_WIDTHS = (896, 4096, 14336)
TILES_PER_CORE = 4


@pytest.mark.parametrize("k", DEEP_WIDTHS)
def test_matches_reference_many_tiles_per_core(seeded, k):
    """The same check as test_matches_reference, but with a full pipeline.

    test_matches_reference uses batch=64, which at K=4096 is 11 tiles spread over
    64 cores: one tile for eleven cores and none for the rest. So the buffer
    rotation (issued % NBuffers), the prefetch and the drain never run there, and
    a fault in any of them cannot fail that test. This sizes the batch so every
    core walks several tiles, and leaves a remainder so the partial tail tile is
    taken too -- except at K=14336, where a tile is one row and no batch can
    leave a remainder.
    """
    fused = build_and_load(k=k, verbose=False)
    rows = fused.rows_for()
    batch = rows * VECTOR_CORES * TILES_PER_CORE + rows // 2 + 1
    tiles = -(-batch // rows)

    # The point of the test is the depth, so assert it rather than trusting that
    # a TILE_ELEMS change left it intact.
    assert tiles / VECTOR_CORES >= 3, (
        f"K={k}: {tiles} tiles over {VECTOR_CORES} cores is too shallow to "
        "exercise the buffer rotation"
    )
    # A tile is `rows` rows, so where rows == 1 every batch is a whole number of
    # tiles and the kernel's partial branch is unreachable by construction. Only
    # claim the tail where one can exist.
    assert batch % rows or rows == 1, f"K={k}: batch {batch} is whole tiles"

    x = torch.randn(batch, k, dtype=torch.bfloat16, device="npu")
    q, s = fused(x)
    torch.npu.synchronize()
    _, q_ref, s_ref = reference(x, k)

    scale_match = (s.cpu() == s_ref.cpu()).float().mean().item()
    assert scale_match > 0.98, (
        f"K={k}, {tiles} tiles: only {scale_match:.3f} of scale bytes match the "
        "fp32-rotated reference"
    )

    got = dequant(q, s, k).reshape(batch, k)
    want = dequant(q_ref, s_ref, k).reshape(batch, k)
    denom = want.abs().mean().clamp_min(1e-6)
    rel = (got - want).abs().mean() / denom
    assert rel < 0.05, f"K={k}, {tiles} tiles: mean rel error {rel:.4f} vs reference"


@pytest.mark.parametrize("k", WIDTHS)
def test_output_is_nontrivial(seeded, k):
    """A kernel that writes nothing, or echoes its input, must fail here."""
    fused = build_and_load(k=k, verbose=False)
    x = torch.randn(32, k, dtype=torch.bfloat16, device="npu")
    q, s = fused(x)
    torch.npu.synchronize()
    assert q.any().item(), f"K={k}: nibbles are all zero"
    assert s.any().item(), f"K={k}: scale bytes are all zero"
    assert len(torch.unique(q.cpu())) > 4, f"K={k}: nibbles are degenerate"


@pytest.mark.parametrize("k", WIDTHS)
def test_rotation_actually_happened(seeded, k):
    """The rotation must change the answer.

    Quantizing x directly and quantizing (x @ H) should differ; if the fused
    output matches the unrotated quantization, `rotate` is a no-op -- which is
    precisely the failure a tolerance-based check would wave through.
    """
    fused = build_and_load(k=k, verbose=False)
    x = torch.randn(64, k, dtype=torch.bfloat16, device="npu")
    q_fused, _ = fused(x)
    q_plain, _ = torch_npu.npu_dynamic_mx_quant(x, dst_type=VENDOR_DST_TYPE)
    torch.npu.synchronize()
    same = (q_fused.cpu() == q_plain.cpu()).float().mean().item()
    assert same < 0.6, (
        f"K={k}: fused output matches the UNROTATED quantization at {same:.3f} "
        "-- the rotation is not happening"
    )


def unroll_width(k, rows):
    """UnrollFor<windows_per_tile, 8>, recomputed on the host.

    windows_per_tile is tile_elems/had_group with had_group = 256, i.e.
    rows*k/256 -- NOT rows*k/32, which is the block count. Getting that wrong is
    what made a first pass at this conclude the unroll was 8 everywhere.
    """
    windows = rows * k // 256
    limit = 8
    while limit > 1 and windows % limit:
        limit //= 2
    return limit


def test_width_matrix_covers_both_unroll_classes(seeded):
    """The matrix must exercise unroll-by-8 AND unroll-by-4.

    Guards the gap itself rather than one instance of it: the original five widths
    were all unroll-by-8, so the other path was dead code in CI while being live
    in production. A width added later -- or a change to TILE_ELEMS, which is what
    actually happened -- must not quietly return the matrix to one class.

    The classes are derived here rather than asserted against a stored list. A
    stored list is a snapshot of TILE_ELEMS, so it goes stale on a tuning change
    and then reports a tile change as a width bug. When this fails it names the
    supported widths that would restore coverage, since that is the fix.
    """
    seen = {}
    for k in WIDTHS:
        rows = build_and_load(k=k, verbose=False).rows_for()
        assert rows > 0, f"K={k}: rows_for returned 0"
        seen.setdefault(unroll_width(k, rows), []).append(k)
    if len(seen) < 2:
        missing = {}
        for k in sorted(SUPPORTED_K):
            rows = build_and_load(k=k, verbose=False).rows_for()
            cls = unroll_width(k, rows)
            if cls not in seen:
                missing.setdefault(cls, []).append(k)
        pytest.fail(
            f"the width matrix only exercises unroll {sorted(seen)}: {seen}. "
            f"Both paths are live in production. Add one of: "
            f"{ {c: v[:6] for c, v in missing.items()} }"
        )


@pytest.mark.parametrize("k", (256, 96, 768))
def test_constant_row_is_a_delta(seeded, k):
    """A constant row becomes one delta per 32-block: a sharp structural check.

    H's first column is all ones, so each block sums into its own element 0 and
    cancels across the other 31. Smearing means the butterfly is pairing wrongly;
    a single delta per *row* would mean it rotated the whole row instead of each
    block, which is the specific bug this variant exists to avoid.

    96 and 768 are unroll-by-4 widths. This check on one of those would have
    caught the sweep overlap directly: an overlapped window gets rotated twice,
    and a twice-rotated constant block is 32x the input in element 0 rather than
    a clean delta.
    """
    fused = build_and_load(k=k, verbose=False)
    x = torch.ones(8, k, dtype=torch.bfloat16, device="npu")
    q, s = fused(x)
    torch.npu.synchronize()
    vals = dequant(q, s, k).reshape(8, k // MX_BLOCK, MX_BLOCK)
    lead, rest = vals[:, :, 0].abs(), vals[:, :, 1:].abs()
    assert (lead > 0).all(), "each block's element 0 should carry its block sum"
    assert rest.max() <= lead.min() * 0.05, (
        f"each 32-block should rotate to a delta; leaked {rest.max():.3f} "
        f"against a lead of {lead.min():.3f}"
    )


def test_unsupported_k_is_rejected(seeded):
    """Widths without an instantiation must raise on the host.

    The dispatch would otherwise fall through silently and hand back the caller's
    buffers untouched.
    """
    for bad in (31, 33, 100, 0, 4095):
        with pytest.raises((ValueError, TypeError)):
            build_and_load(k=bad, verbose=False)


def test_wrong_dtype_is_rejected(seeded):
    fused = build_and_load(k=256, verbose=False)
    for dtype in (torch.float16, torch.float32):
        with pytest.raises(TypeError):
            fused(torch.randn(16, 256, dtype=dtype, device="npu"))
