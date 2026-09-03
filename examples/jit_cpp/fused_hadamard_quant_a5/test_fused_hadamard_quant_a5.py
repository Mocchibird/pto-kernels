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

from jit_util_fused_a5 import (
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


def fwht(x):
    """Textbook iterative Walsh-Hadamard transform, natural order.

    A second implementation on purpose. The kernel does the transform as
    window-local work plus cross-window stages; this does explicit strided
    butterflies, so a mistake in the kernel's decomposition cannot cancel out of
    both sides. `test_fwht_matches_the_explicit_matrix` pins this against the
    Sylvester matrix at the widths where building that matrix is cheap -- at
    K=16384 the matrix alone is a gigabyte, which is why the tests reference this
    instead of the matrix directly.
    """
    y = np.array(x, dtype=np.float64, copy=True)
    k = y.shape[-1]
    h = 1
    while h < k:
        for i in range(0, k, 2 * h):
            a = y[..., i : i + h].copy()
            b = y[..., i + h : i + 2 * h].copy()
            y[..., i : i + h] = a + b
            y[..., i + h : i + 2 * h] = a - b
        h *= 2
    return y


@pytest.mark.parametrize("k", (32, 64, 256, 1024))
def test_fwht_matches_the_explicit_matrix(k):
    """The reference implementation must equal x @ H_k.

    Every other test references `fwht`, so this is what makes those tests mean
    "matches the Hadamard matrix" rather than "matches my other loop".
    """
    rng = np.random.default_rng(k)
    x = rng.standard_normal((4, k))
    want = x @ hadamard_matrix(k)
    got = fwht(x)
    assert np.abs(got - want).max() < 1e-9 * max(np.abs(want).max(), 1.0)


def reference(x, k):
    """Rotate each row by the order-K transform, then quantize.

    One rotation across the whole row, matching the kernel: every output element
    depends on all k inputs. Computed by `fwht`, a different decomposition from
    the kernel's and itself pinned against the explicit Sylvester matrix, so a
    mistake in the kernel cannot cancel out of both sides. fp64 on the host, so
    the reference does not depend on the kernel's bf16 arithmetic.
    """
    rot = torch.from_numpy(fwht(x.float().cpu().numpy())).to(torch.bfloat16).npu()
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


# A spread rather than all eight -- each width is a separate .so compile, so the
# full set costs minutes. The spread has to cover both SLOT-ADDRESSING CLASSES.
#
# A slot addresses group `Slot / chunks` at chunk `Slot % chunks`. For K <= 256 a
# group is a whole window holding several rows and chunks is 1; for K >= 512 a
# group is one row spread over 2, 4, 8 or 16 chunks. Those are different index
# arithmetic, and a matrix covering only one would leave the other live in
# production and dead in CI. That has happened on the predecessor of this
# kernel: every width in a five-width matrix landed in one class, the uncovered
# path was broken, the sweep ran past the end of the tile, and nothing noticed.
#
# Class membership is DERIVED from the width below rather than listed here, so it
# cannot go stale against a tuning change.
WIDTHS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


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


# Deep enough that the pipeline machinery runs, at both slot classes and at both
# ends of the rows range: 256 packs 96 rows per tile and never chunks, 4096 is
# 16 chunks with 6 rows.
DEEP_WIDTHS = (256, 1024, 8192)
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


def chunk_count(k):
    """Shape::chunks, recomputed on the host.

    A group is one row once the row is at least a window wide, so chunks is
    (k/2)/128 there and 1 below it, where several rows share a window instead.
    The two cases take different slot addressing, so a matrix that covered only
    one would leave the other live in production and dead in CI.
    """
    upper = (k if k >= 256 else 256) // 2
    return upper // min(upper, 128)


def test_width_matrix_covers_both_chunk_classes(seeded):
    """The matrix must exercise chunks == 1 AND chunks > 1.

    Guards the gap itself rather than one instance of it: the original five widths
    were in one chunk class, so the other path was dead code in CI while live
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
        seen.setdefault(chunk_count(k) > 1, []).append(k)
    if len(seen) < 2:
        missing = {}
        for k in sorted(SUPPORTED_K):
            cls = chunk_count(k) > 1
            if cls not in seen:
                missing.setdefault(cls, []).append(k)
        pytest.fail(
            f"the width matrix only exercises chunked={sorted(seen)}: {seen}. "
            f"Both paths are live in production. Add one of: "
            f"{ {c: v[:6] for c, v in missing.items()} }"
        )


@pytest.mark.parametrize("k", (256, 512, 4096))
def test_constant_row_is_a_delta(seeded, k):
    """A constant row becomes ONE delta for the whole row.

    H's first column is all ones, so a constant row sums into element 0 and
    cancels across all k-1 others. This is the check that separates this kernel
    from a block-wise rotation: there, a constant row gives k/32 deltas, one per
    block. Here anything past element 0 means the butterfly stopped short of
    spanning the row.

    256 packs rows into a window, 512 chunks by 2 and 4096 by 16, so the three
    cover both slot-addressing classes. Every 32-block after the first is all
    zeros, so only block 0 carries signal.
    """
    fused = build_and_load(k=k, verbose=False)
    x = torch.ones(8, k, dtype=torch.bfloat16, device="npu")
    q, s = fused(x)
    torch.npu.synchronize()
    # dequant returns (batch, k/32, 32); flatten so element 0 is the row's, not
    # block 0's -- indexing the blocked shape takes all 32 of block 0, and 31 of
    # those are legitimately zero
    vals = dequant(q, s, k).reshape(8, k)
    lead, rest = vals[:, 0].abs(), vals[:, 1:].abs()
    assert (lead > 0).all(), "element 0 should carry the whole row's sum"
    assert rest.max() <= lead.min() * 0.05, (
        f"a constant row should rotate to a single delta; leaked "
        f"{rest.max():.3f} against a lead of {lead.min():.3f}. One delta per "
        f"32-block instead means the rotation is still block-wide."
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
