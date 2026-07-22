// fast_hadamard_a5 — A5 (Ascend 950 / dav-c310) register-resident fp16
// Walsh-Hadamard transform for block size N <= 128.
//
// WHY THIS EXISTS
// ---------------
// The `standard/` Hadamard kernel runs each of the log2(N) butterfly stages as
// a TGATHER(even)+TGATHER(odd) into UB scratch + TADD/TSUB back to UB. That is
// log2(N) full UB round-trips per tile, and TGATHER's per-repeat overhead
// dominates for small N. This kernel keeps the ENTIRE butterfly cascade in
// vector registers: one vlds in, log2(N) stages of purely-register shuffle +
// add/sub, one vsts out. No UB traffic and no TGATHER between stages.
//
// ALGORITHM (constant-geometry FWHT, concat-halves recombine, per stage):
//     (e, o) = vdintlv(v, v)      // e=[evens|evens], o=[odds|odds]  (both halves)
//     s      = e + o              // sums   duplicated in lanes 0..63 and 64..127
//     d      = e - o              // diffs  duplicated in lanes 0..63 and 64..127
//     v      = vsel(mask_lo64, s, d)  // v = [ sums(0..63) | diffs(64..127) ]
// The concat-halves recombine (sums to first half, diffs to second half) is the
// textbook constant-geometry FWHT and produces the *natural Sylvester order*
// directly (verified vs scipy.linalg.hadamard) — no bit-reversal correction.
// The duplicate-then-select trick realises the "move diffs to the upper half"
// step with zero UB traffic: vdintlv(v, v) duplicates evens/odds into both
// 64-lane halves, so the diffs already exist in lanes 64..127 and vsel picks
// sums from the low half and diffs from the high half in one op.
//
// SCOPE: this build path targets N == 128 (one length-128 block per 128-lane
// b16 register), which is the clean single-register case (the low-64 select
// mask lines up with the block halves). N < 128 with lane-packed blocks needs a
// per-block half-select mask + shifted diff placement and is left as a
// documented extension (see README).
//
// Build: bisheng --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE
//        -DHAD_N=<n> -DHAD_LOG2N=<log2 n> [-DHAD_INV_SQRT=<1/sqrt(n)>]
//        [-DROWS_PER_TILE=<rows>]

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef HAD_N
#define HAD_N 128
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 7
#endif
// Rows (length-N blocks) per GM<->UB tile. Sized for a large DMA burst: at
// N=128 this is a ROWS_PER_TILE*256 B contiguous transfer (256 rows = 64 KB),
// which the HBM path can drive near peak. Ping/pong uses 2 buffers, so
// 2*ROWS_PER_TILE*256 B must fit the UB budget below. For a given batch, pick
// ROWS_PER_TILE so batch/ROWS_PER_TILE >= (# AIV) to still fill the grid.
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 256
#endif
// Software-pipeline width. The butterfly body is manually unrolled 8-way
// (see hadamard_vf) to hide the vdintlv->vadd->vsub->vsel dependency-chain
// latency across independent registers; REGS_PER_TILE must be a multiple of 8.
#define HAD_UNROLL 8

static_assert(HAD_N >= 2 && HAD_N <= 128, "register kernel supports N in [2,128]");
static_assert((HAD_N & (HAD_N - 1)) == 0, "HAD_N must be a power of two");
// v1 concat-halves recombine (vdintlv(v,v)+vsel low-64) is exact only for the
// single-block-per-register case. Lane-packed N<128 needs a per-block mask.
static_assert(HAD_N == 128, "v1 supports N==128; see README for the N<128 extension");
static_assert((1u << HAD_LOG2N) == (unsigned)HAD_N, "HAD_LOG2N must equal log2(HAD_N)");

#ifdef __CCE_AICORE__

constexpr unsigned LANES_B16 = 128;              // f16 elements per vreg
constexpr unsigned BLK_PER_REG = LANES_B16 / HAD_N;
static_assert(ROWS_PER_TILE % BLK_PER_REG == 0,
              "ROWS_PER_TILE must be a multiple of 128/HAD_N");
constexpr unsigned REGS_PER_TILE = ROWS_PER_TILE / BLK_PER_REG;
static_assert(REGS_PER_TILE % HAD_UNROLL == 0,
              "REGS_PER_TILE must be a multiple of HAD_UNROLL");
constexpr unsigned FLAT = ROWS_PER_TILE * HAD_N; // = REGS_PER_TILE * 128

constexpr unsigned aln256(unsigned b) { return (b + 255u) & ~255u; }
constexpr unsigned X_BYTES = aln256(FLAT * sizeof(half));
constexpr unsigned UB_X0 = 0;
constexpr unsigned UB_X1 = UB_X0 + X_BYTES;      // ping/pong
static_assert(UB_X1 + X_BYTES <= 192u * 1024u, "UB overflow (>192 KB)");

// 1/sqrt(N): host may override via -DHAD_INV_SQRT to avoid device sqrt.
#ifndef HAD_INV_SQRT
#if   HAD_N == 2
#define HAD_INV_SQRT 0.70710678118654752f
#elif HAD_N == 4
#define HAD_INV_SQRT 0.5f
#elif HAD_N == 8
#define HAD_INV_SQRT 0.35355339059327376f
#elif HAD_N == 16
#define HAD_INV_SQRT 0.25f
#elif HAD_N == 32
#define HAD_INV_SQRT 0.17677669529663689f
#elif HAD_N == 64
#define HAD_INV_SQRT 0.125f
#else
#define HAD_INV_SQRT 0.08838834764831845f   // 1/sqrt(128)
#endif
#endif

#ifdef __DAV_VEC__
// Register-resident butterfly over REGS_PER_TILE consecutive 128-lane windows
// of the UB tile at byte offset x_off. No UB traffic between stages; no
// pipe_barrier (dav-c310 vector pipe issues same-pipe RAW in program order).
__tf__ static AICORE void hadamard_vf(__ubuf__ half *x)
{
    __VEC_SCOPE__
    {
        uint32_t l = LANES_B16;
        MaskReg p = CreatePredicate<half>(l);            // all 128 b16 lanes
        uint32_t lh = LANES_B16 / 2;
        MaskReg mlo = CreatePredicate<half>(lh);         // lanes 0..63 active
        const half inv = (half)HAD_INV_SQRT;

        // 8-way software pipeline: issue the SAME op for HAD_UNROLL independent
        // registers back-to-back so the vector pipe stays busy through each
        // op's latency instead of stalling on the vdintlv->vadd->vsub->vsel
        // dependency chain of a single register. Explicit (not array/loop)
        // because __VEC_SCOPE__ requires individually-named RegTensors.
        for (uint16_t base = 0; base < (uint16_t)REGS_PER_TILE; base += 8) {
            RegTensor<half> v0, v1, v2, v3, v4, v5, v6, v7;
            RegTensor<half> e0, e1, e2, e3, e4, e5, e6, e7;
            RegTensor<half> o0, o1, o2, o3, o4, o5, o6, o7;
            RegTensor<half> s0, s1, s2, s3, s4, s5, s6, s7;
            RegTensor<half> d0, d1, d2, d3, d4, d5, d6, d7;
            const uint32_t b = base * LANES_B16;
            vlds(v0, x, b + 0 * LANES_B16, NORM); vlds(v1, x, b + 1 * LANES_B16, NORM);
            vlds(v2, x, b + 2 * LANES_B16, NORM); vlds(v3, x, b + 3 * LANES_B16, NORM);
            vlds(v4, x, b + 4 * LANES_B16, NORM); vlds(v5, x, b + 5 * LANES_B16, NORM);
            vlds(v6, x, b + 6 * LANES_B16, NORM); vlds(v7, x, b + 7 * LANES_B16, NORM);
            for (uint16_t st = 0; st < (uint16_t)HAD_LOG2N; ++st) {
                vdintlv(e0, o0, v0, v0); vdintlv(e1, o1, v1, v1);
                vdintlv(e2, o2, v2, v2); vdintlv(e3, o3, v3, v3);
                vdintlv(e4, o4, v4, v4); vdintlv(e5, o5, v5, v5);
                vdintlv(e6, o6, v6, v6); vdintlv(e7, o7, v7, v7);
                vadd(s0, e0, o0, p); vadd(s1, e1, o1, p);
                vadd(s2, e2, o2, p); vadd(s3, e3, o3, p);
                vadd(s4, e4, o4, p); vadd(s5, e5, o5, p);
                vadd(s6, e6, o6, p); vadd(s7, e7, o7, p);
                vsub(d0, e0, o0, p); vsub(d1, e1, o1, p);
                vsub(d2, e2, o2, p); vsub(d3, e3, o3, p);
                vsub(d4, e4, o4, p); vsub(d5, e5, o5, p);
                vsub(d6, e6, o6, p); vsub(d7, e7, o7, p);
                vsel(v0, s0, d0, mlo); vsel(v1, s1, d1, mlo);
                vsel(v2, s2, d2, mlo); vsel(v3, s3, d3, mlo);
                vsel(v4, s4, d4, mlo); vsel(v5, s5, d5, mlo);
                vsel(v6, s6, d6, mlo); vsel(v7, s7, d7, mlo);
            }
            vmuls(v0, v0, inv, p, MODE_ZEROING); vmuls(v1, v1, inv, p, MODE_ZEROING);
            vmuls(v2, v2, inv, p, MODE_ZEROING); vmuls(v3, v3, inv, p, MODE_ZEROING);
            vmuls(v4, v4, inv, p, MODE_ZEROING); vmuls(v5, v5, inv, p, MODE_ZEROING);
            vmuls(v6, v6, inv, p, MODE_ZEROING); vmuls(v7, v7, inv, p, MODE_ZEROING);
            vsts(v0, x, b + 0 * LANES_B16, NORM_B16, p); vsts(v1, x, b + 1 * LANES_B16, NORM_B16, p);
            vsts(v2, x, b + 2 * LANES_B16, NORM_B16, p); vsts(v3, x, b + 3 * LANES_B16, NORM_B16, p);
            vsts(v4, x, b + 4 * LANES_B16, NORM_B16, p); vsts(v5, x, b + 5 * LANES_B16, NORM_B16, p);
            vsts(v6, x, b + 6 * LANES_B16, NORM_B16, p); vsts(v7, x, b + 7 * LANES_B16, NORM_B16, p);
        }
    }
}
#endif  // __DAV_VEC__

#endif  // __CCE_AICORE__

__global__ AICORE void fast_hadamard_a5(__gm__ void *x_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    set_mask_norm();
    set_vector_mask(-1, -1);

    // Flat (1, FLAT) contiguous GM<->UB burst (rows are contiguous in GM).
    using Sh = pto::Shape<1, 1, 1, 1, FLAT>;
    using St = pto::Stride<1, 1, 1, FLAT, 1>;
    using FlatTile = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;

    const unsigned x_off[2] = {UB_X0, UB_X1};
    const event_t ev[2] = {EVENT_ID0, EVENT_ID1};

    // On A5, get_block_idx()/get_block_num() already enumerate all vector
    // subblocks (AIVs): a block_dim=8 launch yields block ids 0..15, and the two
    // AIVs of one AIC receive distinct ids (confirmed in sim — AIC0's subblocks
    // took tiles 0 and 1). So, unlike the dav-c220 standard kernel, we do NOT
    // combine get_subblockid()/get_subblockdim() here; doing so would double-count
    // and make cores redundantly reprocess tiles. Each id owns a disjoint tile set.
    const uint32_t cid = get_block_idx();
    const uint32_t num_cores = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;   // batch MUST be a multiple of ROWS_PER_TILE

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += num_cores, ++it) {
        const int pp = it & 1;
        const uint64_t off = static_cast<uint64_t>(tb) * FLAT;
        GlobalTensor<half, Sh, St> g(reinterpret_cast<__gm__ half *>(x_gm) + off, Sh());

        wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
        FlatTile xt;  TASSIGN(xt, x_off[pp]);
        TLOAD(xt, g);
        set_flag(PIPE_MTE2, PIPE_V, ev[pp]);
        wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);

        hadamard_vf((__ubuf__ half *)(uintptr_t)x_off[pp]);

        set_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        TSTORE(g, xt);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
    (void)x_gm; (void)batch;
#endif
}

extern "C" void call_fast_hadamard_a5(uint32_t block_dim, void *stream,
                                      uint8_t *x, uint32_t batch)
{
    fast_hadamard_a5<<<block_dim, nullptr, stream>>>(x, batch);
}

// Copy-floor reference: identical tiling / ping-pong GM->UB->GM bounce with NO
// compute. Times the DMA path alone so a benchmark can attribute the Hadamard's
// bandwidth gap to memory vs compute. Same FLAT tile and double-buffering.
__global__ AICORE void copy_ref_a5(__gm__ void *x_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    using Sh = pto::Shape<1, 1, 1, 1, FLAT>;
    using St = pto::Stride<1, 1, 1, FLAT, 1>;
    using FlatTile = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;

    const unsigned x_off[2] = {UB_X0, UB_X1};
    const event_t ev[2] = {EVENT_ID0, EVENT_ID1};
    const uint32_t cid = get_block_idx();
    const uint32_t num_cores = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += num_cores, ++it) {
        const int pp = it & 1;
        const uint64_t off = static_cast<uint64_t>(tb) * FLAT;
        GlobalTensor<half, Sh, St> g(reinterpret_cast<__gm__ half *>(x_gm) + off, Sh());
        wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
        FlatTile xt;  TASSIGN(xt, x_off[pp]);
        TLOAD(xt, g);
        set_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]);
        wait_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]);
        TSTORE(g, xt);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
    (void)x_gm; (void)batch;
#endif
}

extern "C" void call_copy_ref_a5(uint32_t block_dim, void *stream,
                                 uint8_t *x, uint32_t batch)
{
    copy_ref_a5<<<block_dim, nullptr, stream>>>(x, batch);
}
