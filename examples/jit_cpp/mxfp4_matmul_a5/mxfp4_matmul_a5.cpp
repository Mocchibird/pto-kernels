// MXFP4 matmul on A5: y = A @ B with BOTH operands MXFP4, on the cube's
// native microscaled path, TMATMUL_MX.
//
//   A: (M, K) E2M1 nibbles, two per byte, + one E8M0 scale per 32 along K
//   B: (K, N) the same, stored DN -- see LAYOUTS
//   y: (M, N) bf16, accumulated in fp32
//
// The scaling is in the datapath, so this is not a dequantize-and-matmul.
// From PTO's CPU reference (pto/cpu/TMatmul.hpp:44-67):
//
//     acc += a(i, k) * b(k, j) * aScale(i, k / 32) * bScale(k / 32, j)
//
// LAYOUTS -- asymmetric, and not implied by the type names:
//   A data   Layout::ND       row-major (M, K)
//   A scale  Layout::MX_A_ND  row-major (M, K/32)
//   B data   Layout::DN       NOT ND
//   B scale  Layout::MX_B_DN  NOT MX_B_ND
// Feeding row-major B compiles and computes something else.
//
// FP4 POINTER ARITHMETIC IS IN BYTES. sizeof(float4_e2m1x2_t) is 1 and it
// holds two elements, so a K-element step is (K >> 1) bytes and a scale step
// is (K >> 5).
//
// SCALE TILES BIND BY ADDRESS. TMATMUL_MX takes them for the type check but
// never forwards them to the mad_mx builtin (a5/TMatmul.hpp:259-271). The
// hardware reads them at GetScaleAddr(operand.data()), a >> 4 of the
// operand's L0 address (a5/utils.hpp:82-87): the MX scale store is a 1/16
// shadow of L0A/L0B. TASSIGNing them anywhere else compiles and multiplies
// by whatever happens to be there.
//
// Alignment, all asserted by CheckMadMxValid: K a multiple of 64 ELEMENTS,
// N a multiple of 64 for fp4, M a multiple of 16, accumulator fp32.
//
// Arch: cube plus a store, so dav-c310. No MIX, no cross-core sync.
//
// Measured throughput, the sweeps behind the defaults, and the approaches
// tried and rejected are in README.md.
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t SCALE_FACTOR = 32;       // elements per E8M0 scale
constexpr uint32_t SHIFT_SCALE_FACTOR = 5;  // log2(SCALE_FACTOR)
constexpr uint32_t SHIFT_FP4 = 1;           // two nibbles per byte

// Output-tile shape. The L1 fill for the whole matmul is
// M*N*K*bytes_per_elem * (1/BASE_M + 1/BASE_N), and L0C, which holds one
// output tile as fp32, is what bounds both dimensions.
#ifndef MXMM_BASE_M
#define MXMM_BASE_M 128
#endif
#ifndef MXMM_BASE_N
#define MXMM_BASE_N 256
#endif
constexpr unsigned SMALL_M = MXMM_BASE_M;  // 16-aligned
constexpr unsigned SMALL_N = MXMM_BASE_N;  // 64-aligned for fp4

// A taller tile for large M: less L1 re-fetch, half the block count.
constexpr unsigned BIG_M = 2u * SMALL_M;
constexpr unsigned BIG_N = SMALL_N;

// A 16-row tile for small M, so a caller with M below SMALL_M is not padded
// up to it. 16 is the alignment floor TMATMUL_MX imposes.
#ifndef MXMM_TINY_M
#define MXMM_TINY_M 16
#endif
constexpr unsigned TINY_M = MXMM_TINY_M;
constexpr unsigned TINY_N = SMALL_N;
constexpr unsigned BASE_K = 256;  // the cube's K tile, a multiple of 64

// The L1 slab's K extent, independent of the cube's BASE_K. A GM->L1 copy of
// a (BASE_M, K_L1) fp4 tile moves K_L1/2 contiguous bytes per row, and the
// cube still consumes BASE_K at a time from a K_SUB-step inner loop.
#ifndef MXMM_K_L1
#define MXMM_K_L1 512
#endif
constexpr unsigned K_L1 = MXMM_K_L1;
constexpr unsigned K_SUB = K_L1 / BASE_K;
static_assert(K_L1 % BASE_K == 0, "the slab must be whole cube K tiles");
static_assert(K_SUB >= 1u, "at least one sub-tile per slab");

// How many L1 sets, and so how far ahead the GM->L1 load runs: a slab is
// extracted LOOKAHEAD passes after it is loaded. With one set the slab is
// loaded at the top of its own iteration and extracted immediately after, so
// the load no longer overlaps the extract and multiply, and half the L1 is
// free for a wider slab.
#ifndef MXMM_L1_SETS
#define MXMM_L1_SETS 2
#endif
constexpr uint32_t L1_SETS = (uint32_t)(MXMM_L1_SETS);
constexpr uint32_t LOOKAHEAD = L1_SETS - 1u;
static_assert(L1_SETS >= 1u && L1_SETS <= 2u, "one or two L1 sets");

// Output tiles are walked SWIZZLE_GROUP rows down before stepping across, so
// the blocks resident at one time form a square-ish patch of the output
// rather than a full-width strip. 1 restores the plain row-major walk. The
// default is chosen per instantiation from n_tiles; -DMXMM_SWIZZLE overrides
// it.
constexpr uint32_t SWIZZLE_WIDE_TILES = 48u;
constexpr uint32_t SWIZZLE_WIDE = 2u;
constexpr uint32_t SWIZZLE_NARROW = 8u;
constexpr unsigned BASE_SK = BASE_K / SCALE_FACTOR;

static_assert(SMALL_M % 16 == 0 && BIG_M % 16 == 0, "M must be 16-aligned");
static_assert(TINY_M % 16 == 0, "M must be 16-aligned");
static_assert(SMALL_M % TINY_M == 0, "the coarser tiles are multiples of it");
static_assert(SMALL_N % 64 == 0 && BIG_N % 64 == 0, "fp4 needs N 64-aligned");
static_assert(BIG_M % SMALL_M == 0,
              "the big tile's M must be a multiple of the small one's, so "
              "one admissibility check covers both");
static_assert(BASE_K % 64 == 0, "TMATMUL_MX needs K a multiple of 64");

// K and N are template parameters so the GM strides stay static; they are
// per-layer constants. M is RUNTIME, because block_dim is m_tiles * n_tiles
// and a compile-time M would cap parallelism at N/BASE_N.
//
// M_MAX only sizes the BaseShape2D extents. Every GM offset here is computed
// explicitly from BASE_M/BASE_N/BASE_K, so addressing does not depend on it;
// it has to be an upper bound on the real M, which the host checks.
template <unsigned M_MAX, unsigned K, unsigned N, unsigned BM, unsigned BN>
__global__ AICORE void mxfp4_matmul(__gm__ void *a_gm, __gm__ void *a_scale_gm,
                                    __gm__ void *b_gm, __gm__ void *b_scale_gm,
                                    __gm__ void *out_gm, uint32_t m_total,
                                    uint32_t core_count) {
  // The body reads BASE_M/BASE_N throughout, so binding them here makes the
  // tile shape a template parameter without touching an offset or Shape type.
  constexpr unsigned BASE_M = BM;
  constexpr unsigned BASE_N = BN;
#if defined(__DAV_CUBE__)
  using Fp4 = float4_e2m1x2_t;
  using E8m0 = float8_e8m0_t;
  constexpr unsigned SK = K / SCALE_FACTOR;

  // DYNAMIC shape, whole-matrix stride. Pairing a static tile shape with a
  // full-matrix BaseShape2D reads A's first 8 rows as zero. It is also why
  // the scale assert accepts these: staticShape[4] == 2 OR -1
  // (a5/TLoad.hpp:85).
  using DynShape = pto::Shape<1, 1, 1, -1, -1>;
  using GmA = GlobalTensor<Fp4, DynShape,
                           BaseShape2D<Fp4, M_MAX, K, Layout::ND>, Layout::ND>;
  using GmB = GlobalTensor<Fp4, DynShape, BaseShape2D<Fp4, K, N, Layout::DN>,
                           Layout::DN>;
  // The scale tensors' SHAPE must come from TileShape2D, not a hand-written
  // pto::Shape: TLoad asserts staticShape[4] == 2 for every MX_*_ND/DN layout
  // (a5/TLoad.hpp:85), because scales are grouped in pairs along K. That
  // pairing is why K must be a multiple of 64 and not merely of 32.
  using GmAS = GlobalTensor<E8m0, DynShape,
                            BaseShape2D<E8m0, M_MAX, SK, Layout::MX_A_ND>,
                            Layout::MX_A_ND>;
  using GmBS =
      GlobalTensor<E8m0, DynShape, BaseShape2D<E8m0, SK, N, Layout::MX_B_DN>,
                   Layout::MX_B_DN>;
  using GmOut =
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, BASE_M, BASE_N>,
                   BaseShape2D<bfloat16_t, M_MAX, N, Layout::ND>, Layout::ND>;

  // L1. A is ColMajor with a RowMajor fractal, B RowMajor with a ColMajor
  // one; CheckMadMxValid asserts that asymmetry on the L0 tiles downstream.
  // The 512-byte fractal has to be explicit: omitting it reads A's first 8
  // rows as zero.
  //
  // L1 holds a K_L1-wide slab; L0 below holds BASE_K, and the extract picks
  // sub-tile j out of the slab at column offset j*BASE_K.
  using MatA = Tile<TileType::Mat, Fp4, BASE_M, K_L1, BLayout::ColMajor, BASE_M,
                    K_L1, SLayout::RowMajor, 512>;
  using MatB = Tile<TileType::Mat, Fp4, K_L1, BASE_N, BLayout::RowMajor, K_L1,
                    BASE_N, SLayout::ColMajor, 512>;
  // The scales are held for the WHOLE K of an output tile, not per slab: one
  // pair of loads covers every slab, so the burst is longer and the
  // descriptor count per output tile falls from 4*k_slabs to 2*k_slabs + 2.
  using MatAS = Tile<TileType::Mat, E8m0, BASE_M, SK, BLayout::RowMajor, BASE_M,
                     SK, SLayout::RowMajor, 32>;
  using MatBS = Tile<TileType::Mat, E8m0, SK, BASE_N, BLayout::ColMajor, SK,
                     BASE_N, SLayout::ColMajor, 32>;

  // L0, using the Compact variants the tuned reference uses.
  using Left = TileLeft<Fp4, BASE_M, BASE_K, BASE_M, BASE_K>;
  using Right = TileRight<Fp4, BASE_K, BASE_N, BASE_K, BASE_N>;
  using LeftS = TileLeftScale<E8m0, BASE_M, BASE_SK, BASE_M, BASE_SK>;
  using RightS = TileRightScale<E8m0, BASE_SK, BASE_N, BASE_SK, BASE_N>;
  using Acc = TileAcc<float, BASE_M, BASE_N, BASE_M, BASE_N>;

  // L1 is double buffered so the K loop can overlap its GM->L1 load with its
  // extract and multiply. One set is 96 KB at the default slab and tile.
  MatA a_l1, a_l1b;
  MatB b_l1, b_l1b;
  // One scale pair, not one per set: it is loaded once per output tile and
  // read by every extract, so it does not ping-pong with the data slabs.
  MatAS as_l1;
  MatBS bs_l1;
  // L0 is double buffered too. One fp4 operand tile is 32 KB and L0A and L0B
  // are 64 KB each, so two sets fit exactly, which is what holds BASE_K at
  // 256.
  Left a_l0, a_l0b;
  Right b_l0, b_l0b;
  LeftS as_l0, as_l0b;
  RightS bs_l0, bs_l0b;
  Acc c_l0;

  constexpr uint32_t a_l1_bytes = (BASE_M * K_L1) >> SHIFT_FP4;
  constexpr uint32_t b_l1_bytes = (K_L1 * BASE_N) >> SHIFT_FP4;
  // Data slabs first, L1_SETS of them, then the single whole-K scale pair.
  constexpr uint32_t l1_set_bytes = a_l1_bytes + b_l1_bytes;
  constexpr uint32_t scale_bytes = BASE_M * SK + SK * BASE_N;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, a_l1_bytes);
  TASSIGN(a_l1b, l1_set_bytes);
  TASSIGN(b_l1b, l1_set_bytes + a_l1_bytes);
  TASSIGN(as_l1, L1_SETS * l1_set_bytes);
  TASSIGN(bs_l1, L1_SETS * l1_set_bytes + BASE_M * SK);
  // L1 is 512 KB on this part. Holding the scales for the whole K costs 16*K
  // bytes, so past K=16384 they no longer fit beside two data sets -- and
  // this refuses to build rather than faulting on device with 507015.
  static_assert(L1_SETS * l1_set_bytes + scale_bytes <= 512u * 1024u,
                "the data sets plus the whole-K scale pair must fit the 512 KB "
                "L1; reduce L1_SETS or K");
  constexpr uint32_t a_l0_bytes = (BASE_M * BASE_K) >> SHIFT_FP4;
  constexpr uint32_t b_l0_bytes = (BASE_K * BASE_N) >> SHIFT_FP4;
  static_assert(2u * a_l0_bytes <= 64u * 1024u, "two A tiles must fit L0A");
  static_assert(2u * b_l0_bytes <= 64u * 1024u, "two B tiles must fit L0B");
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(a_l0b, a_l0_bytes);
  TASSIGN(b_l0b, b_l0_bytes);
  TASSIGN(c_l0, 0x0);
  TASSIGN(as_l0, GetScaleAddr(a_l0.data()));
  TASSIGN(bs_l0, GetScaleAddr(b_l0.data()));
  TASSIGN(as_l0b, GetScaleAddr(a_l0b.data()));
  TASSIGN(bs_l0b, GetScaleAddr(b_l0b.data()));

  const uint32_t m_tiles = m_total / BASE_M;
  constexpr uint32_t n_tiles = N / BASE_N;
#ifdef MXMM_SWIZZLE
  constexpr uint32_t SWIZZLE_GROUP = (uint32_t)(MXMM_SWIZZLE);
#else
  constexpr uint32_t SWIZZLE_GROUP =
      n_tiles >= SWIZZLE_WIDE_TILES ? SWIZZLE_WIDE : SWIZZLE_NARROW;
#endif
  static_assert(SWIZZLE_GROUP >= 1u, "a group spans at least one tile row");
  constexpr uint32_t k_tiles = K / BASE_K;
  constexpr uint32_t k_slabs = K / K_L1;
  static_assert(K % K_L1 == 0u, "K must be whole L1 slabs");
  const uint32_t out_tiles = m_tiles * n_tiles;
  const uint32_t per_group = SWIZZLE_GROUP * n_tiles;

  // Pipeline flags. Every pipe is in order, so one counter per direction is
  // enough: a wait releases the oldest outstanding token, which is the tile
  // whose turn it is.
  //
  //   MTE2 -> MTE1  ID0  a data slab has landed in L1
  //                 ID1  the whole-K scale pair has landed
  //   MTE1 -> MTE2  ID2  an L1 data set is free to refill
  //                 ID3  the scale buffer is free to refill
  //   MTE1 -> M     ID4  an L0 extract is done
  //   M    -> MTE1  ID5  the cube has released an L0 set
  //   M    -> FIX   ID0  the accumulator is complete
  //   FIX  -> M     ID6  the store has read the accumulator
  //
  // MTE2 runs ahead of MTE1 by design, so the scale buffer needs the same
  // release handshake the data sets get: without ID3 a tile's scale load can
  // land on top of scales the previous tile is still extracting.
  bool scales_held = false;
  for (uint32_t t = get_block_idx(); t < out_tiles; t += core_count) {
    const uint32_t in_group = t % per_group;
    const uint32_t first_mt = (t / per_group) * SWIZZLE_GROUP;
    const uint32_t group_mts =
        m_tiles - first_mt < SWIZZLE_GROUP ? m_tiles - first_mt : SWIZZLE_GROUP;
    const uint32_t mt = first_mt + in_group % group_mts;
    const uint32_t nt = in_group / group_mts;

#define MXMM_TILE_GM(kk)                                                   \
  GmA a_g((__gm__ Fp4 *)a_gm +                                             \
              (((uint64_t)mt * BASE_M * K + (uint64_t)(kk)) >> SHIFT_FP4), \
          DynShape(BASE_M, K_L1));                                         \
  GmB b_g((__gm__ Fp4 *)b_gm +                                             \
              (((uint64_t)nt * BASE_N * K + (uint64_t)(kk)) >> SHIFT_FP4), \
          DynShape(K_L1, BASE_N));                                         \
  (void)0

// The scale tensors are indexed by the output tile alone, not by the slab:
// A's scales by its row panel and B's by its column panel, spanning all of K.
#define MXMM_FILL_SCALES()                                             \
  do {                                                                 \
    GmAS as_g((__gm__ E8m0 *)a_scale_gm +                              \
                  (((uint64_t)mt * BASE_M * K) >> SHIFT_SCALE_FACTOR), \
              DynShape(BASE_M, SK));                                   \
    GmBS bs_g((__gm__ E8m0 *)b_scale_gm +                              \
                  (((uint64_t)nt * BASE_N * K) >> SHIFT_SCALE_FACTOR), \
              DynShape(SK, BASE_N));                                   \
    TLOAD<MatAS, GmAS>(as_l1, as_g);                                   \
    TLOAD<MatBS, GmBS>(bs_l1, bs_g);                                   \
  } while (0)

// sub is the tile's index within the slab and tile its index within the whole
// K. A holds K along its columns and B along its rows, so the offset goes in
// a different argument for each. The data offset is per slab; the scale
// offset is per output tile, because one scale buffer spans all of K.
#define MXMM_DRAIN_TO_L0(a1, b1, a0, b0, as0, bs0, sub, tile) \
  do {                                                        \
    const uint16_t koff_ = (uint16_t)((sub) * BASE_K);        \
    const uint16_t soff_ = (uint16_t)((tile) * BASE_SK);      \
    TEXTRACT(a0, a1, 0, koff_);                               \
    TEXTRACT(b0, b1, koff_, 0);                               \
    TEXTRACT(as0, as_l1, 0, soff_);                           \
    TEXTRACT(bs0, bs_l1, soff_, 0);                           \
  } while (0)

// One GM->L1 copy per slab into the set its index selects.
#define MXMM_LOAD(sl, sel)                     \
  do {                                         \
    MXMM_TILE_GM((uint64_t)(sl) * K_L1);       \
    if ((sel) == 0u) {                         \
      TLOAD(a_l1, a_g);                        \
      TLOAD(b_l1, b_g);                        \
    } else {                                   \
      TLOAD(a_l1b, a_g);                       \
      TLOAD(b_l1b, b_g);                       \
    }                                          \
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0); \
  } while (0)

// e is the global K-tile index. A slab's load is waited for once, before its
// first sub-tile, and the set released once, after its last -- so ID0 and ID2
// count slabs while ID4 and ID5 count cube tiles.
#define MXMM_EXTRACT(e, a0, b0, as0, bs0)                         \
  do {                                                            \
    const uint32_t e_ = (e);                                      \
    const uint32_t slab_ = e_ / K_SUB;                            \
    const uint32_t sub_ = e_ % K_SUB;                             \
    if (sub_ == 0u) {                                             \
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);                 \
    }                                                             \
    if (slab_ % L1_SETS == 0u) {                                  \
      MXMM_DRAIN_TO_L0(a_l1, b_l1, a0, b0, as0, bs0, sub_, e_);   \
    } else {                                                      \
      MXMM_DRAIN_TO_L0(a_l1b, b_l1b, a0, b0, as0, bs0, sub_, e_); \
    }                                                             \
    if (sub_ + 1u == K_SUB) {                                     \
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);                  \
    }                                                             \
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);                       \
  } while (0)

    // One pair of scale loads for the whole output tile, before the data
    // prologue so it is the first thing MTE2 has queued.
    if (scales_held) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    }
    MXMM_FILL_SCALES();
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    scales_held = true;

    // Prologue: LOOKAHEAD slabs are in flight before the first multiply, so a
    // tile's extract waits on a load issued LOOKAHEAD passes earlier instead
    // of one it started itself.
    for (uint32_t j = 0u; j < LOOKAHEAD && j < k_slabs; ++j) {
      MXMM_LOAD(j, j % L1_SETS);
    }
    // MTE1 blocks here until the scales are in L1; the prologue's data loads
    // are already issued on MTE2, so this costs them nothing.
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
      const bool odd = (kt & 1u) != 0u;
      // One GM->L1 copy per slab, issued when the slab's first cube tile
      // comes up rather than once per cube tile.
      if (kt % K_SUB == 0u) {
        const uint32_t slab_ahead = kt / K_SUB + LOOKAHEAD;
        if (slab_ahead < k_slabs) {
          if (slab_ahead >= L1_SETS) {  // that set must be fully extracted
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          }
          MXMM_LOAD(slab_ahead, slab_ahead % L1_SETS);
        }
      }
      // Tile kt was extracted on the previous pass, so this pass extracts
      // kt+1 into the OTHER L0 set and the cube's multiply of kt runs beside
      // it. wait_flag stalls its target pipe, not the scalar unit, so the
      // multiply still issues to M while MTE1 sits in the extract.
      if (kt == 0u) {
        MXMM_EXTRACT(0u, a_l0, b_l0, as_l0, bs_l0);
      }
      if (kt + 1u < k_tiles) {
        if (kt >= 1u) {
          // tile kt-1 has released that L0 set
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        }
        if (odd) {  // kt+1 has the opposite parity, so the other L0 set
          MXMM_EXTRACT(kt + 1u, a_l0, b_l0, as_l0, bs_l0);
        } else {
          MXMM_EXTRACT(kt + 1u, a_l0b, b_l0b, as_l0b, bs_l0b);
        }
      }
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      if (odd) {  // kt == 0 is even, so an odd tile always accumulates
        TMATMUL_MX(c_l0, c_l0, a_l0b, as_l0b, b_l0b, bs_l0b);
      } else if (kt == 0u) {
        TMATMUL_MX(c_l0, a_l0, as_l0, b_l0, bs_l0);
      } else {
        TMATMUL_MX(c_l0, c_l0, a_l0, as_l0, b_l0, bs_l0);
      }
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    }
    // ID2 carries min(k_slabs, L1_SETS) tokens out of the loop: every slab's
    // last extract posts one and only the loads far enough ahead consume one.
    // Draining keeps a count from carrying into the next output tile, where a
    // later wait would return early.
    for (uint32_t d = 0u; d < L1_SETS && d < k_slabs; ++d) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    }
    // Every extract above read the scale buffer, and MTE1 is in order, so one
    // post here releases it for the next output tile.
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    // Every multiply posts on ID5 and only the extracts of tile 2 and later
    // consume one, so min(k_tiles, 2) tokens are outstanding here.
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    if constexpr (k_tiles > 1u) {
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    }
#undef MXMM_TILE_GM
#undef MXMM_FILL_SCALES
#undef MXMM_DRAIN_TO_L0
#undef MXMM_LOAD
#undef MXMM_EXTRACT
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    GmOut out_g((__gm__ bfloat16_t *)out_gm +
                ((uint64_t)mt * BASE_M * N + (uint64_t)nt * BASE_N));
    TSTORE(out_g, c_l0);
    // The next output tile's first multiply is non-accumulating and
    // overwrites c_l0, so it waits for the store; its loads and extracts do
    // not.
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  }

  // One ID3 post is outstanding: every tile posts and every tile but the
  // first consumed one. Drain it so a later launch's first wait cannot return
  // early.
  if (scales_held) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  }
#else
  (void)a_gm;
  (void)a_scale_gm;
  (void)b_gm;
  (void)b_scale_gm;
  (void)out_gm;
  (void)core_count;
#endif
}

// The shape this translation unit is built for. K must be a multiple of K_L1,
// so it cannot be under one L1 slab; 512 is the narrowest that builds at the
// default slab width.
#ifndef MXMM_TEST_K
#define MXMM_TEST_K 512
#endif
#ifndef MXMM_TEST_N
#define MXMM_TEST_N 512
#endif
constexpr unsigned TEST_M_MAX = 65536;  // an upper bound, not the shape
constexpr unsigned M_TILE_ROWS = TINY_M;
constexpr unsigned TEST_K = MXMM_TEST_K, TEST_N = MXMM_TEST_N;

// Which output tile suits this M. The big one moves less data but halves the
// block count, so it only pays where the shape still fills the cores; the
// threshold is the launcher's own block cap.
constexpr uint32_t TARGET_BLOCKS = 64u;
inline uint32_t mxfp4_matmul_pick_m_tile(uint32_t m) {
  if (m % BIG_M == 0u && (m / BIG_M) * (TEST_N / BIG_N) >= TARGET_BLOCKS) {
    return BIG_M;
  }
  if (m % SMALL_M == 0u) {
    return SMALL_M;
  }
  return TINY_M;
}

// What a caller with an awkward M should round up to. The tiny tile is only
// right up to its own height: each output tile streams a whole B column
// panel, so M=64 as four 16-row tiles reads B four times where one padded
// 128-row tile reads it once.
inline uint32_t mxfp4_matmul_pick_m_round(uint32_t m) {
  if (m == 0u) return TINY_M;
  if (m <= TINY_M) return TINY_M;
  return ((m + SMALL_M - 1u) / SMALL_M) * SMALL_M;
}

static inline void mxfp4_matmul_launch(uint32_t block_dim, void *stream,
                                       uint8_t *a_gm, uint8_t *a_scale_gm,
                                       uint8_t *b_gm, uint8_t *b_scale_gm,
                                       uint8_t *out_gm, uint32_t m, uint32_t k,
                                       uint32_t n) {
  // The host rejects what has no instantiation: a launcher that returns
  // quietly hands the caller an untouched output buffer back.
  if (block_dim == 0u) return;
  if (k != TEST_K || n != TEST_N) return;
  // M is runtime, but still has to be whole tiles and within the declared
  // maximum.
  if (m == 0u || m % M_TILE_ROWS != 0u || m > TEST_M_MAX) return;
  // NEVER guard this launch with a device-pass macro.
  const uint32_t tile = mxfp4_matmul_pick_m_tile(m);
  if (tile == BIG_M) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, BIG_M, BIG_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim);
  } else if (tile == SMALL_M) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, SMALL_M, SMALL_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim);
  } else {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, TINY_M, TINY_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim);
  }
}

extern "C" void call_mxfp4_matmul(uint32_t block_dim, void *stream,
                                  uint8_t *a_gm, uint8_t *a_scale_gm,
                                  uint8_t *b_gm, uint8_t *b_scale_gm,
                                  uint8_t *out_gm, uint32_t m, uint32_t k,
                                  uint32_t n) {
  mxfp4_matmul_launch(block_dim, stream, a_gm, a_scale_gm, b_gm, b_scale_gm,
                      out_gm, m, k, n);
}

extern "C" uint32_t mxfp4_matmul_m_tile() { return M_TILE_ROWS; }
// Callers should pass mxfp4_matmul_m_round(m) as the launch M, with an output
// buffer of that many rows, rather than picking a multiple themselves.
extern "C" uint32_t mxfp4_matmul_m_round(uint32_t m) {
  return mxfp4_matmul_pick_m_round(m);
}
extern "C" uint32_t mxfp4_matmul_m_tile_for(uint32_t m) {
  return mxfp4_matmul_pick_m_tile(m);
}
// the N extent that goes with it, so a caller can size its grid as
// (m / m_tile_for) * (N / n_tile_for) without knowing the rule
extern "C" uint32_t mxfp4_matmul_n_tile_for(uint32_t m) {
  const uint32_t tile = mxfp4_matmul_pick_m_tile(m);
  if (tile == BIG_M) return BIG_N;
  return tile == SMALL_M ? SMALL_N : TINY_N;
}
extern "C" uint32_t mxfp4_matmul_m_max() { return TEST_M_MAX; }
extern "C" uint32_t mxfp4_matmul_k() { return TEST_K; }
extern "C" uint32_t mxfp4_matmul_n() { return TEST_N; }
