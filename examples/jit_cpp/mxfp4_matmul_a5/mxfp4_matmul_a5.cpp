// MXFP4 matmul on A5: y = A @ B with BOTH operands MXFP4, on the cube's
// native microscaled path.
//
//   A: (M, K) E2M1 nibbles, two per byte, + one E8M0 scale per 32 along K
//   B: (K, N) the same, stored DN -- see LAYOUTS
//   y: (M, N) bf16, accumulated in fp32
//
// A5 has a microscaled matmul in hardware, TMATMUL_MX, whose semantics are
// MXFP4 block-32 exactly. From PTO's own CPU reference
// (pto/cpu/TMatmul.hpp:44-67):
//
//     acc += a(i, k) * b(k, j) * aScale(i, k / 32) * bScale(k / 32, j)
//
// So the scaling is in the datapath; this is not a dequantize-and-matmul. And
// aScale is (M, K/32), exactly what our activation kernel writes, so the two
// halves of the chain meet with no repacking.
//
// STRUCTURE follows the vendor's tuned reference at pto-isa/kernels/manual/a5/
// matmul_mxfp4_performance/mxmatmul_performance_kernel.cpp, because a first
// version written from the header signatures alone got FOUR separate things
// wrong and produced plausible-looking garbage (rel 1.5, not noise). Each is
// called out below so the reasoning survives.
//
// LAYOUTS -- asymmetric, and not guessable from the type names
//   A data   Layout::ND       plain row-major (M, K)
//   A scale  Layout::MX_A_ND  plain row-major (M, K/32)
//   B data   Layout::DN       NOT ND
//   B scale  Layout::MX_B_DN  NOT MX_B_ND
// B being DN costs nothing, since weights are prepared offline, but feeding
// row-major B silently computes something else.
//
// STRIDES come from a BaseShape2D of the WHOLE matrix, not of the tile.
// Deriving them from the tile width makes every row after the first read from
// the wrong place, which was most of what went wrong the first time.
//
// FP4 POINTER ARITHMETIC IS IN BYTES. sizeof(float4_e2m1x2_t) is 1 and it holds
// two elements, so a K-element step is (K >> 1) bytes and a scale step is
// (K >> 5). Advancing by element counts double-steps.
//
// SCALE TILES BIND BY ADDRESS. TMATMUL_MX takes them, but TMATMUL_MX_IMPL never
// forwards them to the mad_mx builtin (a5/TMatmul.hpp:259-271) -- they exist
// for the type check. The hardware reads them at GetScaleAddr(operand.data()),
// a
// >> 4 of the operand's L0 address (a5/utils.hpp:82-87): the MX scale store is
// a 1/16 shadow of L0A/L0B. TASSIGNing them anywhere else compiles and
// multiplies by whatever happens to be there.
//
// Alignment, all asserted by CheckMadMxValid: K a multiple of 64 ELEMENTS (from
// BASEK, and because the scale count K/32 must be even), N a multiple of 64 for
// fp4, M a multiple of 16, accumulator fp32.
//
// Arch: cube plus a store, so dav-c310. No MIX, no cross-core sync.
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t SCALE_FACTOR = 32;       // elements per E8M0 scale
constexpr uint32_t SHIFT_SCALE_FACTOR = 5;  // log2(SCALE_FACTOR)
constexpr uint32_t SHIFT_FP4 = 1;           // two nibbles per byte

// Output-tile shape. The L1 fill for the whole matmul is
// M*N*K*bytes_per_elem * (1/BASE_M + 1/BASE_N), so both dimensions cut
// re-fetch and L0C, which holds one output tile as fp32, is what bounds
// them. Overridable so the tradeoff can be swept.
#ifndef MXMM_BASE_M
#define MXMM_BASE_M 128
#endif
#ifndef MXMM_BASE_N
#define MXMM_BASE_N 256
#endif
// Two output-tile shapes, chosen at launch. The big one cuts L1 re-fetch by
// (1/BASE_M + 1/BASE_N) and measured 1.49x at M=4096 K=N=8192, but it also
// halves the block count, and at M=256 that starved the 32 cube cores and
// measured 0.65x. So the launcher picks per M rather than one being right.
constexpr unsigned SMALL_M = MXMM_BASE_M;  // 16-aligned
constexpr unsigned SMALL_N = MXMM_BASE_N;  // 64-aligned for fp4
// A K_L1=1024 slab needs 544 KB for two L1 sets at a (256,256) output tile,
// over the 512 KB L1, so testing it requires holding the tile at (128,256) --
// which costs 1.5x the fill per output element, (1/128 + 1/256) against
// (1/256 + 1/256). Defining this collapses the big tile onto the small one so
// only one shape is instantiated and the assert has a chance of passing.
#ifdef MXMM_NO_BIG_TILE
constexpr unsigned BIG_M = SMALL_M;
#else
constexpr unsigned BIG_M = 2u * SMALL_M;
#endif
constexpr unsigned BIG_N = SMALL_N;

// A third, 16-row output tile for small M. Below M=128 the launcher had to pad
// up to the 128-row tile, which does 128 rows of arithmetic for a caller that
// wants one -- and at K=N=2048 to 4096 that padding cost more than the format
// saved, leaving those shapes at 0.91-0.98x of bf16. Sixteen is the alignment
// floor TMATMUL_MX imposes, so it is the finest tile available.
//
// The extra fill this costs is real, (1/16 + 1/256) against (1/128 + 1/256)
// per output element, but at small M the traffic is dominated by streaming B
// regardless, and B's share does not change with the M tile.
#ifndef MXMM_TINY_M
#define MXMM_TINY_M 16
#endif
constexpr unsigned TINY_M = MXMM_TINY_M;
constexpr unsigned TINY_N = SMALL_N;
constexpr unsigned BASE_K = 256;  // multiple of 64

// How many L1 sets, and so how far ahead the GM->L1 load runs: a tile is
// extracted LOOKAHEAD passes after it is loaded.
//
// Two, not three. Three measured 1.00x at every shape from K=N=4096 to 8192,
// because the loads are already at the memory system's limit rather than
// waiting on latency: with the extract and the multiply both switched off the
// kernel still takes 687 of its 687 us at K=N=8192, moving 1.14 GB at
// 1660 GB/s against a 1600 GB/s peak. A deeper prefetch hides latency, and
// there is none here to hide. The third set stays wired up behind the knob
// because it costs 68 KB of L1 to enable and the balance is proven.
// The L1 slab's K extent, independent of the cube's BASE_K. A GM->L1 copy of
// a (BASE_M, K_L1) fp4 tile moves K_L1/2 contiguous bytes per row, so at the
// default 256 that is 128 B -- a quarter of a 512-byte line. Widening it
// lengthens every DMA burst without changing the byte count, and the cube
// still consumes BASE_K at a time from a K_SUB-step inner loop. The vendor's
// kernel_qbmm_mx does exactly this, sizing its kL1 against half of L1.
// 512, not 256. Measured 1.35x at K=N=4096 and 1.57x at 8192, and the
// loads-only arm moves the same bytes 1.60x faster, so the limit was
// per-descriptor issue cost rather than bandwidth. 1024 would give full
// 512-byte bursts but needs 544 KB for two sets against a 512 KB L1.
#ifndef MXMM_K_L1
#define MXMM_K_L1 512
#endif
constexpr unsigned K_L1 = MXMM_K_L1;
constexpr unsigned K_SUB = K_L1 / BASE_K;
static_assert(K_L1 % BASE_K == 0, "the slab must be whole cube K tiles");
static_assert(K_SUB >= 1u, "at least one sub-tile per slab");

#ifndef MXMM_L1_SETS
#define MXMM_L1_SETS 2
#endif
constexpr uint32_t L1_SETS = (uint32_t)(MXMM_L1_SETS);
constexpr uint32_t LOOKAHEAD = L1_SETS - 1u;
// One set is allowed. With LOOKAHEAD 0 the slab is loaded at the top of its
// own iteration and extracted immediately after, so the load no longer
// overlaps the extract and multiply -- which costs little when the load is
// already 97% of the runtime, and buys a 1024-wide slab at the wide output
// tile: 272 KB for one set against 544 for two.
static_assert(L1_SETS >= 1u, "at least one L1 set");
static_assert(L1_SETS <= 3u, "only two and three sets are wired up");

// Output tiles are walked SWIZZLE_GROUP rows down before stepping across, so
// the blocks resident at one time form a square-ish patch of the output rather
// than a full-width strip. 1 restores the plain row-major walk.
//
// The right group depends on how many N tiles there are, so the default is
// chosen per instantiation from n_tiles rather than fixed: a group of 2 is
// worth 7-11% once n_tiles reaches 48, and is neutral to slightly worse below
// that, where 8 holds. 16 is worse at every width measured. -DMXMM_SWIZZLE
// overrides it. The numbers behind the threshold are in README.md.
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

// K and N are template parameters so the GM strides stay static -- they are
// per-layer constants. M is RUNTIME, which is the whole point: block_dim is
// m_tiles * n_tiles, so a compile-time M of 128 capped parallelism at N/256,
// i.e. four cores at N=1024. M is the token count of the batch, and letting it
// grow is what fills the machine.
//
// M_MAX only sizes the BaseShape2D extents. Every GM offset here is computed
// explicitly from BASE_M/BASE_N/BASE_K, so addressing does not depend on it; it
// has to be an upper bound on the real M, which the host checks.
template <unsigned M_MAX, unsigned K, unsigned N, unsigned BM, unsigned BN>
// groups: how many independent problems of this same shape sit back to back
// in memory. One launch covers all of them, because the host can only issue a
// kernel every 5 us or so, and at K=N=512 that is 94% of a matmul's cost.
// groups == 1 is the ordinary path and compiles to the same work list.
__global__ AICORE void mxfp4_matmul(__gm__ void *a_gm, __gm__ void *a_scale_gm,
                                    __gm__ void *b_gm, __gm__ void *b_scale_gm,
                                    __gm__ void *out_gm, uint32_t m_total,
                                    uint32_t core_count, uint32_t groups) {
  // The body reads BASE_M/BASE_N throughout; binding them here means the
  // tile shape becomes a template parameter without touching a single
  // offset, Shape type or TASSIGN below.
  constexpr unsigned BASE_M = BM;
  constexpr unsigned BASE_N = BN;
#if defined(__DAV_CUBE__)
  using Fp4 = float4_e2m1x2_t;
  using E8m0 = float8_e8m0_t;
  constexpr unsigned SK = K / SCALE_FACTOR;

  // DYNAMIC shape, whole-matrix stride -- exactly what both reference sources
  // do. Pairing a STATIC tile shape with a full-matrix BaseShape2D left A's
  // first 8 rows reading as zero; the dynamic form passes the real extents at
  // construction instead of encoding them in the type. It is also why the scale
  // assert accepts these: staticShape[4] == 2 OR -1 (a5/TLoad.hpp:85).
  using DynShape = pto::Shape<1, 1, 1, -1, -1>;
  using GmA = GlobalTensor<Fp4, DynShape,
                           BaseShape2D<Fp4, M_MAX, K, Layout::ND>, Layout::ND>;
  using GmB = GlobalTensor<Fp4, DynShape, BaseShape2D<Fp4, K, N, Layout::DN>,
                           Layout::DN>;
  // The scale tensors' SHAPE must come from TileShape2D, not a hand-written
  // pto::Shape: TLoad asserts staticShape[4] == 2 for every MX_*_ND/DN layout
  // (a5/TLoad.hpp:85), because scales are grouped in pairs along K. That
  // pairing is also why K/32 has to be even, i.e. why K must be a multiple of
  // 64 and not merely of 32.
  using GmAS = GlobalTensor<E8m0, DynShape,
                            BaseShape2D<E8m0, M_MAX, SK, Layout::MX_A_ND>,
                            Layout::MX_A_ND>;
  using GmBS =
      GlobalTensor<E8m0, DynShape, BaseShape2D<E8m0, SK, N, Layout::MX_B_DN>,
                   Layout::MX_B_DN>;
  using GmOut =
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, BASE_M, BASE_N>,
                   BaseShape2D<bfloat16_t, M_MAX, N, Layout::ND>, Layout::ND>;

  // L1. A is ColMajor with a RowMajor fractal, B RowMajor with a ColMajor one;
  // the asymmetry is what CheckMadMxValid asserts on the L0 tiles downstream.
  // The explicit 512-byte fractal matters. Omitting it, as the vendor's tuned
  // reference does, left A's FIRST 8 ROWS reading as zero -- output rows 0..7
  // came back written-but-zero at every probe shift while rows 8+ were exactly
  // right, so the K pairing was already identity and only A's row placement was
  // wrong. The conformance test passes 512 explicitly; the vendor relies on a
  // default that evidently is not the same here.
  // L1 holds a K_L1-wide slab; L0 below still holds BASE_K, and the extract
  // picks sub-tile j out of the slab at column offset j*BASE_K.
  constexpr unsigned SK_L1 = K_L1 / SCALE_FACTOR;
  using MatA = Tile<TileType::Mat, Fp4, BASE_M, K_L1, BLayout::ColMajor, BASE_M,
                    K_L1, SLayout::RowMajor, 512>;
  using MatB = Tile<TileType::Mat, Fp4, K_L1, BASE_N, BLayout::RowMajor, K_L1,
                    BASE_N, SLayout::ColMajor, 512>;
  // The scales are held for the WHOLE K of an output tile, not per slab.
  // Per slab they were 256 rows of 16 bytes each -- 3% of the bytes moved and
  // 50% of the DMA descriptors issued, on a kernel whose loads are bound by
  // descriptor issue rather than bandwidth. Held whole, they are one load of
  // 256 rows of SK_ALL bytes, so the burst becomes a full cache line at
  // K >= 16384 and the descriptor count per output tile falls from
  // 4 * k_slabs units to 2 * k_slabs + 2. The vendor does the same thing at a
  // cadence of 4 slabs (mxmatmul_performance_kernel.cpp:20, mxScalePara);
  // whole-K is the same idea taken as far as L1 allows.
  constexpr uint32_t SK_ALL = K / SCALE_FACTOR;
  using MatAS = Tile<TileType::Mat, E8m0, BASE_M, SK_ALL, BLayout::RowMajor,
                     BASE_M, SK_ALL, SLayout::RowMajor, 32>;
  using MatBS = Tile<TileType::Mat, E8m0, SK_ALL, BASE_N, BLayout::ColMajor,
                     SK_ALL, BASE_N, SLayout::ColMajor, 32>;

  // L0, using the Compact variants the tuned reference uses.
  using Left = TileLeft<Fp4, BASE_M, BASE_K, BASE_M, BASE_K>;
  using Right = TileRight<Fp4, BASE_K, BASE_N, BASE_K, BASE_N>;
  using LeftS = TileLeftScale<E8m0, BASE_M, BASE_SK, BASE_M, BASE_SK>;
  using RightS = TileRightScale<E8m0, BASE_SK, BASE_N, BASE_SK, BASE_N>;
  using Acc = TileAcc<float, BASE_M, BASE_N, BASE_M, BASE_N>;

  // L1 is double buffered so the K loop can overlap its GM->L1 load with
  // its extract and multiply. One set is 51 KB.
  MatA a_l1, a_l1b, a_l1c;
  MatB b_l1, b_l1b, b_l1c;
  // one scale pair, not one per set: it is loaded once per output tile and
  // read by every extract, so it does not ping-pong with the data slabs
  MatAS as_l1;
  MatBS bs_l1;
  // L0 is double buffered too. One fp4 operand tile is 32 KB and L0A and L0B
  // are 64 KB each, so two sets fit exactly, which is why BASE_K stays 256:
  // (256,512,256) measures 1692 TF/s on the cube against 1653 here, but needs
  // all 64 KB per side and leaves no room for the second set.
  Left a_l0, a_l0b;
  Right b_l0, b_l0b;
  LeftS as_l0, as_l0b;
  RightS bs_l0, bs_l0b;
  Acc c_l0;

  constexpr uint32_t a_l1_bytes = (BASE_M * K_L1) >> SHIFT_FP4;
  constexpr uint32_t b_l1_bytes = (K_L1 * BASE_N) >> SHIFT_FP4;
  // Data slabs first, L1_SETS of them, then the single whole-K scale pair.
  constexpr uint32_t l1_set_bytes = a_l1_bytes + b_l1_bytes;
  constexpr uint32_t scale_bytes = BASE_M * SK_ALL + SK_ALL * BASE_N;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, a_l1_bytes);
  TASSIGN(a_l1b, l1_set_bytes);
  TASSIGN(b_l1b, l1_set_bytes + a_l1_bytes);
  // The third set is assigned unconditionally and simply goes untouched at
  // L1_SETS == 2; an unused TASSIGN costs nothing at runtime.
  TASSIGN(a_l1c, 2u * l1_set_bytes);
  TASSIGN(b_l1c, 2u * l1_set_bytes + a_l1_bytes);
  TASSIGN(as_l1, L1_SETS * l1_set_bytes);
  TASSIGN(bs_l1, L1_SETS * l1_set_bytes + BASE_M * SK_ALL);
  // L1 is 512 KB on this part, established by bisection: two sets of a
  // K_L1=512 slab are 272 KB and run exactly, two of a K_L1=1024 slab are
  // 544 KB and fault at launch with 507015.
  //
  // Holding the scales for the whole K costs 16*K bytes, which is 256 KB at
  // K=16384 and exactly fills L1 beside two 128 KB data sets. Past that width,
  // or with a third data set, it does not fit -- and this refuses to build
  // rather than faulting on device.
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
  // Strides between consecutive problems, in bytes, from the shape alone: fp4
  // operands are half a byte per element and E8M0 scales one per 32.
  const uint64_t a_step = ((uint64_t)m_total * K) >> SHIFT_FP4;
  const uint64_t as_step = ((uint64_t)m_total * K) >> SHIFT_SCALE_FACTOR;
  constexpr uint64_t b_step = ((uint64_t)K * N) >> SHIFT_FP4;
  constexpr uint64_t bs_step = ((uint64_t)K * N) >> SHIFT_SCALE_FACTOR;
  const uint64_t o_step = (uint64_t)m_total * N * sizeof(bfloat16_t);
  const uint32_t work_items = out_tiles * groups;
  // ID1 says the whole-K scales for this output tile have landed and ID3 that
  // MTE1 has finished reading the previous tile's, so MTE2 may overwrite them.
  // MTE2 runs ahead of MTE1 by design, which is the whole point of the L1
  // pipeline, so the scale buffer needs the same release handshake the data
  // sets get on ID2 -- without it a tile's scale load can land on top of
  // scales the previous tile is still extracting.
  bool scales_held = false;
  for (uint32_t w = get_block_idx(); w < work_items; w += core_count) {
    const uint32_t grp = w / out_tiles;
    const uint32_t t = w % out_tiles;
    __gm__ uint8_t *a_gm_g = (__gm__ uint8_t *)a_gm + grp * a_step;
    __gm__ uint8_t *as_gm_g = (__gm__ uint8_t *)a_scale_gm + grp * as_step;
    __gm__ uint8_t *b_gm_g = (__gm__ uint8_t *)b_gm + grp * b_step;
    __gm__ uint8_t *bs_gm_g = (__gm__ uint8_t *)b_scale_gm + grp * bs_step;
    __gm__ uint8_t *out_gm_g = (__gm__ uint8_t *)out_gm + grp * o_step;
    const uint32_t in_group = t % per_group;
    const uint32_t first_mt = (t / per_group) * SWIZZLE_GROUP;
    const uint32_t group_mts =
        m_tiles - first_mt < SWIZZLE_GROUP ? m_tiles - first_mt : SWIZZLE_GROUP;
    const uint32_t mt = first_mt + in_group % group_mts;
    const uint32_t nt = in_group / group_mts;
    // The K loop is software pipelined across the two L1 sets: tile kt+1's
    // GM->L1 load is issued before tile kt is extracted and multiplied, so
    // the DMA and the rest overlap instead of taking each other's time.
    // Measured single-buffered at K=N=8192, the loads cost 872 us and the
    // extract-plus-multiply 599, and the total was their sum, 1472.
    //
    // One flag pair per buffer, so a wait cannot consume the other's token:
    //   EVENT_ID0/1  buffer 0/1's load has landed   (MTE2 -> MTE1)
    //   EVENT_ID2/3  buffer 0/1 is free to refill   (MTE1 -> MTE2)
    //   EVENT_ID4    L0 extract done                (MTE1 -> M)
    //   EVENT_ID5    the cube is done reading L0    (M -> MTE1)
// Timing-only address collapses. MXMM_B_SAME_PANEL points every block at B
// column panel 0, so B's footprint per launch falls from N/BASE_N panels to
// one; MXMM_A_SAME_PANEL does the same for A's row panels. The access pattern,
// the byte count issued and the instruction stream are all unchanged -- only
// the range of addresses touched shrinks. If the load time falls, the memory
// system is serving repeats from L2 and our traffic is being evicted; if it
// does not, these DMA reads do not benefit from L2 and no traversal or cache
// hint can help.
#ifdef MXMM_A_SAME_PANEL
#define MXMM_MT_LOAD 0u
#else
#define MXMM_MT_LOAD mt
#endif
#ifdef MXMM_B_SAME_PANEL
#define MXMM_NT_LOAD 0u
#else
#define MXMM_NT_LOAD nt
#endif

#define MXMM_TILE_GM(kk)                                                 \
  GmA a_g((__gm__ Fp4 *)a_gm_g +                                         \
              (((uint64_t)MXMM_MT_LOAD * BASE_M * K + (uint64_t)(kk)) >> \
               SHIFT_FP4),                                               \
          DynShape(BASE_M, K_L1));                                       \
  GmB b_g((__gm__ Fp4 *)b_gm_g +                                         \
              (((uint64_t)MXMM_NT_LOAD * BASE_N * K + (uint64_t)(kk)) >> \
               SHIFT_FP4),                                               \
          DynShape(K_L1, BASE_N));                                       \
  (void)0

// The scale tensors are indexed by the output tile alone, not by the slab:
// A's scales by its row panel and B's by its column panel, spanning all of K.
#define MXMM_SCALE_GM()                                                        \
  GmAS as_g((__gm__ E8m0 *)as_gm_g +                                           \
                (((uint64_t)MXMM_MT_LOAD * BASE_M * K) >> SHIFT_SCALE_FACTOR), \
            DynShape(BASE_M, SK_ALL));                                         \
  GmBS bs_g((__gm__ E8m0 *)bs_gm_g +                                           \
                (((uint64_t)MXMM_NT_LOAD * BASE_N * K) >> SHIFT_SCALE_FACTOR), \
            DynShape(SK_ALL, BASE_N))

// ATTRIBUTION SWITCHES, all three timing-only and all producing wrong output:
//   -DMXMM_NO_LOAD     drops GM->L1, so the difference is the load
//   -DMXMM_NO_EXTRACT  drops L1->L0, so the difference is the extract
//   -DMXMM_NO_MATMUL   drops all but the first multiply, so the difference is
//                      the cube issue itself
// The bottleneck moved once the loads were overlapped, so which stage binds now
// has to be measured rather than assumed.
#ifdef MXMM_NO_LOAD
#define MXMM_FILL(a1, b1) ((void)0)
#define MXMM_FILL_SCALES() ((void)0)
#else
#define MXMM_FILL(a1, b1) \
  do {                    \
    TLOAD(a1, a_g);       \
    TLOAD(b1, b_g);       \
  } while (0)
// Once per output tile, covering every slab's scales in one pair of loads.
#define MXMM_FILL_SCALES()           \
  do {                               \
    MXMM_SCALE_GM();                 \
    TLOAD<MatAS, GmAS>(as_l1, as_g); \
    TLOAD<MatBS, GmBS>(bs_l1, bs_g); \
  } while (0)
#endif

#ifdef MXMM_NO_EXTRACT
#define MXMM_DRAIN_TO_L0(a1, b1, a0, b0, as0, bs0, sub, tile) ((void)0)
#else
// sub is the tile's index within the slab and tile its index within the whole
// K. A holds K along its columns and B along its rows, so the offset goes in a
// different argument for each. The data offset is per slab; the scale offset
// is per output tile, because one scale buffer spans all of K.
#define MXMM_DRAIN_TO_L0(a1, b1, a0, b0, as0, bs0, sub, tile) \
  do {                                                        \
    const uint16_t koff_ = (uint16_t)((sub) * BASE_K);        \
    const uint16_t soff_ = (uint16_t)((tile) * BASE_SK);      \
    TEXTRACT(a0, a1, 0, koff_);                               \
    TEXTRACT(b0, b1, koff_, 0);                               \
    TEXTRACT(as0, as_l1, 0, soff_);                           \
    TEXTRACT(bs0, bs_l1, soff_, 0);                           \
  } while (0)
#endif

// ID4 counts completed extracts and ID5 completed multiplies. Both pipes run
// in order, so one counter each is enough: a wait releases the oldest
// outstanding token, which is exactly the tile whose turn it is.
#ifdef MXMM_NO_SYNC
#define MXMM_POST_EXTRACT() ((void)0)
#define MXMM_AWAIT_EXTRACT() ((void)0)
#define MXMM_POST_MULTIPLY() ((void)0)
#define MXMM_AWAIT_MULTIPLY() ((void)0)
#else
#define MXMM_POST_EXTRACT() set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4)
#define MXMM_AWAIT_EXTRACT() wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4)
#define MXMM_POST_MULTIPLY() set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5)
#define MXMM_AWAIT_MULTIPLY() wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5)
#endif

// ID0 counts landed loads and ID2 freed L1 sets. With three sets a pair of
// flags per set would need six; one counter each way is enough because MTE2 and
// MTE1 are both in order, so the oldest outstanding token always belongs to the
// tile whose turn it is.
#define MXMM_LOAD(sl, sel)                     \
  do {                                         \
    const uint32_t sel_ = (sel);               \
    MXMM_TILE_GM((uint64_t)(sl) * K_L1);       \
    if (sel_ == 0u) {                          \
      MXMM_FILL(a_l1, b_l1);                   \
    } else if (sel_ == 1u) {                   \
      MXMM_FILL(a_l1b, b_l1b);                 \
    } else {                                   \
      MXMM_FILL(a_l1c, b_l1c);                 \
    }                                          \
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0); \
  } while (0)
// e is the global K-tile index. A slab's load is waited for once, before its
// first sub-tile, and the set is released once, after its last -- so ID0 and
// ID2 count slabs while ID4 and ID5 still count cube tiles.
#define MXMM_EXTRACT(e, a0, b0, as0, bs0)                         \
  do {                                                            \
    const uint32_t e_ = (e);                                      \
    const uint32_t slab_ = e_ / K_SUB;                            \
    const uint32_t sub_ = e_ % K_SUB;                             \
    const uint32_t sel_ = slab_ % L1_SETS;                        \
    if (sub_ == 0u) {                                             \
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);                 \
    }                                                             \
    if (sel_ == 0u) {                                             \
      MXMM_DRAIN_TO_L0(a_l1, b_l1, a0, b0, as0, bs0, sub_, e_);   \
    } else if (sel_ == 1u) {                                      \
      MXMM_DRAIN_TO_L0(a_l1b, b_l1b, a0, b0, as0, bs0, sub_, e_); \
    } else {                                                      \
      MXMM_DRAIN_TO_L0(a_l1c, b_l1c, a0, b0, as0, bs0, sub_, e_); \
    }                                                             \
    if (sub_ + 1u == K_SUB) {                                     \
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);                  \
    }                                                             \
    MXMM_POST_EXTRACT();                                          \
  } while (0)

    // One pair of scale loads for the whole output tile, before the data
    // prologue so it is the first thing MTE2 has queued.
    if (scales_held) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    }
    MXMM_FILL_SCALES();
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    scales_held = true;

    // Prologue: LOOKAHEAD tiles are in flight before the first multiply, so
    // a tile's extract waits on a load issued LOOKAHEAD passes earlier
    // instead of one it started itself.
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
#ifdef MXMM_SINGLE_L0
      MXMM_EXTRACT(kt, a_l0, b_l0, as_l0, bs_l0);
      MXMM_AWAIT_EXTRACT();
#else
      // Tile kt was extracted on the previous pass, so this pass extracts
      // kt+1 into the OTHER L0 set and the cube's multiply of kt runs beside
      // it. wait_flag stalls its target pipe, not the scalar unit, so the
      // multiply still issues to M while MTE1 sits in the extract.
      if (kt == 0u) {
        MXMM_EXTRACT(0u, a_l0, b_l0, as_l0, bs_l0);
      }
      if (kt + 1u < k_tiles) {
        if (kt >= 1u) {
          MXMM_AWAIT_MULTIPLY();  // tile kt-1 has released that L0 set
        }
        if (odd) {  // kt+1 has the opposite parity, so the other L0 set
          MXMM_EXTRACT(kt + 1u, a_l0, b_l0, as_l0, bs_l0);
        } else {
          MXMM_EXTRACT(kt + 1u, a_l0b, b_l0b, as_l0b, bs_l0b);
        }
      }
      MXMM_AWAIT_EXTRACT();
#endif
#ifdef MXMM_NO_MATMUL
      if (kt == 0u) {
        TMATMUL_MX(c_l0, a_l0, as_l0, b_l0, bs_l0);
      }
#elif defined(MXMM_SINGLE_L0)
      if (kt == 0u) {
        TMATMUL_MX(c_l0, a_l0, as_l0, b_l0, bs_l0);
      } else {
        TMATMUL_MX(c_l0, c_l0, a_l0, as_l0, b_l0, bs_l0);
      }
#else
      if (odd) {  // kt == 0 is even, so an odd tile always accumulates
        TMATMUL_MX(c_l0, c_l0, a_l0b, as_l0b, b_l0b, bs_l0b);
      } else if (kt == 0u) {
        TMATMUL_MX(c_l0, a_l0, as_l0, b_l0, bs_l0);
      } else {
        TMATMUL_MX(c_l0, c_l0, a_l0, as_l0, b_l0, bs_l0);
      }
#endif
      MXMM_POST_MULTIPLY();
#ifdef MXMM_SINGLE_L0
      // one L0 set, so the next extract cannot start until the cube is done
      MXMM_AWAIT_MULTIPLY();
#endif
    }
    // ID2 carries min(k_tiles, L1_SETS) tokens out of the loop: every extract
    // posts one and only the loads far enough in consume one. Draining keeps
    // a count from carrying into the next output tile, where a later wait
    // would return early.
    for (uint32_t d = 0u; d < L1_SETS && d < k_slabs; ++d) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    }
    // Every extract in the loop above read the scale buffer, and MTE1 is in
    // order, so one post here releases it for the next output tile.
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
#ifndef MXMM_SINGLE_L0
    // Every multiply posts on ID5 and only the extracts of tile 2 and later
    // consume one, so min(k_tiles, 2) tokens are outstanding here.
    MXMM_AWAIT_MULTIPLY();
    if constexpr (k_tiles > 1u) {
      MXMM_AWAIT_MULTIPLY();
    }
#endif
#undef MXMM_TILE_GM
#undef MXMM_SCALE_GM
#undef MXMM_FILL
#undef MXMM_FILL_SCALES
#undef MXMM_DRAIN_TO_L0
#ifndef MXMM_NO_STORE
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    GmOut out_g((__gm__ bfloat16_t *)out_gm_g +
                ((uint64_t)mt * BASE_M * N + (uint64_t)nt * BASE_N));
    TSTORE(out_g, c_l0);
    // The next output tile's first multiply is non-accumulating and overwrites
    // c_l0, so it waits for the store; its loads and extracts do not.
#ifdef MXMM_STORE_BLOCKS_MTE2
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
#else
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
#endif
#endif
  }

  // One ID3 post is outstanding: every tile posts and every tile but the first
  // consumed one. Drain it so a later launch's first wait cannot return early.
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

// One instantiation while the numerics are being established. Generalising
// means a dispatch fold like the fused kernel's, once this is right.
// K = BASE_K and N = BASE_N deliberately: with one tile in every dimension the
// TLOAD is not WINDOWED -- the L1 tile spans a whole row of the matrix, exactly
// as in the conformance test, which is the one thing this kernel does
// differently. If this is exact and K=512 is not, the strided load is the
// fault.
// The default shape must satisfy K % K_L1 == 0, so it cannot be under one
// L1 slab. 512 is the narrowest K that builds at the default K_L1.
#ifndef MXMM_TEST_K
#define MXMM_TEST_K 512
#endif
#ifndef MXMM_TEST_N
#define MXMM_TEST_N 512
#endif
constexpr unsigned TEST_M_MAX = 65536;  // an upper bound, not the shape
// The launcher's alignment requirement is now the finest tile, so a caller
// with M=16 needs no padding at all and one with M=1 pads to 16 rather than
// 128.
constexpr unsigned M_TILE_ROWS = TINY_M;
constexpr unsigned TEST_K = MXMM_TEST_K, TEST_N = MXMM_TEST_N;

// Which output tile suits this M. The big one moves less data but halves the
// block count, so it only pays where the shape still fills the cores. The
// threshold is the launcher's own block cap: below it fewer blocks means
// idle cores and the saving is swamped. Measured 1.49x at M=4096 K=N=8192
// where both shapes saturate, and 0.65x at M=256 where the big tile leaves
// half the 32 cube cores idle.
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

// What a caller with an awkward M should round up to.
//
// The tiny tile is only right up to its own height. Each output tile streams a
// whole B column panel, so M=64 as four 16-row tiles reads B four times, while
// one 128-row tile reads it once and merely wastes rows. B dominates the
// traffic at small M, so the wasted rows are cheaper: measured at K=N=6144,
// four 16-row tiles ran 0.66x of bf16 where the padded 128-row tile ran 1.63x.
inline uint32_t mxfp4_matmul_pick_m_round(uint32_t m) {
  if (m == 0u) return TINY_M;
  if (m <= TINY_M) return TINY_M;
  return ((m + SMALL_M - 1u) / SMALL_M) * SMALL_M;
}

static inline void mxfp4_matmul_launch(uint32_t block_dim, void *stream,
                                       uint8_t *a_gm, uint8_t *a_scale_gm,
                                       uint8_t *b_gm, uint8_t *b_scale_gm,
                                       uint8_t *out_gm, uint32_t m, uint32_t k,
                                       uint32_t n, uint32_t groups) {
  // The host rejects what has no instantiation: a launcher that returns quietly
  // hands the caller an untouched output buffer back.
  if (block_dim == 0u) return;
  if (k != TEST_K || n != TEST_N) return;
  // M is runtime now, but still has to be whole tiles and within the declared
  // maximum: a launcher that returns quietly hands back an untouched buffer.
  if (m == 0u || m % M_TILE_ROWS != 0u || m > TEST_M_MAX) return;
  // NEVER guard this launch with a device-pass macro.
  const uint32_t tile = mxfp4_matmul_pick_m_tile(m);
  if (tile == BIG_M) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, BIG_M, BIG_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim, groups);
  } else if (tile == SMALL_M) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, SMALL_M, SMALL_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim, groups);
  } else {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, TINY_M, TINY_N>
        <<<block_dim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                         out_gm, m, block_dim, groups);
  }
}

// Launch the same kernel n times from C, so a caller measuring throughput
// pays the Python and ctypes cost once instead of n times. At K=N=512 the
// whole matmul is half a megaflop and 524 KB of weights, so essentially all of
// the measured time is per-call overhead, and this separates the part that
// belongs to the launch path from the part that belongs to the host binding.
// One problem, the ordinary call.
extern "C" void call_mxfp4_matmul(uint32_t block_dim, void *stream,
                                  uint8_t *a_gm, uint8_t *a_scale_gm,
                                  uint8_t *b_gm, uint8_t *b_scale_gm,
                                  uint8_t *out_gm, uint32_t m, uint32_t k,
                                  uint32_t n) {
  mxfp4_matmul_launch(block_dim, stream, a_gm, a_scale_gm, b_gm, b_scale_gm,
                      out_gm, m, k, n, 1u);
}

// G problems of identical shape, laid out back to back, in a single launch.
// block_dim should be sized against G * out_tiles rather than out_tiles.
extern "C" void call_mxfp4_matmul_grouped(uint32_t groups, uint32_t block_dim,
                                          void *stream, uint8_t *a_gm,
                                          uint8_t *a_scale_gm, uint8_t *b_gm,
                                          uint8_t *b_scale_gm, uint8_t *out_gm,
                                          uint32_t m, uint32_t k, uint32_t n) {
  if (groups == 0u) return;
  mxfp4_matmul_launch(block_dim, stream, a_gm, a_scale_gm, b_gm, b_scale_gm,
                      out_gm, m, k, n, groups);
}

extern "C" void call_mxfp4_matmul_repeat(uint32_t reps, uint32_t block_dim,
                                         void *stream, uint8_t *a_gm,
                                         uint8_t *a_scale_gm, uint8_t *b_gm,
                                         uint8_t *b_scale_gm, uint8_t *out_gm,
                                         uint32_t m, uint32_t k, uint32_t n) {
  for (uint32_t i = 0u; i < reps; ++i) {
    call_mxfp4_matmul(block_dim, stream, a_gm, a_scale_gm, b_gm, b_scale_gm,
                      out_gm, m, k, n);
  }
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

// ---------------------------------------------------------------------------
// The layout traps above are the ones that cost real time; each is called
// out at the declaration it applies to. Measured throughput, the tuning
// sweeps behind the defaults, and the approaches that were tried and
// rejected are in README.md rather than here.
