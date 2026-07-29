// fast_hadamard_128_cube_a5 — A5 (Ascend 950 / dav-c310) CUBE-core fp16 Walsh-Hadamard
// transform for N == 128, computed as a matmul against the Sylvester matrix.
//
// WHY
// ---
// The register-resident VF kernel (fast_hadamard_128_a5.cpp) is compute-bound on the
// vector pipe at ~1.5 TB/s (~half of the ~3 TB/s HBM copy floor). The WHT is
// really Y = X @ H, H = Sylvester(128)/sqrt(128) (symmetric, +/-1). The cube
// (matrix) unit does that matmul cheaply relative to the memory traffic, so this
// path is MEMORY-bound and can get closer to copy speed.
//
// MATH: cube TMATMUL(C,A,B) computes C[M,N] = A[M,K] @ B[K,N], accumulated over
// K sub-tiles with TMATMUL_ACC. H is symmetric so orientation is a no-op; H is
// pre-scaled by 1/sqrt(128) on the host so C = X @ H = natural-order WHT.
//
// K is tiled in 64-wide sub-tiles (KQ), the cube's L0 fractal granularity. H's
// two K-halves are pre-extracted into resident L0B tiles once and reused for
// every row-tile. In-place on x.
//
// PIPELINE: double-buffered across row-tiles so the four stages overlap:
//   MTE2 load(it) || MTE1 extract(it-?) || M matmul || FIX store(it-1)
// L1A/L0A/L0C are 2-deep ping-pong; H (L0B) is resident/shared.
//
// GUARDS: the kernel definition and the launcher are visible on BOTH host and
// device passes; only the device-only kernel BODY is under __DAV_CUBE__ (defined
// only in the cube device pass). Guarding the <<<>>> launch would compile it out
// on the host pass and make the kernel a silent no-op.
//
// Build: bisheng --cce-aicore-arch=dav-c310-cube -O2 -std=c++17 -fPIC ...

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef HAD_N
#define HAD_N 128
#endif
static_assert(HAD_N == 128, "cube v1 supports N==128");

#ifndef M_TILE
#define M_TILE 128
#endif

constexpr int N = HAD_N;    // 128
constexpr int K = HAD_N;    // 128
constexpr int KQ = 64;      // K sub-tile (cube L0 fractal granularity)
constexpr int KPHASES = K / KQ;  // 2

namespace hc {
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }
}  // namespace hc

// L1 byte layout: A ping/pong (2 * 32 KB) then B/H (32 KB).
constexpr uintptr_t A_BYTES = (uintptr_t)M_TILE * K * sizeof(half);   // 32 KB
constexpr uintptr_t B_L1_START = 2 * A_BYTES;                         // 64 KB
// L0A sub-tile (M_TILE*KQ*2 = 16 KB); L0B sub-tile (KQ*N*2 = 16 KB);
// L0C tile (M_TILE*N*4 = 64 KB).
constexpr uintptr_t A_SUB_BYTES = (uintptr_t)M_TILE * KQ * sizeof(half);  // 16 KB
constexpr uintptr_t B_SUB_BYTES = (uintptr_t)KQ * N * sizeof(half);       // 16 KB
constexpr uintptr_t C_BYTES = (uintptr_t)M_TILE * N * sizeof(float);      // 64 KB

__global__ AICORE void hadamard_cube_a5(__gm__ void *x_gm, __gm__ void *h_gm,
                                        uint32_t batch)
{
#if defined(__DAV_CUBE__)
    __gm__ half *xh = reinterpret_cast<__gm__ half *>(x_gm);
    __gm__ half *hh = reinterpret_cast<__gm__ half *>(h_gm);

    using L1A = Tile<TileType::Mat, half, M_TILE, K, BLayout::ColMajor,
                     M_TILE, K, SLayout::RowMajor, 512>;   // X[M,K] ND->NZ
    using L1B = Tile<TileType::Mat, half, K, N, BLayout::RowMajor,
                     K, N, SLayout::ColMajor, 512>;        // H[K,N] DN
    using L0A = TileLeft<half, M_TILE, KQ>;                // 128x64
    using L0B = TileRight<half, KQ, N>;                    // 64x128
    using L0C = TileAcc<float, M_TILE, N>;                 // 128x128 fp32
    using DynStrideND = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using DynStrideDN = pto::Stride<1, 1, 1, 1, DYNAMIC>;
    using GlobA = GlobalTensor<half, TileShape2D<half, M_TILE, K, Layout::ND>,
                               DynStrideND, Layout::ND>;
    using GlobB = GlobalTensor<half, TileShape2D<half, K, N, Layout::DN>,
                               DynStrideDN, Layout::DN>;
    using GlobC = GlobalTensor<half, TileShape2D<half, M_TILE, N, Layout::ND>,
                               DynStrideND, Layout::ND>;

    const uint32_t cid = get_block_idx();
    const uint32_t num_cores = get_block_num();
    const uint32_t tiles = batch / M_TILE;   // batch MUST be a multiple of M_TILE

    // ---- L1 / L0 buffers ----
    L1A a_l1[2];  TASSIGN(a_l1[0], (uintptr_t)0); TASSIGN(a_l1[1], A_BYTES);
    L1B b_l1;     TASSIGN(b_l1, B_L1_START);
    // a_l0[pp][j]: 2 ping-pong buffers x 2 K-halves -> 4 * 16 KB = 64 KB (full L0A)
    L0A a_l0[2][KPHASES];
    TASSIGN(a_l0[0][0], (uintptr_t)0);              TASSIGN(a_l0[0][1], A_SUB_BYTES);
    TASSIGN(a_l0[1][0], 2 * A_SUB_BYTES);           TASSIGN(a_l0[1][1], 3 * A_SUB_BYTES);
    L0B b_l0[KPHASES];  TASSIGN(b_l0[0], (uintptr_t)0); TASSIGN(b_l0[1], B_SUB_BYTES);
    L0C c_l0[2];  TASSIGN(c_l0[0], (uintptr_t)0); TASSIGN(c_l0[1], C_BYTES);

    // ---- Preload H once: GM -> L1B -> two K-half L0B tiles (resident) ----
    {
        GlobB gh(hh, {}, DynStrideDN(K));
        TLOAD(b_l1, gh);
        hc::SetFlag<PIPE_MTE2, PIPE_MTE1>(4);
        hc::WaitFlag<PIPE_MTE2, PIPE_MTE1>(4);
        for (int j = 0; j < KPHASES; ++j)
            TEXTRACT(b_l0[j], b_l1, j * KQ, 0);
        hc::SetFlag<PIPE_MTE1, PIPE_M>(4);
        hc::WaitFlag<PIPE_MTE1, PIPE_M>(4);
    }

    // ---- init "buffer free" flags for the 2-deep pipeline ----
    hc::SetFlag<PIPE_MTE1, PIPE_MTE2>(0);   // L1A[0] free
    hc::SetFlag<PIPE_MTE1, PIPE_MTE2>(1);   // L1A[1] free
    hc::SetFlag<PIPE_M, PIPE_MTE1>(0);      // L0A[0] free
    hc::SetFlag<PIPE_M, PIPE_MTE1>(1);      // L0A[1] free
    hc::SetFlag<PIPE_FIX, PIPE_M>(0);       // L0C[0] free
    hc::SetFlag<PIPE_FIX, PIPE_M>(1);       // L0C[1] free

    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += num_cores, ++it) {
        const int pp = it & 1;
        const uint64_t off = static_cast<uint64_t>(tb) * M_TILE * K;

        // -- load X tile: GM -> L1A[pp] --
        hc::WaitFlag<PIPE_MTE1, PIPE_MTE2>(pp);   // L1A[pp] free
        {
            GlobA ga(xh + off, {}, DynStrideND(K));
            TLOAD(a_l1[pp], ga);
        }
        hc::SetFlag<PIPE_MTE2, PIPE_MTE1>(pp);

        // -- extract both K-halves: L1A[pp] -> L0A[pp][0..1] --
        hc::WaitFlag<PIPE_MTE2, PIPE_MTE1>(pp);   // load done
        hc::WaitFlag<PIPE_M, PIPE_MTE1>(pp);      // L0A[pp] free
        for (int j = 0; j < KPHASES; ++j)
            TEXTRACT(a_l0[pp][j], a_l1[pp], 0, j * KQ);
        hc::SetFlag<PIPE_MTE1, PIPE_MTE2>(pp);    // L1A[pp] free
        hc::SetFlag<PIPE_MTE1, PIPE_M>(pp);       // extract done

        // -- matmul over K: C = A0@H0 + A1@H1 --
        hc::WaitFlag<PIPE_MTE1, PIPE_M>(pp);
        hc::WaitFlag<PIPE_FIX, PIPE_M>(pp);       // L0C[pp] free
        TMATMUL(c_l0[pp], a_l0[pp][0], b_l0[0]);
        TMATMUL_ACC(c_l0[pp], c_l0[pp], a_l0[pp][1], b_l0[1]);
        hc::SetFlag<PIPE_M, PIPE_MTE1>(pp);       // L0A[pp] free
        hc::SetFlag<PIPE_M, PIPE_FIX>(pp);        // matmul done

        // -- store: L0C[pp] fp32 -> GM fp16 (in place) --
        hc::WaitFlag<PIPE_M, PIPE_FIX>(pp);
        {
            GlobC gc(xh + off, {}, DynStrideND(N));
            TSTORE(gc, c_l0[pp]);
        }
        hc::SetFlag<PIPE_FIX, PIPE_M>(pp);        // L0C[pp] free
    }

    // ---- drain ----
    hc::WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    hc::WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
    hc::WaitFlag<PIPE_M, PIPE_MTE1>(0);
    hc::WaitFlag<PIPE_M, PIPE_MTE1>(1);
    hc::WaitFlag<PIPE_FIX, PIPE_M>(0);
    hc::WaitFlag<PIPE_FIX, PIPE_M>(1);
#else
    (void)x_gm; (void)h_gm; (void)batch;
#endif
}

extern "C" void call_hadamard_cube_a5(uint32_t block_dim, void *stream,
                                      uint8_t *x, uint8_t *h, uint32_t batch)
{
    hadamard_cube_a5<<<block_dim, nullptr, stream>>>(x, h, batch);
}
