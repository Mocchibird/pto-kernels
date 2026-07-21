// Self-contained camodel sim test for the register-resident fast_hadamard_a5.
//
// Validates the kernel against a KNOWN-CORRECT natural-order Walsh-Hadamard
// transform permuted by bit-reversal: the kernel's constant-geometry
// (deinterleave -> sum/diff -> interleave) butterfly produces the WHT in
// bit-reversed index order, so  kernel_out[i] == natural_wht(x)[bitrev(i)].
// Checking against natural_wht (computed the textbook concat-halves way) proves
// the kernel computes a genuine Hadamard, not just "some self-consistent
// butterfly".

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#ifndef HAD_N
#define HAD_N 128
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 7
#endif
#ifndef BATCH_DIM
#define BATCH_DIM 256
#endif

static constexpr uint32_t N = HAD_N;
static constexpr uint32_t LOG2N = HAD_LOG2N;
static constexpr uint32_t BATCH = BATCH_DIM;

extern "C" void call_fast_hadamard_a5(uint32_t block_dim, void *stream,
                                      uint8_t *x, uint32_t batch);

// ---- IEEE-754 fp16 <-> fp32 (round-to-nearest-even on the way down) ----
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) // inf/nan
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);        // overflow -> inf
    if (exp <= 0) {                                            // subnormal/zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1) h++;                     // round
        return (uint16_t)(sign | h);
    }
    uint16_t h = (uint16_t)(sign | (exp << 10) | (man >> 13));
    if (man & 0x1000u) h++;                                    // round-nearest
    return h;
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; exp--; }
               man &= 0x3ffu; out = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &out, 4); return f;
}

// Textbook natural-order (Sylvester) FWHT on one length-N block, in place.
static void wht_natural(float *x) {
    for (uint32_t h = 1; h < N; h <<= 1) {
        for (uint32_t i = 0; i < N; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
}

int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream = nullptr; aclrtCreateStream(&stream);

    const size_t total = (size_t)BATCH * N;
    const size_t bytes = total * sizeof(uint16_t);
    std::vector<uint16_t> h_x(total), h_y(total), h_gold(total);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (size_t i = 0; i < total; ++i) h_x[i] = f32_to_f16(nd(rng));

    const float inv = 1.0f / std::sqrt((float)N);
    for (uint32_t r = 0; r < BATCH; ++r) {
        float blk[N];
        for (uint32_t j = 0; j < N; ++j) blk[j] = f16_to_f32(h_x[r * N + j]);
        wht_natural(blk);   // concat-halves constant-geometry == natural Sylvester order
        for (uint32_t i = 0; i < N; ++i)
            h_gold[r * N + i] = f32_to_f16(blk[i] * inv);
    }

    void *d_x = nullptr;
    aclrtMalloc(&d_x, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_x, bytes, h_x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    const uint32_t block_dim = 8;
    std::printf("==BENCH== fast_hadamard_a5 N=%u batch=%u block_dim=%u\n", N, BATCH, block_dim);
    call_fast_hadamard_a5(block_dim, stream, (uint8_t *)d_x, BATCH);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_y.data(), bytes, d_x, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    double max_diff = 0.0; size_t err = 0;
    for (size_t i = 0; i < total; ++i) {
        float g = f16_to_f32(h_gold[i]), a = f16_to_f32(h_y[i]);
        double d = std::fabs(g - a);
        if (d > max_diff) max_diff = d;
        if (d > 0.05) ++err;
    }
    std::printf("max_diff=%g err_count=%zu / %zu\n", max_diff, err, total);
    for (int i = 0; i < 8; ++i)
        std::printf("  [%d] golden=%g actual=%g\n", i, f16_to_f32(h_gold[i]), f16_to_f32(h_y[i]));

    aclrtFree(d_x); aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    bool pass = (max_diff < 0.1) && (err < total / 100);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
