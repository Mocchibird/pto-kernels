// camodel sim harness for fused_hadamard_mxfp4_a5.
// Runs the fused WHT+MXFP4 kernel under the simulator (no real device),
// dequantizes fp4(e2m1)+e8m0 and checks against the natural-order WHT.
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
#ifndef BATCH_DIM
#define BATCH_DIM 64
#endif
#ifndef IN_BF16
#define IN_BF16 0
#endif
static constexpr uint32_t N = HAD_N;
static constexpr uint32_t BATCH = BATCH_DIM;
static constexpr uint32_t BLK = 32, NBLK = N / BLK;

extern "C" void call_fused_hadamard_mxfp4_a5(uint32_t block_dim, void *stream,
                                             uint8_t *x, uint8_t *q, uint8_t *s, uint32_t batch);

static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) { if (exp < -10) return (uint16_t)sign; man |= 0x800000u; int sh = 14 - exp;
        uint32_t h = man >> sh; if ((man >> (sh - 1)) & 1) h++; return (uint16_t)(sign | h); }
    uint16_t h = (uint16_t)(sign | (exp << 10) | (man >> 13));
    if (man & 0x1000u) h++; return h;
}
static uint16_t f32_to_bf16(float f) { uint32_t x; std::memcpy(&x, &f, 4);
    return (uint16_t)((x + 0x7fffu + ((x >> 16) & 1)) >> 16); }

static const float E2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static void wht_natural(float *x) {
    for (uint32_t h = 1; h < N; h <<= 1)
        for (uint32_t i = 0; i < N; i += (h << 1))
            for (uint32_t j = i; j < i + h; ++j) { float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
}

int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream = nullptr; aclrtCreateStream(&stream);

    const size_t total = (size_t)BATCH * N;
    const uint32_t SSTRIDE = 16;   // u16 per row (32-byte padded scale slot)
    std::vector<uint16_t> h_x(total);
    std::vector<uint8_t> h_q(BATCH * (N / 2));
    std::vector<uint16_t> h_s(BATCH * SSTRIDE);
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.0f);
    std::vector<float> xf(total);
    for (size_t i = 0; i < total; ++i) { xf[i] = nd(rng);
        h_x[i] = IN_BF16 ? f32_to_bf16(xf[i]) : f32_to_f16(xf[i]); }

    // reference: natural WHT * 1/sqrt(N)
    const float inv = 1.f / std::sqrt((float)N);
    std::vector<float> gold(total);
    for (uint32_t r = 0; r < BATCH; ++r) { float blk[N];
        for (uint32_t j = 0; j < N; ++j) blk[j] = xf[r * N + j];
        wht_natural(blk);
        for (uint32_t i = 0; i < N; ++i) gold[r * N + i] = blk[i] * inv; }

    void *d_x = nullptr, *d_q = nullptr, *d_s = nullptr;
    const size_t s_bytes = (size_t)BATCH * SSTRIDE * 2;
    aclrtMalloc(&d_x, total * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_q, BATCH * (N / 2), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_s, s_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(d_q, BATCH * (N / 2), 0, BATCH * (N / 2));
    aclrtMemset(d_s, s_bytes, 0, s_bytes);
    aclrtMemcpy(d_x, total * 2, h_x.data(), total * 2, ACL_MEMCPY_HOST_TO_DEVICE);

    const uint32_t block_dim = 8;
    std::printf("==SIM== fused WHT+MXFP4 N=%u batch=%u in=%s\n", N, BATCH, IN_BF16 ? "bf16" : "fp16");
    call_fused_hadamard_mxfp4_a5(block_dim, stream, (uint8_t *)d_x, (uint8_t *)d_q, (uint8_t *)d_s, BATCH);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_q.data(), h_q.size(), d_q, h_q.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(h_s.data(), s_bytes, d_s, s_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // dequant (try both nibble orders) and compare
    double best_l2 = 1e30; int best_ord = -1;
    for (int ord = 0; ord < 2; ++ord) {
        double num = 0, den = 0;
        for (uint32_t r = 0; r < BATCH; ++r)
            for (uint32_t i = 0; i < N; ++i) {
                uint8_t byte = h_q[r * (N / 2) + i / 2];
                uint8_t nib = (ord == 0) ? ((i & 1) ? (byte >> 4) : (byte & 0xF))
                                         : ((i & 1) ? (byte & 0xF) : (byte >> 4));
                float mag = E2M1[nib & 7]; float sgn = (nib & 8) ? -1.f : 1.f;
                float sc = std::ldexp(1.f, (int)h_s[r * SSTRIDE + i / BLK] - 127);
                float v = sgn * mag * sc; float g = gold[r * N + i];
                num += (v - g) * (v - g); den += g * g;
            }
        double l2 = std::sqrt(num / (den > 0 ? den : 1));
        std::printf("  nibble_ord=%d l2_rel=%g\n", ord, l2);
        if (l2 < best_l2) { best_l2 = l2; best_ord = ord; }
    }
    std::printf("s[0..3]=%u %u %u %u\n", h_s[0], h_s[1], h_s[2], h_s[3]);
    { int nz = 0; for (size_t i = 0; i < h_q.size(); ++i) if (h_q[i]) nz++;
      std::printf("q nonzero bytes: %d / %zu\n", nz, h_q.size()); }
    std::printf("row0 q bytes: ");
    for (int i = 0; i < 64; ++i) std::printf("%02x ", h_q[i]);
    std::printf("\n");
    // dump row0 first 8 dequant vs gold (best order)
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = h_q[i / 2];
        uint8_t nib = (best_ord == 0) ? ((i & 1) ? (byte >> 4) : (byte & 0xF))
                                      : ((i & 1) ? (byte & 0xF) : (byte >> 4));
        float mag = E2M1[nib & 7]; float sgn = (nib & 8) ? -1.f : 1.f;
        float sc = std::ldexp(1.f, (int)h_s[i / BLK] - 127);
        std::printf("  [%d] gold=%.4f deq=%.4f (nib=%u e8m0=%u)\n", i, gold[i], sgn * mag * sc, nib, h_s[i / BLK]);
    }
    std::printf("BEST l2_rel=%g ord=%d  %s\n", best_l2, best_ord, best_l2 < 0.25 ? "PASS" : "FAIL");
    aclrtFree(d_x); aclrtFree(d_q); aclrtFree(d_s);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return best_l2 < 0.25 ? 0 : 1;
}
