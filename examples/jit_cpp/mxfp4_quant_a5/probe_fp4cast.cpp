// Probe: bf16 -> f4e2m1x2 cast behaviour, which decides two open design
// questions.
//
// R6a  Does the cast SATURATE inputs above the largest fp4 magnitude (6.0)?
//      MXFP4's FLOOR scale rule lets amax/X reach 7.0 (measured), so anything
//      above 6.0 must clamp to 6.0 or the block is wrong.
// R6b  Is set_ctrl(1<<50) ("clip into MAX_NORM") needed for that, and does it
//      clobber the mask mode configured immediately before it? The WIP issues
//      it AFTER set_mask_norm()/set_vector_mask(-1,-1), and it is the only
//      set_ctrl in the repo.
//
// -DUSE_CTRL=1 issues set_ctrl after the mask pair (the WIP's order).
// -DUSE_CTRL=2 issues it BEFORE the mask pair. Same output => no clobber.
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef USE_CTRL
#define USE_CTRL 0
#endif
#ifndef PART_SEL
#define PART_SEL PART_P0
#endif

#ifdef __CCE_AICORE__
constexpr unsigned LANES = 128;      // bf16 lanes cast per call
constexpr unsigned IN_ELEMS = 128;   // bf16 in
constexpr unsigned OUT_BYTES = 128;  // dump the whole f4e2m1x2 register
#endif

__global__ AICORE void fp4cast(__gm__ void *in_gm, __gm__ void *out_gm) {
#ifdef __DAV_VEC__
#if USE_CTRL == 2
  set_ctrl(static_cast<uint64_t>(1) << 50);
#endif
  set_mask_norm();
  set_vector_mask(-1, -1);
#if USE_CTRL == 1
  set_ctrl(static_cast<uint64_t>(1) << 50);
#endif
  if (get_block_idx() != 0) return;

  using ShIn = pto::Shape<1, 1, 1, 1, IN_ELEMS>;
  using StIn = pto::Stride<1, 1, 1, IN_ELEMS, 1>;
  using TIn = Tile<TileType::Vec, bfloat16_t, 1, IN_ELEMS, BLayout::RowMajor, 1,
                   IN_ELEMS>;
  using ShOut = pto::Shape<1, 1, 1, 1, OUT_BYTES>;
  using StOut = pto::Stride<1, 1, 1, OUT_BYTES, 1>;
  using TOut = Tile<TileType::Vec, uint8_t, 1, OUT_BYTES, BLayout::RowMajor, 1,
                    OUT_BYTES>;

  TIn xt;
  TASSIGN(xt, 0u);
  GlobalTensor<bfloat16_t, ShIn, StIn> gi((__gm__ bfloat16_t *)in_gm, ShIn());
  TLOAD(xt, gi);
  TOut ot;
  TASSIGN(ot, 512u);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  __VEC_SCOPE__ {
    uint32_t la = LANES;
    uint32_t lb = OUT_BYTES;
    MaskReg pIn = CreatePredicate<bfloat16_t>(la);  // 128 x 16-bit
    MaskReg pOut =
        CreatePredicate<uint8_t>(lb);  // 128 x 8-bit  <- was wrong before
    vector_bf16 v;
    vector_f4e2m1x2 q;
    vector_u8 z;
    vdup(z, (uint8_t)0xEE, pOut, MODE_ZEROING);  // sentinel: untouched = 0xEE
    vsts(z, (__ubuf__ uint8_t *)(uintptr_t)512u, 0, NORM_B8, pOut);
    vlds(v, (__ubuf__ bfloat16_t *)(uintptr_t)0u, 0, NORM);
    vcvt(q, v, pIn, ROUND_R, PART_SEL);
    vsts((vector_u8 &)q, (__ubuf__ uint8_t *)(uintptr_t)512u, 0, NORM_B8, pOut);
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  GlobalTensor<uint8_t, ShOut, StOut> go((__gm__ uint8_t *)out_gm, ShOut());
  TSTORE(go, ot);
#else
  (void)in_gm;
  (void)out_gm;
#endif
}

extern "C" void call_fp4cast(uint32_t bd, void *s, uint8_t *in, uint8_t *out) {
  fp4cast<<<bd, nullptr, s>>>(in, out);
}
