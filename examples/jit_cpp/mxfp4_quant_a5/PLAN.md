<!-- Authored 2026-07-29 from five parallel research passes (format spec, baseline
landscape, WIP audit, repo conventions, ISA surface) over the register-branch
work-in-progress, then corrected against device measurements taken the same day.
Claims marked "device-verified" were measured on bz.39; everything else is an
open question in section 7, not a settled number. -->

# MXFP4 Quantization on Ascend 950 / A5 (dav-c310-vec) — Implementation Plan

**Repo root:** `/Users/hyunmin/Projects/Work/Huawei/pto-kernels`
**New directories this plan creates:**

- `/Users/hyunmin/Projects/Work/Huawei/pto-kernels/examples/jit_cpp/mxfp4_quant_a5` (Stage 1)
- `/Users/hyunmin/Projects/Work/Huawei/pto-kernels/examples/jit_cpp/fused_hadamard_mxfp4_quant_a5` (Stage 2)

**Repo state (re-verified 2026-07-29 after the plan was drafted).** `examples/jit_cpp/fast_hadamard_a5` on branch `fast-hadamard-a5-pr` (tip **`60338d9`**, all 13 CI checks green, open as PR #221 to `huawei-csl:main`) is a pruned two-kernel example, 11 files: `README.md`, `__init__.py`, `benchmark.py`, `copy_ref_256_a5.cpp`, `fast_hadamard_256_a5.cpp`, `jit_util_copy256_a5.py`, `jit_util_hadamard256_a5.py`, `plot_hadamard256_a5.py`, `run_benchmark.sh` (755), `test_copy256_a5.py`, `test_hadamard256_a5.py`. The transform and its copy-floor reference are now **separate translation units**, each exporting exactly one launcher — copy that separation in both new dirs.

**All MXFP4 WIP survives only on branch `fast-hadamard-a5-register`** (tip **`a19a396`**; the MXFP4 files were introduced by `c79a077` and are unchanged at the tip). Read them with `git show fast-hadamard-a5-register:examples/jit_cpp/fast_hadamard_a5/<file>`. Every WIP citation below was taken that way and **spot-verified against the register branch on 2026-07-29** (the `vmins(mf,30)` / `vadds(bd,110)` clamp mismatch, `static_assert(HAD_N == 128)` at `:30`, and `set_ctrl(1<<50)` after the mask pair at `:187` all confirmed at the cited lines).

**Environment gotcha that cost this plan some accuracy — know it before you trust a path.** `/Users/hyunmin/Projects/Work/Huawei` is **bind-mounted to `/root/repos` inside the dev container**, so the "container repo" and the macOS clone are ONE working tree sharing one `.git`. A `git checkout` in either moves the other. That is why this plan says the tree "moved twice during research", and it is why every citation here is pinned to a branch or commit rather than to the working tree. Do not assume the tree is on the branch you last put it on.

**DECISION REQUIRED BEFORE STAGE 1 (blocking, user only):** which branch do the two new directories go on? `fast-hadamard-a5-pr` is the clean base and is what this plan assumes (new branch off it, e.g. `mxfp4-quant-a5`). The MXFP4 arithmetic is lifted by *reading* `fast-hadamard-a5-register`, not by merging it.

---

## 0. Measured facts — Stage 0 probe, device-verified 2026-07-29

Run `probe_conventions.py` on bz.39 to reproduce. **The format contract in §2 was
assumed when drafted and is now measured; every row below agreed with it.** These
supersede the corresponding open questions in §7.

**R2 — `dst_type=296` RESOLVED.** It is `torch_npu.float4_e2m1fn_x2` (found by scanning
`torch`/`torch_npu` for int attributes equal to 296; the op exposes no signature and no
docstring). The op returns **two** uint8 tensors:

| K | q shape | q bytes/row | scale shape | scale bytes/row |
|---|---|---|---|---|
| 256 | `(batch, 128)` | 128 = K/2 | `(batch, 4, 2)` | 8 = K/32 |
| 128 | `(batch, 64)`  | 64 = K/2  | `(batch, 2, 2)` | 4 = K/32 |

So `q` is genuinely **4-bit packed**, and the scale count is exactly K/32. **The byte
accounting in §1.2 is confirmed:** `bytes/row = 2K + K/2 + K/32 = 2.53125K`. For K=128
that is 324 — so the WIP's `TOTAL_B=324` (`bench_mxfp4_baseline.py:11-12`) was **correct
all along**; R2 resolved in its favour, and no ratio derived from it was inflated.

*New finding, feeds R9:* the vendor's scale tensor is shaped `(batch, K/64, 2)`, **not**
flat `(batch, K/32)`. The count matches but the layout does not, so a tensor-vs-tensor
comparison must reshape one side. Decide our own layout deliberately rather than by
accident. (The probe initially flagged this as a MISMATCH; that was the probe's own wrong
assertion — it compared only the last dim. The op is fine.)

**R3 — scale rule RESOLVED: OCP Algorithm 1, FLOOR.** `byte = floor(log2(amax)) - 2 + 127`
matched the measurement in **14 of 14** cases, and the predicted `amax/X ∈ [4,8)` window
was observed exactly:

| amax | scale byte | X | amax/X |
|---|---|---|---|
| 0.25 | 123 | 2^-4 | 4.0 |
| 1.0 | 125 | 2^-2 | 4.0 |
| 1.5 | 125 | 2^-2 | 6.0 |
| 4.0 | 127 | 1 | 4.0 |
| 6.0 | 127 | 1 | 6.0 |
| **7.0** | **127** | **1** | **7.0** |
| 8.0 | 128 | 2 | 4.0 |
| 1024 | 135 | 2^8 | 4.0 |

**The `amax = 7.0` row promotes R6 from a curiosity to a requirement.** `7.0/X = 7.0`
exceeds the largest fp4 magnitude (6.0), so for any block whose `amax ∈ [6,8)·2^k` the
cast **must saturate to 6.0**, not wrap or produce garbage. Clip-on-cast is therefore
**mandatory**, which is exactly what the WIP's `set_ctrl(1<<50)` ("clip into MAX_NORM on
cast", `fused_hadamard_mxfp4_a5.cpp:187`) is for. R6's question — whether `set_ctrl`
writes the CTRL register wholesale and thereby clobbers the mask mode set immediately
before it — is now load-bearing and must be settled on device before Stage 1 is trusted.

**Nibble order CONFIRMED — element 0 is the LOW nibble.** With `e0=1.0, e1=2.0, e31=6.0`
(scale byte 127, X=1) the first output byte is `0x42`: low nibble `0x2` (code for 1.0),
high nibble `0x4` (code for 2.0). So `byte[k] = (code[2k+1] << 4) | code[2k]` — the
convention §2.4 pinned. Note this means `bench256_fused.py:47` (`nib[:,0::2] = lo`) had it
**right**, and the defect was purely the auto-fitting test that made the question
unfalsifiable.

**R5 — rounding RESOLVED: round-to-nearest-even.** All seven exact fp4 midpoints broke the
tie to the even code field: `0.25→0`, `0.75→1.0`, `1.25→1.0`, `1.75→2.0`, `2.5→2.0`,
`3.5→4.0`, `5.0→4.0`. Round-half-away would have taken the larger code every time; it did
not. The host reference must implement RNE.

### ISA probe, device-verified 2026-07-30 (`probe_fp4cast.cpp` + `probe_fp4cast_run.py`)

Resolved from the bisheng builtin header
(`tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`):

- **R10 RESOLVED — `vmul` has a bf16 form.** `__VF_BINARY_OP(vmul, bf16, bf16, v128)`
  (:6800), plus a bf16 `vmuls` (:7797). So the integer-domain reduction is available with
  full E8M0 range and single rounding, which is also the answer to **R4**: use the bf16
  path and there is no double rounding to model.
- **The cast to fp4 takes bf16 only.** `__VF_VCVTFF_RND_PP(bf16, f4e2m1x2, ...)` exists
  (:5096); the `f16` form is present-but-commented-out (:5007). Signature:
  `vcvt(vector_f4e2m1x2 &dst, vector_bf16 src, vector_bool mask, rnd, pp, mode)`,
  `rnd ∈ {ROUND_R,A,F,C,Z}`, `pp ∈ {PART_P0..P3}`.
- **`vcgmax` u16 exists** (:8020) — needed, because there is no bf16 `vcgmax`, so the
  block absmax must be reduced in the integer domain (`vand 0x7FFF` then u16 `vcgmax`).
- **`float8_e8m0_t` is a first-class register type** with VLD/VLDX2/VINTLV/VDINTLV.

Measured on device:

- **`PART_Px` writes every 4th byte at byte-offset x.** One `vcvt` converts **64** bf16
  lanes into 32 bytes at byte-stride 4; four calls (P0..P3) fill one 256-code
  `f4e2m1x2` register. The register is therefore **byte-interleaved across parts** —
  feeding P0..P3 four sequential 64-element blocks scrambles the output. Feed part `x`
  the elements whose codes belong at bytes `4k+x`, i.e. elements `8k+2x` and `8k+2x+1`.
- **The cast SATURATES.** `[0, 6, 6.5, 7, 8, 16, 100, -7]` → bytes `70 77 77 f7`, i.e.
  everything out of range clamps to ±6.0 and `-7 → -6.0`.
- **R6 RESOLVED, and it is a non-issue.** Because the cast saturates natively, the
  `amax/X ∈ [6,8)` case needs no explicit clipping. `set_ctrl(1<<50)` produced
  byte-identical output whether omitted, issued after the mask pair (the WIP's order), or
  issued before it — so it is inert on this path and can simply be dropped rather than
  reasoned about. The clobber concern does not arise.
- Nibble order **low-first** confirmed again here (byte `0x10` = first element in the low
  nibble), matching the vendor-op measurement above.

*Caveat on how this was obtained:* the first version of this probe used a `bfloat16_t`
predicate to store `uint8_t` data and assumed `PART_P0` wrote contiguously. It produced
scrambled output (`1.0 → 0`, `2.0 → -6`, `100 → 1.5`) that briefly read as "the cast does
not saturate". Dump raw bytes against a known ramp with a sentinel fill; do not decode
through an assumed layout.

**Still open:** R7 (sign of zero), R8–R17. **R4 and R6 are closed** by the above; R10 is
closed in favour of the bf16 integer-domain design.

**Still open after this probe:** R4 (double rounding via the bf16 cast pivot — unaffected
by the above), R7 (sign of zero), R8–R17.

---

## 1. Goal and success criteria

### 1.1 What we are building

Two kernels, in this order, both PTO-ISA C++ JIT-compiled with `bisheng` for `--cce-aicore-arch=dav-c310-vec` and loaded via `ctypes`:

1. **`mxfp4_quant_a5`** — standalone OCP-MXFP4 quantization: `(batch, K) fp16|bf16` → `(batch, K/2) uint8` packed E2M1 nibbles + `(batch, K/32) uint8` E8M0 scales.
2. **`fused_hadamard_mxfp4_quant_a5`** — the same quantization fused with the N=256 (and N=128) Walsh–Hadamard transform, no intermediate GM round trip.

Stage 1 ships and is verified before Stage 2 starts. This ordering is a correctness requirement, not tidiness: with a Hadamard in the path, a butterfly output permutation, a `PART_P0`/`PK4_B32` lane-order surprise, and a nibble-order mismatch are **indistinguishable index permutations**. With no transform, a ramp input makes all 32 codes in a block distinct and makes packing order, block boundaries, and the scale byte individually falsifiable.

### 1.2 The ceiling: memory-bound roofline (know this before optimizing)

MXFP4 quantization is overwhelmingly read-bound. Per row of K elements with a packed uint8 scale array:

```
bytes/row = K * sizeof(in_t)  +  K/2      +  K/32
          = 2K (fp16/bf16)    +  0.5K     +  0.03125K
          = 2.53125 * K bytes/row          (79.0% of it is reads)
```

Measured HBM references for this device, from `git show fast-hadamard-a5-register:examples/jit_cpp/fast_hadamard_a5/BENCHMARKS.md`: pure `GM→UB→GM` copy floor **2.87 TB/s** at batch 65536, **3.03 TB/s** peak (batch 65536, grid sweep), HBM ceiling ~3.3 TB/s, and a documented **~11 µs fixed launch-overhead floor**.

Roofline times for K=256 (648 B/row), bracketing 2.87–3.03 TB/s:

| batch | bytes | roofline time | usable for bandwidth claims? |
|---|---|---|---|
| 16384 | 10.62 MB | 3.50 – 3.70 µs | **NO** — below the ~11 µs launch floor; both contenders measure launch latency |
| 65536 | 42.47 MB | 14.0 – 14.8 µs | yes |
| 131072 | 84.93 MB | 28.0 – 29.6 µs | yes |
| 262144 | 169.87 MB | 56.1 – 59.2 µs | yes |

If we keep the WIP's padded 32-byte-per-128-lane scale slot (`fused_hadamard_mxfp4_a5.cpp:55`, `:158`, `:245`) the row cost becomes 704 B instead of 648 — **+8.6% self-inflicted traffic**, moving the batch-65536 floor to 15.2–16.1 µs. Stage 1 does not do this (§4.3).

### 1.3 Falsifiable success criteria

All numbers below are **to be measured**; none are predictions of what we will achieve. Every claim is stated so it can fail.

**C1 — Correctness (hard gate, blocks everything else).**
`pytest test_mxfp4_quant_a5.py` exits 0 with:
- E8M0 scale bytes **100% bit-exact** vs the host reference on every test input, including the adversarial set (§4.5). They are integers; any mismatch is a bug, not a tolerance.
- E2M1 nibbles **100% bit-exact** vs a host reference that models the *same* cast chain the kernel uses (§2.5). Not a tolerance.
- Nibble order asserted against one pinned convention (§2.4). No searching, no relabeling.

**C2 — At the hardware limit.** `t_ours ≤ 1.05 × t_dma_floor` at batch ∈ {65536, 262144} for K ∈ {256, 4096}, where `t_dma_floor` is a **measured** compute-free kernel with the identical TLOAD/TSTORE pattern and identical output byte counts (`copy_ref_mxfp4_a5.cpp`, §4.6). This is a stronger and more durable claim than any ratio against a vendor op, and it is the one we should lead with.

**C3 — Beats torch_npu, apples-to-apples.** `t_ours < t_torch_npu` at batch ∈ {65536, 262144}, for both fp16 and bf16 input, with:
- both contenders allocating outputs per call, **and separately** both preallocated (two rows, both reported — see the allocation-asymmetry defect at `bench_fused_vs_mxfp4.py:74-75` vs `:78-80`);
- one pinned `block_dim` (64) for us, no best-of-N sweep against the baseline's single shot (`bench_fused_vs_mxfp4.py:63`, `:93-101`);
- median of ≥7 per-launch event pairs, p05/p95 also reported (BENCHMARKS.md documents a timer glitch that read ~2× too fast on this box);
- identical output semantics — i.e. only after §4.1 has established that we and the baseline implement the same scale rule, rounding, and nibble order. **A speed claim against an operator whose semantics we have not matched is not a claim.**

**C4 — Beats the vendor MXFP4 op. DECIDED 2026-07-29 (was R1).** The research established that CANN's own `pto/npu/a5/TQuant.hpp` has **no MXFP4 path at all** (`QuantType` is only `{MXFP8, INT8_SYM, INT8_ASYM}`, `:22-27`; `static_assert(float32_t)` at `:302`, `:390`), so "beat AscendC" had no literal referent. **Resolution: the vendor contender is the CANN-shipped `aclnn` operator behind `torch_npu.npu_dynamic_mx_quant`**, called directly with preallocated outputs. That is the only genuine vendor MXFP4 implementation on this box, so C4 and C3 collapse into a single honest comparison against it.

Consequences: we do **not** author a `TQuant.hpp`-style AscendC kernel, and no table row is labeled "AscendC" — the row is named for the actual op. A PTO-ISA-vs-AscendC *programming-model* claim is explicitly out of scope; if it is ever wanted it needs its own comparison, and beating a kernel we wrote ourselves would not establish it.

**C5 — Fusion is worth it (Stage 2 only).** `t_fused < t_wht + t_quant` measured on the same box, by a margin > 1.15×. The theoretical ceiling on this speedup is **2.58×** (§5.1); anything measured above ~2.6× is a measurement error. §5.2 pre-registers the prediction that we may land *below* 1.15× because the fused kernel is vector-issue-bound, and states what we do if that happens.

**Decision rule if the baseline is already at the roofline.** If the first measurement shows `t_torch_npu ≤ 1.05 × t_dma_floor` at batch 65536, then quant-only has nothing left to win and C3 becomes "parity at the roofline". Say so plainly; do not chase noise. The value then moves entirely to C5 (fusion removes 61% of the traffic) and to small/medium batch.

---

## 2. Format contract

This section is the single source of truth. The host reference (`mxfp4_ref.py`) and the kernel are both written against it, and the test asserts they agree.

### 2.1 Element type: E2M1 ("fp4")

4 bits: sign[3], exponent[2:1] (bias 1), mantissa[0]. 16 codes, no Inf, no NaN.

| mag field | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |

`0b0001` (0.5) is the only subnormal. Code `0x8` is a valid **−0**: any byte-level comparator must treat `0x0` and `0x8` as numerically equal but **must not** accept one where the reference produced the other (we assert exact bytes, so the kernel must reproduce the reference's sign of zero — see §7 R7).

Since there are no Inf/NaN encodings, out-of-range magnitudes **saturate to ±6**. On device this is the hardware clip-to-MAX_NORM behaviour enabled by `set_ctrl(1u<<50)` (`fused_hadamard_mxfp4_a5.cpp:187`) — the only `set_ctrl` call anywhere in the repo, and therefore unproven (§7 R6).

### 2.2 Block size and scale type

- Block size **k = 32** elements, one shared scale per block, blocks are consecutive 32-element runs of the row. K must be a multiple of 32 so blocks never straddle rows.
- Scale is **E8M0**: unsigned 8-bit exponent, bias 127, value `2^(X-127)`. `X=0x00` is `2^-127` (E8M0 has **no zero** and **no Inf**); `X=0xFF` is **NaN**.

### 2.3 Scale selection and the bias arithmetic (the part that must be exact)

**Committed rule (default): OCP MX Spec v1.0 §6.3 Algorithm 1.**

```
shared_exp = floor(log2(amax))            amax = max_i |v_i| over the 32-element block
X          = 2^(shared_exp - emax_E2M1)   emax_E2M1 = 2  (largest normal 6.0 = 1.5 * 2^2)
byte       = shared_exp + 125             (= shared_exp - 2 + 127)
=> amax / X  ∈  [4, 8)
```

Consequence to accept by construction: elements in `[7,8)·X` clamp to 6, i.e. up to 25% error on the block maximum. This is what Algorithm 1 does; it is not a bug.

**Device implementation, derived from the source dtype's biased exponent field.** Let `b` = the biased exponent field extracted from the bit pattern of `amax`. Then, generally:

```
byte           = b + (127 - 2 - bias_src)
mult_exp_field = 2*bias_src + 2 - b            (mantissa field = 0 → exact power of two)
mult           = 1/X  exactly
```

| source | bias_src | byte | mult exp field | representable window for b |
|---|---|---|---|---|
| fp16 | 15 | `b + 110` | `32 - b` | `b ∈ [2, 31]` |
| bf16 | 127 | `b - 2` | `256 - b` | `b ∈ [2, 254]` |

Both magic constants in the WIP are therefore **exactly right** for fp16-normal `amax`: `110 = 127-2-15` and `32 = 2·15+2` (`fused_hadamard_mxfp4_a5.cpp:155-158`, comment at `:7`). Verified: `b=15` (amax ∈ [1,2)) → byte 125 → `X = 2^-2 = 0.25`, `mult = 2^(32-15-15) = 2^2 = 4`, `amax·mult ∈ [4,8)` ✓.

**The clamp rule (this is the confirmed defect, stated as the contract):** clamp `b` into the window **first**, then derive *both* the byte and the multiplier field from the clamped `b`. Never clamp the multiplier field. The window is not arbitrary — it is exactly the condition that `1/X` is a finite non-Inf, non-subnormal number in the multiply dtype. For fp16, `b=1` would give field 31 = `0x7C00` = +Inf and `b=0` would give field 32 = `0x8000` = −0.0, so *a* clamp is genuinely required.

**Documented deviations from OCP that follow from clamping (must be tested, not hidden):**
- fp16 path: `b` clamped to `[2,31]` ⇒ emitted byte ∈ `[112,141]` ⇒ `X ∈ [2^-15, 2^14]`, versus the format's `[2^-127, 2^127]`. Blocks with `amax < 2^-13` get a shared exponent floored at −15. The stored scale and applied multiplier remain **exact inverses**, so nothing is off by a power of two; the block simply uses fewer codes. Blocks with `amax < 2^-17` quantize entirely to zero.
- Reading `b=0` from an fp16-subnormal `amax` yields `e = -15` regardless of the true `floor(log2 amax) ∈ [-24,-15]`, so subnormal blocks get a wrong exponent even after the clamp is fixed. The clamp masks this.
- bf16 path: `b ∈ [2,254]` covers essentially the whole E8M0 range, so bf16 has no such deviation. **This is a reason to prefer a bf16-domain path (§4.4).**

**Optional alternate rule, behind `-DMX_SCALE_RULE=RCEIL`.** The OCP spec presents Algorithm 1 as *a* valid conversion, not the only one; the "max-fit / rceil" rule `X = 2^ceil(log2(amax/6))` (NVIDIA MXFP4/NVFP4 recipes, torchao `ScaleCalculationMode.RCEIL`) maps `amax/X ∈ (3,6]` and is also in the wild. A one-code difference in the byte is a **2× error**, so if the baseline uses rceil we must be able to switch. The implementation is two extra ops in the amortized pass:

```
rceil:  byte = b + (127-2-bias_src) + t,   mult_exp_field = 2*bias_src + 2 - b - t
        t = [ mantissa_field(amax) > (1<<(mant_bits-1)) ]     # i.e. significand m > 1.5
        (fp16: t = [mant > 512];  clamp b to [2,30] because field must stay ≥ 1)
```
Derivation: for `amax = m·2^e`, `m ∈ [1,2)`, `ceil(log2(m/6)) = -2` iff `m ≤ 1.5`, else `-1`. Check `amax = 6·2^k`: `m=1.5, e=k+2, t=0` → byte `= k+127` → `X = 2^k` → `amax/X = 6` exactly ✓.

**Default: FLOOR.** Switch only if §4.1 measures the baseline as rceil.

**NaN/Inf: `-DMX_NAN_CONFORMANT` (default 0).** With `vabs`/magnitude-masking, `b == 31` (fp16) or `b == 255` (bf16) holds **iff** `amax` is Inf or NaN, because the largest finite value has field 30 / 254. So a single `vcmps_eq` + `vsel` in the amortized pass can force `byte = 0xFF` (E8M0 NaN) for those blocks — CANN's own fp8 path uses exactly this pattern (`pto/npu/a5/TQuant.hpp:158-166`, `vb32_b8_nan`). Default is **off** (byte 141 / 253, block saturates to ±6·2^14) because the goal is to *match the baseline*, and we do not yet know what the baseline does. Flip the default once §4.1 answers it.

### 2.4 Nibble order — PINNED ONCE, ASSERTED, NEVER AUTO-FITTED

**Committed convention:**

```
byte[k] = (code[2k+1] << 4) | code[2k]        # element 2k → LOW nibble (bits 3:0)
```

This is torch's `float4_e2m1fn_x2` packing and CUTLASS `float_e2m1_t` array packing. The OCP spec does **not** define a sub-byte memory layout, so this is our choice, made once.

It is also what the ISA structure implies: `vcvt(vector_f4e2m1x2&, vector_bf16, ..., PART_Pk)` is a 4:1 in-lane narrowing (`__clang_cce_vector_intrinsics.h:5096-5115`), and `PART_P0`/`PART_P1` map to *ascending byte positions* within the b32 lane (established by `pto/npu/a5/TQuant.hpp:229-233`, which does `PART_P0` + `PART_P1` + `vor` + `PK_B32` to emit 128 bytes in order). By the same ascending rule, bf16 lane `2i` lands in the low nibble of byte `i`. **This is inference, not measurement.**

**Rules, non-negotiable:**
1. A single deterministic probe (§4.1, `probe_conventions.py`) settles it: one 32-element block whose 32 elements map to distinct/known E2M1 codes, run the kernel, compare all 16 output bytes exactly. A wrong order is loudly detectable (it swaps adjacent elements; `l2_rel ≈ √2 ≈ 1.4` on random data vs ≈0.12 when correct).
2. The measured answer is written into `README.md` with the date and stack versions, and hard-coded in `mxfp4_ref.py` as a module constant with no alternate branch.
3. The test asserts it. **No `for order in (True, False)` loop. No "keep whichever scores better".** This is the specific unfalsifiability defect in `test_fused_mxfp4.py:107` and `sim_fused/main.cpp:91-106`.
4. Correction to the review brief, so we fix the right thing: `bench256_fused.py:47` (`nib[:,0::2]=lo`) is the **same** convention as `test_fused_mxfp4.py`'s `order=True` and `sim_fused`'s `ord==0`, **not** the opposite. All three agreed; the defect is that two of them would have accepted either. Do not "fix" `bench256_fused.py` by flipping it.

### 2.5 Element rounding — and the double-rounding problem

**Rounding mode: round-to-nearest-even on the E2M1 grid, then saturate.** Because the grid is non-uniform, the decision points and tie targets are worth hard-coding into the reference (ties go to the neighbour with an even 3-bit magnitude field):

| midpoint | 0.25 | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 | 5.0 | >6 |
|---|---|---|---|---|---|---|---|---|
| RNE target | **0.0** | 1.0 | 1.0 | 2.0 | 2.0 | 4.0 | 4.0 | 6.0 |

The `0.25 → 0` tie is the one most reference implementations get wrong.

**The cast chain is a double rounding, and this is a real numeric deviation.** On A5 the only available narrowing cast to fp4 takes **bf16** (`__VF_VCVTFF_RND_PP(bf16, f4e2m1x2)` at `__clang_cce_vector_intrinsics.h:5096-5115`; the f16→fp4 form exists but is **commented out** in the header). So an fp16 path must go `f16 → bf16 → f4e2m1`, two roundings. Every E2M1 midpoint (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) is exactly bf16-representable, so a half-bf16-ulp band on one side of each midpoint rounds *onto* the midpoint and then takes the tie rule instead of the correct direction. Example: f16 `2.5009765625` → bf16 `2.5` → tie → `2.0`, where single rounding gives `3.0`. Rate on random data: roughly **0.1–1%** of elements one code off a single-rounded reference.

**Consequences we commit to:**
- `mxfp4_ref.py` models the **exact same chain**, step by step, in the same dtypes: `f16 multiply (RNE) → bf16 (RNE) → e2m1 (RNE, saturate)`. Against that reference the gate is **bit-exact**, not a tolerance.
- Against torch_npu (which very likely rounds from f32 in one step) bit-exactness is **unattainable on the fp16 path**. The cross-check against the baseline is therefore a **mismatch histogram** with two hard assertions: scale bytes 100% identical, and every nibble mismatch is at most ±1 code, at a rate below a threshold pinned from the first measurement. Anything outside ±1 code is a hard failure.
- **`ROUND_R` is assumed to be RNE and this is unverified** (§7 R5). If the probe shows otherwise, we change the reference to match the hardware — we do **not** loosen the gate.

### 2.6 The one format-level fact that buys real performance

E2M1 codes depend only on the ratio `v_i / amax`. Therefore **any positive scalar prefactor is invisible to the codes**, and a *power-of-two* prefactor shifts only the stored byte, by an exact integer. Formally, if the kernel quantizes `z = y · 2^p` where `y` is what it actually has in registers:

```
byte = b_y + (127 - 2 - bias_src) + p       mult_exp_field = 2*bias_src + 2 - b_y   (UNCHANGED)
```

The multiplier depends only on `b_y`; the prefactor lives entirely in the byte. This makes the orthonormal `1/sqrt(N)` scale **free and exact** for `N = 4^k` (N=256: subtract 4 from the byte, delete the `vmuls` entirely) and reduces N=128 to a single `sqrt(2)`, foldable into the multiplier's mantissa at zero extra full-width cost (§5.3). It also gives us free f16 headroom in the fused kernel.

### 2.7 Output layout

- `q`: `(batch, K/2)` `uint8`, contiguous, C-order, nibble order per §2.4.
- `scale`: `(batch, K/32)` `uint8`, contiguous, C-order, one packed E8M0 byte per block. **Not** the WIP's padded 16-bit-per-block 32-byte slot (`fused_hadamard_mxfp4_a5.cpp:55`, `:158`), which leaves 24 of every 32 bytes uninitialized in UB and still `TSTORE`s them to the caller's tensor, and inflates writes by 41%.
- **Open:** if the eventual consumer is an MXFP4 matmul that wants a swizzled/blocked scale layout, this changes and the write-traffic accounting changes with it (§7 R9). Default to the packed layout, which is also what makes a direct tensor-vs-tensor comparison against the baseline possible.

---

## 3. Stage 0 — prerequisites

Stage 0 is one on-device session plus one decision. Nothing in Stage 1 should be written before §3.3 completes.

### 3.1 Confirmed defects: fix, or deliberately leave behind

| # | Defect | Location (branch `fast-hadamard-a5-register`) | Disposition |
|---|---|---|---|
| D1 | `vmins(mf,30)` clamps the **multiplier field** while the byte is derived from the **unclamped** `bd`, so stored scale and applied multiplier stop being exact inverses. Dequant is exactly **2× too small at bd=1** (amax ∈ [2^-14,2^-13)) and **4× too small at bd=0** (amax fp16-subnormal). All-zero blocks are accidentally safe (0·anything = 0). | `fused_hadamard_mxfp4_a5.cpp:156` vs `:157`; same pair at `fused_hadamard256_mxfp4_a5.cpp:129-130` vs `:131` | **FIXED by contract §2.3**: clamp `b` first, derive both from it. The existing `vmaxs(mf,1)` at `:155` is already dead code (`vand 0x1F` guarantees `bd ≤ 31` hence `mf ≥ 1`) and is deleted. |
| D2 | Both harnesses **search both nibble orders and keep the better score**, so a packing bug is unfalsifiable and gets reported OK under a different label. | `test_fused_mxfp4.py:107`; `sim_fused/main.cpp:91-106` | **NOT CARRIED.** New dirs start clean. §2.4 pins one order and asserts it. (Note: `bench256_fused.py:47` is the *same* order as both candidates, not the opposite — the brief is wrong on that detail.) |
| D3 | Pooled-buffer benchmark reuses buffers across ~44 in-place launches of an **unnormalized** transform, so timed data saturates to inf/NaN. | `bench256_grid.py:57-79` (8 + 7·50 = 358 launches / POOL=8 = 44.75 per buffer; fp16 overflows after 4 passes at gain √256=16); `bench256.py:73-87` (13.75 per buffer) | **NOT APPLICABLE** to the quant kernels (out-of-place: read `x`, write separate `q`/`s`), so do not spend effort re-fixing it here. The new harness nevertheless **asserts every pooled input buffer is bitwise unchanged** after timing (§4.6), which makes the whole class impossible to reintroduce. |
| D4 | Scripts compute an ok/FAIL verdict, print it, then exit 0. | `test_fused_mxfp4.py:121-126` (computes `ok`, never uses it, `main()` returns None); `bench256_fused.py:82` | **NOT CARRIED.** All verdicts go through `pytest` asserts; any standalone script ends `sys.exit(1)` on failure. Scoping correction: `sim_fused/main.cpp:125` **does** `return best_l2 < 0.25 ? 0 : 1` — it is a real gate, just a weak one (best-of-two orders, and the 0.25 threshold barely exceeds MXFP4's own intrinsic ~0.10–0.15 L2). |
| D5 | Pass criterion is structurally blind. Post-WHT amax of 32 normalized Gaussians has `bd ≈ 16`, so `bd ∈ {0,1}` never occurs and D1 cannot move a single-threshold `l2 < 0.25` test. Correct MXFP4 on Gaussian data lands at `l2_rel ≈ 0.10–0.14`. | `test_fused_mxfp4.py:121`; `sim_fused/main.cpp:122` | **REPLACED** by bit-exact comparison + adversarial blocks (§4.5). |
| D6 | Degenerate 2-buffer pipeline: `set_flag(MTE2,V)` immediately followed by `wait_flag(MTE2,V)` on the same buffer in the same iteration, so DMA never overlaps compute. | `fused_hadamard_mxfp4_a5.cpp:221-223`; `fused_hadamard256_mxfp4_a5.cpp:177-178` | **SUPERSEDED** by the `ISSUE_LOAD(k+PREFETCH)`-before-wait pattern that already exists at HEAD in `fast_hadamard_256_a5.cpp:102-136` (NBUF=4, PREFETCH=2, 8 event IDs). BENCHMARKS.md records this as worth 2.2 → 2.7 TB/s at batch 64k. |
| D7 | Padded 32-byte int16 scale slot: 24 of every 32 bytes never written yet still DMA'd; 8× scale write amplification; not an MX-consumable layout. | `fused_hadamard_mxfp4_a5.cpp:55`, `:158`, `:245`; readers slice `s_u16[:, :NBLK]` at `test_fused_mxfp4.py:76` | **SUPERSEDED** by packed uint8 + a single aligned `PK_B16` store per group (§4.3). |
| D8 | `PHASE_SEL` compiles the quant phase out entirely while the benchmark table keeps the same label, so a stale shell variable manufactures a winning row. Defined mid-loop inside the kernel body. | `fused_hadamard_mxfp4_a5.cpp:225-236`; driven by `bench_fused_vs_mxfp4.py:25` | **SUPERSEDED**: ablations become **separately named kernels** (`mxfp4_reduce_only`, `mxfp4_cast_only`) in their own TU with their own exported symbols, so a stale env var cannot mislabel a row. Never used for reported numbers. |
| D9 | `static_assert(ROWS_PER_TILE % HAD_UNROLL == 0)` guards the wrong quantity while the row loop hardcodes 8 → latent OOB at `HAD_UNROLL=2, ROWS_PER_TILE=2`. `tiles = batch / ROWS_PER_TILE` silently drops the tail. | `fused_hadamard_mxfp4_a5.cpp:72` vs `:87`, `:75`; `:206`; `fused_hadamard256_mxfp4_a5.cpp:168` (no assert at all) | **FIXED**: assert on the quantity the loop actually uses, and pad the batch in the Python wrapper (the `matmul_swizzle` convention already used at `jit_util_hadamard256_a5.py:145-158`). |
| D10 | Shared broadcast temp `hi` across all 8 unrolled bodies creates a WAW/WAR chain that serializes the 40 `vintlv` ops the unroll exists to overlap. ~88 live vectors in the N=128 quant phase (must spill); `DOU4` defined for exactly this and never used. | `fused_hadamard_mxfp4_a5.cpp:139`, `:145`–`:171`, `:76` | **SUPERSEDED**: the `vintlv` broadcast chain is replaced entirely by a single `vlds(..., E2B_B16)` load-pipe op (§4.4), so the temp does not exist. |
| D11 | `__init__.py` present in `examples/jit_cpp/fast_hadamard_a5` at HEAD breaks the documented `pytest test_hadamard256_a5.py` invocation: pytest walks up to `examples/jit_cpp` for the sys.path basedir and the flat `from jit_util_hadamard256_a5 import ...` (`test_hadamard256_a5.py:16`) raises `ModuleNotFoundError`. Proven by copying the dir with stub torch: as-is → error; after deleting `__init__.py` → 8 tests collected. | `examples/jit_cpp/fast_hadamard_a5/__init__.py` (present at `fef2cae`) | **NOT REPRODUCED.** Neither new dir gets an `__init__.py`. No sibling example has one. |
| D12 | Unsafe artifact cache key: artifact named `fht256_a5_r{rows_per_tile}.o/.so`, staleness test is only `so.st_mtime >= src.st_mtime`. `nbuf`, `prefetch`, and any other `-D` are **not in the key**, so changing NBUF/PREFETCH silently reuses a stale `.so`. | `jit_util_hadamard256_a5.py:57-62` | **FIXED** in both new `jit_util`s: key on `sha256(src bytes + repr(full flag list))`. This will silently invalidate benchmark conclusions if not fixed. |
| D13 | The `__global__` body is wrapped in `#ifdef __DAV_VEC__`. Compile with anything but `--cce-aicore-arch=dav-c310-vec` (e.g. the Makefile's `compile_a5_%` target, which uses `dav-c310` with **no** `-vec`) and you get an empty kernel that links, launches, and returns unmodified data. | `fast_hadamard_256_a5.cpp:51`, `:90`, `:138`; repo `Makefile` `compile_a5_%` | **GUARDED**: every test asserts non-trivial output (a zeroed/unchanged `q` buffer must fail). Matches the user's own memory note on the host/device-pass guard. |

### 3.2 What is lifted vs superseded

**Lifted (arithmetic and ideas, retyped into the new kernels; read via `git show fast-hadamard-a5-register:examples/jit_cpp/fast_hadamard_a5/<file>`):**
- exponent extraction from the fp16 bit pattern by integer ops — `fused_hadamard_mxfp4_a5.cpp:150-151` (drop the redundant `vand 0x1F`: `vabs`/magnitude-mask already cleared the sign, so the shift is clean);
- the bias arithmetic `byte = bd+110`, `field = 32-bd`, and the exponent-field construction of the reciprocal (`vsub` then `vshls 10`) — `:155-158`. **Never** `vrec`/`vdiv`: both exist on A5 but are approximate, which would make the stored byte and the applied multiplier non-exact inverses, i.e. exactly D1's failure class by another route;
- the bf16 pivot and the `PK4_B32` 64-byte fp4 store — `:169-170`, predicate at `:126`;
- `set_ctrl(1u<<50)` for clip-into-MAX_NORM — `:187` (pending §7 R6);
- from HEAD, `fast_hadamard_256_a5.cpp:102-136` — the `ISSUE_LOAD`/NBUF/PREFETCH software pipeline, verbatim;
- from HEAD, `copy_ref_256_a5.cpp` — the pattern for a separate-TU DMA-floor twin with its own `static_assert` on the UB budget.

**Superseded, do not carry:** the `vdintlv(cg,cg)+vmax` combine (`:148-149`) → replaced by the DINTLV-**load** trick, which needs zero combine steps; the 5×`vintlv` 32× broadcast (`:162-165`) → replaced by one `vlds E2B_B16`; the padded scale slot; the 2-buffer pipeline; `PHASE_SEL`; and the stale header comment claiming "alignment-exempt ONEPT stores" when the code uses `NORM_B16` (`:53-54` vs `:158`) — a design-drift signal, plus dead `NBLK` at `:52` and dead `DOU4` at `:76`.

**Not recreated:** `test_fused_mxfp4.py`, `bench256_fused.py`, `bench_fused_vs_mxfp4.py`, `bench_mxfp4_baseline.py`, `sim_fused/`, `sim_test/`. **The new dirs start clean.** No sim harness is tracked in any example at HEAD; whether to recreate one is R11.

### 3.3 Stage 0 deliverable: `probe_conventions.py` and a pinned `CONVENTIONS` record

One on-device script whose output is committed into `README.md` (with date, CANN version, torch/torch_npu versions, driver, SoC, NPU index). Everything downstream depends on it. Details of what it measures are in §4.1. **Do not write Stage 1's reference or test before this has run**, because the answers determine the reference's scale rule, rounding model, nibble order, and output dtypes.

---

## 4. Stage 1 — `examples/jit_cpp/mxfp4_quant_a5`

### 4.1 First: measure the baseline's semantics (`probe_conventions.py`)

The only baseline the repo ever calls is `torch_npu.npu_dynamic_mx_quant(x, block_size=32, dst_type=296)`, invoked identically at `bench_mxfp4_baseline.py:44`, `bench_fused_vs_mxfp4.py:80`, `bench256_fused.py:95`. `296` is an undocumented magic integer with **zero** other occurrences in the repo; it is not an `aclDataType` (`FLOAT4_E2M1` is a small enum) and not a `torch::ScalarType`. Every byte-count constant derived from it (`TOTAL_B=324` at `bench_mxfp4_baseline.py:11-12`) is currently *assumed*, never verified — the script fetches `(q, sc)` and prints shapes at `:52-54` but never asserts. If `296` silently selected an 8-bit destination, every published ratio is wrong in our favour.

`probe_conventions.py` answers, in one run:

1. **Signature and dtype table.**
   `print(torch.ops.npu.npu_dynamic_mx_quant.default._schema)`; then
   `grep -rn 'dst_type' $(python3 -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))")` to find the int→dtype mapping for `296`.
2. **Output layout.** `q.shape/.dtype/.stride()`, `scale.shape/.dtype/.stride()`. Assert them (in particular: is the scale `uint8`, `int8`, or `torch.float8_e8m0fnu`? is it padded?).
3. **Scale rule: FLOOR vs RCEIL.** Blocks with `amax ∈ [7,8)·2^k` → byte `k+125` (floor) vs `k+126` (rceil). Also `amax = 6·2^k`, `1.5·2^k`, `1.0·2^k`.
4. **Rounding mode.** Blocks containing exactly `(0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)·X` → RNE vs round-half-away vs truncate. Include the `0.25` tie specifically.
5. **Nibble order.** One block of 32 elements whose codes are all known and pairwise-distinguishable; compare all 16 bytes exactly.
6. **Clamp window / underflow.** `amax` at `2^-15`, `2^-14`, `2^-13`, and fp16-subnormal; all-zero block; single-nonzero block.
7. **NaN/Inf.** Does the baseline emit `0xFF`?
8. **`aclnn` entry point** (for C4 option (a)):
   `ls $ASCEND_HOME_PATH/include/aclnnop | grep -i mx` and
   `nm -D --defined-only $ASCEND_HOME_PATH/lib64/libopapi.so | grep -i -e MxQuant -e DynamicMx`.
   CANN convention is two-phase `aclnn<Op>GetWorkspaceSize(...)` then `aclnn<Op>(ws, ws_size, executor, stream)`, which lets us pass **preallocated** outputs — the only way to compare "the operator itself" symmetrically.
9. **ISA probes** (a tiny throwaway `.cpp` + ctypes, or a scratch kernel in this dir): `vcgmax` f16 group size and output lane packing (load lane index `i` as f16, `vcgmax`, store 8 results, expect `15,31,47,…,127` in lanes 0..7); `ROUND_R` tie behaviour on `f16→bf16` and `bf16→fp4`; whether `vmul` has a bf16 form; the exact `vsts` overload incantation for `PK_B16` of a `vector_u16` and `PK4_B32` of fp4 (see §4.4 note).

**Output:** a committed table. Until it exists, the plan's defaults (§2) stand, and every one of them is marked as a default-to-be-confirmed rather than a fact.

### 4.2 File list (matches repo conventions, verified against `fef2cae`)

`/Users/hyunmin/Projects/Work/Huawei/pto-kernels/examples/jit_cpp/mxfp4_quant_a5/`

| file | role |
|---|---|
| `README.md` | H1 `mxfp4_quant_a5 — <one line>`; prose on the ISA trick; `## Files`; `## Build & run`; `## Notes` (format contract summary, the documented deviations, the measured headline vs the copy floor, the pinned conventions table from §4.1). Mirrors `fast_hadamard_a5/README.md` and `causal_conv1d/README.md`. |
| `mxfp4_quant_a5.cpp` | the kernel. Exports `extern "C" void call_mxfp4_quant(uint32_t bd, void* stream, uint8_t* x, uint8_t* q, uint8_t* s, uint32_t batch)`. |
| `copy_ref_mxfp4_a5.cpp` | DMA-floor twin: identical TLOAD/TSTORE pattern and identical output byte counts, **zero** vector-execute work. Exports `call_copy_mxfp4_floor`. Own TU with its own UB `static_assert`, per `copy_ref_256_a5.cpp:38-39`. |
| `mxfp4_ablate_a5.cpp` | separately-named ablation kernels `call_mxfp4_reduce_only` / `call_mxfp4_cast_only`, replacing `PHASE_SEL` (D8). |
| `jit_util_mxfp4_quant_a5.py` | standalone A5 bisheng build + ctypes load; hash-keyed artifact cache (D12); batch padding wrapper; `argtypes` one-per-line with trailing comments (the `jit_util_layernorm.py:26-37` style). |
| `jit_util_copy_ref_mxfp4_a5.py` | same for the floor twin. |
| `mxfp4_ref.py` | the host reference (numpy only, no torch import): `e2m1_encode`, `e2m1_decode`, `quantize_block`, `quantize(x, in_dtype, scale_rule, prescale_exp)`, `dequantize`, `pack_nibbles`, `unpack_nibbles`. One pinned nibble order as a module constant. **The single source of truth**; Stage 2 imports it. |
| `conftest.py` | byte-identical copy of the guarded 993-byte variant (`causal_conv1d/conftest.py`, md5 `65dc45eb71b21cc5324a4d5972f802b6`): `--npu` addoption wrapped in `try/except ValueError`, session `npu_device` fixture, autouse `torch.npu.set_device`. `fast_hadamard_a5` lacks this, which is why its test hardcodes `.npu()`. |
| `test_mxfp4_quant_a5.py` | the gate. `torch = pytest.importorskip("torch")`, `pytest.importorskip("torch_npu")`, then `from mxfp4_ref import ...` and `from jit_util_mxfp4_quant_a5 import ...` with `# noqa: E402`. Real `assert`s only. |
| `benchmark.py` | the harness (§4.6). Emits CSV to `build/`. |
| `plot_mxfp4_quant_a5.py` | matplotlib-only plot from the CSV; generated PNGs go to the companion `pto-kernels-plots` repo (keeps us under pre-commit's 500 KB `check-added-large-files`). |
| `probe_conventions.py` | §4.1. |
| `run_benchmark.sh`, `run_probes.sh` | mode **755**, `#!/usr/bin/env bash`, `set -euo pipefail`, `SCRIPT_DIR=...`, `: "${ASCEND_HOME_PATH:=...}"`, `cd "$SCRIPT_DIR"`, `exec python3 ...`. |
| `.gitignore` | `build/`, `outputs/`, `__pycache__/`, `*.pyc`, `*.so`, `*.csv`, `*.png`. |
| — | **NO `__init__.py`** (D11). **No shebangs on `.py`** (ruff EXE001 fires on `fast_hadamard_a5/benchmark.py`, mode 644 with a shebang). |

### 4.3 Tiling, memory layout, and pipeline

**Shapes.** `x: (batch, K)` fp16 or bf16, contiguous. `q: (batch, K/2)` uint8. `s: (batch, K/32)` uint8. Because rows are contiguous and `K % 32 == 0`, the flat 32-element runs of `x` coincide exactly with per-row MX blocks, and a tile of `R` rows maps to `R·K/2` contiguous `q` bytes and `R·K/32` contiguous `s` bytes — so all three tensors are single 1-D `TLOAD`/`TSTORE` tiles.

**Constraints (all `static_assert`ed):**
- `K % 256 == 0`, **or** `K == 128` with `R` even. `vlds(..., DINTLV_B16)` consumes 512 contiguous bytes = 256 fp16 = exactly 8 MX blocks (`__VF_VLDSX2` at `__clang_cce_vector_intrinsics.h:1566-1580`; `nElemPerVlds = CCE_VL/BLOCK_BYTE_SIZE = 8` at `pto/npu/a5/TRowExpand.hpp:97`), so the tile's element extent must be a multiple of 256.
- `GROUP = 4096` elements (= 128 blocks) is the amortization unit for the scale math; require `R*K % GROUP == 0`.
- UB budget: `NBUF * align512(R*K*2 + R*K/2 + R*K/32) + scratch ≤ UB_USABLE_BYTES` (default 192 KB, overridable per `fast_hadamard_256_a5.cpp:42-44`).

Worked example, `K=256, R=64, NBUF=4`: per buffer `32768 (x) + 8192 (q) + 512 (s) = 41472 B` → aligned 41472 → ×4 = 165 888 B, plus a **single shared** scratch (maxima 512×2 B + duplicated multipliers 1024×2 B ≈ 3 KB) — the scratch can be shared because all vector work for a tile completes serially in the one V stream before the next tile's vector work begins. Total ≈ 169 KB, fits 192 KB. For `K=4096`, `R=1` (one row = exactly one GROUP); for `K=2048`, `R=2`; for `K=128`, `R≥32`. The Python side picks `R` so that `R*K*2 ≈ 32 KB` subject to `R*K % 4096 == 0`.

**Pipeline: copy `fast_hadamard_256_a5.cpp:102-136` verbatim.** `NBUF=4`, `PREFETCH=2`, 8 distinct event IDs, `ISSUE_LOAD(k+PREFETCH)` issued **before** `wait_flag(MTE2,V)` on tile `k`. One `MTE3→MTE2` token per buffer covers `x`, `q`, and `s` since they share the buffer slot. Per tile: `wait(MTE3→MTE2)` → `TLOAD x` → `set/wait(MTE2→V)` → passes A/B/C → `set/wait(V→MTE3)` → `TSTORE q`, `TSTORE s` → `set(MTE3→MTE2)`.

`NBUF=4` is optimal, but **not** for the reason `BENCHMARKS.md` gives. That file blames the NBUF>4 device fault (error 507035) on one event ID per buffer serving all three pipe handoffs — that is **wrong**. The real cause was a fixed `unsigned xoff[4]` table indexed by `K % NBUF`, i.e. an out-of-bounds read; `ev[8]` was always correctly sized. Fixed in PR #221 (`60338d9`) by computing offsets with the `XOFF()` macro. **Device-verified 2026-07-29:** with that fix NBUF=6 runs correctly and is ~1% *slower* than NBUF=4 (2634 vs 2668 GB/s at batch 65536, ROWS_PER_TILE=64), and raising the UB budget to the physical 248 KB changes nothing (2622 GB/s). The transform is HBM-bound, so depth beyond 4 buys nothing — sweep NBUF freely, but treat a win as surprising.

**`mem_bar(VST_VLD)`** between pass A and pass B, and between pass B and pass C — pass A writes the maxima scratch and pass B reads it; pass B writes the multiplier scratch and pass C reads it. This is the barrier the WIP N=128 kernel was missing (its only `mem_bar` is `VST_VLD` at `fused_hadamard_mxfp4_a5.cpp:115`, with nothing protecting tile *n*'s quant reads of `W` from tile *n+1*'s transform writes).

### 4.4 Inner loop — three passes

The structural insight that makes this cheap: after a deinterleave load, `vlds(e, o, ptr, 0, DINTLV_B16)` puts `src[2j]` in `e[j]` and `src[2j+1]` in `o[j]`, so `m[j] = max(|src[2j]|, |src[2j+1]|)` and one **32-byte hardware block** of `m` (16 lanes) covers **exactly 32 source elements = exactly one MX block**. `vcgmax` reduces each 32-byte block to one value, compacted into lanes 0..7 (`__VF_VCG_P_OP` at `__clang_cce_vector_intrinsics.h:7957-8032`, guard includes `__NPU_ARCH__==3510`; A5 is `__NPU_ARCH__ == 3510` per `bisheng -dM -E --cce-aicore-arch=dav-c310-vec`). So **8 block maxima per 512 input bytes with zero combine steps** — this replaces the WIP's `vdintlv(cg,cg)+vmax` pair.

And the inverse primitive lives on the **load pipe**: `vlds(dst, ptr, off, E2B_B16)` reads 8 consecutive u16/f16 and broadcasts element `j` to all 16 lanes of the `j`-th 32-byte block (`pto/npu/a5/TRowExpand.hpp:97,110,122`; `Dist` enum at `__clang_cce_vector_intrinsics.h:78-79`; availability for 3510 at `:1068-1073`). That replaces the WIP's 5-op `vintlv` broadcast chain (and D10's serializing shared temp) with **one load-pipe op**. There is no cheaper register-to-register alternative: `vcp` (CHN4TO8/16/32 channel-copy upsample) and `vcbmax` are both **excluded** on 3510 (`:6433-6450`, `:7244-7265`), and `vgatherb` takes a `__ubuf__` base, i.e. it is a UB gather, not a register permute (`:2860-2935`, `:2976-3045`).

**Unify the two input dtypes in the integer domain.** For non-negative values the u16 bit pattern of both fp16 and bf16 is monotone in magnitude, so magnitude-masking with `vand 0x7FFF` plus **integer** `vmax`/`vcgmax` on `u16` (available on 3510 for u16/s16) gives the block max as a bit pattern, for either dtype, with the same op count as `vabs`+`vmax`. This also means NaN/Inf are picked up correctly (their patterns are the largest). Exponent extraction is then `vshrs 10` (fp16) or `vshrs 7` (bf16).

**Pass A** — per 512 input bytes (256 elements), 16 iterations per GROUP:

```
vlds(e, o, xb + off, 0, DINTLV_B16);            // 256 elems -> even/odd, 128 lanes each
vand(ae, (u16&)e, c7FFF, pAll);                 // magnitude mask (dtype-agnostic)
vand(ao, (u16&)o, c7FFF, pAll);
vmax(m, ae, ao, pAll);                          // u16 max == magnitude max
vcgmax(cg, m, pAll);                            // cg[0..7] = 8 block maxima (patterns)
vstus(alnReg, 8, cg, mbuf);                     // accumulate 16 B into the align register
```
…then once per GROUP: `vstas(alnReg, mbuf, 0);` → one aligned 256-byte flush of 128 maxima.

Cost: **4 exec + 2 ld/st per 256 elements.**

*Why `vstus`/`vstas`:* 8 f16 maxima = 16 bytes, below the 32-byte `NORM` alignment quantum. The alignment-register accumulate/flush pair is exactly what CANN's own reducer uses (`__VF_VSTUS` at `__clang_cce_vector_intrinsics.h:2280-2310`, `__VF_VSTAS` at `:2500-2520`; used at `pto/npu/a5/TQuant.hpp:120-124`). `ONEPT_B16` stores are the alignment-exempt fallback. **Confirm `vstus` semantics with count=8 by compile+probe before relying on it** (§7 R8).

**Pass B** — once per GROUP (128 blocks / 4096 elements), so amortized to ~0.002 exec/element:

```
vlds(bm, mbuf, 0, NORM);                        // 128 block-max patterns
vshrs(bd, bm, SHIFT, pAll, MODE_ZEROING);       // SHIFT = 10 (fp16) | 7 (bf16)
vmaxs(bd, bd, BLO, pAll);                       // *** THE FIX: clamp b, not mf ***
vmins(bd, bd, BHI, pAll);                       //     fp16 [2,31] | bf16 [2,254]
                                                //  (RCEIL: also -1 on BHI, and += t)
vadds(sb, bd, BYTE_BIAS, pAll);                 // BYTE_BIAS = 110 + PRESCALE_EXP - L  (fp16)
vsts(sb, sbuf, grp*128, PK_B16, pAll);          // 128 packed e8m0 bytes, ALIGNED
vsub(mf, cMULT, bd, pAll);                      // MULT const = 32 (fp16) | 256 (bf16)
vshls(mf, mf, SHIFT, pAll);                     // exact power-of-two multiplier
vintlv(d0, d1, mf, mf);                         // duplicate each multiplier x2
vsts(d0, mbuf2,   0, NORM_B16, pAll);
vsts(d1, mbuf2, 128, NORM_B16, pAll);
```

Note how this kills D7 outright: 128 e8m0 bytes emitted as **one aligned 128-byte `PK_B16` store**, no padded slot, no `ONEPT`, no uninitialized bytes. (`PK_B16` takes the low byte of each of 128 b16 lanes → 128 bytes; `DistVST` enum at `__clang_cce_vector_intrinsics.h:91-108`.)

The `vintlv(mf, mf)` duplication exists so that a natural-order 128-lane register — 128 consecutive elements = 4 MX blocks, each spanning **two** 32-byte hardware blocks — can be served by a single `E2B_B16` broadcast whose 8 scalars are `[m0,m0,m1,m1,m2,m2,m3,m3]`.

**Pass C, variant V3 (natural-order; recommended default)** — per 256 elements:

```
vlds(v0, xb + off,       0, NORM);              // 128 elems, natural order
vlds(v1, xb + off + 128, 0, NORM);
vlds(b0, mbuf2, boff,     E2B_B16);             // 4 mults, each replicated over 2 hw blocks
vlds(b1, mbuf2, boff + 8, E2B_B16);
vmul(v0, v0, b0, pAll);                         // fp16 (or bf16, see below)
vmul(v1, v1, b1, pAll);
vcvt(c0, v0, pAll, ROUND_R);                    // fp16 -> bf16   [SKIP on the bf16 path]
vcvt(c1, v1, pAll, ROUND_R);
vcvt(q0, c0, pAll, ROUND_R, PART_P0);           // bf16 -> f4e2m1x2, 64 B payload
vcvt(q1, c1, pAll, ROUND_R, PART_P0);
vsts((vector_u8&)q0, qb + qoff,      0, PK4_B32, p64);   // 64 B
vsts((vector_u8&)q1, qb + qoff + 64, 0, PK4_B32, p64);
```

Cost: **6 exec + 6 ld/st per 256 elements** (fp16 path). On the bf16 path the two `f16→bf16` casts vanish → **4 exec + 6 ld/st**, and the rounding becomes single.

**Pass C, variant V1 (DINTLV; the ISA lens's original)** — re-reads with `DINTLV_B16` (1 load for 2 registers), one `E2B_B16` broadcast serving **both** `e` and `o` (because `e[j]`/`o[j]` both live in source block `j/16`, exactly the `E2B_B16` layout), 2 `vmul`, one `vintlv` to restore natural order, then the same 4 casts and 2 stores: **7 exec + 4 ld/st**.

So V3 trades **1 exec op for 2 ld/st ops** versus V1. Which wins depends on which pipe binds (§4.7). **A/B them at identical unroll; do not guess.**

**Store variant to test:** `PART_P0` + `PART_P1` into two registers, `vor`, then one `PK_B32` 128-byte store (the `pto/npu/a5/TQuant.hpp:229-233` pattern) is 3 exec + 1 st versus 2 exec + 2 st — the same total, shifting one op from ld/st to exec. Prefer 2× `PK4_B32` if exec binds.

**Two `vsts` overload gotchas that cost the ISA lens a compile error, both resolved by compiling, not reasoning:** the base pointer's element type must match the source vector's **lane** type regardless of the pack dist, and `offset` is in units of the **pointer's** element type, not bytes (`__VF_VSTS` at `__clang_cce_vector_intrinsics.h:1960-1990`: `offset * sizeof(LT)`). For `PK4_B32` of fp4 the working form was to cast the register to `vector_u8` with a `__ubuf__ uint8_t*` base. The `PK_B16`-of-`vector_u16` form (128 b16 lanes → 128 bytes) needs the same treatment determined by probe.

**Micro-optimizations deliberately NOT taken (with reasons, so they are not rediscovered):**
- Replacing `vmul` by a power-of-two exponent-field **integer add** on the u16 pattern: same op count (1), and it is *wrong* for exact zeros (`0x0000 + (k<<10)` becomes a normal number) and for subnormal inputs (no implicit leading 1). Zero gain, real hazard.
- Keeping the maxima register-resident across a GROUP to skip the UB round trip: would need 32 live data registers plus temps for a 4096-element group, and there is no pinned lane-pack primitive to compact eight 8-lane `vcgmax` results into one register. Revisit only if profiling shows the scratch round trip matters.

### 4.5 The host reference and the correctness gate

**`mxfp4_ref.py`** — numpy only, no device, no torch. Two independent references, which is the point:

1. **Bit-chain reference** (what the gate compares against): reproduces the kernel's exact sequence — magnitude-mask, extract `b`, clamp, `byte = b + BYTE_BIAS`, build the multiplier from its **exact bit pattern**, `scaled = np.float16(v) * np.float16(mult)` (RNE), `bf = to_bf16_rne(scaled)`, `code = e2m1_rne_saturating(bf)`, `byte[k] = (code[2k+1]<<4) | code[2k]`.
2. **Spec reference** (independent cross-check of the bias arithmetic only): in float64, `byte = floor(log2(amax)) + 125`, clamped to the same window. Asserted equal to reference 1's bytes for every block whose `amax ∈ [2^-13, 65504]`. This catches a bias-constant error that reference 1 would happily reproduce because it shares the kernel's arithmetic.

**`test_mxfp4_quant_a5.py`** — pytest, real asserts, exits nonzero on failure (D4). Parametrized over `dtype ∈ {fp16, bf16}`, `K ∈ {128, 256, 2048, 4096}`, `batch ∈ {64, 1000, 4096, 65536}` (including non-tile-multiples to exercise the padding wrapper, per `test_hadamard256_a5.py:41`).

| test | assertion |
|---|---|
| `test_nibble_order_pinned` | one crafted block with 32 known distinct codes → **all output bytes exactly equal** the pinned packing. Single convention, no loop. This is the D2 regression. |
| `test_scale_bytes_bitexact` | scale bytes **100%** equal to reference, on random and adversarial input. |
| `test_codes_bitexact` | fp4 nibbles **100%** equal to the bit-chain reference. |
| `test_spec_cross_check` | reference-1 bytes == reference-2 (f64) bytes in the valid window. |
| `test_output_is_nontrivial` | `q` is not all-zero and not equal to the pre-launch buffer contents — catches the silent-no-op arch-flag failure (D13). |
| `test_adversarial_blocks` | the block set below. |
| `test_vs_torch_npu` | scale bytes 100% identical; nibble mismatch histogram printed; **hard fail** on any mismatch > ±1 code; **hard fail** if the ±1 rate exceeds the threshold pinned in §4.1. Skipped with a clear message if §4.1 showed a convention mismatch we chose not to match. |

**Adversarial block set** (random `N(0,1)` reaches none of these; this is why D1 was invisible):

1. `amax` at exactly `2^-15`, `2^-14`, `2^-13` — the clamp window, i.e. D1's failure sites (`bd = 0, 1, 2`).
2. `amax` fp16-subnormal (`2^-20`, `2^-24`) — the wrong-exponent deviation.
3. All-zero block. Single-nonzero block. All-identical block.
4. `amax ∈ [6,8)·2^k` — the clamp-to-6 band implied by Algorithm 1.
5. Every exact E2M1 midpoint `(0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)·X` — the rounding contract, including the `0.25→0` tie.
6. Values sitting one f16 ulp either side of each midpoint — the double-rounding band (§2.5).
7. One huge outlier plus 31 tiny values — extreme dynamic range within a block.
8. `amax` near fp16 max (65504) and, for bf16, `|x| > 65504` and `|x| < 6e-5` — the range deviation of any fp16-domain path.
9. `+Inf`, `−Inf`, `NaN` blocks; `−0.0` elements (sign-of-zero, §7 R7).

### 4.6 Benchmark harness

**Contender roster** (the roster is what removes the fairness defects, not any single fix):

| row | contender | outputs | what it establishes |
|---|---|---|---|
| 1 | ours (PTO) | preallocated | the kernel's own cost |
| 2 | ours (PTO) | `torch.empty` per call | makes the allocation term **visible** instead of hidden |
| 3 | `torch_npu.npu_dynamic_mx_quant` as-called | allocates (inherent) | the number a real model sees |
| 4 | `aclnn` op called directly | preallocated | "the operator itself", symmetric — needs §4.1 item 8 |
| 5 | `copy_ref_mxfp4_a5` DMA floor | preallocated | **C2**: are we done? |
| 6 | *(optional)* our AscendC `TQuant`-style kernel | preallocated | C4 option (b), separately labeled |

Rows 1 vs 5 decide whether there is anything left to win. Rows **2 vs 3** are the only apples-to-apples "we beat torch_npu" — the existing table compares row 1 against row 3 (`bench_fused_vs_mxfp4.py:74-75` prealloc vs `:78-80` allocating closure), which is the single dominant fairness defect.

**Measurement rules:**
- Identical preallocated `q`/`s` buffers reused by every contender that supports them.
- Per-launch event pairs; **median of ≥7 trials**; report **p05/p95** so a timer glitch is visible rather than averaged in. BENCHMARKS.md documents a glitch that read ~2× too fast on this box.
- Cache flush between iterations (`benchmarking/utils.py:8-43`, `do_bench`: 256 MB zero-fill at `:22-31`, raw per-iteration list via `aggregation="none"` at `:42-43`). The bespoke `time_us` in every WIP script instead wraps one event pair around a 100-rep loop and never flushes. **Pick per-launch timing for all contenders** — mixing loop-amortized and per-launch styles across contenders is on its own enough to manufacture a win, and per-launch is what makes the ~11 µs launch floor visible rather than hidden.
- Each contender gets its **own** pool counter starting at the same index (the shared `it["k"]` at `bench_fused_vs_mxfp4.py:79`/`:84` gives the two contenders different pool phases and cache states).
- Pool larger than L2 so reads hit HBM; **after** timing, assert every pooled input is bitwise unchanged vs a CPU copy taken before warmup (D3 insurance).
- **One pinned `block_dim` = 64** (the device's 64 AIV cores), from a separate tuning run. No best-of-N against a baseline that has no tunable.
- `batch ≥ 65536` for all bandwidth claims; report 16384 in a separate section explicitly labeled **launch-bound**.
- The correctness gate runs **first**; `sys.exit(1)` before any timing if it fails.
- Print a stack stamp into the same output as the numbers: CANN version, torch / torch_npu versions, driver, SoC, NPU index, git SHA, `block_dim`, `ROWS_PER_TILE`, `NBUF`, `PREFETCH`, `K`, dtype, scale rule.
- Sweep **both** input dtypes. `bench_mxfp4_baseline.py:31` hardcodes fp16 while `bench_fused_vs_mxfp4.py:65-67` loops over both — quoting an fp16 kernel number against an implicitly-bf16 baseline is a silent mismatch.

**Reporting rules:** wall-clock µs is the **primary** number (layout-independent, cannot be gamed). Then achieved GB/s computed from **each contender's own actual byte count**. Then percent-of-roofline against the measured row-5 floor. Never compute two contenders' GB/s from the same idealized 648 B/row while they write different volumes — and note that the existing `in_GB/s` column (`bench_fused_vs_mxfp4.py:89-91`, `in_bytes = batch*N*2`) is **read-bytes only**, so it is not a bandwidth and is not comparable to the read+write TB/s in BENCHMARKS.md.

### 4.7 Pre-registered performance analysis (arithmetic shown; nothing invented)

Per-element op counts from §4.4 (pass B amortized over 4096 elements):

| variant | exec/elem | ld+st/elem |
|---|---|---|
| V3, fp16 in | (4+6)/256 + ~0.002 = **0.041** | (2+6)/256 + ~0.001 = **0.032** |
| V1, fp16 in | (4+7)/256 + ~0.002 = **0.045** | (2+4)/256 + ~0.001 = **0.024** |
| V3, bf16 in | (4+4)/256 + ~0.002 = **0.033** | **0.032** |

The only demonstrated reference point on this device is `fast_hadamard_256_a5.cpp` at 2.70 TB/s / 94% of copy: 8 stages × (1 `vlds` DINTLV + `vadd` + `vsub` + 2 `vsts`) per 256 fp16 = **0.0625 exec/elem, 0.094 ld+st/elem**, at 2.70 TB/s ÷ 4 B/elem = 675 Ge/s ⇒ **≈42 G exec/s and ≈63 G ld-st/s sustained**.

To be HBM-bound at the 2.87 TB/s copy floor, MXFP4 quant must sustain 2.87e12 / 2.53125 = **1.13 Te/s**, requiring:

| variant | exec ops/s needed | ld+st ops/s needed |
|---|---|---|
| V3 fp16 | 47 G/s (**+12%** vs 42 G/s demonstrated) | 37 G/s (slack) |
| V1 fp16 | 51 G/s (**+21%**) | 28 G/s (slack) |
| V3 bf16 | 37 G/s (**below** the demonstrated point) | 37 G/s (slack) |

**Honest caveat that the research's own conclusion glossed over:** 42 G exec/s is a **lower bound** on the exec pipe's ceiling, not the ceiling, precisely *because* the WHT was memory-bound at 94% of copy — it was never exec-limited, so it never revealed the pipe's maximum. We therefore **cannot** conclude that MXFP4 quant will be vector-issue-bound; we can only say it needs 12–21% more exec throughput than any point we have measured, with load/store comfortably slack. **This is the single most important thing to measure early** (§6, step 3), because it decides whether optimization effort goes into removing exec ops (V3, bf16 path, `PART_P0`-pair stores) or is wasted.

The bf16 row is the standout: **if `vmul` has a bf16 form** (§4.1 item 9), the bf16 path needs *less* exec throughput than the demonstrated point, gets full E8M0 range, **and** eliminates the double rounding — three wins from one probe. Prioritize that probe.

Whether the exec and ld/st pipes truly issue in parallel is load-bearing for this whole argument (it is what makes the `E2B_B16` broadcast a win over `vintlv`). The WHT data point is consistent with parallel issue but does not prove it. **A/B the `E2B_B16` broadcast against a 4-chained-`vintlv` broadcast at identical unroll to settle it** — if there is no difference, they share issue slots and the analysis changes.

Also unknown: how many architectural vector registers A5 exposes. A ~40-live-register 4-way-unrolled probe compiled clean with no spills (`.text` 0x360), but no documented count and no spill diagnostic exists, and there is **no working disassembler** for this device target (`llvm-objdump` prints `<not available>` for every instruction in `__aicore_rel_binary`). So op counts cannot be verified statically — sweep the unroll factor and watch for a throughput cliff.

---

## 5. Stage 2 — `examples/jit_cpp/fused_hadamard_mxfp4_quant_a5`

Starts only after Stage 1 passes C1 and has a recorded C2/C3 number.

### 5.1 What fusion buys, computed explicitly

Per row of K=256 fp16, two-pass path (in-place WHT, then quantize):

```
WHT   : 512 read + 512 write                      = 1024 B
quant : 512 read + 128 (fp4) + 8 (scale) write    =  648 B
                                            total = 1672 B/row
fused : 512 read + 128 + 8 write                  =  648 B/row
ratio = 1672 / 648 = 2.580x traffic reduction
```

At batch 65536: **109.6 MB → 42.47 MB**. Roofline times at 2.87 TB/s: 38.2 µs → 14.8 µs.

**This 2.58× is a hard ceiling on the achievable speedup, and it is a falsifiability tool:** any measured fused speedup above ~2.6× over the two-op path is a measurement error, and any fused/quant-only ratio below 1.0 means the quant-only contender was mis-measured. Because fusion adds **zero** GM traffic, the fused kernel's byte count is *identical* to quant-only's — the fused and quant-only rows in the results table must share the same `bytes/row` denominator.

The concrete starting line to beat, from the register-branch commit message for `c79a077` (**re-measure it; it was recorded behind a gate that exits 0 and with the D6 non-overlapped pipeline**): the WIP fused N=256 kernel beat torch_npu **quant-only** by 1.4× at batch 16384 and 1.03× at 65536 (fp16). That 1.03× is a useful planning datapoint — a kernel that also does a full WHT already *matched* quant-only, which is itself evidence that quant-only has real headroom.

### 5.2 Pre-registered prediction, and what we do if fusion loses

Adding an N=256 WHT costs ~16 exec + ~24 ld/st ops per 256 elements (§4.7), so the fused kernel is roughly:

```
exec/elem  ≈ 0.0625 + 0.041 = 0.104      (V3 fp16, before the levers in §5.3)
ld+st/elem ≈ 0.094  + 0.032 = 0.126
```

If the exec ceiling is near the only demonstrated point (42–50 G/s), the element rate is 400–480 Ge/s ⇒ 1.01–1.22 TB/s effective on 2.53125 B/elem ⇒ **35–42 µs at batch 65536**. The two-pass path is 24.9 µs (measured WHT) + t_quant. **So the fused kernel may land at only ~0.95–1.15× the two-pass path**, i.e. the 2.58× traffic win largely eaten by being vector-issue-bound.

This is a prediction, not a result, and it rests on a lower bound (§4.7). But it is pre-registered so it can be wrong in public:

- **If measured fused speedup > 1.15×:** ship it, publish the ratio and the % of the 2.58× ceiling attained.
- **If ≤ 1.15×:** apply the §5.3 levers, re-measure once.
- **If still ≤ 1.15×:** say so in the README — *"at N=256 on A5 the WHT+MXFP4 fusion is vector-issue-bound and the 2.58× traffic win does not translate; use the two kernels separately"* — and record the measured exec-throughput ceiling as the reason. A documented negative result here is worth more than a marginal ratio, and it is exactly the kind of claim the existing harnesses could not have made.

### 5.3 Design

**Kernel:** `fused_hadamard_mxfp4_quant_a5.cpp`, one TU, exporting `call_fused_hadamard_mxfp4_quant`. Butterfly = `bfly256` lifted **verbatim** from `fast_hadamard_256_a5.cpp:57-85` (proven 94% of copy, 7e-4 rel error), operating in place on the UB `x` tile; then passes A/B/C from §4.4 on the same buffer. Pipeline, event protocol, and UB accounting exactly as §4.3.

**Lever 1 — the `1/sqrt(N)` normalization is FREE and EXACT.** By §2.6, a power-of-two prefactor moves only the stored byte. For N=256, `L = log2(sqrt(256)) = 4`:

```
byte = b' + 110 - 4 = b' + 106        multiplier field = 32 - b'   (UNCHANGED)
```

The full-width `vmuls(v, v, (half)HAD_INV)` at `fused_hadamard_mxfp4_a5.cpp:110` / `fused_hadamard256_mxfp4_a5.cpp:119` is **deleted**, along with its rounding.

**Lever 2 — free f16 headroom, which also fixes a latent overflow.** The WIP runs the butterfly unnormalized and applies `1/sqrt(N)` only afterwards, so f16 intermediates grow up to N×: **N=128 overflows f16 for max|x| ≳ 512 and N=256 for max|x| ≳ 256** (worse for N=256, which round-trips f16 through UB for all 8 stages before `quant256` applies `HAD_INV` on load). Once `inf` appears, `bd=31` and the block silently dequantizes to `6·2^14`. Fix: prescale the input by an exact power of two `2^-p` on the first stage's load. With prescale `2^-p` and normalization `2^-L`:

```
byte = b' + 110 + p - L          multiplier field = 32 - b'        (still unchanged)
```

Cost: one `vmuls` per 256 elements on stage 0 only ≈ 0.004 exec/elem (~4% of the quant exec budget). Expose as `-DMX_PRESCALE_EXP=p`, default 0 (no multiply, correct for the O(1) activations these kernels target), set 8 for full N=256 headroom. Zero numeric cost — the prefactor is absorbed exactly.

**Lever 3 — N=128 needs one `sqrt(2)`, foldable into the multiplier mantissa.** `1/sqrt(128) = 2^-3.5` is not a power of two, so one `sqrt2` remains. Fold it into the per-block multiplier at zero full-width cost. Derivation: `amax_y = m·2^(b-15)`, `m ∈ [1,2)`; `amax_z = amax_y·2^-3.5 = (m·sqrt2)·2^(b-19)`; `m·sqrt2 ∈ [1.414, 2.828)` so it renormalizes iff `m ≥ sqrt2`:

```
t    = [ m >= sqrt(2) ] = [ mant_field(amax) >= 425 ]     # 1024*(sqrt2 - 1) = 424.26
byte = b + 106 + t
mult = f16 bits ((32 - b - t) << 10) | 0x1A8              # 0x1A8 = 424 = f16 mantissa of sqrt2
t computed branch-free as:  t = (mant + 599) >> 10        # 2 amortized ops in pass B
```

Verified by hand: `amax_y = 1.0` (`m=1, b=15, t=0`) → `amax_z = 0.08839`, `floor(log2) = -4`, byte should be 121; formula gives `15+106+0 = 121` ✓; `X = 2^-6`, `amax_z/X = 5.657 ∈ [4,8)` ✓; `mult = sqrt2·2^(17-15) = 5.657` ✓. And `amax_y = 1.5` (`mant=512, t=1`) → byte `15+106+1 = 122`, `amax_z/X = 4.243 ∈ [4,8)` ✓.

**Consequence for the reference:** `0x1A8` gives `1.4140625`, not `sqrt(2) = 1.41421356` — a relative perturbation of `1.1e-4`. The host reference must use the **exact bit pattern `0x1A8`**, not `sqrt(2)` in f64, or the gate will show spurious mismatches on elements within `1.1e-4` of an fp4 midpoint. With the exact bits modelled, the gate stays bit-exact.

**Lever 4 — fuse pass A into the last butterfly stage.** The final stage's `s`/`d` registers can feed the magnitude-mask/`vmax`/`vcgmax` directly, removing one `vlds` per 256 elements from pass A (0.004 ld/st per element). Modest; do it only if profiling shows the load pipe binding.

**Stretch option, explicitly out of initial scope — put the Hadamard on the cube unit.** `fast_hadamard_128_cube_a5.cpp` reached 2.03 TB/s (76% of copy) at N=128 by computing `Y = X@H` on the matrix unit, and the user's own notes record that cube beats vector for the plain WHT at N=128. If the fused kernel proves vector-issue-bound (§5.2), moving the transform entirely off the vector pipe is the strongest structural lever available. It is out of initial scope because it requires mixing `--cce-aicore-arch=dav-c310-cube` and `-vec` code paths, which is unproven in this repo and is precisely where the user's recorded "host/device-pass guard that silently no-ops kernels" gotcha bites. Revisit as a Stage 3 only after §5.2's measurement.

### 5.4 Validation: composition of two verified stages

Three gates, in increasing strength:

**G1 — bit-identical composition (the strong one, and it needs no host float simulation).** For the same input `x` on device:

```
fused(x)                     vs      quant( wht( x * 2^-p ) )   with byte offset (p - L)
```

Both paths perform the *same* arithmetic in the *same* order (the fused kernel's butterfly is `bfly256` verbatim, and its quant passes are Stage 1's verbatim), so:
- fp4 **nibbles must be bit-identical**;
- scale **bytes must differ by exactly the constant `(p - L)`**.

`b'` is the same quantity in both paths (the f16 exponent field of `amax` of the same `y'`), so the `[2,31]` clamp fires identically in both, and `byte ∈ [112,141] + (p-L)` with `p-L ∈ [-4,4]` never leaves `[0,255]`. This gate is exact, cheap, and depends on nothing unverified.

**G2 — bit-exact vs a host reference.** `mxfp4_ref.quantize(f16_butterfly_sim(x * 2^-p), byte_offset = p - L)`, where `f16_butterfly_sim` reproduces the 8-stage even/odd butterfly in numpy `float16` in the same stage order. numpy f16 arithmetic is IEEE RNE, as the device `vadd`/`vsub` should be, so this is feasible bit-exactly. If G2 fails while G1 passes, the discrepancy is in the butterfly's f16 rounding model, not in the quantizer — which is exactly the isolation we want.

**G3 — float sanity.** Dequantized fused output vs the exact f64 `x @ H / sqrt(N)`: `l2_rel` must land in the MXFP4-intrinsic band **0.10–0.14** (SQNR ~17–19 dB). This is a *sanity band, not a gate* — it is precisely the criterion that was structurally blind to D1 (post-WHT `amax` of 32 normalized Gaussians has `bd ≈ 16`, so `bd ∈ {0,1}` never occurs and a 2×/4× clamp error cannot move a `l2 < 0.25` threshold). Report it; never gate on it alone.

Plus: the full Stage 1 adversarial set (§4.5) re-run through the fused path, with the transform's own overflow bound added (inputs at and just past `max|x| ≈ 256/2^-p` for N=256).

### 5.5 File list

Same shape as §4.2: `README.md`, `fused_hadamard_mxfp4_quant_a5.cpp`, `jit_util_fused_hadamard_mxfp4_quant_a5.py`, `conftest.py`, `test_fused_hadamard_mxfp4_quant_a5.py`, `benchmark.py`, `plot_*.py`, `run_benchmark.sh` (755), `.gitignore`, no `__init__.py`.

**Reference sharing.** `mxfp4_ref.py` stays canonical in `mxfp4_quant_a5/` and the fused dir imports it via a `sys.path.insert` of `../mxfp4_quant_a5` — the precedent is `layernorm`/`swiglu` doing exactly that against `../fast_hadamard/jit_util_common.py`. This trades a little dir self-containment for **one** source of truth on the format contract, which is the right trade given that three divergent host dequantizers with no shared source of truth is one of the confirmed defects. Note the trade in both READMEs.

The fused benchmark additionally needs a **two-pass contender**: the Stage-1 quant kernel run on the output of `fast_hadamard_256_a5`'s `call_hadamard256`, timed as two launches, so C5's denominator is measured on this box rather than assumed.

---

## 6. Verification plan

Ordered. Each step gates the next. Nothing is reported from a step whose predecessor failed.

**On the host (no device):**
1. `pre-commit run --all-files` — nothing excludes `examples/`. Hooks: check-yaml, end-of-file-fixer, trailing-whitespace, check-json, check-merge-conflict, check-added-large-files (500 KB), check-toml, detect-private-key, check-ast, **black 25.12.0**, **clang-format v19.1.7** on `types_or: [c++, c]`, cmake-format 0.6.13. The tree at HEAD is 100% clean on all of these (137 files black-clean; 0 clang-format replacements on every kernel); the reviewed WIP failed with 14 black files, 8/8 dirty `.cpp` (38–352 replacements each), and an end-of-file-fixer violation in `sim_test/run.sh`. `.clang-format` is Google style with the default **80**-column limit; use `// clang-format off/on` around hand-aligned intrinsic blocks (the `layernorm/kernel_layernorm.cpp:11-13` precedent).
2. `black --check .`
3. `prospector` at the repo root — strictness medium, `max-line-length 140` with E501 disabled, `ignore-paths` is **only** `build/`, `scripts/`, `csrc/`, so **`examples/` IS linted**. Every example dir currently reports 0 messages; the WIP reported 80. Target 0. Exactly two suppressions are needed and prospector honours `# noqa` for all its sub-tools (verified by stripping them: 0 → 2 messages): `import torch_npu  # noqa` and `..._as_parameter_  # noqa`. Rules that specifically bite kernel-JIT Python: `multiple-imports` (one import per line), `protected-access`, `unused-import`, **`cell-var-from-loop`** (closures capturing loop vars — the pooled-buffer benchmark pattern), `subprocess-run-check` (always `check=True`), `unspecified-encoding`, `f-string-without-interpolation`, `use-maxsplit-arg`, `unsubscriptable-object`, E305/E306.
4. `ruff check .` under **both** the classic default set (E4/E7/E9/F — the tree passes, the WIP had 95 errors: E702×65, E401×11, E701×11, F401×5, F821×2, F541×1) **and**, best-effort, ruff 0.16 defaults (I001 import sorting, RUF100 no-match noqa, BLE001 no bare `except Exception`, UP006 `list`/`dict` over `typing.List`, EXE001 no shebang at mode 644, B023 no closures over loop vars). *Unresolved:* the CI job uses unpinned `astral-sh/ruff-action@v3`, and ruff 0.16 flags **151 pre-existing errors repo-wide**, so either CI pins something older or the job is already red on main (§7 R12). Being clean under both costs little.
   Note the two `F821` errors ruff found in the WIP bench scripts (`bench256_grid.py:65`, `bench256_nbuf.py:76`) were **real bugs**, which is the argument for the broader set.
5. Host-only unit tests of `mxfp4_ref.py` (encode/decode round trip over all 16 codes; every midpoint tie; reference-1 vs reference-2 agreement; pack/unpack inverse). These need no device and no `torch_npu` — keep the module numpy-only so they run in CI.

**On device (bz.39 — R13 **resolved** 2026-07-29: use the **base** miniconda `python3` after `source /usr/local/Ascend/cann-9.0.0/set_env.sh`. Do **not** `conda activate ascend` — that env is matplotlib-only with no torch, so `pytest` silently collects 0 items via `importorskip`. All device runs go through the lock: `task-submit --device 1 --run "bash script.sh"`):**

6. **`probe_conventions.py`** (§4.1). Commit the results table. **Blocking for everything below.**
7. Compile-only smoke: build every TU at every `(K, ROWS_PER_TILE, NBUF, PREFETCH, dtype)` the harness will use, and check the `.so` exports the expected symbol. Cheap; catches the `vsts` overload and predicate-type errors that cost the ISA lens a compile cycle.
8. **`pytest test_mxfp4_quant_a5.py`** — must exit 0. This is the gate; nothing is timed before it passes.
9. **Ablation measurement** (`mxfp4_ablate_a5.cpp`): reduce-only, cast-only, full. This gives the per-pass cost directly and, critically, an empirical read on the vector-execute throughput ceiling — the number §4.7 says everything hinges on and that we currently only have a lower bound for.
10. **A/B measurements**, one variable at a time: V1 vs V3 pass C; `E2B_B16` broadcast vs 4-chained `vintlv` (settles whether exec and ld/st issue in parallel); `PK4_B32`×2 vs `PART_P0/P1`+`vor`+`PK_B32`; unroll factor sweep (watch for a register-pressure cliff); `ROWS_PER_TILE` / `NBUF` grid (NBUF≥6 is safe once offsets come from `XOFF()` instead of a 4-entry table — see §4.3; it measured ~1% slower on the transform, so a win here would be surprising).
11. **`benchmark.py`** (§4.6) — full roster, both dtypes, K sweep, batch ≥ 65536 for claims. Emits CSV + stack stamp.
12. Re-measure the register-branch baselines as the recorded starting line: torch_npu quant-only, and the WIP fused N=256 claim (1.4× @16384, 1.03× @65536).
13. **`pytest test_fused_hadamard_mxfp4_quant_a5.py`** — G1/G2/G3 (§5.4). Must exit 0.
14. Fused benchmark including the two-pass contender; evaluate C5 against its 2.58× ceiling.

**What gates the PR:**

- steps 1–5 clean;
- step 6's conventions table committed into `README.md` with date + stack versions;
- steps 8 and 13 exit 0 on device;
- a `BENCHMARKS`-style table committed with wall-clock µs, per-contender GB/s from each contender's own byte count, and % of the measured DMA floor;
- the pinned nibble order asserted by a test, and no auto-fitting code anywhere in either dir;
- every documented deviation from OCP (§2.3) written down in `README.md` with the test that covers it.

**Note on CI coverage — there is none for this.** GitHub CI only builds (cmake/wheel) and lints. GitLab CI runs `pytest tests/` (root only) plus a `test_examples_on_x86_910B4` matrix limited to `EXAMPLE: ["scan", "silu_dynamic"]` on a 910B4 runner — a dav-c310 kernel must **not** be added to that matrix (wrong SoC). Doxygen's `INPUT` is `csrc` only and `CMakeLists.txt` never references `examples/`. So **`pytest` exit status is the only automatic protection** against the print-a-verdict-then-exit-0 defect class, which is why every verdict in both new dirs must be an `assert`.

---

## 7. Risks and open questions

Each carries the resolution path. Items are ordered by how much downstream work they block.

**R1 — What is the "AscendC" baseline? RESOLVED 2026-07-29 (user decision).**
The vendor contender is the `aclnn` op behind `torch_npu.npu_dynamic_mx_quant`; option (a). No self-authored AscendC kernel, no row labeled "AscendC". See C4. Remaining sub-task: confirm the op is callable directly with preallocated outputs (probe item 8), so the allocation-asymmetry defect at `bench_fused_vs_mxfp4.py:74-80` cannot recur.

**R2 — What is `dst_type=296`? RESOLVED 2026-07-29 — see §0.** It is `torch_npu.float4_e2m1fn_x2`; output is 4-bit packed at K/2 bytes/row with K/32 scale bytes/row, so `2.53125K` stands and the WIP's `TOTAL_B=324` was correct. Original note follows.

**R2 — What is `dst_type=296`? (blocks every byte-count constant)**
Undocumented magic int, three call sites, zero explanation, not an `aclDataType`, not a `ScalarType`. If it selects an 8-bit destination the baseline writes 2× the bytes and every published ratio is wrong in our favour. *Resolve:* §4.1 items 1–2. Until then, no constant derived from it is trustworthy — including `TOTAL_B=324` at `bench_mxfp4_baseline.py:11-12`.

**R3 — Which scale rule, FLOOR or RCEIL? RESOLVED 2026-07-29 — FLOOR, see §0** (14/14 match on `floor(log2(amax)) - 2 + 127`). Original note follows.

**R3 — Which scale rule does the baseline use, FLOOR or RCEIL?**
A one-code byte difference is a 2× error, so this decides whether we can claim parity at all. *Resolve:* §4.1 item 3. Mitigated in advance by `-DMX_SCALE_RULE` (§2.3) so switching is two amortized ops, not a redesign.

**R4 — Is bit-exact parity with torch_npu required, or is tolerance acceptable? (user decision)**
On the fp16 path bit-exactness is **unattainable** because the only cast to fp4 takes bf16, forcing a double rounding (~0.1–1% of elements one code off a single-rounded reference). *Resolve:* if bit-exact is required, either the reference models the two-step rounding (which §4.5 does, giving bit-exactness against *our* reference) or the fp4 code must be computed with integer ops from (exponent difference, mantissa) rather than via the bf16 pivot — a significant redesign. **The bf16-input path may sidestep this entirely** (single rounding, §4.4) — probe it first.

**R5 — Is the rounding RNE? RESOLVED 2026-07-29 — yes, see §0** (all 7 fp4 midpoints tie to even). The remaining sub-question, whether a direct f16→fp4 cast exists that avoids the bf16 pivot, is still open and belongs to R4. Original note follows.

**R5 — Does `ROUND_R` mean round-to-nearest-even? What other modes exist?**
Assumed RNE throughout; unverified. Also unknown: whether any direct f16→fp4 or f32→fp4 cast exists that avoids the bf16 pivot (the header has the f16→fp4 form **commented out**, which is suggestive but not proof about other entry points). *Resolve:* §4.1 item 9 — cast known midpoints and read back. If it is not RNE, change the reference to match the hardware; do not loosen the gate.

**R6 — Does `set_ctrl()` write the CTRL register wholesale or OR into it? PROMOTED TO BLOCKING 2026-07-29.** §0 measured a block with `amax/X = 7.0 > 6.0`, so clip-on-cast is *required*, not optional — this must be settled before Stage 1 is trusted. Original note follows.

**R6 — Does `set_ctrl()` write the CTRL register wholesale or OR into it?**
`set_ctrl(1u<<50)` at `fused_hadamard_mxfp4_a5.cpp:187` is the **only** `set_ctrl` call in the repo, and it is issued **after** `set_mask_norm()` / `set_vector_mask(-1,-1)`. If it writes wholesale it clears the mask mode those two calls just established. Every other kernel in `csrc/kernel/` uses only the mask pair (`kernel_abs.cpp:34-35`). The clip-to-MAX_NORM behaviour that Algorithm 1's `[4,8)` band **depends on** rides on this. *Resolve:* device A/B — feed a block with `amax` in `[7,8)·X` and check whether out-of-range elements saturate to 6 or blow up, with `set_ctrl` before vs after the mask pair.

**R7 — Sign of zero, and `−0` codes.**
We assert exact bytes, so the kernel must reproduce the reference's sign of zero. `−0.0` input × a positive multiplier should stay `−0.0` and cast to code `0x8`. Whether the hardware cast emits `0x8` or `0x0` for `−0.0`, and what it does for values that round to zero from below, is unverified. *Resolve:* an adversarial-block probe (§4.5 item 9). If the hardware normalizes, the reference must too — and note that `0x0`/`0x8` are numerically equal, so this affects only bit-exactness, never accuracy.

**R8 — `vstus`/`vstas` semantics with count=8 on f16.**
The maxima scratch store is 16 bytes, below the 32-byte `NORM` quantum, and the plan's Pass A depends on the alignment-register accumulate/flush pair (`__VF_VSTUS` `:2280-2310`, `__VF_VSTAS` `:2500-2520`, used at `TQuant.hpp:120-124`). *Resolve:* compile probe first, then a device probe storing a known ramp and reading back all 256 bytes. Fallbacks in order: `ONEPT_B16` stores; or the `pset_b32(PAT_VL8)`-at-stride-8 pattern `TQuant.hpp:83-84` uses (which would need a compaction pass); or shrink GROUP to 32 blocks so the scale store is one aligned 32-byte write and accept a less-amortized pass B.

**R9 — What layout does the eventual consumer want for the scale tensor?**
If an MXFP4 matmul requires a swizzled/blocked or transposed E8M0 layout, the packed `(batch, K/32)` uint8 recommendation is wrong and the write-traffic accounting changes with it. *Resolve:* ask the consumer's owner; until then default to packed contiguous, which is both the ecosystem convention and what makes a direct tensor-vs-tensor comparison against the baseline possible. Keep the layout behind a single `-D` so a change is local.

**R10 — bf16 dynamic range: f16 pivot vs a native bf16/integer path. (biggest design fork)**
The fp16 pivot caps E8M0 to `[2^-15, 2^14]` and clips `|x| > 65504` / collapses `|x| < 6e-5` **before the reduction even runs** — a full-range bf16 reference would disagree on extreme blocks. §4.4's integer-domain reduction (`vand 0x7FFF` + u16 `vmax`/`vcgmax`, `vshrs 7`, `byte = b-2`, `field = 256-b`) gives the full E8M0 range **and** single rounding **and** the lowest exec op count of any variant — but it needs `vmul` to have a bf16 form. *Resolve:* §4.1 item 9. If `vmul` bf16 does not exist, fallbacks are: bf16→f32 (`UNPK_B16` + `PART_EVEN/ODD`), multiply in f32, f32→bf16 → fp4 (2× the pass-C ops per 128 elements, full range, single rounding at the fp4 step); or keep the f16 pivot and **document the range deviation with a test that pins it**. Do not leave this to fall out of the bit arithmetic.

**R11 — Should a `sim_*/` camodel harness be recreated?**
It exists only on the abandoned register branch, is not tracked in any example at HEAD, and its `run.sh` was the one file failing pre-commit's end-of-file-fixer — yet the register-branch README called it *"the authoritative on-device numeric proof"*. *Resolve:* user decision. Default: **no**. Its one genuine advantage (a gate that returns nonzero) is fully covered by pytest, and its actual verdict was best-of-two-nibble-orders, i.e. the defect we are removing.

**R12 — Is the ruff CI job green on main today?**
`astral-sh/ruff-action@v3` is unpinned; `pyproject.toml` has only `[tool.ruff] exclude = ["build", ".skills"]` with no `lint.select`; ruff 0.16 defaults flag 151 pre-existing errors repo-wide (I001×74, RUF100×44, BLE001×7, …). Either CI pins something older or the job is already red. *Resolve:* read a recent CI run's log. Mitigated by writing clean under both rule sets (§6 step 4).

**R13 — Device access incantation. RESOLVED 2026-07-29.**
Use the **base** miniconda `python3` (`/home/hchang/miniconda/bin/python3`) after `source /usr/local/Ascend/cann-9.0.0/set_env.sh`: it has torch 2.9.0+cpu + torch_npu with `torch.npu.is_available()==True`, plus numpy and pytest 9.0.3. Do **not** `conda activate ascend` — verified empirically that env has **no torch at all** (it is the matplotlib-only plotting env), so `pytest` collects 0 items and benchmarks die with `ModuleNotFoundError: No module named 'torch'`. Device runs go through the lock: `task-submit --device 1 --run "bash script.sh"`. Pushes from bz.39 still need `ssh.github.com:443` + a forwarded agent.

**R14 — `conftest.py` + `--npu` fixture, or hardcode `.npu()`? (minor)**
Every sibling except `fast_hadamard_a5` carries the byte-identical 993-byte guarded `conftest.py`. Adding it is strictly better (device selectable, several example dirs collectable in one pytest run) but means the tests take an `npu_device` parameter. *Resolve:* add it. Decided; noted here only because it diverges from the dir this work descends from.

**R15 — Should Stage 1 be K-generic from day one? (scope)**
"Beat torch_npu" is only meaningful at realistic K (2048–8192), and neither WIP kernel supports it — both hard-wire the transform width as the quant width (`static_assert(HAD_N == 128)` at `fused_hadamard_mxfp4_a5.cpp:30`; `SUBROWS` at `fused_hadamard256_mxfp4_a5.cpp:36`). §4.3 makes Stage 1 K-generic (`K % 256 == 0`, plus K=128) from the start, which is **new work, not a lift**. *Risk:* it enlarges Stage 1. *Mitigation:* the tiling is parameterized by `ROWS_PER_TILE` with the group constraint, so K-generality costs one `static_assert` and a Python-side `R` chooser, not a second kernel.

**R16 — Vector register count and the maximum viable unroll.**
Unknown; no documented count, no spill diagnostic, and no working disassembler for this device target (`llvm-objdump` prints `<not available>` for the `__aicore_rel_binary` section, so op counts cannot be verified statically). A ~40-live-register 4-way-unrolled probe compiled clean. The WIP's N=128 quant phase had ~88 live vectors and its own comment said the unroll was register-limited to 4 (`DOU4` at `fused_hadamard_mxfp4_a5.cpp:76`) while using `DOU8` everywhere — the two WIP kernels disagree on the tuned unroll factor. *Resolve:* sweep the unroll and watch for a throughput cliff (§6 step 10); use `.text` size as a crude static proxy.

**R17 — The number that decides Stage 2's fate is a lower bound, not a ceiling.**
§4.7's 42 G exec/s comes from a kernel that was memory-bound at 94% of copy and therefore never exec-limited. Every "will it be vector-bound?" conclusion in this plan — including §5.2's pre-registered prediction that fusion may not win — rests on it. *Resolve:* §6 step 9's ablation measures the exec ceiling directly. Do this **before** investing in either pass-C variant or in Stage 2's levers; it is the cheapest measurement with the largest fan-out in this plan.

---

## Appendix — quick reference

**Bias arithmetic (memorize these four):**

```
fp16:  byte = b + 110 + p - L      mult f16 exp field = 32  - b      b clamped to [2, 31]
bf16:  byte = b -   2 + p - L      mult bf16 exp field = 256 - b     b clamped to [2, 254]
       (p = -log2(prescale applied to input);  L = log2(sqrt(N)), 0 for standalone quant)
       110 = 127-2-15    32 = 2*15+2    -2 = 127-2-127    256 = 2*127+2
```

**Clamp `b`, never the multiplier. Derive both from the clamped `b`.**

**Nibble order:** `byte[k] = (code[2k+1] << 4) | code[2k]`. Pinned. Asserted. Never fitted.

**Bytes/row:** `K * sizeof(in_t) + K/2 + K/32` = `2.53125K` for 16-bit input. K=256 → **648 B/row**.

**Fusion ceiling:** `1672 / 648 = 2.58×`. Anything above ~2.6× measured is a measurement error.

**Roofline, K=256, batch 65536:** 42.47 MB ⇒ **14.0–14.8 µs** at the measured 2.87–3.03 TB/s copy floor. Batch 16384 is launch-bound (~11 µs floor > 3.7 µs roofline) — exclude from bandwidth claims.

**Do not:** index a buffer-offset table by `K % NBUF` (the real 507035 cause — use `XOFF()`) · add `__init__.py` · add a shebang to a `.py` · use `vrec`/`vdiv` for the reciprocal · search nibble orders · gate on an aggregate L2 · report a number from a step whose correctness gate did not pass.
