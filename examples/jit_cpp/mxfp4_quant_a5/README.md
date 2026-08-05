# mxfp4_quant_a5 — MXFP4 block quantization on Ascend A5

Quantizes a `(batch, K)` **bfloat16** matrix to OCP MXFP4 on the Ascend 950 / A5
(`dav-c310`) vector core, JIT-compiled with `bisheng` and loaded through `ctypes`.

Each consecutive run of **32 elements** shares one E8M0 scale byte; each element
becomes one E2M1 nibble. Outputs are `q` `(batch, K/2)` uint8 and `scale`
`(batch, K/32)` uint8.

**Shape contract:** `batch` is dynamic; `K` is a compile-time template argument
(one instantiation per supported width, **128 … 4096**, dispatched at run time so
there is no rebuild per width); the MXFP4 block size 32 is static.

> **Status: correct, not yet optimised.** The gate is bit-exact and 32/32 tests
> pass on real hardware. There is **no performance claim and no benchmark** in this
> change — the kernel knowingly carries one extra UB round trip (see
> "The alignment tax"). A benchmark and a comparison against
> `torch_npu.npu_dynamic_mx_quant` are follow-up work.

## Files

- `mxfp4_quant_a5.cpp` — the kernel. Every derived size and its `static_assert`
  lives in one `QuantShape`, checked per instantiation, so an invalid combination
  cannot be built. No `#define` beyond the arch guards.
- `mxfp4_ref.py` — host reference, numpy only. Two independent references: a
  **bit chain** that reproduces the kernel's exact sequence, and a **float64 spec
  formula** that shares no arithmetic with it, so a wrong bias constant cannot hide.
- `jit_util_mxfp4_a5.py` — build + load. The callable pads the batch to a multiple
  of `ROWS_PER_TILE` and slices back, so any batch size works.
- `test_mxfp4_quant_a5.py` — 32 tests, bit-exact against the reference.

## Build & run

Requires a real A5 device with `torch`/`torch_npu` and the CANN toolkit; set
`ASCEND_HOME_PATH` (or `ASCEND_TOOLKIT_HOME`).

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
pytest test_mxfp4_quant_a5.py          # 32 tests, bit-exact
```

```python
from jit_util_mxfp4_a5 import build_and_load
quant = build_and_load(k=4096)
q, scale = quant(x)                    # x: (batch, 4096) bfloat16, contiguous
```

## Format contract, as measured

Every value below was measured on device against the CANN operator behind
`torch_npu.npu_dynamic_mx_quant`, and is reproduced independently by `mxfp4_ref.py`.
None of it is assumed.

**Scale rule — OCP MX v1.0 §6.3 Algorithm 1, FLOOR.**
`byte = floor(log2 amax) + 125`, i.e. `b - 2` where `b` is the bf16 amax's biased
exponent. Matched in 14 of 14 probe cases:

| amax | 0.25 | 1.0 | 1.5 | 4.0 | 6.0 | 7.0 | 8.0 | 1024 |
|---|---|---|---|---|---|---|---|---|
| scale byte | 123 | 125 | 125 | 127 | 127 | 127 | 128 | 135 |

The `amax = 7.0` row is the one that matters: `7.0/X = 7.0` exceeds the largest
E2M1 magnitude (6.0), so the cast **must** saturate — and it does, on its own. No
clipping code and no `set_ctrl` are needed.

**Reciprocal.** `1/X` is the bf16 with exponent field `256 - b`, mantissa 0. Both
the byte and the reciprocal are derived from the **same clamped `b`** (window
`[2, 254]`, which is exactly where `1/X` stays a finite non-subnormal bf16), so
they remain exact inverses. Never clamp the reciprocal field separately.

**Nibble order.** `byte[k] = (code[2k+1] << 4) | code[2k]` — element `2k` is the
**low** nibble. Measured: `e0=1.0, e1=2.0` at scale byte 127 gives first byte
`0x42`. Pinned once and asserted; never auto-fitted to whatever scores better.

**Rounding — round-to-nearest-even** on the E2M1 grid, ties to the even magnitude
field:

| midpoint | 0.25 | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 | 5.0 | > 6 |
|---|---|---|---|---|---|---|---|---|
| target | **0.0** | 1.0 | 1.0 | 2.0 | 2.0 | 4.0 | 4.0 | 6.0 |

**Why bf16 input and not fp16.** The only narrowing cast to fp4 on this device
takes bf16, so a bf16 input is a *single* rounding and the gate can be genuinely
bit-exact. An fp16 path would go `f16 → bf16 → fp4`: every E2M1 midpoint is exactly
bf16-representable, so a half-ulp band rounds *onto* a midpoint and then takes the
tie rule instead of the correct direction — roughly 0.1–1% of elements one code off
a single-rounded reference. fp16 is therefore excluded by design, not omission.

## How it works

Four passes over a 16 KB bf16 UB tile, pipelined against DMA with 4 buffers and 2
tiles prefetched:

| pass | does | width |
|---|---|---|
| A | magnitude max per 32-element block | 256 elements → 8 maxima |
| A2 | compact the padding pass A must leave | 4 groups → 32 maxima |
| B | maxima → scale byte + bf16 reciprocal | 128 blocks |
| C | scale, cast to E2M1, pack nibbles | 128 elements → 64 bytes |

Pass A relies on a 2:1 fold so that 16 lanes correspond to one block, which is what
`vcgmax`'s group size on b16 requires. Pass C does one `vcvt` (128 bf16 → 64 bytes
at byte stride 4) and one `vselr` to gather them contiguous.

### The alignment tax

`vsts` requires a **32-byte-aligned** UB destination, and `vcgmax` on b16 yields
only 8 results (16 bytes) — so consecutive groups would land 16 bytes apart and
fault the vector core. Pass A therefore writes each group into a 32-byte slot and
**pass A2 exists purely to squeeze the holes back out**. `Tile` also refuses a
sub-32-byte DMA, so the padding cannot instead be skipped on the way to GM.

That extra UB round trip is the known inefficiency. The idea worth trying is
storing the maxima zero-extended to b32, which makes each group exactly 32 bytes
and contiguous, deleting pass A2 at the cost of moving pass B into the b32 domain.

## Correctness

`pytest` → **32 passed** on real A5 hardware.

The gate is **bit-exact**, not a tolerance: scale bytes and E2M1 nibbles are
integers, so any mismatch is a bug. Coverage:

- every supported `K` (128 … 4096) and batch sizes 64 / 128 / 1000 / 4097 / 65536,
  including a non-multiple that exercises the padding wrapper and one shape with
  more logical work than physical cores;
- eight **adversarial block families** that random `N(0,1)` never reaches: the
  clamp window (`2^-15`, `2^-14`, `2^-13`), subnormal amax, the clip-to-6 band,
  all seven E2M1 midpoints, all-zero, a huge outlier, near-bf16-max, and `-0.0`;
- the nibble order asserted against one pinned convention;
- the bit-chain and float64 spec references asserted equal to each other;
- rejection of an unsupported `K`, a wrong dtype, and a non-contiguous input — the
  kernel can report none of these, so the host must;
- `ROWS_PER_TILE` queried from the built `.so` and pinned against the Python
  value, so the two cannot drift;
- a cross-check against `torch_npu.npu_dynamic_mx_quant`, the CANN operator that
  is the only other MXFP4 implementation on this device. Measured 2026-08-05 on
  CANN 9.0.0: the vendor op, this kernel and the host reference are **bit-identical**
  on the bf16 path. It is a cross-check rather than the gate -- it runs on the same
  hardware, so a shared driver bug would pass it, and matching the vendor is a
  weaker claim than implementing Algorithm 1 correctly.

Per `.skills/testing-pto-kernels`, device runs repeat (`PTO_DEVICE_REPEATS`,
default 5) because a four-pass pipeline is where a missing `set_flag`/`wait_flag`
shows up nondeterministically, and each synchronize is bounded by
`PTO_SYNC_TIMEOUT_S`.

**Quantization quality** on `N(0,1)`, `K=4096` — reported because max error alone
says only which value landed worst in a 16-level grid:

| relative RMSE | R² |
|---|---|
| 0.115 | 0.987 |

## Notes

- Two host-side guards exist because the failure is otherwise silent: `TSTORE`
  needs a **512-byte-aligned** GM destination and simply does not write otherwise,
  and the ctypes launch is not ordered against torch's copy into the padded buffer
  (without a synchronize the kernel quantizes zeros and returns a plausible
  all-zero result).
- The vendor op returns `scale` shaped `(batch, K/64, 2)` — same count, different
  layout — so a tensor-vs-tensor comparison must reshape one side.
