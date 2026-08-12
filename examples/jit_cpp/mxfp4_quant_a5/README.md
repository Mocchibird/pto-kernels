# mxfp4_quant_a5 — MXFP4 block quantization on Ascend A5

bf16 -> 4-bit E2M1 nibbles plus one E8M0 scale per 32 elements, on
the Ascend 950 / A5 (`dav-c310`) vector core, JIT-compiled with
`bisheng` and loaded via `ctypes`. `K` is a template parameter over 26
widths; one .so holds an instantiation per width and the launcher
dispatches on it, so there is no rebuild per size.

Run the numbers below with `./run_benchmark.sh --batch-sweep`.

`(batch, K)` bfloat16 → `q` `(batch, K/2)` uint8 + `scale` `(batch, K/32)` uint8.

`batch` dynamic, `K` a compile-time template argument (26 widths, 64…14336, dispatched at run time — one `.so` serves every width), block size 32 static. bf16 in, A5 only.

## Files

| file | |
|---|---|
| `mxfp4_quant_a5.cpp` | the kernel; every derived size and its `static_assert` in one `QuantShape`, no `#define` |
| `test_mxfp4_quant_a5.py` | 88 tests, bit-exact against `torch_npu.npu_dynamic_mx_quant` |
| `jit_util_mxfp4_a5.py` | build + load, pads the batch and slices back |
| `benchmark.py`, `run_benchmark.sh` | regenerate the tables below |

## Correctness

`pytest` → **88 passed** on real A5. Scale bytes and E2M1 nibbles identical to `torch_npu`. Covers every supported `K`, batches 1/7/33/64/128/1000/4097/12345/65536, eight adversarial block families, the partial-tile tail, the host padding path, the active-stream invariant, and rejection of unsupported `K`, wrong dtype and non-contiguous input. CI builds and lints but cannot run these — the gate needs the hardware.

Quality on `N(0,1)`, K=4096: relative RMSE **0.115**, R² **0.987**.

## Performance

Measured on **CANN 9.1.0-beta.3 with PTO 9.1.0**, the toolchain the CI containers use. Fixed batch of 65,536 rows. Bandwidth counts `2K` read + `K/2 + K/32` written = 2.53125 B/element, one formula for every arm. Medians over 3 separate processes x 64 interleaved brackets with a rotating contender order; the ratio is the median paired per-bracket ratio with a percentile bootstrap 95% interval, and `(ns)` marks an interval that spans parity.

![MXFP4 on A5, CANN 9.1.0-beta.3](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_by_k.png)

### Two comparisons, because there are two questions

The kernel can be called two ways, and mixing them makes any comparison meaningless:

| arm | call path |
|---|---|
| **raw launch** | a bare `ctypes` launch, outputs preallocated. No Python wrapper. |
| **API** | `quant(x)` — the documented entry point: argument checks, padding arithmetic, two `torch.empty` allocations, output slicing. |

`torch_npu` allocates inherently and offers no preallocated mode, so it is only comparable to our **API** arm. PTO's `TQuant` is reached through the **same bare launch** as our raw arm — the identical source built twice, with only the four compute passes swapped for the tile op — so tiling, buffering and every `TLOAD`/`TSTORE` are byte-identical and that pairing isolates **compute**.

Our wrapper costs **2.9x** at K=64 (2080 → 707 GB/s) and disappears by K=2048, where the kernel runs long enough to hide it. That is an API cost, not a kernel cost, and `quant(x, out=(q, s))` already skips the allocating half.

### vs PTO `TQuant` — compute only, matched raw launch

| K | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ours (raw) (GB/s) | **2080** | **2429** | **3067** | **3158** | **3209** | **3079** |
| PTO `TQuant` (GB/s) | 1955 | 2428 | 3054 | 3156 | 3209 | 2760 |
| ratio | **1.06** | **1.00**&nbsp;(ns) | **1.01**&nbsp;(ns) | **1.00**&nbsp;(ns) | **1.01** | **1.10** |
| 95% interval | 1.06–1.07 | 0.99–1.01 | 1.00–1.01 | 0.99–1.00 | 1.00–1.02 | 1.03–1.14 |

Parity across the middle of the range and ahead at both ends: **1.06x** at K=64 and **1.10x** at K=2048. Replacing our four hand-written passes with the vendor tile op would cost throughput at those two widths and change nothing elsewhere. Output is bit-identical at every shape.

### vs `torch_npu` — user-facing, both allocating

| K | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **707** | **1399** | **2801** | **3076** | **3152** | **2847** |
| `torch_npu` (GB/s) | 638 | 1265 | 2603 | 3301 | 3046 | 2901 |
| ratio | **1.12** | **1.10** | **1.08** | **0.93** | **1.03** | **0.98** |
| 95% interval | 1.10–1.15 | 1.09–1.14 | 1.06–1.10 | 0.91–0.94 | 1.03–1.04 | 0.96–0.99 |

Ahead at K≤256 (**1.12x**–**1.08x**) and at K=1024, behind at K=512 (**0.93x**) and marginally at K=2048 (**0.98x**). The K=512 deficit reproduced in every process: beta.3's vendor kernel is genuinely faster there, at exactly two of its 256-element column tiles.

Output is **bit-identical to both vendor implementations at every shape**.

## Rows per launch, at K=4096

The same two comparisons over the batch list `fast_hadamard_a5` (#221) uses. Only
4096 and 8192 of those values are legal widths here, so this is the batch axis.

![MXFP4 on A5 by batch, CANN 9.1.0-beta.3](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_by_batch.png)

### vs PTO `TQuant` — compute only

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (raw) (GB/s) | **3121** | **3097** | **3217** | **2994** | **2995** | **2867** |
| PTO `TQuant` (GB/s) | 3002 | 3107 | 3186 | 2762 | 2623 | 2623 |
| ratio | **1.02** | **1.00**&nbsp;(ns) | **1.03** | **1.08** | **1.11** | **1.10** |
| 95% interval | 1.01–1.05 | 0.99–1.00 | 1.01–1.05 | 1.02–1.13 | 1.10–1.13 | 1.09–1.11 |

Never behind: parity at 8192 and ahead by up to **1.11x**. Taken
with the width sweep, our four passes match or beat the vendor tile op at every
shape measured on either axis.

### vs `torch_npu` — user-facing

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **2736** | **3173** | **3157** | **2830** | **2873** | **2862** |
| `torch_npu` (GB/s) | 2423 | 3143 | 3107 | 2906 | 2841 | 2700 |
| ratio | **1.13** | **0.98**&nbsp;(ns) | **1.01** | **0.98** | **1.01**&nbsp;(ns) | **1.06** |
| 95% interval | 1.11–1.16 | 0.98–1.02 | 1.00–1.03 | 0.97–0.99 | 1.00–1.02 | 1.05–1.07 |

Between **0.98x** and **1.13x**. The batch=4096
row is the least firm on this axis: cross-process spread there is 15.7% for our raw
arm and 24.7% for `torch_npu`, against under 5% everywhere else.

## Is the `torch_npu` peak at K=512 real?

Yes. It is the sharpest feature on either curve and it is the only width where we
lose meaningfully, so it was worth 5 more processes across the multiples of
256 around it.

![torch_npu K=512 peak probe](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_peak.png)

| K | 256 | 512 | 768 | 1024 | 1280 | 1536 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **2748** | **3214** | **3255** | **3137** | **2990** | **2868** |
| `torch_npu` (GB/s) | 2406 | 3423 | 2940 | 3041 | 2848 | 2955 |
| ratio | **1.13** | **0.93** | **1.11** | **1.03** | **1.05** | **0.97** |
| 95% interval | 1.11–1.16 | 0.92–0.94 | 1.10–1.12 | 1.01–1.03 | 1.05–1.06 | 0.97–0.98 |

`torch_npu` reaches **3423 GB/s** at K=512 against **2990** averaged
over its neighbours at 768 and 1024 — a **14%** spike, and it
reproduced in all 5 processes with a cross-process spread of only
**3.2%** (3368, 3378, 3423, 3442, 3477 GB/s). So it is
the vendor kernel, not a measurement artifact.

Our own curve is smooth across the same widths, which is why the ratio dips only
here and at K=1536. Both of those are genuine losses on this toolchain.

