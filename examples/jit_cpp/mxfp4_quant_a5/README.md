# mxfp4_quant_a5 — MXFP4 block quantization on Ascend A5

bf16 -> 4-bit E2M1 nibbles plus one E8M0 scale per 32 elements, on
the Ascend 950 / A5 (`dav-c310`) vector core, JIT-compiled with
`bisheng` and loaded via `ctypes`. `K` is a template parameter over 26
widths; one .so holds an instantiation per width and the launcher
dispatches on it, so there is no rebuild per size.

Reproduce every number below with `./run_benchmark.sh`; see
"Reproducing the tables" at the end.

MXFP4 block quantization for Ascend 950 / A5 (`dav-c310`), JIT-compiled with `bisheng`, loaded via `ctypes`.

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

Measured on **CANN 9.1.0-beta.3 with PTO 9.1.0**, the toolchain the CI containers use. Fixed batch of 65,536 rows. Bandwidth counts `2K` read + `K/2 + K/32` written = 2.53125 B/element, one formula for every arm.

Each contender is timed in 64 brackets, interleaved one bracket at a time with a rotating order, and the whole sweep is repeated in independent processes: 3 carry the TQuant arm, and `torch_npu` gets 18 at the narrow widths. `ratio` is the median process, `across processes` the full spread between them, and `processes agreeing` how many landed on the majority side of parity; `(ns)` marks under 80% agreement. A within-process confidence interval is deliberately not quoted, because `torch_npu` selects a different kernel in some processes than in others and no within-process statistic can see that.

![MXFP4 on A5, CANN 9.1.0-beta.3](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_by_k.png)

### Two comparisons, because there are two questions

The kernel can be called two ways, and mixing them makes any comparison meaningless:

| arm | call path |
|---|---|
| **raw launch** | a bare `ctypes` launch, outputs preallocated. No Python wrapper. |
| **API** | `quant(x)` — the documented entry point: argument checks, padding arithmetic, two `torch.empty` allocations, output slicing. |

`torch_npu` allocates inherently and offers no preallocated mode, so it is only comparable to our **API** arm. PTO's `TQuant` is reached through the **same bare launch** as our raw arm — the identical source built twice, with only the four compute passes swapped for the tile op — so tiling, buffering and every `TLOAD`/`TSTORE` are byte-identical and that pairing isolates **compute**.

Our wrapper costs **3.0x** at K=64 (2074 → 685 GB/s) and disappears by K=2048, where the kernel runs long enough to hide it. That is an API cost, not a kernel cost, and `quant(x, out=(q, s))` already skips the allocating half.

### vs PTO `TQuant` — compute only, matched raw launch

| K | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ours (raw) (GB/s) | **2074** | **2428** | **3069** | **3160** | **3210** | **3078** |
| PTO `TQuant` (GB/s) | 1964 | 2440 | 3061 | 3170 | 3205 | 2681 |
| ratio | **1.05** | **1.00** | **1.01** | **1.00**&nbsp;(ns) | **1.01** | **1.13** |
| across processes | 1.05–1.06 | 1.00–1.00 | 1.00–1.03 | 1.00–1.00 | 1.00–1.02 | 1.11–1.14 |
| processes agreeing | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |

Parity across the middle of the range and ahead at both ends: **1.05x** at K=64 and **1.13x** at K=2048. Replacing our four hand-written passes with the vendor tile op would cost throughput at those two widths and change nothing elsewhere. Output is bit-identical at every shape.

### vs `torch_npu` — user-facing, both allocating

| K | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **685** | **1389** | **2777** | **3118** | **3183** | **2883** |
| `torch_npu` (GB/s) | 614 | 1251 | 2532 | 3385 | 3061 | 2936 |
| ratio | **1.11** | **1.11** | **1.08** | **0.92** | **1.04** | **0.98** |
| across processes | 0.85–1.20 | 0.83–1.18 | 0.85–1.14 | 0.89–0.96 | 1.03–1.04 | 0.98–0.99 |
| processes agreeing | 16/18 | 16/18 | 16/18 | 18/18 | 3/3 | 3/3 |

Ahead at K≤256 (**1.11x**–**1.08x**) and at K=1024, behind at K=512 (**0.92x**) and marginally at K=2048 (**0.98x**). The K=512 deficit reproduced in every process: beta.3's vendor kernel is genuinely faster there, at exactly two of its 256-element column tiles.

Output is **bit-identical to both vendor implementations at every shape**.

## Rows per launch, at K=4096

The same two comparisons over the batch list `fast_hadamard_a5` (#221) uses. Only
4096 and 8192 of those values are legal widths here, so this is the batch axis.

![MXFP4 on A5 by batch, CANN 9.1.0-beta.3](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_by_batch.png)

### vs PTO `TQuant` — compute only

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (raw) (GB/s) | **3200** | **3150** | **3266** | **3046** | **2982** | **2876** |
| PTO `TQuant` (GB/s) | 3038 | 3140 | 3119 | 2827 | 2628 | 2623 |
| ratio | **1.00**&nbsp;(ns) | **1.00**&nbsp;(ns) | **1.05** | **1.07** | **1.12** | **1.10** |
| across processes | 1.00–1.18 | 1.00–1.00 | 1.00–1.05 | 1.07–1.09 | 1.12–1.13 | 1.10–1.10 |
| processes agreeing | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |

Never behind: parity at 8192 and ahead by up to **1.12x**. Taken
with the width sweep, our four passes match or beat the vendor tile op at every
shape measured on either axis.

### vs `torch_npu` — user-facing

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **2780** | **3193** | **3176** | **2869** | **2833** | **2866** |
| `torch_npu` (GB/s) | 2511 | 3210 | 3098 | 2930 | 2821 | 2699 |
| ratio | **1.14**&nbsp;(ns) | **1.00**&nbsp;(ns) | **1.00** | **0.98** | **1.02**&nbsp;(ns) | **1.06** |
| across processes | 0.92–1.16 | 0.99–1.02 | 1.00–1.01 | 0.98–0.98 | 1.00–1.02 | 1.06–1.06 |
| processes agreeing | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |

Between **0.98x** and **1.14x**. The batch=4096
row is the least firm on this axis: cross-process spread there is 15.7% for our raw
arm and 24.7% for `torch_npu`, against under 5% everywhere else.

## Is the `torch_npu` peak at K=512 real?

Yes. It is the sharpest feature on either curve and it is the only width where we
lose meaningfully, so it was worth 5 more processes across the multiples of
256 around it.

![torch_npu K=512 peak probe](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_peak.png)

| K | 256 | 512 | 768 | 1024 | 1280 | 1536 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **2753** | **3095** | **3218** | **3160** | **2979** | **2845** |
| `torch_npu` (GB/s) | 2356 | 3395 | 2964 | 3037 | 2867 | 2960 |
| ratio | **1.15** | **0.93** | **1.11** | **1.03** | **1.06** | **0.97** |
| across processes | 0.86–1.23 | 0.90–0.98 | 1.05–1.12 | 1.02–1.04 | 1.05–1.07 | 0.96–0.98 |
| processes agreeing | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

`torch_npu` reaches **3395 GB/s** at K=512 against **3001** averaged
over its neighbours at 768 and 1024 — a **13%** spike, and it
reproduced in all 5 processes with a cross-process spread of only
**9.2%** (3148, 3362, 3395, 3422, 3438 GB/s). So it is
the vendor kernel, not a measurement artifact.

Our own curve is smooth across the same widths, which is why the ratio dips only
here and at K=1536. Both of those are genuine losses on this toolchain.


## Reproducing the tables

On a real A5 with CANN 9.1.0-beta.3 sourced. `benchmark.py` builds this source
twice by itself -- once as committed, once with `-DMXFP4_TQUANT` -- so the TQuant
arm needs no extra file:

```bash
./run_benchmark.sh --axis k     --tag 1      # -> build/pairs_k_1.csv
./run_benchmark.sh --axis batch --tag 1      # -> build/pairs_batch_1.csv
# the K=512 probe: the API pair over the multiples of 256 around the peak
./run_benchmark.sh --axis k --pairs api \
    --ks 256,512,768,1024,1280,1536 --tag peak1
# and the narrow widths in many processes, one tag each, to see whether the
# vendor arm is stable there -- it is not
./run_benchmark.sh --axis k --pairs api --ks 64,128,256,512 --tag m01
```

Repeat with `--tag 2`, `--tag 3`, ... one process each; every figure here is a
median over 3 processes (5 for the peak probe). Each arm is gated for
bit-exactness against `torch_npu` before it is timed, so a wrong kernel cannot
produce a fast number. On CANN 9.0.0 the TQuant arm is skipped with a message --
9.0.0 has no MXFP4 quantizer -- and the `api` pair still runs.

Plotting lives in the companion
[`pto-kernels-plots`](https://github.com/Mocchibird/pto-kernels-plots/tree/main/mxfp4_quant_a5)
repo, next to the figures and the raw CSVs.
