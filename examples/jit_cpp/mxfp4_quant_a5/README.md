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

`pytest` → **88 passed** on real A5, on **two different parts and two different toolkits**: an Ascend 950DT on CANN 9.0.0 / 9.1.0-beta.3, and an Ascend 950PR (`Ascend950PR_9589`) on CANN 9.1.0 release. Scale bytes and E2M1 nibbles identical to `torch_npu`. Covers every supported `K`, batches 1/7/33/64/128/1000/4097/12345/65536, eight adversarial block families, the partial-tile tail, the host padding path, the active-stream invariant, and rejection of unsupported `K`, wrong dtype and non-contiguous input. CI builds and lints but cannot run these — the gate needs the hardware.

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

On par through the middle of the range and ahead at both ends -- **1.05x** at K=64 and **1.13x** at K=2048. Since the two builds differ only in the compute passes, that gap is compute, not DMA. Output is bit-identical at every shape.

### vs `torch_npu` — user-facing, both allocating

| K | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **685** | **1389** | **2777** | **3118** | **3183** | **2883** |
| `torch_npu` (GB/s) | 614 | 1251 | 2532 | 3385 | 3061 | 2936 |

Ahead at K≤256 (**1.11x**–**1.08x**) and at K=1024; behind at K=512 (**0.92x**) and marginally at K=2048 (**0.98x**). One caveat worth stating: `torch_npu` is not a stable baseline at narrow widths -- it picks a faster kernel in about one process in 15, and at K=512 it takes that path every time, which is the one width where it clearly wins.

Output is **bit-identical to both vendor implementations at every shape**.

The kernel is written as PTO tiles rather than as a closed op, so the quantizer can be fused into a larger kernel later -- a rotation, a norm or a GEMM epilogue writing MXFP4 directly -- without paying a second pass over HBM. On a memory-bound op that is where the remaining win is, since a standalone quantize already runs at DMA speed.

## Rows per launch, at K=4096

The same two comparisons over the batch list `fast_hadamard_a5` (#221) uses. Only
4096 and 8192 of those values are legal widths here, so this is the batch axis.

![MXFP4 on A5 by batch, CANN 9.1.0-beta.3](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_beta3_by_batch.png)

### vs PTO `TQuant` — compute only

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (raw) (GB/s) | **3200** | **3150** | **3266** | **3046** | **2982** | **2876** |
| PTO `TQuant` (GB/s) | 3038 | 3140 | 3119 | 2827 | 2628 | 2623 |

Never behind: parity at 8192 and ahead by up to **1.12x**. Taken
with the width sweep, our four passes match or beat the vendor tile op at every
shape measured on either axis.

### vs `torch_npu` — user-facing

| rows | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| ours (API) (GB/s) | **2780** | **3193** | **3176** | **2869** | **2833** | **2866** |
| `torch_npu` (GB/s) | 2511 | 3210 | 3098 | 2930 | 2821 | 2699 |

Between **0.98x** and **1.14x**. The batch=4096
row is the least firm on this axis: cross-process spread there is 15.7% for our raw
arm and 24.7% for `torch_npu`, against under 5% everywhere else.

## Reproducing the tables

On a real A5 with CANN 9.1.0-beta.3 sourced. `benchmark.py` builds this source
twice by itself -- once as committed, once with `-DMXFP4_TQUANT` -- so the TQuant
arm needs no extra file:

```bash
./run_benchmark.sh --axis k     --tag 1      # -> build/pairs_k_1.csv
./run_benchmark.sh --axis batch --tag 1      # -> build/pairs_batch_1.csv
# the narrow widths in several processes, because torch_npu is not stable there
./run_benchmark.sh --axis k --pairs api --ks 64,128,256,512 --tag m01
```

Repeat with `--tag 2`, `--tag 3`, ... one process each; every figure here is a
median over 3 processes, and 15 for the narrow widths of the
`torch_npu` comparison. Each arm is gated bit-exact against `torch_npu` before it
is timed, so a wrong kernel cannot produce a fast number.

PTO 9.1.0 shipped two `TQuant_MXFP4_E2M1_Impl` signatures -- the release headers
added a `bool Exp2DStrided` template parameter that 9.1.0-beta.3 does not have --
so `benchmark.py` compiles the variant both ways and keeps whichever the local
headers accept. The numbers above come from beta.3; the release form was verified
separately on an Ascend 950PR. On CANN 9.0.0 the TQuant arm is skipped with a
message, since 9.0.0 has no MXFP4 quantizer, and the `torch_npu` pair still runs.

Plotting lives in the companion
[`pto-kernels-plots`](https://github.com/Mocchibird/pto-kernels-plots/tree/main/mxfp4_quant_a5)
repo, next to the figures and the raw CSVs.
