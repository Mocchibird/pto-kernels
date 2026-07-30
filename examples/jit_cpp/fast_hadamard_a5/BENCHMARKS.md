# fast_hadamard_a5 — kernel benchmarks (Ascend 950 / A5, dav-c310)

Walsh–Hadamard transform kernels, fp16, in-place, measured on a real 950 device.
All numbers: `block_dim=64` (= the device's ~64 AI cores), warmup 5–10 / 50–100
reps, throughput = `2·batch·N·2 bytes / time` (load + store). Each kernel is
compared against a pure GM→UB→GM **copy floor** with the *same* tiling (the HBM
bandwidth ceiling). Reproduce with `run_plain_all.sh` (drives `bench_cube.py`,
`bench128_dintlv.py`, `bench256.py`).

## Results — batch 65536 (the amortized regime)

| Kernel | N | Technique | dur µs | TB/s | % of copy | copy floor | correctness (rel) |
|--------|---|-----------|-------:|-----:|----------:|-----------:|-------------------|
| `fast_hadamard_256_a5.cpp` | 256 | deinterleave-load butterfly | ~24.9 | **2.70** | **94%** | 2.87 | 7e-4 |
| `fast_hadamard_128_cube_a5.cpp` | 128 | cube / matmul vs Sylvester H | ~16.5 | **2.03** | **76%** | 2.68 | 3e-4 |
| `fast_hadamard_128_dintlv_a5.cpp` | 128 | deinterleave-load butterfly | ~17.7 | 1.90 | 70% | 2.68 | 8e-4 |
| `fast_hadamard_128_a5.cpp` | 128 | register-resident VF butterfly | ~22.2 | 1.51 | 57% | 2.68 | 9e-4 |

Batch 16384 is launch-overhead-bound (~11 µs fixed floor): the N=128 kernels all
sit at ~0.73–0.78 TB/s there and the N=256 kernel is already at its copy floor
(~1.6 TB/s). Batch 65536 is the meaningful comparison.

## Which kernel to use

- **Need N=256** → `fast_hadamard_256_a5.cpp` (2.70 TB/s, essentially copy speed).
- **Need N=128** → `fast_hadamard_128_cube_a5.cpp` (2.03 TB/s) is fastest.
- `fast_hadamard_128_a5.cpp` is the register-VF baseline and also provides the
  `copy_ref_a5` DMA floor used by the N=128 benches.

These are **not interchangeable**: N=128 and N=256 are different transforms.

## Why the numbers land where they do

The WHT is memory-bound, so "close to copy" == "good".

- **Deinterleave-load** (`_256`, `_128_dintlv`): each of the log2(N) stages does
  the even/odd split on the `vlds DINTLV_B16` load and the concat-halves
  recombine on the `vsts` store, so only `vadd`/`vsub` touch the vector-execute
  pipe. At N=256 a row fills two 128-lane registers → full-width ops → 94% of
  copy. At N=128 a row is one register and the split is 64+64, so ops run
  half-width (`LANES = HAD_N/2 = 64`) → only 70% of copy.
- **Cube** (`_128_cube`): computes `Y = X @ H` (H = Sylvester(128)/√128,
  symmetric, pre-scaled) on the matrix unit; the arithmetic is nearly free
  relative to the DMA, so it's memory-bound at 76% of copy. Best for N=128.
- **Register-VF** (`_128`): keeps the 7-stage butterfly register-resident
  (`vdintlv`→`vadd`/`vsub`→`vsel`, ~28 vec ops/row). Compute-bound on the vector
  pipe → only 57% of copy.

Notes: the `_128` and `_128_cube` kernels emit the **normalized** WHT (÷√128);
the two deinterleave-load kernels are **unnormalized** (`x @ Sylvester`) — a
final-stage `vmuls` by 1/√128 would make them drop-in normalized. Device build
flags: vector kernels `--cce-aicore-arch=dav-c310-vec -DREGISTER_BASE`; cube
kernel `--cce-aicore-arch=dav-c310-cube`.

## fast_hadamard_256_a5 — batch × ROWS_PER_TILE sweep

Grid sweep over batch (2^10..2^18) × `ROWS_PER_TILE ∈ {16,32,64,128}` (NBUF auto-fit
to the 192 KB UB), block_dim 64. Copy floor = a fixed, UB-valid ROWS=64 build,
median of 7 trials, measured once per batch. Reproduce: `run_grid256.sh`
(data → `build/grid256.csv`), `run_check256.sh` (correctness),
`python3 plot_hadamard256_grid.py` (heatmap + bandwidth line → `build/hadamard256_grid.png`).
Interactive version: https://claude.ai/code/artifact/e2b9f64a-1a91-4a24-9f83-11f630896d56

- **Already near-optimal / memory-bound.** `ROWS_PER_TILE ∈ {16,32,64,128}` all track
  the copy floor once past the small-batch launch-overhead region. Copy floor peaks at
  **3.03 TB/s** (batch 65536); hadamard reaches ~2.4–3.0 TB/s at large batch (ratio
  0.78–1.15). It dips to ~0.78–0.92 only at batch 65536, where the copy floor peaks.
  No meaningful speedup available. Default `ROWS_PER_TILE=64` is fine; `32`/`128` are
  comparable.
- **`ROWS_PER_TILE=256` (NBUF=1) is buffering-limited** (single-buffered transform,
  ~2.2 TB/s) — dropped from the sweep.
- **Correctness verified for every config** (batch 4096): max rel error 8.5e-4 vs
  x·Sylvester(256), *identical* across all ROWS_PER_TILE (tiling doesn't perturb the
  math); `copy256` preserves data bit-exactly (max|Δ|=0) at every config.
- **UB budget / pipeline depth** (`bench256_nbuf.py`): the A5 has **248 KB** UB, not
  the 192 KB the kernels hard-code. Raising the budget and deepening the pipeline
  helps only up to `NBUF=4` (e.g. batch 64k: `NBUF=2` 2.2 TB/s → `NBUF=4` 2.7 TB/s).
  `NBUF>=6` device-faulted with error 507035, and **the explanation previously given
  here was wrong**: it blamed the kernel reusing one event ID per buffer across all
  three pipe handoffs. It was not the event protocol — `ev[8]` was correctly sized
  and the token accounting balances for any NBUF. The real cause was a fixed
  `unsigned xoff[4]` table of UB offsets indexed by `K % NBUF`, i.e. an
  out-of-bounds read for NBUF>4, which at ROWS=64 also slips past the UB
  static_assert because 6 × 32 KB is exactly 192 KB. Fixed on branch
  `fast-hadamard-a5-pr` (PR #221) by computing offsets with the `XOFF()` macro.
  **Device-verified 2026-07-29** with that fix: NBUF=6 runs correctly and is ~1%
  *slower* than NBUF=4 (2634 vs 2668 GB/s at batch 65536, ROWS_PER_TILE=64), and
  raising the budget to the physical 248 KB changes nothing (2622 GB/s). So the
  original conclusion — UB capacity is not the bottleneck — was right, but the
  reason was not: the transform is simply HBM-bound, and four buffers already
  saturate the load and store pipes. `UB_USABLE_BYTES` remains overridable.
- Measurement note: an earlier version recompiled the copy reference at each ROWS and
  used a single timed loop, which produced physically-impossible >5 TB/s copy reads
  (an over-budget 2-buffer copy at ROWS=256 + timer glitches). Fixed via the fixed
  ROWS=64 copy build + median-of-trials + a larger buffer pool; copy floor now stays
  under the ~3.3 TB/s HBM ceiling.
