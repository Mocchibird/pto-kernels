# fast_hadamard_a5 — register-resident fp16 Walsh–Hadamard (Ascend 950 / A5)

A register-resident fast Walsh–Hadamard transform (FWHT) for the Ascend 950
(dav-c310) vector core. Unlike the [`../fast_hadamard/standard`](../fast_hadamard)
kernel — which runs each of the `log2(N)` butterfly stages as a
`TGATHER(even)+TGATHER(odd)` into UB scratch plus `TADD`/`TSUB` back to UB (one
UB round-trip per stage, dominated by `TGATHER` per-repeat overhead) — this
kernel keeps the **entire butterfly cascade in vector registers**:

```
vlds (once)  ->  [ log2(N) stages, all in registers ]  ->  vsts (once)
```

No UB traffic and no `TGATHER` between stages. It is a good first A5 kernel and
composes with MXFP quantization (e.g. MXFP4 training) — the transform result can
be cast/quantized straight out of the register.

## Algorithm

Constant-geometry FWHT with a **concat-halves** recombine, per stage:

```
(e, o) = vdintlv(v, v)      // e = [evens|evens], o = [odds|odds]  (both 64-lane halves)
s      = e + o              // sums  (duplicated in lanes 0..63 and 64..127)
d      = e - o              // diffs (duplicated in lanes 0..63 and 64..127)
v      = vsel(mask_lo64, s, d)   // v = [ sums(0..63) | diffs(64..127) ]
```

The concat-halves recombine (sums to the first half, diffs to the second half)
is the textbook constant-geometry FWHT and produces the **natural Sylvester
order** directly — verified against `scipy.linalg.hadamard`, no bit-reversal
correction needed. After the stages, `vmuls(v, v, 1/sqrt(N))` applies the
orthonormal scale.

The trick that keeps it register-only is `vdintlv(v, v)`: deinterleaving `v`
against itself duplicates the evens/odds into **both** 64-lane halves, so the
diffs already exist in lanes 64..127 and a single `vsel` with a low-64 mask
assembles `[sums | diffs]` with zero UB traffic — the register equivalent of the
`standard/` kernel's `ColValid` "write first/second half" trick.

> **Do not** use an interleave recombine (`vintlv`, `v[2i]=s[i], v[2i+1]=d[i]`).
> A deinterleave-then-interleave stage partially inverts itself and does **not**
> compute a Hadamard (the output magnitudes do not match a WHT). This was
> confirmed both numerically and on the simulator.

No `pipe_barrier` between stages: on dav-c310 the vector pipe issues same-pipe
RAW in program order, so the `vdintlv → vadd/vsub → vsel` chain is naturally
ordered. GM↔UB uses a flat contiguous burst with a 2-buffer (ping/pong)
load/compute/store pipeline.

## Scope

v1 targets **N == 128** (one length-128 block per 128-lane b16 register), the
clean single-register case where the low-64 select mask lines up with the block
halves. This is the KVarN `D=128` size and the maximum single-register width.

**N < 128, lane-packed** (e.g. MXFP4 block size 32, four blocks per register) is
a documented extension: it needs a per-block half-select mask (active for the
first `N/2` lanes of *each* block) and a shifted diff placement, because the
`vdintlv(v,v)` duplication has period 64, not period `N/2`. The butterfly math
itself is unchanged.

## Register layout note (A5 quirk)

A5 vector register lanes are physically 32-bit. A 16-bit type sits in the low
half of a 32-bit lane with a 16-bit gap; a 16→32 cast fills that gap. This
kernel stays entirely in the **packed b16 view** (`NORM` load/store, dense
128-lane `vdintlv`/`vsel`) and never casts, so the gap does not affect it. It
*does* matter for an MXFP-fused variant (f16→f32→fp8/fp4): use `UNPK_B16` +
`PART_EVEN/ODD` `vcvt` with a 128-lane cast mask there.

## Performance (Ascend950PR_9599 camodel, msprof op simulator)

Profiled at N=128, batch=256. The register approach is **compute-bound with
near-zero overhead** — the gather/dispatch tax of the `standard/` TGATHER kernel
is gone:

| Pipe | % of busy-core cycles | Notes |
|------|----------------------|-------|
| RVECEX (vector execute) | **79.7%** | the butterfly: `vdintlv` 27%, `vadd` 17%, `vsub` 17%, `vsel` 15%, `vmuls` 3% |
| MTE2 (GM→UB) | 7.2% | one flat burst per tile |
| SCALAR | 6.5% | address / loop setup |
| RVECLD / RVECST | 3.2% / 3.2% | `vlds` / `vsts` |
| PUSHQ (VF dispatch) | 0.1% | — |

There is **no gather** (`RV_VSQZ`), **no per-gather predicate regeneration**
(`RV_PLT`/`RV_PAND`), and VF dispatch is negligible — the exact costs that
dominate the TGATHER `standard/` kernel (~20% gather + ~17% mask-gen + ~13%
dispatch, with real arithmetic only ~4%). Here ~97% of cycles are genuine vector
work. `vdintlv` is the single priciest op (~11 cyc/call) but one deinterleave
per stage is irreducible.

Every compute op is a single 128-lane fp16 SIMD instruction on RVECEX: one
`vadd`/`vsub` computes all N/2 butterflies of a stage at once. The `log2(N)`
stages are serial (inherent to the FWHT).

**Grid fill matters more than the kernel.** Wall time is set by how many of the
16 AIV (8 AIC × 2 vector subblocks) are busy = number of tiles = `batch /
ROWS_PER_TILE`. Under-tiling is the trap: with `ROWS_PER_TILE=128` and batch=256
only 2 of 16 AIV ran (12,924 ticks); lowering to `ROWS_PER_TILE=16` gives 16
tiles → all AIV busy → **3,939 ticks (3.3× faster)**, same numerics. The default
is 16 for this reason; tune it so `batch / ROWS_PER_TILE >= 16`.

## Build & test (self-contained sim)

```bash
cd sim_test
ASCEND_HOME_PATH=$ASCEND_TOOLKIT_HOME ./run.sh      # N_DIM/BATCH_DIM overridable
```

Runs on the `Ascend950PR_9599` camodel and checks the fp16 output against a CPU
natural-order Sylvester WHT. Current result: N=128, batch=256 → `max_diff ≈
1.9e-3` (fp16 rounding), `err_count = 0/32768`, `PASS`.

## Benchmark on a 950 device (real hardware)

On an actual Ascend 950 server (with `torch` + `torch_npu` and the CANN
toolkit), one command compiles, correctness-checks, and benchmarks — reporting
achieved HBM bandwidth:

```bash
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/jit_cpp/fast_hadamard_a5
./run_benchmark.sh --npu 0 --block-dim 20
# sweep sizes + write CSV:
./run_benchmark.sh --npu 0 --block-dim 20 \
    --batches 1024,4096,16384,65536 --repeats 50 --csv bw.csv
```

Output per size: `duration_us`, `GB/s`, `TB/s`, where bandwidth = `2·batch·N·2 B`
(load + store) ÷ measured time (`torch.npu.Event`). It first verifies the output
against the natural Sylvester WHT and aborts if that fails.

### Performance tuning (approaching HBM bandwidth)

Two levers matter for utilization; both are in the kernel now:

- **Tile size** (`ROWS_PER_TILE`, default 256 → a 64 KB DMA burst). Small tiles
  (the earlier 16 → 4 KB) leave the HBM path idle and pay per-tile sync
  overhead. Larger tiles drive bigger contiguous bursts. Trade-off: for a given
  batch, keep `batch / ROWS_PER_TILE >= (#AIV)` so the grid still fills.
- **Register software-pipelining** (`hadamard_vf` is unrolled 4-way). The
  butterfly is a `vdintlv→vadd→vsub→vsel` dependency chain; issuing each op for
  4 independent registers back-to-back hides the per-op latency instead of
  stalling on it.

Use `--copy-floor` to measure the pure GM→UB→GM DMA ceiling (same tiling) and
print `had/copy` — the fraction of copy speed the transform achieves. If
`had/copy` is near 1.0 the kernel is memory-bound (done); if it's low, compute
(the butterfly) is still the limit.

```bash
./run_benchmark.sh --npu 0 --block-dim 32 --batches 16384,65536 --copy-floor
```

- **`--block-dim`** is the launch grid = number of AIC on your 950 (each spawns
  2 AIV); set it to the device's AIC count for full occupancy. The kernel is
  correct for any value.
- This is **self-contained**: it compiles with the A5 toolchain directly
  (`dav-c310-vec`, `-DREGISTER_BASE`) rather than the shared `jit_util_common`,
  which targets `dav-c220`.
- Unlike the camodel (which models compute only, not HBM), a real 950 exposes
  the memory-bound behavior — this is where the ">3 TB/s / ≈copy-speed" claim
  is actually measurable.

## Files

- `fast_hadamard_a5.cpp` — the kernel (register-resident butterfly + ping/pong DMA).
- `sim_test/main.cpp`, `sim_test/run.sh` — self-contained on-device correctness test.
- `benchmark.py` — on-device benchmark: compile + correctness + bandwidth (real 950).
- `run_benchmark.sh` — one-command wrapper (sets env, runs `benchmark.py`).
- `jit_util_hadamard_a5.py` — self-contained A5 build (`compile_kernel`) + ctypes loader.
- `test_hadamard_a5.py` — numpy reference check (host-side transform math).
