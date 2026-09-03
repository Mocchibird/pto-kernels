# fused_hadamard_quant_a5 - an order-K Hadamard and MXFP4 in one launch

`x -> order-K Hadamard -> E2M1 nibbles + one E8M0 scale per 32`, as a
single kernel on the Ascend 950 / A5 (`dav-c310-vec`) vector core, JIT-compiled
with `bisheng` and loaded through `ctypes`. `K` is a template parameter over 26
widths; one `.so` holds an instantiation per width and the launcher dispatches on
it, so there is no rebuild per size.

`fused_hadamard_quant_b32_a5` is the companion that rotates independent
32-element blocks instead. Prefer that one for a `K` that is not a power of two;
prefer this one when the method calls for a rotation across the whole row.

The rotation is row wide: every output element depends on all `K` inputs, which
is why `K` must be a power of two.

The transform runs in two phases, because Sylvester factors as
`H_K = H_(K/256) (x) H_256`. Phase 1 does the order-256 transform inside every
256-element window, each window an independent deinterleave-load, add/sub,
concat-halves-store repeated eight times. Phase 2 pairs windows `(a, a|t)`
elementwise for `log2(K/256)` further stages. Windows are independent in phase 1
and window pairs are independent in phase 2, so no register holds more than one
window and the row width is not capped -- a single-phase butterfly keeping a
whole row in registers stops at 4096, where a row is already 16 chunks against
16 register slots. The stage count is the same either way:
`8 + log2(K/256) = log2(K)`.

The MXFP4 group stays 32 and no longer lines up with the rotation, which costs
nothing: the quantizer takes the rotated tile in 32-element blocks whatever
produced it.

## Fusing the pair is 2.45x the two separate launches

Unfused, this is two passes over HBM: the butterfly writes the rotated tile out
and the quantizer reads it straight back. Fused, that tile never leaves UB and
only the nibbles and scales are written. Bytes per element tell the whole story:
6.53 unfused against 2.53 fused.

| K | 2 launches | fused | vs 2 | rel err | spread |
|---|--:|--:|--:|--:|--:|
| 1024 | 38.8 | 28.4 | 1.37x | 0.0 | 17.7% |
| 4096 | 297.7 | 121.7 | **2.45x** | 0.0 | 2.8% |
| 8192 | 612.3 | 247.4 | **2.47x** | 0.0 | 2.2% |
| 16384 | 1209.9 | 493.8 | **2.45x** | 0.0 | 1.0% |

M = 16384, microseconds per launch, what `benchmark.py` prints. Both arms agree
to a relative error of 0.0, checked before either is timed. Byte traffic
predicts 6.53 / 2.53 = 2.58x, and the three clean widths measure 2.45-2.47x.

K=1024 is lower for a reason worth knowing rather than hiding. The unfused
intermediate is `2*M*k` bytes, so at K=1024 it is 32 MB against a 128 MiB L2 and
the unfused arm reads much of it from cache rather than HBM -- which flatters the
arm fusing is measured against and understates the result. Its 17.7% bracket
spread is the same thing showing up as noise.
The 19.4% bracket spread on that row is the same thing showing up as noise.
K=4096 and above is where the comparison is clean, and that is where the
prediction and the measurement meet.

## It runs at about the speed of a copy of its input

| K | fused | d2d copy | vs copy | fused GB/s | copy GB/s |
|---|--:|--:|--:|--:|--:|
| 1024 | 121.5 | 191.5 | **1.58x** | 1398 | 1402 |
| 4096 | 122.9 | 191.7 | **1.56x** | 1382 | 1400 |
| 8192 | 122.5 | 192.3 | **1.57x** | 1386 | 1396 |
| 16384 | 122.5 | 189.3 | **1.55x** | 1387 | 1418 |

64Mi elements per launch. The fused column is flat -- 121.5 to 122.9 us across a
16x range of row width -- because the transform is entirely hidden under the DMA
at every width. Both arms reach much the same bandwidth, the kernel 1382-1398
GB/s against the copy's 1396-1418, so the kernel is not moving bytes faster than
a copy; it is moving 1.58x fewer of them, 2.53 B/element against 4.0.

Getting there took two changes, and their sizes are worth recording. The
cross-window stages were originally addressed by a shift-and-OR index computed
per register slot inside the unrolled fold, which cost 266 us of a 388 us kernel
at K=16384; walking nested loops over `base + m*step` instead, with the same
memory pattern and the same number of passes, cut that phase to about 23 us.
Fusing the passes on top (`FUSED_CROSS_FUSE`) added a further 1.16x. Set it to 1
to get one stage per pass and measure the difference.

Measured on an `Ascend950PR_9589`: 64 vector cores, 128 MiB L2, 1.65 GHz, HBM
peak 1.6 TB/s, so the kernel reaches 86-87% of peak. The copy is a
reference for what moving the bytes costs, not a proven lower bound --
it is a vendor kernel doing a simpler job. HBM peak is the closer thing
to a real ceiling, and that is the number above. Other A5 parts have
different HBM, and absolute GB/s from one part should not be compared against
another -- the ratios above are the portable numbers.

## Correctness

The kernel cannot be bit-exact against a torch expression: it rotates in bf16
with a specific operand order and no torch formulation reproduces that tree. So
`test_fused_hadamard_quant_a5.py` establishes it three ways, strongest first.

1. **Scale bytes** must match a reference that rotates in fp32 and quantizes with
   `torch_npu`. A scale is a power of two derived from a block maximum, so bf16
   rounding inside the butterfly almost never moves it -- disagreeing scales mean
   a wrong rotation, not different rounding. Threshold 98%; measured 99.8%.
2. **Dequantized values** must track that reference to within MXFP4's own
   resolution. This catches a correct-looking permutation, which a check on the
   packed bytes would not. Threshold 5%; measured 0.36%.
3. **The output must be non-trivial.** A kernel that writes nothing, or writes
   its input back, is the characteristic silent failure on this hardware and
   would pass a loose tolerance. Separate tests assert the nibbles are neither
   all-zero nor degenerate, and that the result differs from quantizing without
   the rotation.

Two structural cases are covered because both have hidden real bugs here. The
width list spans both **unroll classes** -- the butterfly unrolls by 8 or by 4
depending on `rows_for(k) * k / 256`, they are different code paths, and class
membership is derived from the built `.so` rather than hardcoded, because raising
the tile size once moved four widths between classes and left the matrix
single-class. And one test uses a batch deep enough that every core walks
several tiles, since a shallow batch leaves the buffer rotation, the prefetch and
the drain unexercised.

```bash
python3 -m pytest -q test_fused_hadamard_quant_a5.py
```

## Running the benchmark

```bash
./run_benchmark.sh              # or: python3 benchmark.py --device 0
```

Needs a CANN whose PTO carries MXFP4 (`Exp2DStrided` in `pto/npu/a5/TQuant.hpp`).
9.1.0 and 9.2.0 both do; 9.0.0 does not.

## Tunables

All compile-time, with the shipped defaults. Every combination is checked by
`static_assert`, so a tile that will not fit UB or a prefetch depth that would
deadlock fails to compile rather than misbehaving.

| flag | default | what it is |
|---|---|---|
| `FUSED_TILE_ELEMS` | 24576 | elements per UB tile, 48 KB in bf16 |
| `FUSED_BUFFERS` | 3 | UB pipeline buffers |
| `FUSED_PREFETCH` | 2 | tiles in flight ahead |

The same source builds the reduced kernels the ladder needs:
`FUSED_ROTATE_ONLY` drops the quantizer and `FUSED_NO_ROTATE` drops the
butterfly, so the two arms differ in what they fuse and in nothing else.
`FUSED_BUFFERS` above 4 does not build at K=4096 -- five slots need 311,040
bytes of UB against 253,952 available.

The butterfly is the unnormalised Sylvester matrix, so its output is `sqrt(K)`
larger than an orthogonal Hadamard's, and that factor is left to the caller.
Scale `x` by `1/sqrt(K)` going in if orthogonal semantics are wanted. Whether
`E8M0` could have absorbed it instead depends on the width: at `K` = 64, 256,
1024 or 4096 the factor is 8, 16, 32 or 64 and is itself a power of two, but at
32, 128, 512 or 2048 it is not, so the scale cannot take it and the nibbles
would genuinely differ. Leaving it out is the one behaviour that holds at every
supported width.
