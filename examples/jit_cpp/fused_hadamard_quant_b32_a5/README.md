# fused_hadamard_quant_b32_a5 - a block-32 Hadamard and MXFP4 in one launch

`x -> block-32 Hadamard -> E2M1 nibbles + one E8M0 scale per 32`, as a
single kernel on the Ascend 950 / A5 (`dav-c310-vec`) vector core, JIT-compiled
with `bisheng` and loaded through `ctypes`. `K` is a template parameter over 28
widths from 32 to 16384; one `.so` holds an instantiation per width and the
launcher dispatches on it, so there is no rebuild per size. The rotation is 32
wide however long the row is, so the width costs nothing in registers and is
bounded only by the tile and DMA arithmetic.

`fused_hadamard_quant_a5` is the companion that rotates the whole row instead.
Pick this one when `K` is not a power of two, or when the widest rotation is not
wanted: MXFP4's scale covers 32 elements, so a 32-wide rotation already matches
the quantizer's granularity, and on heavy-tailed data it measured 4-5% lower
quantization error than a full-row rotation at K=4096, because spreading an
outlier across the whole row lifts every block's shared scale instead of just
one block's.

The rotation is 32 wide rather than row wide, which is what lets `K` be any
multiple of 32 instead of a power of two: a row is a run of independent 32-blocks,
the butterfly window is 256 = eight blocks, and a block never straddles a row.
The MXFP4 group is also 32, so one scale covers exactly one rotated block.

## Fusing the pair is 2.45-2.54x the two separate launches

Unfused, this is two passes over HBM: the butterfly writes the rotated tile out
and the quantizer reads it straight back. Fused, that tile never leaves UB and
only the nibbles and scales are written. Bytes per element tell the whole story:
6.53 unfused against 2.53 fused.

| K | 2 launches | fused | vs 2 | rel err | spread |
|---|--:|--:|--:|--:|--:|
| 32 | 29.1 | 13.6 | 2.14x | 0.0 | 9.2% |
| 1024 | 37.5 | 27.2 | 1.38x | 0.0 | 14.5% |
| 4096 | 293.4 | 119.7 | **2.45x** | 0.0 | 3.7% |
| 16384 | 1206.1 | 474.6 | **2.54x** | 0.0 | 1.3% |

M = 16384, microseconds per launch, what `benchmark.py` prints. Both arms agree
to a relative error of 0.0, checked before either is timed. Byte traffic
predicts 6.53 / 2.53 = 2.58x, and the two clean widths measure 2.45x and 2.54x.

The other two rows are not traffic results, for different reasons. At K=32 a row
is 0.5M elements at M=16384 and the fused arm's 13.6 us is the dispatch floor, so
its 2.14x is two launches against one rather than anything about bytes. At K=1024
the unfused intermediate is `2*M*k` = 32 MB against a 128 MiB L2, so the unfused
arm reads much of it from cache rather than HBM, which flatters the arm fusing is
measured against; the 14.5% bracket spread on that row is the same thing showing
up as noise. The copy section below has neither problem, since it runs 64Mi
elements whatever `K` is.

## It runs at about the speed of a copy of its input

| K | fused | d2d copy | vs copy | fused GB/s | copy GB/s |
|---|--:|--:|--:|--:|--:|
| 32 | 122.1 | 191.1 | **1.57x** | 1391 | 1404 |
| 1024 | 122.4 | 192.4 | **1.57x** | 1388 | 1395 |
| 4096 | 117.1 | 192.8 | **1.65x** | 1450 | 1392 |
| 16384 | 117.3 | 192.4 | **1.64x** | 1448 | 1395 |

64Mi elements per launch. Both arms reach much the same bandwidth -- the kernel
1398-1477 GB/s against the copy's 1394-1420 -- so the kernel is not moving bytes
faster than a copy, it is moving 1.58x fewer of them: 2.53 B/element against 4.0.
The butterfly and the quantizer are both hidden under the DMA. That
also means there is nothing to win from instruction selection here; the tile size
is what mattered, and a `vsts` to `vscatter` swap with bit-identical output cost
+29 us.

Measured on an `Ascend950PR_9589`: 64 vector cores, 128 MiB L2, 1.65 GHz, HBM
peak 1.6 TB/s, so the kernel reaches 88-91% of peak. The copy is a
reference for what moving the bytes costs, not a proven lower bound --
it is a vendor kernel doing a simpler job. HBM peak is the closer thing
to a real ceiling, and that is the number above. Other A5 parts have
different HBM, and absolute GB/s from one part should not be compared against
another -- the ratios above are the portable numbers.

## Correctness

The kernel cannot be bit-exact against a torch expression: it rotates in bf16
with a specific operand order and no torch formulation reproduces that tree. So
`test_fused_hadamard_quant_b32_a5.py` establishes it three ways, strongest first.

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
python3 -m pytest -q test_fused_hadamard_quant_b32_a5.py
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

The butterfly is the unnormalised Sylvester matrix, so its output is `sqrt(32)`
larger than an orthogonal block Hadamard's. That is left to the caller rather
than absorbed: MXFP4's `E8M0` scale is a power of two and `sqrt(32)` is not, so
the scale cannot take it up and the nibbles would genuinely differ. Scale `x` by
`1/sqrt(32)` going in if orthogonal semantics are wanted.
