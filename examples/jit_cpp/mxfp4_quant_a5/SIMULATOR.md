# Running A5 kernels on the Ascend 950 simulator (`msprof op simulator`)

How to execute and verify an A5 / `dav-c310` PTO kernel with **no A5 hardware**, using
the CANN cycle-accurate (CA) model.

Everything below was run end to end on a CPU-only aarch64 container (CANN 9.0.0,
no NPU, no `npu-smi`) and the results are the real output of those commands.

Two kernels are wired up, and between them they cover both shapes of correctness
reference you are likely to need:

| Kernel | Command | Reference |
| --- | --- | --- |
| `mxfp4_quant_a5` | `examples/jit_cpp/mxfp4_quant_a5/run_sim.sh` | numpy restatement of the spec, **bit-exact** (§1) |
| `fast_hadamard_a5` | `examples/jit_cpp/fast_hadamard_a5/run_sim.sh` | `x @ H` in numpy, **relative tolerance** (§7) |

Section 8 covers plain `torch_npu` under the simulator — which framework
operations work and which silently do not.

---

## 1. The exact command

```bash
cd examples/jit_cpp/mxfp4_quant_a5
./run_sim.sh
```

That wrapper is the recommended entry point. What it actually runs, expanded:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
source "${ASCEND_HOME_PATH}/bin/setenv.bash"

# 1. The simulator's shared libraries must be on the loader path BEFORE msprof starts.
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH}"

# 2. Tells the repo's helpers to relax device timeouts and drop repeat counts.
export PTO_SIMULATOR=1

# 3. Build the device .so with bisheng OUTSIDE msprof (see gotcha 6).
python3 -c "from jit_util_mxfp4_a5 import compile_kernel; print(compile_kernel())"

# 4. Run the Python process under the CA model.
msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --timeout=120 \
  --output=./outputs/msprof_mxfp4 \
  python3 ./run_sim_mxfp4_a5.py --output-json ./outputs/msprof_mxfp4/result.json
```

`msprof op simulator` wraps the whole Python process and injects the CA-model
runtime in place of the real driver, so ordinary `torch_npu` allocations,
`.npu()` copies, and the `ctypes` kernel launch all execute against the
simulator. **You do not change the kernel or the launch code at all.**

Useful knobs on the wrapper:

```bash
./run_sim.sh --k 4096 --block-dim 2      # a different row width / core count
./run_sim.sh --batch 512                 # one custom batch instead of the default cases
MSPROF_TIMEOUT=300 ./run_sim.sh          # raise the whole-process bound
MSPROF_OUT_DIR=/tmp/sim ./run_sim.sh     # see gotcha 3 — matters on bind mounts
MSPROF_SOC_VERSION=Ascend950PR_9571 ./run_sim.sh
```

---

## 2. Requirements

| Requirement | This container | How to check |
| --- | --- | --- |
| CANN toolkit | 9.0.0 (`V100R001C10SPC001B250`), aarch64 | `cat $ASCEND_HOME_PATH/aarch64-linux/ascend_toolkit_install.info` |
| `msprof` | `$ASCEND_HOME_PATH/bin/msprof` | `msprof op simulator --help` |
| `bisheng` | `$ASCEND_HOME_PATH/bin/bisheng` | compiles `--cce-aicore-arch=dav-c310-vec` |
| **A5 simulator package** | `tools/simulator/Ascend950PR_9599/` | `ls $ASCEND_HOME_PATH/tools/simulator` |
| Python | 3.11 | |
| `torch` | 2.9.0**+cpu** — a CPU build is fine | |
| `torch_npu` | 2.9.0.post2 | supplies the allocator/stream the CA model hooks |
| `numpy` | 2.4.4 | the correctness reference |
| NPU device | **none needed** | `npu-smi` is absent here |
| Host | 12 cores / 7 GB RAM was enough for these shapes | bigger shapes cost much more of both |

The simulator package is the one piece that is easy to miss. All the
`Ascend950PR_*` names are symlinks to a single `dav_3510` CA model, so any of
them works as `--soc-version`; the repo standard is `Ascend950PR_9599`. If the
directory does not exist, the simulator component was not installed with the
toolkit and no amount of flag-fiddling will help.

`torch_npu` is required **even though there is no NPU**: the CA model substitutes
the driver underneath it, not the framework above it.

---

## 3. What a passing run looks like

```text
[compile] up-to-date, reusing mxfp4_a5.so
[sim] k=64 rows/tile=256 block_dim=2
[INFO] <ProfInit> Start profiling on kernel: _Z11mxfp4_quantILj64ELj256ELj4ELj2EEvPvS0_S0_j
[info] [0000000325] [block_start] : AIV, task_id=0, core_id=0, block_id=0
[info] [0000000325] [block_start] : AIV, task_id=0, core_id=1, block_id=1
[info] [0000003351] [block_end]   : AIV, task_id=0, core_id=1, block_id=1
[info] [0000004713] [block_end]   : AIV, task_id=0, core_id=0, block_id=0
  full_tile: PASS (256x64)
  partial_tile: PASS (272x64)
  adversarial: PASS (256x64)
2026-08-07 11:06:48 [INFO]  Profiling running finished. All task success.
[verdict] PASS
```

- The bracketed numbers are **simulated cycles**, not wall-clock.
- One `block_start`/`block_end` pair per AIV core per launch. Two cores here
  because `--block-dim 2`; three launches because there are three test cases.
- `[verdict] PASS` is the wrapper's own gate (see gotcha 1).

Measured results for `mxfp4_quant_a5`, bit-exact against the numpy reference,
zero mismatched scale bytes or nibbles:

| Case | Covers | `k=64` | `k=4096` |
| --- | --- | --- | --- |
| `full_tile` | one whole 16384-element tile | PASS (256x64) | PASS (4x4096) |
| `partial_tile` | the kernel's partial-last-tile tail | PASS (272x64) | PASS (5x4096) |
| `adversarial` | clamp window, saturation band, every E2M1 midpoint, ±0, `3e38` | PASS (256x64) | PASS (4x4096) |

`k=64` is the narrowest instantiation and `k=4096` the production row width.

Wall clock: **40–75 s per run** here for three launches of one tile each,
including report parsing. Cheap by CA-model standards only because the shapes are
deliberately tiny.

### The profiling report

When the parse succeeds (gotcha 3), msprof prints a per-core summary and writes
real per-instruction data under `<OPPROF_*>/simulator/`:

```text
[INFO]  Extract 345 relations from kernel
[INFO]  Core operator results run in simulator as follow:
core_name           duration_time(us)   running_time(us)
core0.veccore0      1.57                1.51
core0.veccore1      0.28                0.26
[INFO]  Profiling data parse finished.
```

| Artifact | Contents |
| --- | --- |
| `simulator/<core>/<core>_instr_exe.csv` | one row per instruction: `instr, addr, pipe, call_count, cycles, running_time(us), detail` |
| `simulator/<core>/<core>_code_exe.csv` | per source line — needs `-g` at compile time to be useful |
| `simulator/trace.json` | Chrome-trace timeline; open in `chrome://tracing` or Perfetto |

That CSV is the useful one for optimization work. Cycles by pipe for
`mxfp4_quant_a5` at `k=4096`, one tile on `core0.veccore0`:

```text
RVECEX 4937   MTE2 3071   SCALAR 2546   RVECLD 2081   RVECST 1381   VECTOR 1016
top instructions by cycles: RV_VLD 1939, SET_FLAG 1535,
                            MOV_SRC_TO_DST_ALIGNv2 1535, RV_VST 1381, RV_VMUL 1016
```

Treat these as *model* cycles for spotting pipe imbalance and stalls, not as a
performance number — accept performance only on real silicon.

`Kernel missed debug_line information` / `Code call stack is empty` are expected
unless you add `-g` to the `bisheng` flags; they do not indicate a failure.

---

## 4. Why a separate runner instead of the pytest suite

`test_mxfp4_quant_a5.py` compares against `torch_npu.npu_dynamic_mx_quant`, the
vendor operator, which needs real A5 silicon. Under the CA model that operator is
unavailable, and the suite is deliberately written to **fail rather than skip**
when the reference is missing — a green suite that asserted nothing is worse than
a red one.

So `run_sim_mxfp4_a5.py` restates OCP MX v1.0 §6.3 Algorithm 1 (FLOOR) in numpy
and compares **bit-exactly**, not within a tolerance. That is legitimate here
rather than a weaker check: the MX scale is a power of two, so the kernel's bf16
multiply is exact, and float64 reproduces it exactly up to the final E2M1
rounding. The reference was independently pinned against the two constants the
device test also pins (`amax=6.0` → scale byte 127, element 0 → low nibble
`0x42`) before being trusted.

**This is a correctness smoke test, not an acceptance gate.** It proves the
kernel computes the right bits and that its DMA/pipeline synchronization does not
deadlock. It does *not* replace the on-device suite: the CA model does not
reproduce real-silicon timing, the vendor operator, or multi-core races at 64
cores. Run the pytest suite on real A5 hardware before accepting a change.

---

## 5. Gotchas

Each of these cost real debugging time.

1. **`msprof`'s exit status is not your kernel's verdict.** It reports on
   profiling, and nothing it returns reflects the values your kernel computed.
   Measured here: on the gotcha-3 run it logged
   `[ERROR] Profiling data parse failed` and **still exited 0**. So a zero exit
   proves neither that the numerics are right nor that the report was written.
   Always gate on something your own script wrote: `run_sim.sh` requires
   `result.json` to exist and contain `"result": "PASS"`, and treats a missing
   file as failure.

2. **The simulator runtime aborts during its own teardown.** After the kernel and
   all comparisons finish, process exit raises
   `terminate called after throwing an instance of 'std::bad_function_call'`
   and the child dies with status 6. msprof then logs
   `Running task failed, data parsing start` and **discards the profiling data**.
   Fix: leave before the interpreter's exit handlers run.

   ```python
   sys.stdout.flush(); sys.stderr.flush()
   os._exit(exit_code)      # last line of the runner
   ```

   With that in place the same run reports `Profiling running finished. All task
   success.` This is a simulator teardown bug, not a kernel bug — but it looks
   exactly like a crash in your kernel, so know it on sight.

3. **Do not put `--output` on a bind-mounted volume.** With `--output` inside the
   repo (a Docker Desktop `fakeowner`/virtiofs mount) the CA model's
   `aicore_binary.o` came back unreadable — `lstat` returned `EACCES` even to
   root — so msprof's readability check failed, the per-kernel
   `device0/<kernel>/` layout was never assembled, and the report died with:

   ```text
   [ERROR] path: .../dump/aicore_binary.o is not readable
   [ERROR] Parse kernel data failed, kernel name is _Z11mxfp4_quant...
   [ERROR] Profiling data parse failed. Please check
   ```

   **The numerics were still correct and still verified** — only the profiling
   report was lost, which is why gotcha 1 matters. Pointing `--output` at a
   container-local filesystem fixed it completely and produced the full report:

   ```bash
   MSPROF_OUT_DIR=/tmp/sim_out ./run_sim.sh
   ```

   Check your own mount with `findmnt -T . -no FSTYPE`; `overlay`/`ext4` is fine,
   `fakeowner`/`virtiofs`/`9p` is not. Note the two failure modes look similar in
   the log but are independent: gotcha 2 loses the data before parsing starts,
   this one fails during the parse.

4. **One run writes ~7000 dump files** under
   `<output>/OPPROF_<timestamp>_<id>/dump/` — 6 MB, or ~16 MB once the reports
   parse. `outputs/` and `OPPROF_*/` are in `.gitignore` — keep it that way, and
   clean up between runs.

5. **Keep the shapes tiny.** This is a cycle-level model: it simulates every
   instruction on every core. `mxfp4_quant_a5` has a fixed 16384-element
   (32 KB) tile, so one tile is the smallest useful unit of work regardless of
   `k` — `k=64` needs 256 rows to fill one, `k=4096` needs only 4. Prefer
   `--block-dim 2` over the real 64 cores; you are validating logic, not
   measuring throughput.

6. **Compile outside `msprof`.** `bisheng` under the injected simulator runtime
   is only slower, and a compile error is far easier to read without the
   profiler wrapping the output. `run_sim.sh` does the build in a separate
   process first; `compile_kernel()` caches on mtime, so it is nearly free when
   the source has not changed.

7. **`--soc-version` must name a directory that actually exists** under
   `tools/simulator/`, and the matching `.../<soc>/lib` must be on
   `LD_LIBRARY_PATH` *before* `msprof` starts. Getting this wrong yields loader
   errors that do not mention the simulator at all.

8. **Do not link the kernel `.so` against `runtime_camodel` in this mode.**
   `msprof op simulator` injects the CA-model runtime itself; linking it too is
   for the standalone C++ path (see below).

9. **Device-to-device copies silently do nothing.** This is the nastiest one, and
   it cost the most time here. `dst.copy_(src)` and `dst[:n] = src` between two
   device tensors complete without error and leave the destination **unchanged**
   (§8 measures it). Host-to-device copies are fine. It matters because the
   idiomatic host-side padding wrapper is
   `buf = torch.zeros(...); buf[:batch] = x` — a D2D copy — so on the CA model the
   kernel receives an all-zero buffer, computes a perfectly correct transform of
   zeros, and the runner reports a numerical failure that has nothing to do with
   the kernel. Stage padding on the **host** instead (build the padded array in
   numpy, then one `.npu()` copy), or pass whole-tile batches so the padding path
   never runs.

---

## 6. Adapting this to another A5 kernel

The pattern is mechanical:

1. Compile for A5: `--cce-aicore-arch=dav-c310-vec` (`-cube` for Cube-only,
   `dav-c310` for a mixed kernel), plus the flags in
   `jit_util_mxfp4_a5.py:FIXED_FLAGS`.
2. Write a plain runner script that allocates with `torch_npu`, launches through
   `ctypes`, calls `torch.npu.synchronize()`, compares against a host reference,
   and writes a PASS/FAIL JSON.
3. End that script with `os._exit(code)` (gotcha 2).
4. Run it under `msprof op simulator --soc-version=Ascend950PR_9599`.
5. Gate on the JSON, not on `msprof`'s status (gotcha 1).

`run_sim.sh` + `run_sim_mxfp4_a5.py` in this directory are a copyable template.

Two other CA-model entry points exist and are documented in
`.skills/testing-pto-kernels/reference/launch-methods.md`:

| Path | When |
| --- | --- |
| `msprof op simulator` | default; wraps a Python/`ctypes` process, gives dumps + profile |
| `cannsim record -s Ascend950` | A5 SoC traces (`trace_core*.json`); wraps an executable, args go through `-u "..."` |
| link `-lruntime_camodel` | a standalone ACL C++ `main()` with no Python at all |

Ready-made A5 examples to sanity-check your environment before debugging your own
kernel — this one passes here in about a minute:

```bash
cd .skills/testing-pto-kernels/reference/dynamic_multi_core/a5
./run_sim.sh msprof --n 128
```

---

## 7. A second kernel: `fast_hadamard_a5`

Same pattern, different kernel — the fast Walsh-Hadamard transform. The exact
command:

```bash
cd examples/jit_cpp/fast_hadamard_a5
./run_sim.sh                              # n=256, 2 cores (both verified below)
./run_sim.sh --n 64                       # a different block size
MSPROF_OUT_DIR=/tmp/sim_out ./run_sim.sh  # gotcha 3
# --block-dim N and --batch N are forwarded too, as in §1
```

Expanded, it is the identical `msprof` invocation as §1 with a different runner:

```bash
msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --timeout=120 \
  --output=./outputs/msprof_hadamard \
  python3 ./run_sim_hadamard_a5.py --output-json ./outputs/msprof_hadamard/result.json
```

This kernel needs **no vendor operator** — the reference is `x @ H` for the ±1
Hadamard matrix, which numpy computes directly — so the same reference the device
suite uses works here unchanged. The comparison is therefore a **relative
tolerance** (0.03, the device suite's own bound), not bit-exact: the kernel
accumulates in fp16 across log2(N) butterfly stages and is not expected to match a
float32 reference bit for bit.

Measured:

| Case | Covers | `n=256` | `n=64` |
| --- | --- | --- | --- |
| `two_tiles` | two whole tiles across both cores | PASS (1.0e-3) | PASS (7.4e-4) |
| `host_padded_tail` | a tail that does not fill its tile | PASS (7.3e-4) | PASS (7.1e-4) |
| `packed_rows_hostile_padding` | `n<256` only: `inf`/`nan` padding must not bleed into real rows through a shared vector window | n/a | PASS (7.0e-4) |

Figures in parentheses are relative error against the 0.03 bound.

**One caveat, and it is the reason gotcha 9 exists.** Every case passes a whole
number of tiles, so the host wrapper's own padding path (`buf[:batch] = x`) never
runs. That path is a device-to-device copy, which this CA model does not implement
— it leaves the buffer zeroed, the kernel then correctly transforms zeros, and the
runner reports `rel_error ≈ 1.0`. That is exactly what happened on the first
attempt here, and it is **not** a kernel bug: `padded_batch=33` failed while the
numerically identical `two_tiles=64` passed. Padding is staged on the host
instead, and **the wrapper's padding path remains a real-device test only.**

---

## 8. Plain `torch_npu` under the simulator

Yes — ordinary `torch_npu` code runs under `msprof op simulator`, not just custom
kernels. `msprof` swaps the driver underneath the framework, so allocation,
transfers, and CANN's own operators all execute on the CA model. Worth knowing
because a runner leans on these paths even when the kernel under test is your own.

Measured by `sim_torch_npu_probe.py` in this directory — **13 of 15 work**:

```bash
cd examples/jit_cpp/mxfp4_quant_a5
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH}"
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=900 \
  --output=/tmp/probe python3 sim_torch_npu_probe.py --output-json /tmp/probe.json
```

| Operation | Status |
| --- | --- |
| `torch.zeros` on device | ok |
| host → device copy (whole tensor) | ok |
| **device → device `.copy_`** | **WRONG — silently leaves the destination unchanged** |
| **device → device slice assign** | **WRONG — silently leaves the destination unchanged** |
| host → device slice assign | ok |
| pad on host, one h2d copy | ok |
| elementwise add | ok |
| multiply by scalar | ok |
| `sum` reduction | ok |
| fp16 → fp32 cast | ok |
| `matmul` 16×16 (Cube, vs numpy) | ok |
| `softmax` (composite op) | ok |
| `torch.npu.synchronize()` | ok |
| `current_stream` pointer | ok |
| `get_device_properties` | ok |

So real compute works — including CANN's own Cube matmul and composite ops like
softmax. The two failures are both the same thing, and they are **silent**: no
exception, no warning, just a destination that never changed. That is the failure
mode to internalise, because it looks like a broken kernel rather than a missing
runtime feature.

Practical rules:

- Build buffers on the host, then a single `.npu()` copy. Never `d2d`.
- Where a wrapper does D2D copies internally, either pass shapes that avoid the
  path or accept that the path is device-only.
- Keep tensors tiny. Every framework operator dispatches a real kernel that the
  model simulates instruction by instruction — the 16×16 matmul is not free.
- The probe prints and flushes per operation, so a timeout still leaves a record
  of everything that worked. Re-use that habit in your own runners.

---

## 9. Files

| File | Purpose |
| --- | --- |
| `mxfp4_quant_a5/run_sim.sh` | env setup, build, `msprof` invocation, PASS/FAIL gate |
| `mxfp4_quant_a5/run_sim_mxfp4_a5.py` | numpy MXFP4 reference + three cases, writes `result.json` |
| `mxfp4_quant_a5/sim_torch_npu_probe.py` | what plain `torch_npu` can and cannot do on the CA model (§8) |
| `mxfp4_quant_a5/test_mxfp4_quant_a5.py` | the real acceptance suite — **real A5 hardware only** |
| `fast_hadamard_a5/run_sim.sh` | the same wrapper for the Hadamard kernel |
| `fast_hadamard_a5/run_sim_hadamard_a5.py` | `x @ H` reference + cases (§7) |
| `fast_hadamard_a5/test_hadamard_a5.py` | its acceptance suite — real hardware |
