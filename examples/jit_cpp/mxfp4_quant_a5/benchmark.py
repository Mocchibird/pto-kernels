"""Bandwidth benchmark for mxfp4_quant_a5: ours vs PTO TQuant and vs torch_npu.

Reproduces every figure in README.md. Run it through run_benchmark.sh.

TWO COMPARISONS, on deliberately different call paths, because pairing arms that
do not share one measures the wrapper rather than either kernel:

  --pairs raw   ours vs PTO TQuant. Both a bare ctypes launch with preallocated
                outputs, from THIS source built twice -- the second time with
                -DMXFP4_TQUANT, which swaps the four compute passes for the
                vendor tile op and leaves tiling, buffering and every
                TLOAD/TSTORE identical. So this isolates COMPUTE. Needs PTO
                9.1.0; on 9.0.0, which has no MXFP4 quantizer, it is skipped.
  --pairs api   ours vs torch_npu. Both one Python call that allocates its own
                outputs. torch_npu has no preallocated entry point, so this is
                the only fair user-facing pairing.

The two `ours` arms differ only in Python -- argument checks, padding arithmetic,
two allocations and output slicing -- which costs about 2.9x at K=64 and nothing
by K=2048.

Bandwidth counts every byte the operation must move: 2K read plus K/2 + K/32
written, i.e. 2.53125 bytes per element, one formula for every arm.

TIMING. BRACKETS brackets per shape; each fires LAUNCHES launches between two
synchronizes and divides the wall clock by LAUNCHES, identically for every arm.
These are steady-state throughput figures, not single-call latency. Contenders
are interleaved one bracket at a time with a ROTATING order: under a fixed order
the first arm in each bracket absorbs the previous one's cache eviction, which
alone was enough to make a preallocated arm read slower than an allocating one.
The reported ratio is the median paired per-bracket ratio with a percentile
bootstrap 95% interval, and a shape whose interval spans 1.0 is unresolved.

That interval covers variation WITHIN a process only. torch_npu has been seen to
select a different kernel from one process to the next, so run several processes
with different --tag values and compare their spread before believing a small
margin on the api pair.

Every contender is gated bit-exact against torch_npu before it is timed, so a
wrong kernel cannot report a fast number.

The vendor arm is whatever CANN is on ASCEND_HOME_PATH: torch_npu resolves
libopapi_nn.so from there, so its numbers move with the toolkit and rows are
comparable only within one version.

Emits build/pairs_<axis>_<tag>.csv, which is what the plotting scripts read.
"""

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import ctypes
import os
import subprocess

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from jit_util_mxfp4_a5 import MX_BLOCK, SUPPORTED_K, build_and_load, row_quantum

HERE = Path(__file__).resolve().parent
BUILDDIR = HERE / "build"

# The same widths as the fast_hadamard_a5 sweep (PR #221), keeping only those
# with an EVEN block count: torch_npu lays its scales out as (batch, K/64, 2), so
# an odd block count does not fit its layout. The kernel supports all 26 widths;
# the rest are covered by the tests.
HADAMARD_NS = [32, 64, 128, 256, 512, 1024, 2048]
K_LIST = [k for k in HADAMARD_NS if k in SUPPORTED_K and (k // MX_BLOCK) % 2 == 0]
K_SWEEP_BATCH = 65536  # fixed across the K sweep
BATCH_SWEEP_K = 4096
WORKING_SET_BYTES = 1024 * 1024 * 1024
POOL_MIN, POOL_MAX = 2, 64
LAUNCHES = 40
BRACKETS = 64
BOOTSTRAP = 2000
SEED = 20260811
WARMUP = 5
VENDOR_DST_TYPE = 296

BYTES_PER_ELEM = 2.0 + 0.5 + 1.0 / MX_BLOCK


def raw_launcher(so_path, k):
    """ctypes launcher with no wrapper: the path the TQuant arm also uses."""
    from jit_util_mxfp4_a5 import VECTOR_CORES, current_stream_ptr

    handle = ctypes.CDLL(str(so_path))
    launch = handle.call_mxfp4_quant
    launch.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    launch.restype = None

    def call(tensor, packed, scales):
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            tensor.shape[0],
            k,
        )

    return call


class TQuantUnavailable(RuntimeError):
    """PTO has no MXFP4 quantizer -- it arrived in 9.1.0, so 9.0.0 cannot build."""


def build_tquant(k):
    """Build the kernel with its four compute passes replaced by PTO TQuant."""
    home = os.environ["ASCEND_HOME_PATH"]
    out = BUILDDIR / "mxfp4_a5_tquant.so"
    obj = BUILDDIR / "mxfp4_a5_tquant.o"
    BUILDDIR.mkdir(parents=True, exist_ok=True)
    source = HERE / "mxfp4_quant_a5.cpp"
    flags = (
        f"-xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -DMXFP4_TQUANT "
        f"-std=c++17 -O2 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
        f"-mllvm -cce-aicore-stack-size=0x8000 "
        f"-mllvm -cce-aicore-function-stack-size=0x8000 "
        f"-mllvm -cce-aicore-addr-transform "
        f"-mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end "
        f"-I{home}/aarch64-linux/include -I{home}/include"
    ).split()
    compile_step = subprocess.run(
        [f"{home}/bin/bisheng", *flags, "-c", str(source), "-o", str(obj)],
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_step.returncode != 0:
        raise TQuantUnavailable(compile_step.stderr.strip()[-400:])
    subprocess.run(
        [
            f"{home}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{out.name}",
            str(obj),
            "-o",
            str(out),
        ],
        check=True,
    )
    handle = ctypes.CDLL(str(out))
    launch = handle.call_mxfp4_quant
    launch.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    launch.restype = None
    from jit_util_mxfp4_a5 import VECTOR_CORES, current_stream_ptr

    def call(tensor, packed, scales):
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            tensor.shape[0],
            k,
        )

    return call


def batch_for(k):
    """Fixed batch, rounded up to the quantum so no shape takes the pad path."""
    quantum = row_quantum(k)
    return -(-K_SWEEP_BATCH // quantum) * quantum


def make_pool(batch, k):
    per_buffer = batch * k * 2
    depth = max(POOL_MIN, min(POOL_MAX, WORKING_SET_BYTES // max(per_buffer, 1)))
    # generated on device: a host fp32 randn of this size costs minutes and 2x
    # the memory, and the benchmark does not care which random values they are
    pool = [
        torch.randn(batch, k, dtype=torch.float32, device="npu").to(torch.bfloat16)
        for _ in range(depth)
    ]
    torch.npu.synchronize()
    return pool


def interleaved_micros(contenders, pool):
    """Time every contender round-robin, one bracket each, keeping all samples.

    Bracket i of every contender sees the same machine, which paired_speedup
    requires. The order ROTATES per bracket: with a fixed order the first contender
    pays whatever the previous bracket's last one left in cache, which is enough
    to make a preallocated arm read slower than an allocating one.
    """
    for _, call in contenders:
        for i in range(WARMUP):
            call(pool[i % len(pool)])
    torch.npu.synchronize()
    samples = {name: [] for name, _ in contenders}
    for bracket in range(BRACKETS):
        turn = bracket % len(contenders)
        for name, call in contenders[turn:] + contenders[:turn]:
            torch.npu.synchronize()
            start = time.perf_counter()
            for i in range(LAUNCHES):
                call(pool[i % len(pool)])
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / LAUNCHES)
    return samples


def paired_speedup(ours, theirs):
    """Median per-bracket ratio theirs/ours, with a percentile bootstrap 95% CI.

    Paired, because the brackets are interleaved: the ratio within a bracket
    cancels whatever drift both contenders saw. The interval is a bootstrap over
    those ratios, NOT their min-max range -- a range widens as you sample more,
    so using it as the test would make more evidence look like less. Resolved
    means the interval excludes 1.0.
    """
    pairs = [t / o for o, t in zip(ours, theirs) if o > 0 and t > 0]
    if len(pairs) < 3:
        return 0.0, 0.0, 0.0, False
    rng = random.Random(SEED)
    draws = sorted(
        statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(BOOTSTRAP)
    )
    return (
        statistics.median(pairs),
        draws[int(0.025 * BOOTSTRAP)],
        draws[int(0.975 * BOOTSTRAP) - 1],
        not (draws[int(0.025 * BOOTSTRAP)] <= 1.0 <= draws[int(0.975 * BOOTSTRAP) - 1]),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def gate(name, ours_out, other_out):
    """Bit-exactness before timing: a fast arm that computes nothing is the
    signature failure on this part."""
    (oq, os_), (tq, ts) = ours_out, other_out
    torch.npu.synchronize()
    packed = (oq == tq).float().mean().item()
    scale = (os_ == ts).float().mean().item()
    wrote = bool(oq.any().item())
    assert wrote, f"{name}: our arm wrote nothing"
    return packed, scale, wrote


def measure_pair(k, batch, which):
    """One matched pair, both arms on an identical call path.

    raw   ours_raw vs TQuant -- both a bare ctypes launch into the same source
          built twice, outputs preallocated. Only the four compute passes differ,
          so this isolates COMPUTE.
    api   ours vs torch_npu -- both one Python call that allocates its own
          outputs, which is what a caller actually gets. torch_npu has no other
          mode, so this is the only fair user-facing pairing.
    """
    from jit_util_mxfp4_a5 import compile_kernel

    pool = make_pool(batch, k)
    blocks = k // MX_BLOCK
    if which == "raw":
        qa = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
        sa = torch.empty((batch, blocks), dtype=torch.uint8, device="npu")
        qb = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
        sb = torch.empty((batch, blocks), dtype=torch.uint8, device="npu")
        mine = raw_launcher(compile_kernel(verbose=False), k)
        theirs = build_tquant(k)
        contenders = [
            ("ours_raw", lambda x: mine(x, qa, sa)),
            ("tquant", lambda x: theirs(x, qb, sb)),
        ]
        mine(pool[0], qa, sa)
        theirs(pool[0], qb, sb)
        match = gate("raw", (qa, sa), (qb, sb))
    else:
        quant = build_and_load(k=k, verbose=False)
        vendor = getattr(torch_npu, "npu_dynamic_mx_quant", None)
        if vendor is None:
            raise RuntimeError("npu_dynamic_mx_quant missing: no baseline")
        contenders = [
            ("ours", quant),
            ("torch_npu", lambda x: vendor(x, dst_type=VENDOR_DST_TYPE)),
        ]
        oq, os_ = quant(pool[0])
        tq, ts = vendor(pool[0], dst_type=VENDOR_DST_TYPE)
        ts = ts.reshape(ts.shape[0], -1)[:, :blocks]
        match = gate("api", (oq, os_), (tq, ts))

    samples = interleaved_micros(contenders, pool)
    first, second = contenders[0][0], contenders[1][0]
    ratio, low, high, resolved = paired_speedup(samples[first], samples[second])
    rows = []
    for name in (first, second):
        taken = samples[name]
        micros = statistics.median(taken)
        rows.append(
            {
                "pair": which,
                "k": k,
                "batch": batch,
                "contender": name,
                "micros": round(micros, 3),
                "p_lo": round(min(taken), 3),
                "p_hi": round(max(taken), 3),
                "spread_pct": round(100 * (max(taken) - min(taken)) / micros, 1),
                "gbs": round(batch * k * BYTES_PER_ELEM / (micros * 1e-6) / 1e9, 1),
                "packed_match": round(match[0], 6),
                "scale_match": round(match[1], 6),
                "speedup": round(ratio, 4) if name == second else 1.0,
                "speedup_lo": round(low, 4) if name == second else 1.0,
                "speedup_hi": round(high, 4) if name == second else 1.0,
                "resolved": int(resolved) if name == second else 0,
                "brackets_n": len(taken),
                "status": "ok",
            }
        )
    pool.clear()
    torch.npu.empty_cache()
    return rows


def tquant_builds():
    """Report whether the TQuant variant compiles, so the arm can be skipped."""
    try:
        build_tquant(K_LIST[0])
        return True
    except TQuantUnavailable as exc:
        print(f"skipping the raw pair: PTO here has no MXFP4 quantizer\n  {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="two matched-path comparisons")
    parser.add_argument(
        "--tag", default="1", help="suffix for the CSV; use one per process"
    )
    parser.add_argument("--axis", choices=["k", "batch"], default="k")
    parser.add_argument("--pairs", default="raw,api", help="which pairs to run")
    parser.add_argument("--ks", default="", help="comma list overriding K_LIST")
    args = parser.parse_args()

    # PR 221 sweeps ROWS PER LAUNCH over this list at a fixed width. Only 4096
    # and 8192 of those values are legal K here (SUPPORTED_K stops at 14336), so
    # the larger ones are the batch axis, not widths.
    pr221_batches = (4096, 8192, 16384, 32768, 65536, 131072)
    widths = tuple(int(v) for v in args.ks.split(",")) if args.ks else K_LIST
    for w in widths:
        assert w in SUPPORTED_K, f"K={w} has no instantiation"
    shapes = (
        [(k, batch_for(k)) for k in widths]
        if args.axis == "k"
        else [(BATCH_SWEEP_K, b) for b in pr221_batches]
    )
    label = "K" if args.axis == "k" else "batch"

    out = []
    for which in args.pairs.split(","):
        if which == "raw" and not tquant_builds():
            continue
        print(f"\n=== {which} pair, by {label} ===")
        print(
            f"{label:>7} {'ours':>8} {'other':>8} {'ratio':>7} {'95% CI':>16} {'res':>4}"
        )
        for width, batch in shapes:
            rows = measure_pair(width, batch, which)
            out += rows
            a, b = rows[0], rows[1]
            key = width if args.axis == "k" else batch
            print(
                f"{key:>7} {a['gbs']:>8.0f} {b['gbs']:>8.0f} {b['speedup']:>7.3f} "
                f"[{b['speedup_lo']:.3f}, {b['speedup_hi']:.3f}]"
                f"{'yes' if b['resolved'] else 'NO':>5}"
            )
    write_csv(BUILDDIR / f"pairs_{args.axis}_{args.tag}.csv", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
