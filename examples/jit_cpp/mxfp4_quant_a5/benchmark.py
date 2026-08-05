"""Bandwidth benchmark: this kernel vs torch_npu, bf16 -> MXFP4, on Ascend A5.

Bandwidth counts every byte the operation must move: `2K` read plus `K/2 + K/32`
written, i.e. **2.53125 bytes per element**. Each contender's own byte count is
used, so a contender that writes less is not credited with the difference.

Fairness rules, because the easy version of this comparison is misleading:

* Output allocation is reported as a separate row rather than hidden. `torch_npu`
  allocates its outputs inherently, so comparing it against a preallocated kernel
  flatters us. Both allocating rows are the apples-to-apples pair.
* Per-launch `torch.npu.Event` timing for every contender. Wrapping one event pair
  around an N-rep loop amortises the launch cost away and would not be comparable.
* Inputs come from a rotating pool sized to `WORKING_SET_BYTES`, so reads miss
  cache. Both contenders share the identical pool at a given shape, so the
  ours-vs-vendor comparison is sound everywhere. The **footprint itself is not
  constant across the batch sweep** -- once one buffer exceeds `WORKING_SET_BYTES`
  the pool floors at `POOL_MIN`, giving 256 MiB at batch 4k/16k but 1 GiB at 64k
  and 4 GiB at 256k (the `footprint_mib` column records it). So read the batch
  sweep as per-batch comparisons, not as a bandwidth-vs-batch curve.
* `--repeat` takes the median of that many full sweeps; one sweep can disagree
  with the next by more than the effect being measured.

Emits `build/mxfp4_kbench.csv` (bandwidth vs K) and `build/mxfp4_bbench.csv`
(bandwidth vs batch).
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from jit_util_mxfp4_a5 import MX_BLOCK, SUPPORTED_K, build_and_load

HERE = Path(__file__).resolve().parent
BUILDDIR = HERE / "build"

K_LIST = list(SUPPORTED_K)
BATCHES = [4096, 16384, 65536, 262144]
K_SWEEP_BATCH = 65536  # bandwidth-bound for every K
BATCH_SWEEP_K = 4096
WORKING_SET_BYTES = 256 * 1024 * 1024  # footprint, held constant across shapes
POOL_MIN, POOL_MAX = 2, 512  # loose enough that the smallest shape still reaches it
TRIALS = 9  # per-launch event pairs; the median is reported
WARMUP = 3
MIN_DEVICE_MICROS = 20.0  # under this a launch times dispatch, not bandwidth
HBM_BOUND = 3400.0  # a rate above this is not a usable measurement
VENDOR_DST_TYPE = 296  # torch_npu.float4_e2m1fn_x2

BYTES_PER_ELEM = 2.0 + 0.5 + 1.0 / MX_BLOCK  # read bf16, write nibble + scale


def pool_depth(batch, k):
    per_buffer = batch * k * 2
    return max(POOL_MIN, min(POOL_MAX, WORKING_SET_BYTES // per_buffer))


def make_pool(batch, k):
    depth = pool_depth(batch, k)
    pool = [
        torch.randn(batch, k, dtype=torch.float32).to(torch.bfloat16).npu()
        for _ in range(depth)
    ]
    torch.npu.synchronize()
    return pool


def event_median(call, pool):
    """Median microseconds over TRIALS per-launch event pairs, rotating the pool."""
    for i in range(WARMUP):
        call(pool[i % len(pool)])
    torch.npu.synchronize()
    samples = []
    for i in range(TRIALS):
        x = pool[i % len(pool)]
        start, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        start.record()
        call(x)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def verdict(micros, gbs):
    if micros < MIN_DEVICE_MICROS:
        return f"launch-bound({micros:.0f}us)"
    if gbs > HBM_BOUND:
        return f"over-hbm({gbs:.0f})"
    return "ok"


def contenders(k, batch):
    """(label, allocates, callable) for each thing being compared."""
    quant = build_and_load(k=k, verbose=False)
    q = torch.empty((batch, k // 2), dtype=torch.uint8).npu()
    s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8).npu()
    torch.npu.synchronize()
    vendor = getattr(torch_npu, "npu_dynamic_mx_quant", None)

    rows = [
        ("ours", False, lambda x: quant(x, out=(q, s))),
        ("ours", True, quant),
    ]
    if vendor is not None:
        rows.append(("torch_npu", True, lambda x: vendor(x, dst_type=VENDOR_DST_TYPE)))
    return rows


def measure(k, batch):
    pool = make_pool(batch, k)
    footprint = len(pool) * batch * k * 2 / 2**20
    out = []
    for label, allocates, call in contenders(k, batch):
        micros, lo, hi = event_median(call, pool)
        gbs = batch * k * BYTES_PER_ELEM / (micros * 1e-6) / 1e9
        out.append(
            {
                "k": k,
                "batch": batch,
                "contender": label,
                "allocates": int(allocates),
                "pool": len(pool),
                "footprint_mib": round(footprint, 1),
                "micros": round(micros, 2),
                "p_lo": round(lo, 2),
                "p_hi": round(hi, 2),
                "gbs": round(gbs, 1),
                "status": verdict(micros, gbs),
            }
        )
    pool.clear()
    torch.npu.empty_cache()
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def median_of_sweeps(sweep, repeat):
    """Element-wise median across whole sweeps, keyed by the row's identity."""
    passes = [sweep() for _ in range(repeat)]
    keys = ["k", "batch", "contender", "allocates"]
    merged = []
    for base in passes[0]:
        ident = tuple(base[key] for key in keys)
        peers = [
            next(r for r in p if tuple(r[key] for key in keys) == ident) for p in passes
        ]
        row = dict(base)
        for col in ("micros", "p_lo", "p_hi", "gbs"):
            row[col] = round(statistics.median(float(r[col]) for r in peers), 2)
        row["status"] = next((r["status"] for r in peers if r["status"] != "ok"), "ok")
        merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description="MXFP4 bandwidth benchmark.")
    parser.add_argument("--repeat", type=int, default=1, help="median of N sweeps")
    parser.add_argument("--batch-sweep", action="store_true", help="also sweep batch")
    args = parser.parse_args()

    if getattr(torch_npu, "npu_dynamic_mx_quant", None) is None:
        print(
            "warning: torch_npu.npu_dynamic_mx_quant absent; no vendor row",
            file=sys.stderr,
        )

    krows = median_of_sweeps(
        lambda: [r for k in K_LIST for r in measure(k, K_SWEEP_BATCH)], args.repeat
    )
    for row in krows:
        print(
            f"  K={row['k']:<5} {row['contender']:<10} "
            f"alloc={row['allocates']} {row['micros']:>9.2f}us "
            f"{row['gbs']:>7.1f} GB/s  {row['status']}"
        )
    write_csv(BUILDDIR / "mxfp4_kbench.csv", krows)

    if args.batch_sweep:
        brows = median_of_sweeps(
            lambda: [r for b in BATCHES for r in measure(BATCH_SWEEP_K, b)],
            args.repeat,
        )
        for row in brows:
            print(
                f"  batch={row['batch']:<7} {row['contender']:<10} "
                f"alloc={row['allocates']} footprint={row['footprint_mib']}MiB "
                f"{row['gbs']:>7.1f} GB/s  {row['status']}"
            )
        write_csv(BUILDDIR / "mxfp4_bbench.csv", brows)


if __name__ == "__main__":
    main()
