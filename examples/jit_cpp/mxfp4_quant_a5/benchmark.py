"""Bandwidth benchmark: this kernel vs torch_npu, bf16 -> MXFP4, on Ascend A5.

Bandwidth counts every byte the operation must move: `2K` read plus `K/2 + K/32`
written, i.e. 2.53125 bytes per element. Each contender's own byte count is used.

TIMING. LAUNCHES launches fire back to back between two synchronizes and the wall
clock is divided by LAUNCHES, identically for every contender. These are
steady-state throughput figures, not single-call latency. Contenders are
interleaved one bracket at a time, and the reported ratio is the median paired
per-bracket ratio with a bootstrap 95% interval; a shape whose interval spans 1.0
is reported as unresolved.

THE K SWEEP USES A FIXED BATCH, so every row answers the question a caller has:
"at this batch, what do I get at each K?". Total work therefore scales with K,
and the elems_mi column records it.

FAIRNESS. Output allocation is its own row: torch_npu allocates inherently, so
the two allocating rows are the apples-to-apples pair. A device-to-device copy
row is a roofline bound in the CSV, at 4 B/elem rather than 2.53; verdict() flags
any contender that reads above it.

Emits build/mxfp4_kbench.csv (vs K, fixed batch) and build/mxfp4_bbench.csv
(vs batch, at one K).
"""

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

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
EXCLUDED_K = [k for k in HADAMARD_NS if k not in K_LIST]
K_SWEEP_BATCH = 65536  # fixed across the K sweep
BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
BATCH_SWEEP_K = 4096
WORKING_SET_BYTES = 1024 * 1024 * 1024
POOL_MIN, POOL_MAX = 2, 64
LAUNCHES = 40
BRACKETS = 32
BOOTSTRAP = 2000
SEED = 20260811
WARMUP = 5
MIN_BRACKET_MICROS = 2.0
VENDOR_DST_TYPE = 296

BYTES_PER_ELEM = 2.0 + 0.5 + 1.0 / MX_BLOCK
COPY_BYTES_PER_ELEM = 4.0


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


def verdict(micros, gbs, copy_gbs):
    if micros < MIN_BRACKET_MICROS:
        return f"too-fast-to-time({micros:.1f}us)"
    if copy_gbs and gbs > 1.20 * copy_gbs:
        return f"above-copy({gbs:.0f}>{copy_gbs:.0f})"
    return "ok"


def measure(k, batch):
    pool = make_pool(batch, k)
    footprint = len(pool) * batch * k * 2 / 2**20
    quant = build_and_load(k=k, verbose=False)
    q = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
    s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8, device="npu")
    dst = torch.empty((batch, k), dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    vendor = getattr(torch_npu, "npu_dynamic_mx_quant", None)

    rows = [
        ("ours", 0, BYTES_PER_ELEM, lambda x: quant(x, out=(q, s))),
        ("ours", 1, BYTES_PER_ELEM, quant),
    ]
    if vendor is not None:
        rows.append(
            (
                "torch_npu",
                1,
                BYTES_PER_ELEM,
                lambda x: vendor(x, dst_type=VENDOR_DST_TYPE),
            )
        )
    rows.append(("d2d_copy", 0, COPY_BYTES_PER_ELEM, dst.copy_))

    # One interleaved pass over every contender, so each bracket sees the same
    # machine and the paired ratio below is meaningful.
    keys = [(label, allocates) for label, allocates, _, _ in rows]
    try:
        samples = interleaved_micros(
            [(f"{label}/{allocates}", call) for label, allocates, _, call in rows],
            pool,
        )
    except Exception as exc:  # a vendor op may reject an unusual shape
        return [
            {
                "k": k,
                "batch": batch,
                "contender": label,
                "allocates": allocates,
                "bytes_per_elem": per_elem,
                "pool": len(pool),
                "elems_mi": round(batch * k / 2**20, 1),
                "footprint_mib": round(footprint, 1),
                "micros": 0.0,
                "p_lo": 0.0,
                "p_hi": 0.0,
                "spread_pct": 0.0,
                "gbs": 0.0,
                "speedup": 0.0,
                "speedup_lo": 0.0,
                "speedup_hi": 0.0,
                "resolved": 0,
                "status": f"error:{type(exc).__name__}",
            }
            for label, allocates, per_elem, _ in rows
        ]

    # The apples-to-apples pair: both allocating, which torch_npu must do anyway.
    base = samples.get("ours/1")
    out, copy_gbs = [], 0.0
    for (label, allocates), (_, _, per_elem, _) in zip(keys, rows):
        taken = samples[f"{label}/{allocates}"]
        micros = statistics.median(taken)
        lo, hi = min(taken), max(taken)
        gbs = batch * k * per_elem / (micros * 1e-6) / 1e9
        if label == "d2d_copy":
            copy_gbs = gbs
        if base is not None and label != "ours":
            ratio, ratio_lo, ratio_hi, resolved = paired_speedup(base, taken)
        else:
            ratio, ratio_lo, ratio_hi, resolved = 1.0, 1.0, 1.0, 0
        out.append(
            {
                "k": k,
                "batch": batch,
                "contender": label,
                "allocates": allocates,
                "bytes_per_elem": per_elem,
                "pool": len(pool),
                "elems_mi": round(batch * k / 2**20, 1),
                "footprint_mib": round(footprint, 1),
                "micros": round(micros, 2),
                "p_lo": round(lo, 2),
                "p_hi": round(hi, 2),
                "spread_pct": round(100.0 * (hi - lo) / micros, 1),
                "gbs": round(gbs, 1),
                "speedup": round(ratio, 4),
                "speedup_lo": round(ratio_lo, 4),
                "speedup_hi": round(ratio_hi, 4),
                "resolved": int(resolved),
                "status": "ok",
                "brackets": taken,
            }
        )
    for row in out:
        if row["status"] == "ok":
            row["status"] = verdict(row["micros"], row["gbs"], copy_gbs)
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
    """Element-wise median across whole sweeps. Records `repeat` in every row so
    a plot cannot claim a different number than was actually run."""
    passes = [sweep() for _ in range(repeat)]
    keys = ["k", "batch", "contender", "allocates"]
    merged, pooled = [], {}
    for base in passes[0]:
        ident = tuple(base[key] for key in keys)
        peers = [
            next(r for r in p if tuple(r[key] for key in keys) == ident) for p in passes
        ]
        row = dict(base)
        for col in ("micros", "p_lo", "p_hi", "spread_pct", "gbs"):
            row[col] = round(statistics.median(float(r[col]) for r in peers), 2)
        row["status"] = next((r["status"] for r in peers if r["status"] != "ok"), "ok")
        row["sweeps"] = repeat
        # every bracket from every sweep, so the ratio below covers run-to-run
        # drift and not just the within-process spread of one sweep
        pooled[ident] = [b for r in peers for b in r["brackets"]]
        merged.append(row)

    for row in merged:
        base_ident = (row["k"], row["batch"], "ours", 1)
        samples = pooled.get(base_ident)
        mine = pooled[tuple(row[key] for key in keys)]
        if samples is None or row["contender"] == "ours":
            ratio, low, high, resolved = 1.0, 1.0, 1.0, 0
        else:
            ratio, low, high, resolved = paired_speedup(samples, mine)
        row["speedup"] = round(ratio, 4)
        row["speedup_lo"] = round(low, 4)
        row["speedup_hi"] = round(high, 4)
        row["resolved"] = int(resolved)
        del row["brackets"]
    return merged


def report(rows, axis, keys):
    print(
        f"\n  {'shape':<22}{'ours(prealloc)':>16}{'ours':>10}{'torch':>10}"
        f"{'ratio':>8}{'copy':>8}{'spread':>8}"
    )
    for key in keys:
        got = {(r["contender"], r["allocates"]): r for r in rows if r[axis] == key}
        pre = got.get(("ours", 0))
        ours = got.get(("ours", 1))
        ven = got.get(("torch_npu", 1))
        cp = got.get(("d2d_copy", 0))
        if not (pre and ours):
            continue
        label = (
            f"{axis}={key} ({pre['elems_mi']:.0f}M el)"
            if axis == "k"
            else f"batch={key}"
        )
        ratio = f"{ours['gbs'] / ven['gbs']:.3f}" if ven and ven["gbs"] else "-"
        print(
            f"  {label:<22}{pre['gbs']:>16.0f}{ours['gbs']:>10.0f}"
            f"{(ven['gbs'] if ven else 0):>10.0f}{ratio:>8}"
            f"{(cp['gbs'] if cp else 0):>8.0f}{ours['spread_pct']:>7.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="MXFP4 bandwidth benchmark.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-sweep", action="store_true")
    args = parser.parse_args()

    if getattr(torch_npu, "npu_dynamic_mx_quant", None) is None:
        print("warning: npu_dynamic_mx_quant absent; no vendor row", file=sys.stderr)

    print(
        f"=== K sweep at fixed batch {K_SWEEP_BATCH}, "
        f"median of {args.repeat} sweeps ==="
    )
    krows = median_of_sweeps(
        lambda: [r for k in K_LIST for r in measure(k, batch_for(k))], args.repeat
    )
    report(krows, "k", K_LIST)
    write_csv(BUILDDIR / "mxfp4_kbench.csv", krows)

    if args.batch_sweep:
        print(
            f"\n=== batch sweep at K={BATCH_SWEEP_K}, "
            f"median of {args.repeat} sweeps ==="
        )
        brows = median_of_sweeps(
            lambda: [r for b in BATCHES for r in measure(BATCH_SWEEP_K, b)],
            args.repeat,
        )
        report(brows, "batch", BATCHES)
        write_csv(BUILDDIR / "mxfp4_bbench.csv", brows)

    flagged = [r for r in krows if r["status"] != "ok"]
    print(f"\n  rows flagged: {len(flagged)}")
    for row in flagged[:10]:
        print(f"    K={row['k']} {row['contender']}: {row['status']}")
    live = [r for r in krows if r["gbs"]]
    print(f"  worst bracket spread: {max(r['spread_pct'] for r in live):.1f}%")
    print("BENCH DONE")


if __name__ == "__main__":
    main()
