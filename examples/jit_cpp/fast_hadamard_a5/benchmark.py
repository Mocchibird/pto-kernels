#!/usr/bin/env python3
"""On-device benchmark for fast_hadamard_a5 (Ascend 950 / A5).

Compiles the kernel, checks correctness against the natural Sylvester
Walsh-Hadamard, then times it and reports achieved HBM bandwidth.

Usage (on a 950 server, with torch_npu):
    export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
    python benchmark.py --npu 0 --block-dim 20
    python benchmark.py --batches 1024,4096,16384,65536 --repeats 50 --csv out.csv

N is fixed at 128 (the block size this v1 kernel supports).
"""

import argparse
import csv
import sys

import numpy as np

N = 128
BYTES_PER_ELEM = 2  # fp16


def sylvester(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def reference(x_f32):
    """Natural-order WHT / sqrt(N), per row. x_f32: (batch, N) float32."""
    H = sylvester(N)
    return (x_f32 @ H.T) / np.sqrt(N)


def check_correctness(hadamard_func, torch, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch, N)).astype(np.float16)
    gold = reference(x_np.astype(np.float32))
    x = torch.from_numpy(x_np).npu()
    hadamard_func(x, batch)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    max_abs = float(np.abs(out - gold).max())
    denom = float(np.abs(gold).max()) or 1.0
    rel = max_abs / denom
    ok = rel < 0.02  # fp16 accumulation over log2(N)=7 stages
    print(f"[correctness] batch={batch} N={N}: max_abs_diff={max_abs:.4g} "
          f"rel={rel:.4g} -> {'OK' if ok else 'FAIL'}")
    return ok


def time_us(hadamard_func, torch, batch, block_dim, warmup, repeats):
    # in-place kernel: use independent buffers so each timed launch sees fresh data
    pool = [torch.randn(batch, N, dtype=torch.float16).npu()
            for _ in range(warmup + repeats)]
    torch.npu.synchronize()
    for i in range(warmup):
        hadamard_func(pool[i], batch, block_dim=block_dim)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for i in range(repeats):
        hadamard_func(pool[warmup + i], batch, block_dim=block_dim)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / repeats  # ms/rep -> us/rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", default="0")
    ap.add_argument("--batches", default="256,1024,4096,16384,65536")
    ap.add_argument("--block-dim", type=int, default=20,
                    help="launch grid = #AIC on your 950 (each spawns 2 AIV). Tune to the device.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa
    from jit_util_hadamard_a5 import build_and_load

    torch.npu.set_device(int(str(args.npu).split(":")[-1]))
    fn = build_and_load(N, block_dim=args.block_dim)

    if not check_correctness(fn, torch):
        print("Correctness failed; aborting benchmark.", file=sys.stderr)
        sys.exit(1)

    batches = [int(b) for b in args.batches.split(",") if b]
    rows = []
    hdr = f"{'batch':>8}  {'N':>4}  {'dur_us':>10}  {'GB/s':>9}  {'TB/s':>7}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for batch in batches:
        us = time_us(fn, torch, batch, args.block_dim, args.warmup, args.repeats)
        data_bytes = 2 * batch * N * BYTES_PER_ELEM  # load + store
        gbs = (data_bytes / 1e9) / (us / 1e6)
        print(f"{batch:>8}  {N:>4}  {us:>10.3f}  {gbs:>9.1f}  {gbs / 1000:>7.3f}")
        rows.append({"batch": batch, "N": N, "block_dim": args.block_dim,
                     "duration_us": round(us, 4), "bytes": data_bytes,
                     "bandwidth_gbs": round(gbs, 2)})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
