"""
Benchmark fp16 Sinkhorn at transformer-head shapes — torch fp16 vs PTO NPU kernel.

K (head_dim) : 64, 128, 256
L (n_tokens) : 32, 64, 128, 256
Batch        : 1  (single (K, L) matrix per call)

Writes:
  outputs/csv/head_shapes_bench.csv
  outputs/csv/batched_vs_serial.csv
  outputs/plots/head_shapes_*.png
  outputs/plots/batched_vs_serial_*.png
"""

# pylint: disable=wrong-import-position
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch_npu  # noqa

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from jit_util_common import get_current_stream_ptr  # noqa: E402
from jit_util_sinkhorn import jit_compile  # noqa: E402

# --- Sinkhorn hyperparameters ------------------------------------------------

SINKHORN_ORDER = 8
SINKHORN_LR = 0.9
SINKHORN_EPS = 1e-6

# --- Benchmark grids --------------------------------------------------------

HEAD_DIMS = [64, 128, 256]
N_TOKENS = [32, 64, 128, 256]

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
BATCH_K = 128
BATCH_L = 128

KERNEL_WARMUP = 10
KERNEL_REPEATS = 50


# --- torch reference --------------------------------------------------------


def sinq_torch_fp16(matrix, sinkhorn_order=8, sinkhorn_lr=0.9, sinkhorn_eps=1e-6):
    """Vectorised torch SINQ on (N, K, L).  Stays in fp16."""
    K, L = matrix.shape[-2], matrix.shape[-1]
    m = matrix
    mu1 = torch.ones(*matrix.shape[:-2], L, dtype=m.dtype, device=m.device)
    mu2 = torch.ones(*matrix.shape[:-2], K, 1, dtype=m.dtype, device=m.device)
    tgt = (
        torch.minimum(
            m.std(dim=-1).amin(dim=-1, keepdim=True),
            m.std(dim=-2).amin(dim=-1, keepdim=True),
        ).unsqueeze(-1)
        + sinkhorn_eps
    )
    for _ in range(sinkhorn_order):
        cur = m / mu1.unsqueeze(-2) / mu2
        mu1 = mu1 * (cur.std(dim=-2) / tgt.squeeze(-1)) ** sinkhorn_lr
        mu2 = mu2 * ((cur.std(dim=-1) / tgt.squeeze(-1)) ** sinkhorn_lr).unsqueeze(-1)
    return m / mu1.unsqueeze(-2) / mu2, mu1, mu2.squeeze(-1)


# --- timing / metric helpers ------------------------------------------------


def time_npu(fn, warmup=KERNEL_WARMUP, repeats=KERNEL_REPEATS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats  # us


def bytes_per_call(K, L, dtype_bytes):
    return (2 * K * L + L + K) * dtype_bytes


def flops_per_call(K, L, order):
    return K * L * (6 * (order + 1) + 2)


# --- head-shapes bench -----------------------------------------------------


def run_head_shapes(sinq_func, stream_ptr, device):
    rows = []
    header = (
        f"{'K':>4s} {'L':>4s} | "
        f"{'torch_us':>9s} {'torch_GB/s':>10s} {'torch_GFLOPS':>12s} | "
        f"{'npu_us':>9s} {'npu_GB/s':>10s} {'npu_GFLOPS':>12s} | "
        f"{'speedup':>7s}"
    )
    print(header)
    print("-" * len(header))

    for K in HEAD_DIMS:
        for L in N_TOKENS:
            torch.random.manual_seed(42)
            mat = torch.rand(1, K, L, dtype=torch.float16, device=device) + 0.1
            out = torch.empty_like(mat)
            mu1 = torch.empty(1, L, dtype=torch.float16, device=device)
            mu2 = torch.empty(1, K, dtype=torch.float16, device=device)

            t_us = time_npu(
                lambda: sinq_torch_fp16(
                    mat,
                    sinkhorn_order=SINKHORN_ORDER,
                    sinkhorn_lr=SINKHORN_LR,
                    sinkhorn_eps=SINKHORN_EPS,
                )
            )
            k_us = time_npu(
                lambda: sinq_func(
                    mat,
                    out,
                    mu1,
                    mu2,
                    order=SINKHORN_ORDER,
                    lr=SINKHORN_LR,
                    eps=SINKHORN_EPS,
                    stream_ptr=stream_ptr,
                )
            )

            B = bytes_per_call(K, L, 2)
            F = flops_per_call(K, L, SINKHORN_ORDER)
            t_gbs = B / (t_us * 1e3)
            k_gbs = B / (k_us * 1e3)
            t_gflops = F / (t_us * 1e3)
            k_gflops = F / (k_us * 1e3)
            speedup = t_us / k_us

            print(
                f"{K:>4d} {L:>4d} | "
                f"{t_us:>9.2f} {t_gbs:>10.3f} {t_gflops:>12.3f} | "
                f"{k_us:>9.2f} {k_gbs:>10.3f} {k_gflops:>12.3f} | "
                f"{speedup:>7.2f}"
            )
            rows.append(
                {
                    "K": K,
                    "L": L,
                    "torch_us": t_us,
                    "torch_GB_s": t_gbs,
                    "torch_GFLOPS": t_gflops,
                    "npu_us": k_us,
                    "npu_GB_s": k_gbs,
                    "npu_GFLOPS": k_gflops,
                    "speedup": speedup,
                }
            )
    return rows


# --- batched-vs-serial bench -----------------------------------------------


def run_batched_vs_serial(sinq_func, stream_ptr, device):
    print(f"\nK={BATCH_K}, L={BATCH_L}, order={SINKHORN_ORDER}")
    print(
        f"{'N':>5}  {'batched us':>12}  {'per-mat us':>12}  "
        f"{'serial us':>12}  {'per-mat us':>12}  {'speedup':>8}"
    )
    rows = []
    for N in BATCH_SIZES:
        mat = torch.rand(N, BATCH_K, BATCH_L, dtype=torch.float16, device=device) + 0.1
        out = torch.empty_like(mat)
        mu1 = torch.empty(N, BATCH_L, dtype=torch.float16, device=device)
        mu2 = torch.empty(N, BATCH_K, dtype=torch.float16, device=device)

        b_us = time_npu(
            lambda: sinq_func(
                mat,
                out,
                mu1,
                mu2,
                order=SINKHORN_ORDER,
                lr=SINKHORN_LR,
                eps=SINKHORN_EPS,
                stream_ptr=stream_ptr,
            )
        )

        mats_1 = [
            (
                torch.rand(1, BATCH_K, BATCH_L, dtype=torch.float16, device=device)
                + 0.1,
                torch.empty(1, BATCH_K, BATCH_L, dtype=torch.float16, device=device),
                torch.empty(1, BATCH_L, dtype=torch.float16, device=device),
                torch.empty(1, BATCH_K, dtype=torch.float16, device=device),
            )
            for _ in range(N)
        ]

        def serial_fn():
            for m, o, m1, m2 in mats_1:
                sinq_func(
                    m,
                    o,
                    m1,
                    m2,
                    order=SINKHORN_ORDER,
                    lr=SINKHORN_LR,
                    eps=SINKHORN_EPS,
                    stream_ptr=stream_ptr,
                )

        s_us = time_npu(serial_fn)
        speedup = s_us / b_us if b_us > 0 else float("nan")

        print(
            f"{N:>5d}  {b_us:>12.2f}  {b_us/N:>12.2f}  "
            f"{s_us:>12.2f}  {s_us/N:>12.2f}  {speedup:>8.2f}x"
        )
        rows.append(
            {
                "N": N,
                "batched_us": b_us,
                "batched_per_mat_us": b_us / N,
                "serial_us": s_us,
                "serial_per_mat_us": s_us / N,
                "speedup": speedup,
            }
        )
    return rows


# --- plots ------------------------------------------------------------------


def _shape_labels(rows):
    return [f"{r['K']}x{r['L']}" for r in rows]


def plot_speedup(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.full((len(HEAD_DIMS), len(N_TOKENS)), np.nan)
    for r in rows:
        i = HEAD_DIMS.index(r["K"])
        j = N_TOKENS.index(r["L"])
        grid[i, j] = r["speedup"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(
        grid, aspect="auto", cmap="viridis", vmin=1.0, vmax=max(np.nanmax(grid), 1.0)
    )
    ax.set_xticks(range(len(N_TOKENS)), [str(l) for l in N_TOKENS])
    ax.set_yticks(range(len(HEAD_DIMS)), [str(k) for k in HEAD_DIMS])
    ax.set_xlabel("n_tokens")
    ax.set_ylabel("head_dim")
    ax.set_title(f"PTO NPU fp16 vs torch fp16 — speedup (x), order={SINKHORN_ORDER}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.1f}x",
                ha="center",
                va="center",
                color="white" if grid[i, j] < grid.max() * 0.6 else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


def _grouped_bar(rows, torch_key, npu_key, ylabel, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _shape_labels(rows)
    t_vals = [r[torch_key] for r in rows]
    n_vals = [r[npu_key] for r in rows]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, t_vals, w, label="torch fp16", color="#94a3b8")
    ax.bar(x + w / 2, n_vals, w, label="PTO NPU fp16", color="#dc2626")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_xlabel("shape (head_dim x n_tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_batched(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in rows]
    per_mat_batch = [r["batched_per_mat_us"] for r in rows]
    per_mat_serial = [r["serial_per_mat_us"] for r in rows]
    total_batch = [r["batched_us"] for r in rows]
    total_serial = [r["serial_us"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(Ns, per_mat_batch, "o-", color="#dc2626", label="batched (one launch)")
    ax1.plot(Ns, per_mat_serial, "s--", color="#94a3b8", label="serial (N launches)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(Ns, [str(n) for n in Ns])
    ax1.set_xlabel("batch size N")
    ax1.set_ylabel("per-matrix latency (us, log)")
    ax1.set_title(f"Per-matrix cost @ K=L={BATCH_K}, order={SINKHORN_ORDER}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(Ns, total_batch, "o-", color="#dc2626", label="batched")
    ax2.plot(Ns, total_serial, "s--", color="#94a3b8", label="serial")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(Ns, [str(n) for n in Ns])
    ax2.set_xlabel("batch size N")
    ax2.set_ylabel("total latency (us, log)")
    ax2.set_title("Total wall time + speedup")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2_r = ax2.twinx()
    ax2_r.plot(Ns, speedup, "^:", color="#059669", label="speedup (x)")
    ax2_r.set_ylabel("batched speedup over serial (x)", color="#059669")
    ax2_r.tick_params(axis="y", labelcolor="#059669")

    fig.suptitle("PTO NPU sinkhorn — batched vs serial launches (log y)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


# --- main -------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO Sinkhorn (head-shapes + batched sweep)."
    )
    parser.add_argument("--npu", type=str, default="npu:0")
    parser.add_argument("--no-cache-stream", dest="cache_stream", action="store_false")
    parser.add_argument("--warmup", type=int, default=KERNEL_WARMUP)
    parser.add_argument("--repeats", type=int, default=KERNEL_REPEATS)
    parser.add_argument(
        "--skip-batched", action="store_true", help="Skip the batched-vs-serial sweep."
    )
    parser.set_defaults(cache_stream=True)
    return parser.parse_args()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {path}")


def main():
    global KERNEL_WARMUP, KERNEL_REPEATS
    args = _parse_args()
    KERNEL_WARMUP = args.warmup
    KERNEL_REPEATS = args.repeats

    torch.npu.set_device(args.npu)
    device = args.npu
    base = THIS_DIR

    print(f"Using device: {device}")
    print("Compiling kernel_sinkhorn.cpp ...")
    sinq_func = jit_compile(
        str(base / "kernel_sinkhorn.cpp"),
        verbose=True,
        device=device,
    )
    stream_ptr = get_current_stream_ptr() if args.cache_stream else None
    if stream_ptr is not None:
        print("Using cached NPU stream pointer for PTO launches.")

    # --- head shapes ---
    csv_dir = base / "outputs" / "csv"
    plot_dir = base / "outputs" / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    hs_rows = run_head_shapes(sinq_func, stream_ptr, device)
    _write_csv(csv_dir / "head_shapes_bench.csv", hs_rows)

    plot_speedup(hs_rows, plot_dir / "head_shapes_speedup.png")
    _grouped_bar(
        hs_rows,
        "torch_GB_s",
        "npu_GB_s",
        ylabel="effective bandwidth (GB/s)",
        title=f"Sinkhorn fp16 bandwidth — order={SINKHORN_ORDER}, batch=1",
        path=plot_dir / "head_shapes_bandwidth.png",
    )
    _grouped_bar(
        hs_rows,
        "torch_GFLOPS",
        "npu_GFLOPS",
        ylabel="effective GFLOPS",
        title=f"Sinkhorn fp16 compute throughput — order={SINKHORN_ORDER}, batch=1",
        path=plot_dir / "head_shapes_flops.png",
    )

    # --- batched vs serial ---
    if not args.skip_batched:
        bs_rows = run_batched_vs_serial(sinq_func, stream_ptr, device)
        _write_csv(csv_dir / "batched_vs_serial.csv", bs_rows)
        plot_batched(bs_rows, plot_dir / "batched_vs_serial_log.png")


if __name__ == "__main__":
    main()
