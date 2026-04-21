"""
Re-plot Sinkhorn benchmark results from saved CSVs (no NPU needed).

Reads:
  outputs/csv/head_shapes_bench.csv
  outputs/csv/batched_vs_serial.csv

Writes:
  outputs/plots/head_shapes_*.png
  outputs/plots/batched_vs_serial_*.png
"""
import argparse
import csv
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

HEAD_DIMS = [64, 128, 256]
N_TOKENS = [32, 64, 128, 256]
SINKHORN_ORDER = 8
BATCH_K = 128


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(rows, keys):
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = float(r[k])
    return rows


def _shape_labels(rows):
    return [f"{int(float(r['K']))}x{int(float(r['L']))}" for r in rows]


def plot_speedup(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.full((len(HEAD_DIMS), len(N_TOKENS)), np.nan)
    for r in rows:
        K, L = int(float(r["K"])), int(float(r["L"]))
        if K in HEAD_DIMS and L in N_TOKENS:
            grid[HEAD_DIMS.index(K), N_TOKENS.index(L)] = float(r["speedup"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   vmin=1.0, vmax=max(np.nanmax(grid), 1.0))
    ax.set_xticks(range(len(N_TOKENS)), [str(l) for l in N_TOKENS])
    ax.set_yticks(range(len(HEAD_DIMS)), [str(k) for k in HEAD_DIMS])
    ax.set_xlabel("n_tokens")
    ax.set_ylabel("head_dim")
    ax.set_title(f"PTO NPU fp16 vs torch fp16 — speedup (x), order={SINKHORN_ORDER}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}x",
                        ha="center", va="center",
                        color="white" if grid[i, j] < np.nanmax(grid) * 0.6 else "black",
                        fontsize=10)
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


def grouped_bar(rows, key_a, key_b, label_a, label_b, ylabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _shape_labels(rows)
    a_vals = [float(r[key_a]) for r in rows]
    b_vals = [float(r[key_b]) for r in rows]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, a_vals, w, label=label_a, color="#94a3b8")
    ax.bar(x + w / 2, b_vals, w, label=label_b, color="#dc2626")
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

    Ns = [int(float(r["N"])) for r in rows]
    per_mat_batch = [float(r["batched_per_mat_us"]) for r in rows]
    per_mat_serial = [float(r["serial_per_mat_us"]) for r in rows]
    total_batch = [float(r["batched_us"]) for r in rows]
    total_serial = [float(r["serial_us"]) for r in rows]
    speedup = [float(r["speedup"]) for r in rows]

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


def main():
    parser = argparse.ArgumentParser(description="Re-plot Sinkhorn benchmark CSVs.")
    parser.add_argument("--base-dir", type=str, default=str(THIS_DIR))
    args = parser.parse_args()
    base = Path(args.base_dir)

    csv_dir = base / "outputs" / "csv"
    plot_dir = base / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Head shapes
    hs_csv = csv_dir / "head_shapes_bench.csv"
    if hs_csv.exists():
        rows = _load_csv(hs_csv)
        plot_speedup(rows, plot_dir / "head_shapes_speedup.png")
        grouped_bar(
            rows, "torch_GB_s", "npu_GB_s", "torch fp16", "PTO NPU fp16",
            ylabel="effective bandwidth (GB/s)",
            title=f"Sinkhorn fp16 bandwidth — order={SINKHORN_ORDER}, batch=1",
            path=plot_dir / "head_shapes_bandwidth.png",
        )
        grouped_bar(
            rows, "torch_GFLOPS", "npu_GFLOPS", "torch fp16", "PTO NPU fp16",
            ylabel="effective GFLOPS",
            title=f"Sinkhorn fp16 compute throughput — order={SINKHORN_ORDER}, batch=1",
            path=plot_dir / "head_shapes_flops.png",
        )
    else:
        print(f"Warning: {hs_csv} not found, skipping head-shapes plots.")

    # Batched vs serial
    bs_csv = csv_dir / "batched_vs_serial.csv"
    if bs_csv.exists():
        rows = _load_csv(bs_csv)
        plot_batched(rows, plot_dir / "batched_vs_serial_log.png")
    else:
        print(f"Warning: {bs_csv} not found, skipping batched plots.")


if __name__ == "__main__":
    main()
