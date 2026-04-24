"""
Bandwidth benchmark for the two K=4 NPU sinkhorn kernels.

Compares:
  - v1: kernel_sinkhorn.cpp            (minimal PTO demo, BATCH=8 stacking)
  - v2: kernel_sinkhon_v2.cpp          (fast-path + small-batch dispatch)

Shape: (num_tokens, 4, 4) fp16 NPU tensors.  num_tokens sweep follows the
GPU bench so plots align across platforms.

Effective bandwidth is the one-read-one-write global traffic:
    bytes_moved = 2 * num_tokens * 4 * 4 * sizeof(fp16)
    GiB/s       = bytes_moved / elapsed / 1024^3

Outputs:
    outputs/bench_sinkhorn.csv
    outputs/bench_sinkhorn.png
"""
import csv
import ctypes
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from jit_util_sinkhorn    import sinkhorn_normalize, launch_v1_raw      # v1
from jit_util_sinkhorn_v2 import sinkhorn_normalize_v2, launch_v2_raw   # v2


_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "outputs"

N = 4                 # mhc / hidden_size — fixed per deepseek
REPEAT = 10           # sinkhorn iterations, matches upstream test
EPS = 1e-6
BYTES_PER_ELEM = 2    # fp16
WARMUP = 10
REPEATS = 50

# num_tokens sweep: dense powers of 2 from 1 up to 1M so the BW-bound
# regime is fully resolved.  Superset of gpu/bench_sinkhorn.py's grid.
NUM_TOKENS = [
    1 << k for k in range(1, 21)  # 1024 … 1048576
]


def time_npu(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats / 1000.0  # seconds


def effective_gib_s(num_tokens: int, seconds: float) -> float:
    bytes_moved = 2 * num_tokens * N * N * BYTES_PER_ELEM
    return bytes_moved / seconds / (1024 ** 3)


def _warm_compile() -> None:
    """Prime both kernels once so JIT compile cost isn't on the clock."""
    x = torch.randn((1, N, N), dtype=torch.float16, device="npu")
    sinkhorn_normalize(x, REPEAT, EPS)
    sinkhorn_normalize_v2(x, REPEAT, EPS)
    torch.npu.synchronize()


def run_bench():
    _warm_compile()
    rows = []
    header = (
        f"{'num_tokens':>11s} | {'bytes':>12s} | "
        f"{'v1 μs':>9s} {'v1 GiB/s':>9s} | "
        f"{'v2 μs':>9s} {'v2 GiB/s':>9s} | "
        f"{'v2/v1':>6s}"
    )
    print(header)
    print("-" * len(header))

    for nt in NUM_TOKENS:
        torch.manual_seed(42 + nt)
        x   = torch.randn((nt, N, N), dtype=torch.float16, device="npu")
        out = torch.empty_like(x)
        # Pre-build ctypes pointers so the hot path doesn't pay for
        # `ctypes.c_void_p(tensor.data_ptr())` on every iteration.
        x_ptr   = ctypes.c_void_p(x.data_ptr())
        out_ptr = ctypes.c_void_p(out.data_ptr())

        v1_s = time_npu(lambda: launch_v1_raw(x_ptr, out_ptr, nt, REPEAT, EPS))
        v2_s = time_npu(lambda: launch_v2_raw(x_ptr, out_ptr, nt, REPEAT, EPS))

        bytes_moved = 2 * nt * N * N * BYTES_PER_ELEM
        v1_gib = effective_gib_s(nt, v1_s)
        v2_gib = effective_gib_s(nt, v2_s)
        speedup = v1_s / v2_s

        rows.append({
            "num_tokens": nt,
            "bytes_moved": bytes_moved,
            "v1_us": v1_s * 1e6,
            "v1_gib_s": v1_gib,
            "v2_us": v2_s * 1e6,
            "v2_gib_s": v2_gib,
            "speedup_v2_over_v1": speedup,
        })
        print(
            f"{nt:>11d} | {bytes_moved:>12d} | "
            f"{v1_s*1e6:>9.2f} {v1_gib:>9.2f} | "
            f"{v2_s*1e6:>9.2f} {v2_gib:>9.2f} | "
            f"{speedup:>6.2f}x"
        )
    return rows


def plot_bandwidth(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs     = [r["num_tokens"] for r in rows]
    v1_ys  = [r["v1_gib_s"] for r in rows]
    v2_ys  = [r["v2_gib_s"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(xs, v1_ys, "-o", color="#94a3b8", linewidth=1.8,
            label="v1  kernel_sinkhorn")
    ax.plot(xs, v2_ys, "-o", color="#dc2626", linewidth=1.8,
            label="v2  kernel_sinkhon_v2 (K=4)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("num_tokens  (number of 4×4 matrices)")
    ax.set_ylabel("effective bandwidth (GiB/s)")
    ax.set_title(f"NPU sinkhorn fp16 — K={N}, repeat={REPEAT}")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved → {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_bench()

    csv_path = OUTPUT_DIR / "bench_sinkhorn.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved → {csv_path}")
    plot_bandwidth(rows, OUTPUT_DIR / "bench_sinkhorn.png")


if __name__ == "__main__":
    main()
