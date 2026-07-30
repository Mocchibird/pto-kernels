#!/usr/bin/env python3
"""Does a deeper pipeline (more UB buffers, enabled by the A5's real ~248 KB UB
vs the hard-coded 192 KB) speed up fast_hadamard_256_a5?

For each (ROWS_PER_TILE, NBUF) that fits the UB budget it rebuilds the kernel,
checks correctness vs x @ Sylvester(256) (an overflow would corrupt the output),
and times bandwidth at a few batch sizes. Prints per-config so NBUF=4 (current
default) can be compared against 6/8."""
import ctypes, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 256
h = os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]
UB_BUDGET = 248 * 1024  # A5 physical UB per docs
ROWS_LIST = [32, 64]
NBUF_LIST = [
    2,
    4,
]  # NBUF>=6 device-faults (507035): the per-buffer event-ID reuse tops out ~4 outstanding
BATCHES = [4096, 16384, 65536, 262144]
TRIALS = 7


def aln512(b):
    return (b + 511) & ~511


def build(rows, nbuf, prefetch):
    tag = f"{rows}_{nbuf}"
    src = HERE / "fast_hadamard_256_a5.cpp"
    obj = HERE / f"build/nb256_{tag}.o"
    so = HERE / f"build/nb256_{tag}.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DROWS_PER_TILE={rows}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
        f"-DUB_USABLE_BYTES={UB_BUDGET}",
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-Wno-ignored-attributes",
        "-Wno-macro-redefined",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Xhost-start",
        "-Xhost-end",
        f"-I{h}/aarch64-linux/include",
        f"-I{h}/include",
    ]
    subprocess.run(
        [f"{h}/bin/bisheng", "-xcce", *common, "-c", str(src), "-o", str(obj)],
        check=True,
    )
    # copy256 moved into its own TU (copy_ref_256_a5.cpp); link it in so that
    # call_copy256 still resolves from this .so.
    cobj = obj.with_suffix(".copy.o")
    subprocess.run(
        [
            f"{h}/bin/bisheng",
            "-xcce",
            *common,
            "-c",
            str(HERE / "copy_ref_256_a5.cpp"),
            "-o",
            str(cobj),
        ],
        check=True,
    )
    subprocess.run(
        [
            f"{h}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{so.name}",
            str(obj),
            str(cobj),
            "-o",
            str(so),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(so))
    for nm in ("call_hadamard256", "call_copy256"):
        getattr(lib, nm).argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        getattr(lib, nm).restype = None
    return lib


def sylvester(n):
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def sp():
    return torch.npu.current_stream()._as_parameter_


def check(fn, bd, batch=4096):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, N)).astype(np.float16)
    gold = x_np.astype(np.float32) @ sylvester(N)
    x = torch.from_numpy(x_np).npu()
    fn(bd, sp(), ctypes.c_void_p(x.data_ptr()), batch)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    return float(np.abs(out - gold).max()) / (float(np.abs(gold).max()) or 1.0)


def gbs_median(fn, bd, batch):
    data = 2 * batch * N * 2
    pool = [torch.randn(batch, N, dtype=torch.float16).npu() for _ in range(8)]
    torch.npu.synchronize()
    it = {"k": 0}

    def one():
        b = pool[it["k"] % 8]
        it["k"] += 1
        fn(bd, sp(), ctypes.c_void_p(b.data_ptr()), batch)

    for _ in range(8):
        one()
    torch.npu.synchronize()
    samples = []
    for _ in range(TRIALS):
        s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            one()
        e.record()
        torch.npu.synchronize()
        samples.append(data / 1e9 / (s.elapsed_time(e) * 1e3 / 50 / 1e6))
    del pool
    samples.sort()
    return samples[len(samples) // 2]


def main():
    bd = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    header = (
        f"{'ROWS':>4} {'NBUF':>4} {'PF':>3} {'UB_KB':>6} {'rel':>9} {'ok':>3}  "
        + "  ".join(f"{b//1024}k" if b >= 1024 else str(b) for b in BATCHES)
        + "  (GB/s)"
    )
    print(header)
    for rows in ROWS_LIST:
        tile = aln512(rows * N * 2)
        for nbuf in NBUF_LIST:
            if nbuf * tile > UB_BUDGET:
                continue
            prefetch = max(1, nbuf - 1)
            lib = build(rows, nbuf, prefetch)
            rel = check(lib.call_hadamard256, bd)
            ok = "OK" if rel < 0.03 else "FAIL"
            gbs = [gbs_median(lib.call_hadamard256, bd, b) for b in BATCHES]
            ub_kb = nbuf * tile // 1024
            print(
                f"{rows:>4} {nbuf:>4} {prefetch:>3} {ub_kb:>6} {rel:>9.4g} {ok:>3}  "
                + "  ".join(f"{g:>6.0f}" for g in gbs)
            )
            sys.stdout.flush()
    print("NBUF SWEEP DONE")


if __name__ == "__main__":
    main()
