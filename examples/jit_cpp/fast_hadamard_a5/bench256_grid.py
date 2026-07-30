#!/usr/bin/env python3
"""Grid-benchmark fast_hadamard_256_a5 over (batch x ROWS_PER_TILE), cleaned up.

Fixes vs v1 (which produced impossible >HBM copy reads):
  * copy floor measured ONCE per batch from a FIXED, UB-valid ROWS=64 build
    (v1 recompiled copy256 at each ROWS; at ROWS=256 its hardcoded 2-buffer
     ping-pong = 2*128 KB overran the 192 KB UB -> garbage timing).
  * MEDIAN of several trials instead of a single timed loop -> rejects the
    occasional event-timer glitch that read ~2x too fast.
  * larger buffer pool -> working set >> L2, so the copy hits HBM not cache.
  * ROWS_PER_TILE=256 dropped (NBUF=1, buffering-limited, not useful).

Emits CSV: rows,nbuf,batch,had_gbs,copy_gbs,ratio -> build/grid256.csv (+ stdout).
copy_gbs is the fixed ROWS=64 reference for that batch (same across the ROWS axis)."""
import ctypes, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 256
h = os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]
ROWS_LIST = [16, 32, 64, 128]
BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]  # 2^10..2^18
COPY_ROWS = 64  # fixed, UB-valid tiling for the copy-floor reference
POOL = 8  # working set >> L2 to avoid cache-resident (too-fast) copies
TRIALS = 7  # median over trials rejects timer glitches


def nbuf_for(rows):
    return max(1, min(4, (192 * 1024) // (rows * N * 2)))


def build(rows, nbuf, pf, tag):
    src = HERE / "fast_hadamard_256_a5.cpp"
    obj = HERE / f"build/g256_{tag}.o"
    so = HERE / f"build/g256_{tag}.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DROWS_PER_TILE={rows}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={pf}",
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


def sp():
    return torch.npu.current_stream()._as_parameter_


def gbs_median(fn, bd, batch):
    """Median bandwidth over TRIALS, each trial = r reps over a POOL round-robin."""
    data = 2 * batch * N * 2
    r = 50
    pool = [torch.randn(batch, N, dtype=torch.float16).npu() for _ in range(POOL)]
    torch.npu.synchronize()
    it = {"k": 0}

    def one():
        b = pool[it["k"] % POOL]
        it["k"] += 1
        fn(bd, sp(), ctypes.c_void_p(b.data_ptr()), batch)

    for _ in range(8):
        one()
    torch.npu.synchronize()
    gs = []
    for _ in range(TRIALS):
        s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(r):
            one()
        e.record()
        torch.npu.synchronize()
        us = s.elapsed_time(e) * 1e3 / r
        gs.append(data / 1e9 / (us / 1e6))
    del pool
    gs.sort()
    return gs[len(gs) // 2]


def main():
    bd = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    # ---- fixed copy-floor reference (ROWS=64), measured once per batch ----
    cref_lib = build(
        COPY_ROWS, nbuf_for(COPY_ROWS), min(2, nbuf_for(COPY_ROWS) - 1), "copyref"
    )
    copy_ref = {}
    for batch in BATCHES:
        copy_ref[batch] = gbs_median(cref_lib.call_copy256, bd, batch)

    print("rows,nbuf,batch,had_gbs,copy_gbs,ratio")
    out = ["rows,nbuf,batch,had_gbs,copy_gbs,ratio"]
    for rows in ROWS_LIST:
        nbuf = nbuf_for(rows)
        pf = min(2, max(0, nbuf - 1))
        lib = build(rows, nbuf, pf, str(rows))
        for batch in BATCHES:
            if batch % rows != 0:
                continue
            hg = gbs_median(lib.call_hadamard256, bd, batch)
            cg = copy_ref[batch]
            line = f"{rows},{nbuf},{batch},{hg:.1f},{cg:.1f},{hg/cg:.4f}"
            print(line)
            sys.stdout.flush()
            out.append(line)
    (HERE / "build/grid256.csv").write_text("\n".join(out) + "\n")
    print(
        f"# copy-floor peak = {max(copy_ref.values()):.1f} GB/s (should be < ~3300 = HBM ceiling)"
    )
    print("GRID256 DONE")


if __name__ == "__main__":
    main()
