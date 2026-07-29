#!/usr/bin/env python3
"""Validate + benchmark the N=128 deinterleave-load WHT (fast_hadamard_128_dintlv_a5.cpp)."""
import ctypes, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 128
h = os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]


def sylvester(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def build():
    src = HERE / "fast_hadamard_128_dintlv_a5.cpp"; obj = HERE / "build/h128d.o"; so = HERE / "build/h128d.so"
    (HERE / "build").mkdir(exist_ok=True)
    rows = os.environ.get("ROWS", "128"); nbuf = os.environ.get("NBUF", "4"); pf = os.environ.get("PF", "2")
    print(f"[cfg] ROWS={rows} NBUF={nbuf} PREFETCH={pf}")
    common = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE", f"-DROWS_PER_TILE={rows}",
              f"-DPIPELINE_BUFFERS={nbuf}", f"-DPREFETCH_TILES={pf}",
              "-O2", "-std=c++17", "-fPIC", "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{h}/aarch64-linux/include", f"-I{h}/include"]
    subprocess.run([f"{h}/bin/bisheng", "-xcce", *common, "-c", str(src), "-o", str(obj)], check=True)
    subprocess.run([f"{h}/bin/bisheng", "-fPIC", "-shared", "--cce-fatobj-link",
                    "-Wl,-soname,h128d.so", str(obj), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    for nm in ("call_hadamard128_dintlv", "call_copy128_dintlv"):
        getattr(lib, nm).argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        getattr(lib, nm).restype = None
    return lib


def sp():
    return torch.npu.current_stream()._as_parameter_


def time_us(call, w=10, r=100):
    torch.npu.synchronize()
    for _ in range(w): call()
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(r): call()
    e.record(); torch.npu.synchronize()
    return s.elapsed_time(e) * 1e3 / r


def main():
    bd = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    lib = build()
    H = sylvester(N)  # unnormalized (kernel does no scale)
    rng = np.random.default_rng(0); b = 256
    x_np = rng.standard_normal((b, N)).astype(np.float16)
    gold = x_np.astype(np.float32) @ H
    x = torch.from_numpy(x_np).npu()
    lib.call_hadamard128_dintlv(bd, sp(), ctypes.c_void_p(x.data_ptr()), b)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    rel = np.abs(out - gold).max() / (np.abs(gold).max() or 1.0)
    print(f"[hadamard128_dintlv correctness] b={b}: max_rel={rel:.4g} -> {'OK' if rel < 0.03 else 'FAIL'}")
    print("  gold[0,:6]=", np.round(gold[0, :6], 2), " out[0,:6]=", np.round(out[0, :6], 2))
    hdr = f"{'batch':>8}  {'kernel':>14}  {'dur_us':>9}  {'GB/s':>8}  {'vs_copy':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for batch in (16384, 65536):
        data = 2 * batch * N * 2
        POOL = 8
        pool = [torch.randn(batch, N, dtype=torch.float16).npu() for _ in range(POOL)]
        it = {"k": 0}
        def mk(fn):
            def run():
                buf = pool[it["k"] % POOL]; it["k"] += 1
                fn(bd, sp(), ctypes.c_void_p(buf.data_ptr()), batch)
            return run
        cus = time_us(mk(lib.call_copy128_dintlv)); cg = data / 1e9 / (cus / 1e6)
        for name, fn in (("h128_dintlv", lib.call_hadamard128_dintlv), ("copy", lib.call_copy128_dintlv)):
            us = time_us(mk(fn)); g = data / 1e9 / (us / 1e6)
            print(f"{batch:>8}  {name:>14}  {us:>9.3f}  {g:>8.1f}  {g/cg:>8.3f}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
