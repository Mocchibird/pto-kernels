#!/usr/bin/env python3
"""Head-to-head: CUBE-core Hadamard (matmul vs Sylvester H) vs the VF/vector
kernel vs the DMA copy floor, on an Ascend 950 (A5). N=128, in-place, fp16."""
import ctypes, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch_npu  # noqa
from jit_util_hadamard_a5 import compile_kernel  # builds the vector .so

HERE = Path(__file__).resolve().parent
N = 128
INV = 1.0 / math.sqrt(N)
BYTES = 2  # fp16


def sylvester(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def _home():
    return os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]


def build_cube():
    home = _home()
    bisheng, inc = f"{home}/bin/bisheng", f"{home}/aarch64-linux/include"
    src, obj, so = HERE / "fast_hadamard_128_cube_a5.cpp", HERE / "build/hcube.o", HERE / "build/hcube.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = ["--cce-aicore-arch=dav-c310-cube", "-DHAD_N=128", "-O2", "-std=c++17", "-fPIC",
              "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000",
              "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform",
              "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{inc}", f"-I{home}/include"]
    subprocess.run([bisheng, "-xcce", *common, "-c", str(src), "-o", str(obj)], check=True)
    subprocess.run([bisheng, "-fPIC", "-shared", "--cce-fatobj-link",
                    f"-Wl,-soname,hcube.so", str(obj), "-o", str(so)], check=True)
    return so


def load_sym(so, name, with_h):
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, name)
    if with_h:
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    else:
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = None
    return fn


def stream_ptr():
    s = torch.npu.current_stream()
    p = getattr(s, "_as_parameter_", None)
    if p is None:
        raise RuntimeError("no stream ptr")
    return p


def main():
    block_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    batches = [16384, 65536]

    # H = Sylvester/sqrt(N), symmetric, fp16, resident on device.
    Hf = (sylvester(N) * INV).astype(np.float16)
    H_dev = torch.from_numpy(Hf).npu().contiguous()
    h_ptr = ctypes.c_void_p(H_dev.data_ptr())

    # Build/load all three kernels.
    cube_so = build_cube()
    cube = load_sym(cube_so, "call_hadamard_cube_a5", with_h=True)
    vec_so = compile_kernel(N)  # existing vector kernel (dav-c310-vec)
    vec = load_sym(vec_so, "call_fast_hadamard_a5", with_h=False)
    cpy = load_sym(vec_so, "call_copy_ref_a5", with_h=False)

    # ---- correctness of the cube kernel vs the reference WHT ----
    rng = np.random.default_rng(0)
    b = 256
    x_np = rng.standard_normal((b, N)).astype(np.float16)
    gold = x_np.astype(np.float32) @ (sylvester(N) * INV)   # x @ H/sqrt(N)
    x = torch.from_numpy(x_np).npu()
    cube(block_dim, stream_ptr(), ctypes.c_void_p(x.data_ptr()), h_ptr, b)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    max_abs = float(np.abs(out - gold).max())
    denom = float(np.abs(gold).max()) or 1.0
    rel = max_abs / denom
    ok = rel < 0.02
    print(f"[cube correctness] batch={b} N={N}: max_abs_diff={max_abs:.4g} rel={rel:.4g} -> {'OK' if ok else 'FAIL'}")
    # --- diagnostics ---
    noop = float(np.abs(out - x_np.astype(np.float32)).max())  # 0 => kernel did nothing
    print(f"[diag] max|out - input| = {noop:.4g}  (==0 => kernel is a no-op; store not landing)")
    print(f"[diag] H_dev sample: {H_dev.cpu().numpy().reshape(-1)[:4]}  (expect +/-{INV:.4f})")
    print(f"[diag] input[0,:6] = {x_np[0,:6].astype(np.float32)}")
    print(f"[diag] out  [0,:6] = {out[0,:6]}")
    print(f"[diag] gold [0,:6] = {gold[0,:6]}")
    print(f"[diag] out row0 mean/std = {out[0].mean():.4g}/{out[0].std():.4g}; "
          f"gold row0 mean/std = {gold[0].mean():.4g}/{gold[0].std():.4g}")
    if not ok:
        print("cube correctness FAILED; timing anyway for diagnostics", file=sys.stderr)

    def time_us(fn, batch, with_h, warmup=5, repeats=50):
        POOL = min(warmup + repeats, 8)
        pool = [torch.randn(batch, N, dtype=torch.float16).npu() for _ in range(POOL)]
        torch.npu.synchronize()

        def call(buf):
            p = ctypes.c_void_p(buf.data_ptr())
            if with_h:
                fn(block_dim, stream_ptr(), p, h_ptr, batch)
            else:
                fn(block_dim, stream_ptr(), p, batch)
        for i in range(warmup):
            call(pool[i % POOL])
        torch.npu.synchronize()
        s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s.record()
        for i in range(repeats):
            call(pool[i % POOL])
        e.record()
        torch.npu.synchronize()
        return s.elapsed_time(e) * 1e3 / repeats

    hdr = f"{'batch':>8}  {'kernel':>8}  {'dur_us':>10}  {'GB/s':>9}  {'TB/s':>7}  {'vs_copy':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for batch in batches:
        data = 2 * batch * N * BYTES  # load + store
        cus = time_us(cpy, batch, False)
        cgbs = (data / 1e9) / (cus / 1e6)
        for name, fn, wh in [("cube", cube, True), ("vector", vec, False), ("copy", cpy, False)]:
            us = time_us(fn, batch, wh)
            gbs = (data / 1e9) / (us / 1e6)
            print(f"{batch:>8}  {name:>8}  {us:>10.3f}  {gbs:>9.1f}  {gbs/1000:>7.3f}  {gbs/cgbs:>8.3f}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
