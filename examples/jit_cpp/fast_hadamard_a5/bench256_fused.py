#!/usr/bin/env python3
"""Correctness + throughput: fused N=256 Hadamard+MXFP4 vs torch_npu mxfp4-only, (batch,256)."""
import ctypes, math, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 256; BLK = 32; NBLK = N // BLK          # 8 blocks
SUB = 2                                      # 128-subrows per 256-row
SSTRIDE_U16 = 16                             # u16 per subrow scale slot (32 bytes)
INV = 1.0 / math.sqrt(N)
E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], np.float32)
h = os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]


def sylvester(n):
    H = np.array([[1.0]], np.float64)
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H


def build(bf16):
    tag = "bf16" if bf16 else "fp16"
    src = HERE / "fused_hadamard256_mxfp4_a5.cpp"; obj = HERE / f"build/f256_{tag}.o"; so = HERE / f"build/f256_{tag}.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE", f"-DHAD_IN_BF16={1 if bf16 else 0}",
              "-DROWS_PER_TILE=64", "-O2", "-std=c++17", "-fPIC", "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{h}/aarch64-linux/include", f"-I{h}/include"]
    subprocess.run([f"{h}/bin/bisheng", "-xcce", *common, "-c", str(src), "-o", str(obj)], check=True)
    subprocess.run([f"{h}/bin/bisheng", "-fPIC", "-shared", "--cce-fatobj-link",
                    f"-Wl,-soname,f256_{tag}.so", str(obj), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so)); fn = lib.call_fused_hadamard256_mxfp4_a5
    fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = None; return fn


def sp(): return torch.npu.current_stream()._as_parameter_


def dequant(q, s):  # q:(b,128) u8, s:(b,32) u16 -> (b,256) float
    b = q.shape[0]; out = np.zeros((b, N), np.float32)
    for sr in range(SUB):
        qb = q[:, sr*64:(sr+1)*64]
        lo = qb & 0xF; hi = qb >> 4
        nib = np.empty((b, 128), np.uint8); nib[:, 0::2] = lo; nib[:, 1::2] = hi
        mag = E2M1[nib & 7]; sgn = np.where(nib & 8, -1., 1.).astype(np.float32)
        e8 = s[:, sr*SSTRIDE_U16: sr*SSTRIDE_U16+4].astype(np.float32)  # 4 blocks
        scale = np.repeat(np.power(2., e8 - 127.), BLK, axis=1)
        out[:, sr*128:(sr+1)*128] = sgn * mag * scale
    return out


def time_us(call, w=10, r=100):
    torch.npu.synchronize()
    for _ in range(w): call()
    torch.npu.synchronize()
    a, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(r): call()
    e.record(); torch.npu.synchronize(); return a.elapsed_time(e) * 1e3 / r


def main():
    bd = 64
    for bf16 in (False, True):
        dt = torch.bfloat16 if bf16 else torch.float16
        fn = build(bf16)
        print(f"\n###### N=256 fused, input {'bf16' if bf16 else 'fp16'} ######")
        # correctness
        b = 256; rng = np.random.default_rng(0)
        x_np = rng.standard_normal((b, N)).astype(np.float32)
        gold = (x_np @ sylvester(N)) * INV
        x = torch.from_numpy(x_np).to(dt).npu()
        q = torch.zeros(b, N // 2, dtype=torch.uint8).npu()
        s = torch.zeros(b, SUB * SSTRIDE_U16, dtype=torch.int16).npu()
        fn(bd, sp(), ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(s.data_ptr()), b)
        torch.npu.synchronize()
        out = dequant(q.cpu().numpy(), s.cpu().numpy().astype(np.uint16))
        l2 = np.linalg.norm(out - gold) / (np.linalg.norm(gold) or 1.)
        print(f"  correctness l2_rel={l2:.4g} -> {'OK' if l2 < 0.25 else 'FAIL'}")
        print("   gold[0,:6]=", np.round(gold[0, :6], 3), " out=", np.round(out[0, :6], 3))
        # throughput vs torch_npu mxfp4 on (batch,256)
        hdr = f"{'batch':>8}  {'kernel':>22}  {'dur_us':>9}  {'in_GB/s':>8}  {'vs_quant':>8}"
        print(hdr + "\n" + "-" * len(hdr))
        for batch in (16384, 65536):
            POOL = 8
            xs = [torch.randn(batch, N, dtype=dt).npu() for _ in range(POOL)]
            q2 = torch.zeros(batch, N // 2, dtype=torch.uint8).npu()
            s2 = torch.zeros(batch, SUB * SSTRIDE_U16, dtype=torch.int16).npu()
            it = {"k": 0}
            def base():
                bb = xs[it["k"] % POOL]; it["k"] += 1
                return torch_npu.npu_dynamic_mx_quant(bb, block_size=BLK, dst_type=296)
            def fused():
                bb = xs[it["k"] % POOL]; it["k"] += 1
                fn(bd, sp(), ctypes.c_void_p(bb.data_ptr()), ctypes.c_void_p(q2.data_ptr()), ctypes.c_void_p(s2.data_ptr()), batch)
            inb = batch * N * 2
            bus = time_us(base); bg = inb / 1e9 / (bus / 1e6)
            fus = time_us(fused); fg = inb / 1e9 / (fus / 1e6)
            print(f"{batch:>8}  {'mxfp4_only(torch_npu)':>22}  {bus:>9.3f}  {bg:>8.1f}  {1.0:>8.2f}")
            print(f"{batch:>8}  {'fused had256+mxfp4':>22}  {fus:>9.3f}  {fg:>8.1f}  {bus/fus:>8.2f}")
            print("-" * len(hdr))


if __name__ == "__main__":
    main()
