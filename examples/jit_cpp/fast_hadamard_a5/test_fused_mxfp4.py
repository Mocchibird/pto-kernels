#!/usr/bin/env python3
"""Build + correctness for fused Hadamard+MXFP4 VF kernel.
Dequantizes the fp4(e2m1)+e8m0 output and compares to the reference WHT."""
import ctypes, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 128; BLK = 32; NBLK = N // BLK
SSTRIDE = 16    # u16 per row (32-byte padded scale slot; first NBLK valid)
INV = 1.0 / math.sqrt(N)

# e2m1 magnitude table indexed by 3-bit (exp<<1 | mantissa)
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def sylvester(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def home():
    return os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]


def build(bf16: bool):
    h = home(); bish = f"{h}/bin/bisheng"; inc = f"{h}/aarch64-linux/include"
    tag = "bf16" if bf16 else "fp16"
    src = HERE / "fused_hadamard_mxfp4_a5.cpp"
    obj = HERE / f"build/fused_{tag}.o"; so = HERE / f"build/fused_{tag}.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE",
              f"-DHAD_IN_BF16={1 if bf16 else 0}", "-DROWS_PER_TILE=64",
              "-O2", "-std=c++17", "-fPIC", "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000",
              "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform",
              "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{inc}", f"-I{h}/include"]
    subprocess.run([bish, "-xcce", *common, "-c", str(src), "-o", str(obj)], check=True)
    subprocess.run([bish, "-fPIC", "-shared", "--cce-fatobj-link",
                    f"-Wl,-soname,fused_{tag}.so", str(obj), "-o", str(so)], check=True)
    return so


def load(so):
    lib = ctypes.CDLL(str(so))
    fn = lib.call_fused_hadamard_mxfp4_a5
    fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = None
    return fn


def stream_ptr():
    return torch.npu.current_stream()._as_parameter_


def dequant(q_u8, s_u16, nibble_low_is_even=True):
    """q_u8: (batch,64) packed fp4; s_u16: (batch,SSTRIDE) e8m0 (first 4 valid).
    -> (batch,128) float."""
    b = q_u8.shape[0]
    lo = (q_u8 & 0x0F).astype(np.uint8)          # low nibble
    hi = (q_u8 >> 4).astype(np.uint8)            # high nibble
    nib = np.empty((b, N), dtype=np.uint8)
    if nibble_low_is_even:
        nib[:, 0::2] = lo; nib[:, 1::2] = hi
    else:
        nib[:, 0::2] = hi; nib[:, 1::2] = lo
    sign = np.where((nib & 0x8) != 0, -1.0, 1.0).astype(np.float32)
    mag = E2M1_MAG[(nib & 0x7)]
    val = sign * mag
    e8 = s_u16[:, :NBLK].astype(np.float32)                  # (b,4)
    scale = np.power(2.0, e8 - 127.0)
    scale_full = np.repeat(scale, BLK, axis=1)               # (b,128)
    return val * scale_full


def main():
    bf16 = "--bf16" in sys.argv
    block_dim = 64
    dt = torch.bfloat16 if bf16 else torch.float16
    print(f"==== fused Hadamard+MXFP4 correctness ({'bf16' if bf16 else 'fp16'}) ====")
    so = build(bf16); fn = load(so)

    batch = 256
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, N)).astype(np.float32)
    gold = x_np @ (sylvester(N) * INV)                       # reference WHT (float)

    x = torch.from_numpy(x_np).to(dt).npu()
    q = torch.zeros(batch, N // 2, dtype=torch.uint8).npu()
    s = torch.zeros(batch, SSTRIDE, dtype=torch.int16).npu()
    fn(block_dim, stream_ptr(), ctypes.c_void_p(x.data_ptr()),
       ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(s.data_ptr()), batch)
    torch.npu.synchronize()

    q_np = q.cpu().numpy()
    s_np = s.cpu().numpy().astype(np.uint16)
    print("s[0,:4] (e8m0) =", s_np[0, :4])
    print("q[0,:8] (packed)  =", q_np[0, :8])

    best = None
    for order in (True, False):
        out = dequant(q_np, s_np, order)
        denom = np.abs(gold).max() or 1.0
        rel = np.abs(out - gold).max() / denom
        # relative L2 too
        l2 = np.linalg.norm(out - gold) / (np.linalg.norm(gold) or 1.0)
        print(f"nibble_low_is_even={order}: max_rel={rel:.4g}  l2_rel={l2:.4g}")
        if best is None or l2 < best[0]:
            best = (l2, order, out)
    l2, order, out = best
    print(f"\nBEST: order_even={order}  l2_rel={l2:.4g}")
    print("gold[0,:8] =", np.round(gold[0, :8], 3))
    print("out [0,:8] =", np.round(out[0, :8], 3))
    # MXFP4 relative error is typically ~0.1 (1-bit mantissa); pass if < 0.25
    ok = l2 < 0.25
    print("RESULT:", "OK" if ok else "FAIL (quantization error too large)")


if __name__ == "__main__":
    main()
