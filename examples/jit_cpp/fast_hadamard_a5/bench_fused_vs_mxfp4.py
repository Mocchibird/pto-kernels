#!/usr/bin/env python3
"""Head-to-head throughput: fused Hadamard+MXFP4 VF kernel vs torch_npu MXFP4-only.
Goal: fused ~= quant-only  =>  the Hadamard rides free under the quant traffic."""
import ctypes, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 128; BLK = 32; NBLK = N // BLK; SSTRIDE = 16


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
              f"-DPHASE_SEL={os.environ.get('PHASE_SEL','0')}",
              "-O2", "-std=c++17", "-fPIC", "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000",
              "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform",
              "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{inc}", f"-I{h}/include"]
    subprocess.run([bish, "-xcce", *common, "-c", str(src), "-o", str(obj)], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([bish, "-fPIC", "-shared", "--cce-fatobj-link",
                    f"-Wl,-soname,fused_{tag}.so", str(obj), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    fn = lib.call_fused_hadamard_mxfp4_a5
    fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    fn.restype = None
    return fn


def sp():
    return torch.npu.current_stream()._as_parameter_


def time_us(call, warmup=10, repeats=100):
    torch.npu.synchronize()
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        call()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) * 1e3 / repeats


def main():
    block_dims = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["32", "48", "64"])]
    batches = [16384, 65536]
    for bf16 in (False, True):
        dt = torch.bfloat16 if bf16 else torch.float16
        fused = build(bf16)
        print(f"\n################  input dtype = {'bf16' if bf16 else 'fp16'}  ################")
        hdr = f"{'batch':>8}  {'kernel':>22}  {'dur_us':>9}  {'in_GB/s':>8}  {'vs_quant':>8}"
        print(hdr + "\n" + "-" * len(hdr))
        for batch in batches:
            POOL = 8
            xs = [torch.randn(batch, N, dtype=dt).npu() for _ in range(POOL)]
            q = torch.zeros(batch, N // 2, dtype=torch.uint8).npu()
            s = torch.zeros(batch, SSTRIDE, dtype=torch.int16).npu()
            it = {"k": 0}

            def base():
                b = xs[it["k"] % POOL]; it["k"] += 1
                return torch_npu.npu_dynamic_mx_quant(b, block_size=BLK, dst_type=296)

            def make_fused(bd):
                def run():
                    b = xs[it["k"] % POOL]; it["k"] += 1
                    fused(bd, sp(), ctypes.c_void_p(b.data_ptr()),
                          ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(s.data_ptr()), batch)
                return run

            in_bytes = batch * N * 2
            bus = time_us(base)
            bgbs = in_bytes / 1e9 / (bus / 1e6)
            print(f"{batch:>8}  {'mxfp4_only(torch_npu)':>22}  {bus:>9.3f}  {bgbs:>8.1f}  {1.0:>8.2f}")
            best = None
            for bd in block_dims:
                us = time_us(make_fused(bd))
                if best is None or us < best[1]:
                    best = (bd, us)
                gbs = in_bytes / 1e9 / (us / 1e6)
                print(f"{batch:>8}  {'fused had+mxfp4 bd=' + str(bd):>22}  {us:>9.3f}  {gbs:>8.1f}  {bus/us:>8.2f}")
            print(f"{batch:>8}  {'>>> best fused':>22}  {best[1]:>9.3f}  "
                  f"{in_bytes/1e9/(best[1]/1e6):>8.1f}  {bus/best[1]:>8.2f}   (bd={best[0]})")
            print("-" * len(hdr))


if __name__ == "__main__":
    main()
