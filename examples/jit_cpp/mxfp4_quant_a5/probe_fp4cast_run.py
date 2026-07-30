#!/usr/bin/env python3
"""Discover the bf16 -> f4e2m1x2 layout by dumping raw bytes, not by assuming one.

Input is the eight exactly-representable E2M1 magnitudes in order, repeated:
  0, 0.5, 1, 1.5, 2, 3, 4, 6   ->  codes 0,1,2,3,4,5,6,7
So if the cast writes codes contiguously, low nibble first, the output bytes are
0x10 0x32 0x54 0x76 repeating. Anything else reveals the real packing. Untouched
bytes are pre-filled with the 0xEE sentinel, so PART_Px's footprint is visible.
"""
import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa

HERE = Path(__file__).resolve().parent
HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # -> codes 0..7


def build(part, use_ctrl=0):
    tag = f"{part}_{use_ctrl}"
    obj, so = HERE / f"fc{tag}.o", HERE / f"fc{tag}.so"
    flags = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DUSE_CTRL={use_ctrl}",
        f"-DPART_SEL={part}",
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-Wno-ignored-attributes",
        "-Wno-macro-redefined",
        "-Xhost-start",
        "-Xhost-end",
        f"-I{HOME}/aarch64-linux/include",
        f"-I{HOME}/include",
    ]
    src = HERE / "probe_fp4cast.cpp"
    r = subprocess.run(
        [f"{HOME}/bin/bisheng", "-xcce", *flags, "-c", str(src), "-o", str(obj)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        return None, r.stderr.strip().splitlines()[-1]
    r = subprocess.run(
        [
            f"{HOME}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{so.name}",
            str(obj),
            "-o",
            str(so),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        return None, r.stderr.strip().splitlines()[-1]
    lib = ctypes.CDLL(str(so))
    fn = lib.call_fp4cast
    fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = None
    return fn, None


def run(fn, vals):
    src = np.zeros(128, dtype=np.float32)
    src[: len(vals)] = vals
    x = torch.from_numpy(src).to(torch.bfloat16).npu()
    out = torch.full((128,), 0xEE, dtype=torch.uint8).npu()
    fn(
        1,
        torch.npu.current_stream()._as_parameter_,  # noqa
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
    )
    torch.npu.synchronize()
    return out.cpu().numpy()


def hexdump(b, per=16):
    for i in range(0, len(b), per):
        row = " ".join(f"{v:02x}" for v in b[i : i + per])
        print(f"    [{i:3d}] {row}")


def main():
    ramp = (MAGS * 16)[:128]

    print("=== input: codes 0..7 repeating; contiguous low-nibble-first would be")
    print("=== bytes 10 32 54 76 repeating. 0xEE = untouched.")
    for part in ("PART_P0", "PART_P1", "PART_P2", "PART_P3"):
        fn, err = build(part)
        if fn is None:
            print(f"\n  {part}: BUILD FAILED: {err}")
            continue
        out = run(fn, ramp)
        touched = int((out != 0xEE).sum())
        print(f"\n  {part}: {touched}/128 bytes written")
        hexdump(out[:32])
        if touched and touched != 128:
            idx = np.where(out != 0xEE)[0]
            print(f"    written range: [{idx.min()}..{idx.max()}]")

    print("\n=== does set_ctrl(1<<50) change anything? (PART_P0) ===")
    outs = {}
    for uc in (0, 1, 2):
        fn, err = build("PART_P0", uc)
        if fn is None:
            print(f"  USE_CTRL={uc}: BUILD FAILED: {err}")
            continue
        outs[uc] = run(fn, ramp)
    if len(outs) > 1:
        ks = list(outs)
        same = all(np.array_equal(outs[ks[0]], outs[k]) for k in ks[1:])
        print(f"  identical across no-ctrl / after-mask / before-mask: {same}")

    print("\n=== saturation: values above 6.0, same layout as above ===")
    over = [0.0, 6.0, 6.5, 7.0, 8.0, 16.0, 100.0, -7.0]
    fn, _ = build("PART_P0")
    if fn is not None:
        out = run(fn, (over * 16)[:128])
        hexdump(out[:16])
        print(f"    first 8 inputs were: {over}")
    print("FP4CAST DONE")


if __name__ == "__main__":
    main()
