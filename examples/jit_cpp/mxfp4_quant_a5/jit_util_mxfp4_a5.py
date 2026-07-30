#!/usr/bin/env python3
"""Build + ctypes load for mxfp4_quant_a5.cpp (dav-c310 vector core)."""
import ctypes
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
QK = 32
KWIDTH = 256
ROWS_PER_TILE = 16


def _home():
    for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(k):
            return os.environ[k]
    raise RuntimeError("set ASCEND_HOME_PATH")


def build(
    kwidth=KWIDTH,
    rows_per_tile=ROWS_PER_TILE,
    nbuf=4,
    prefetch=2,
    src=None,
    verbose=False,
):
    src = Path(src) if src else HERE / "mxfp4_quant_a5.cpp"
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    tag = f"{src.stem}_k{kwidth}_r{rows_per_tile}_n{nbuf}"
    obj, so = out / f"{tag}.o", out / f"{tag}.so"
    h = _home()
    flags = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DKWIDTH={kwidth}",
        f"-DROWS_PER_TILE={rows_per_tile}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
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
    r = subprocess.run(
        [f"{h}/bin/bisheng", "-xcce", *flags, "-c", str(src), "-o", str(obj)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    r = subprocess.run(
        [
            f"{h}/bin/bisheng",
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
        raise RuntimeError(r.stderr[-1500:])
    lib = ctypes.CDLL(str(so))
    fn = lib.call_mxfp4_quant
    fn.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    fn.restype = None
    if verbose:
        print(f"[build] {so.name}")
    return fn


def run(fn, x_bf16, kwidth=KWIDTH, block_dim=64):
    """x_bf16: torch bf16 (batch, K) on npu. Returns (packed uint8, e8m0 uint8)."""
    import torch

    batch, k = x_bf16.shape
    q = torch.zeros((batch, k // 2), dtype=torch.uint8).npu()
    s = torch.zeros((batch, k // QK), dtype=torch.uint8).npu()
    fn(
        int(block_dim),
        torch.npu.current_stream()._as_parameter_,  # noqa
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        batch,
    )
    torch.npu.synchronize()
    return q, s
