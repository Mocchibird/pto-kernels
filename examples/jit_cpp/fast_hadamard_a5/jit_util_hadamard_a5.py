"""Self-contained build + load for fast_hadamard_a5 on an Ascend 950 (A5) device.

Deliberately standalone: the shared examples/jit_cpp jit_util_common.compile_cpp
targets dav-c220 (-DMEMORY_BASE), but this kernel is A5 (dav-c310-vec,
-DREGISTER_BASE), so we compile with bisheng directly here. Only needs a working
`bisheng` (set ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME) to build, and torch_npu
at run time to launch.
"""

import ctypes
import math
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ascend_home() -> str:
    for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def compile_kernel(n: int = 128, src: Path | None = None, out_dir: Path | None = None,
                   verbose: bool = True, force: bool = False) -> Path:
    """Compile fast_hadamard_a5.cpp to a device .so for the given block size N.

    Skips the bisheng invocation when an up-to-date .so already exists (pass
    force=True to always rebuild)."""
    src = Path(src) if src else HERE / "fast_hadamard_a5.cpp"
    out_dir = Path(out_dir) if out_dir else HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    home = _ascend_home()
    bisheng = f"{home}/bin/bisheng"
    inc = f"{home}/aarch64-linux/include"
    log2n = n.bit_length() - 1
    inv = repr(1.0 / math.sqrt(n))
    obj = out_dir / f"fht_a5_n{n}.o"
    so = out_dir / f"fht_a5_n{n}.so"
    if not force and so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so}")
        return so
    common = [
        "--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE",
        f"-DHAD_N={n}", f"-DHAD_LOG2N={log2n}", f"-DHAD_INV_SQRT={inv}f",
        "-O2", "-std=c++17", "-fPIC",
        "-Wno-ignored-attributes", "-Wno-macro-redefined",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-addr-transform",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Xhost-start", "-Xhost-end", f"-I{inc}", f"-I{home}/include",
    ]
    cc = [bisheng, "-xcce", *common, "-c", str(src), "-o", str(obj)]
    ln = [bisheng, "-fPIC", "-shared", "--cce-fatobj-link",
          f"-Wl,-soname,{so.name}", str(obj), "-o", str(so)]
    if verbose:
        print(f"[compile] N={n} log2={log2n} -> {so}")
    subprocess.run(cc, check=True)
    subprocess.run(ln, check=True)
    return so


def load_lib(so_path: Path, block_dim: int = 20):
    """ctypes-load the .so and return a launch callable.

    The returned func signature is (x, batch, n, log2_n, block_dim, stream_ptr)
    to be drop-in compatible with the repo's run_hadamard_iteration; n / log2_n
    are ignored (compiled into the kernel)."""
    lib = ctypes.CDLL(str(so_path))
    kernel = lib.call_fast_hadamard_a5
    kernel.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel.restype = None

    def _stream_ptr(stream_ptr):
        if stream_ptr is not None:
            return stream_ptr
        import torch  # noqa
        s = torch.npu.current_stream()
        ptr = getattr(s, "_as_parameter_", None)
        if ptr is None:
            raise RuntimeError(
                "Could not resolve the current NPU stream pointer; the kernel "
                "would launch on the default stream and Event timing would be wrong.")
        return ptr

    def hadamard_func(x, batch, n=None, log2_n=None, block_dim=block_dim, stream_ptr=None):
        kernel(int(block_dim), _stream_ptr(stream_ptr),
               ctypes.c_void_p(x.data_ptr()), int(batch))

    hadamard_func.block_dim = block_dim
    return hadamard_func


def build_and_load(n: int = 128, block_dim: int = 20, verbose: bool = True):
    return load_lib(compile_kernel(n, verbose=verbose), block_dim=block_dim)
