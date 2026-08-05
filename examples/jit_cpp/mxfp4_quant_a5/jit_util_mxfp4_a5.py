"""Build + load for the A5 MXFP4 quantization kernel.

batch dynamic; K a compile-time template argument; block size 32 static.
"""

import ctypes
import os
import subprocess
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "mxfp4_quant_a5.cpp"

K = 4096  # default row width; must match DEFAULT_K in the kernel
MX_BLOCK = 32
BLOCK_DIM = 64  # overridden by vector_core_count() where available
TILE_ELEMS = 16384  # must match TILE_ELEMS in the kernel
SUPPORTED_K = (128, 256, 512, 1024, 2048, 4096)

# (block_dim, stream, x, q, s, batch, k)
KERNEL_ARGS = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
]
DEFAULT_ARGS = KERNEL_ARGS[:-1]  # call_mxfp4_quant_default takes no k

# Flags that never vary. Written as one string and split: black re-explodes a list
# of short strings to one entry per line, but leaves this alone.
FIXED_FLAGS = (
    "-O2 -std=c++17 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
    "-mllvm -cce-aicore-stack-size=0x8000 "
    "-mllvm -cce-aicore-function-stack-size=0x8000 "
    "-mllvm -cce-aicore-addr-transform "
    "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
    "-Xhost-start -Xhost-end"
).split()


def ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            return os.environ[key]
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def check_k(k: int) -> int:
    """Validate before any arithmetic that could fail on a bad k."""
    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}, got {k}")
    return k


def rows_for(k: int = K) -> int:
    """ROWS_PER_TILE for row width ``k``: a 16 KB bf16 tile, floored at 1 row."""
    return max(1, TILE_ELEMS // check_k(k))


def compile_kernel(force: bool = False, verbose: bool = True) -> Path:
    """Compile and link the kernel to a device .so, reusing an up-to-date one."""
    out_dir = HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj, so = out_dir / "mxfp4_a5.o", out_dir / "mxfp4_a5.so"
    if not force and so.exists() and so.stat().st_mtime >= SRC.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so.name}")
        return so
    home = ascend_home()
    bisheng = f"{home}/bin/bisheng"
    arch = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE"]
    inc = [f"-I{home}/aarch64-linux/include", f"-I{home}/include"]
    cmd = [bisheng, "-xcce", *arch, *FIXED_FLAGS, *inc]
    subprocess.run([*cmd, "-c", str(SRC), "-o", str(obj)], check=True)
    link = f"-fPIC -shared --cce-fatobj-link -Wl,-soname,{so.name}".split()
    subprocess.run([bisheng, *link, str(obj), "-o", str(so)], check=True)
    return so


def entry(so_path, name, argtypes=None):
    """ctypes-load ``so_path`` and bind the launcher ``name``."""
    fn = getattr(ctypes.CDLL(str(so_path)), name)
    fn.argtypes = list(KERNEL_ARGS if argtypes is None else argtypes)
    fn.restype = None
    return fn


@lru_cache(maxsize=1)
def stream():
    """Cached stream pointer: querying it per launch has measurable cost."""
    import torch  # noqa: F401

    resolved = getattr(torch.npu.current_stream(), "_as_parameter_", None)
    if resolved is None:
        raise RuntimeError("could not resolve the current NPU stream pointer")
    return resolved


def kernel_rows_for(so_path):
    """The kernel's own RowsFor<K>, so a test can pin it against rows_for()."""
    fn = getattr(ctypes.CDLL(str(so_path)), "mxfp4_rows_for")
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32

    def query(k):
        rows = int(fn(k))
        if rows == 0:
            raise ValueError(f"kernel has no instantiation for k={k}")
        return rows

    return query


def load_lib(so_path, block_dim: int = BLOCK_DIM, k: int = K):
    """Return a callable mapping a (batch, k) bf16 tensor to (q, scale)."""
    import torch  # noqa: F401

    # Validated here as well as in compile_kernel: the dispatching launcher's
    # default case is a silent no-op, so an unchecked k would hand back an
    # untouched output buffer rather than failing.
    check_k(k)
    rows = rows_for(k)
    kernel = entry(so_path, "call_mxfp4_quant")

    def run(x, out=None, stream_ptr=None):
        assert (
            x.dim() == 2 and x.shape[1] == k
        ), f"expected (batch, {k}) bfloat16, got {tuple(x.shape)}"
        # The kernel reads the buffer as bf16 and as one flat run, and can report
        # neither: a wider dtype is reinterpreted and a strided view is read as if
        # contiguous, both silently.
        assert x.dtype == torch.bfloat16, f"expected bfloat16, got {x.dtype}"
        assert x.is_contiguous(), "expected a contiguous tensor; call .contiguous()"

        batch = int(x.shape[0])
        padded = -(-batch // rows) * rows
        src = x
        if padded != batch:
            src = torch.zeros((padded, k), device=x.device, dtype=x.dtype)
            src[:batch] = x
            # The ctypes launch is not ordered against torch's copy, so without
            # this the kernel can quantize the still-zero buffer and return all
            # zeros -- which is a plausible-looking result, not an error.
            torch.npu.synchronize()
        if out is None:
            q = torch.empty((padded, k // 2), device=x.device, dtype=torch.uint8)
            s = torch.empty((padded, k // MX_BLOCK), device=x.device, dtype=torch.uint8)
        else:
            q, s = out
            # TSTORE needs a 512-byte-aligned destination. A small torch tensor
            # is not guaranteed to be, and the failure is silent: the DMA simply
            # does not land, leaving whatever was in the buffer.
            for name, t in (("q", q), ("scale", s)):
                assert t.data_ptr() % 512 == 0, (
                    f"{name} output is not 512-byte aligned "
                    f"({t.data_ptr() % 512} off); allocate it larger or let this "
                    "wrapper allocate it"
                )
                assert t.is_contiguous(), f"{name} output must be contiguous"
        kernel(
            int(block_dim),
            stream() if stream_ptr is None else stream_ptr,
            ctypes.c_void_p(src.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
            padded,
            k,
        )
        return q[:batch], s[:batch]

    run.block_dim = block_dim
    run.rows_per_tile = rows
    return run


def build_and_load(block_dim: int = BLOCK_DIM, k: int = K, verbose: bool = True):
    check_k(k)  # before rows_for divides by it
    return load_lib(compile_kernel(verbose=verbose), block_dim=block_dim, k=k)
