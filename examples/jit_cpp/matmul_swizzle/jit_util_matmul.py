import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)

DEFAULT_MAX_BLOCK_DIM = int(os.environ.get("PTO_MATMUL_MAX_BLOCK_DIM", "20"))
DEFAULT_SWIZZLE_DIRECTION = int(os.environ.get("PTO_MATMUL_SWIZZLE_DIRECTION", "1"))
DEFAULT_SWIZZLE_COUNT = int(os.environ.get("PTO_MATMUL_SWIZZLE_COUNT", "3"))


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:
    so_dir = os.path.join(os.path.dirname(kernel_cpp), "outputs", "so")
    os.makedirs(so_dir, exist_ok=True)
    lib_path = os.path.join(so_dir, "matmul_abt_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "--npu-arch=dav-2201",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
    ]

    link_flags = [
        f"-L{ASCEND_TOOLKIT_HOME}/lib64",
        "-lascendcl",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path, *link_flags]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(maxBlockDim, stream, x, y, z, M, N, K, swizzle_direction, swizzle_count)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # maxBlockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x [M, K]
        ctypes.c_void_p,  # y [N, K]
        ctypes.c_void_p,  # z [M, N]
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.c_int,  # swizzle_direction (0=Zn, 1=Nz)
        ctypes.c_int,  # swizzle_count
    ]
    lib.call_kernel.restype = None

    def _launch_kernel(
        a, b, c, m, n, k, max_block_dim, stream_ptr, swizzle_direction, swizzle_count
    ):
        lib.call_kernel(
            max_block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
            swizzle_direction,
            swizzle_count,
        )

    def matmul_abt(
        a,
        b,
        max_block_dim=DEFAULT_MAX_BLOCK_DIM,
        stream_ptr=None,
        swizzle_direction=DEFAULT_SWIZZLE_DIRECTION,
        swizzle_count=DEFAULT_SWIZZLE_COUNT,
    ):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)

        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])

        c = torch.empty((m, n), device=a.device, dtype=a.dtype)
        torch.npu.synchronize()

        _launch_kernel(
            a,
            b,
            c,
            m,
            n,
            k,
            max_block_dim,
            stream_ptr,
            int(swizzle_direction),
            int(swizzle_count),
        )
        return c

    return matmul_abt


def jit_compile(src_path, verbose=True, clean_up=False):
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func


if __name__ == "__main__":
    jit_compile("./matmul_custom_pto.cpp")
