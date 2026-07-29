#!/usr/bin/env python3
"""Correctness check for fast_hadamard_256_a5 across every ROWS_PER_TILE config
used in the grid sweep. Verifies (a) the WHT output vs x @ Sylvester(256), and
(b) that copy256 preserves data. Batch 4096 (multiple of all ROWS)."""
import ctypes, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch_npu  # noqa

HERE = Path(__file__).resolve().parent
N = 256
h = os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"]
ROWS_LIST = [16, 32, 64, 128, 256]
BATCH = 4096


def sylvester(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def nbuf_for(rows):
    return max(1, min(4, (192 * 1024) // (rows * N * 2)))


def build(rows, nbuf, pf):
    src = HERE / "fast_hadamard_256_a5.cpp"
    obj = HERE / f"build/c256_{rows}.o"; so = HERE / f"build/c256_{rows}.so"
    (HERE / "build").mkdir(exist_ok=True)
    common = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE", f"-DROWS_PER_TILE={rows}",
              f"-DNBUF={nbuf}", f"-DPREFETCH={pf}", "-O2", "-std=c++17", "-fPIC",
              "-Wno-ignored-attributes", "-Wno-macro-redefined",
              "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
              "-mllvm", "-cce-aicore-addr-transform", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
              "-Xhost-start", "-Xhost-end", f"-I{h}/aarch64-linux/include", f"-I{h}/include"]
    r = subprocess.run([f"{h}/bin/bisheng", "-xcce", *common, "-c", str(src), "-o", str(obj)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1] if r.stderr else "compile error"
    subprocess.run([f"{h}/bin/bisheng", "-fPIC", "-shared", "--cce-fatobj-link",
                    f"-Wl,-soname,{so.name}", str(obj), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    for nm in ("call_hadamard256", "call_copy256"):
        getattr(lib, nm).argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        getattr(lib, nm).restype = None
    return lib, None


def sp():
    return torch.npu.current_stream()._as_parameter_


def main():
    bd = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    H = sylvester(N)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((BATCH, N)).astype(np.float16)
    gold = x_np.astype(np.float32) @ H
    goldmax = float(np.abs(gold).max()) or 1.0

    print(f"{'ROWS':>5} {'NBUF':>5} {'had_rel':>10} {'had':>5} {'copy_max|dx|':>13} {'copy':>5}")
    for rows in ROWS_LIST:
        nbuf = nbuf_for(rows); pf = min(2, max(0, nbuf - 1))
        lib, err = build(rows, nbuf, pf)
        if lib is None:
            print(f"{rows:>5} {nbuf:>5}   BUILD FAIL: {err}"); continue
        # (a) transform correctness
        x = torch.from_numpy(x_np.copy()).npu()
        lib.call_hadamard256(bd, sp(), ctypes.c_void_p(x.data_ptr()), BATCH)
        torch.npu.synchronize()
        out = x.cpu().numpy().astype(np.float32)
        rel = float(np.abs(out - gold).max()) / goldmax
        had_ok = rel < 0.03
        # (b) copy preserves data
        y_np = rng.standard_normal((BATCH, N)).astype(np.float16)
        y = torch.from_numpy(y_np.copy()).npu()
        lib.call_copy256(bd, sp(), ctypes.c_void_p(y.data_ptr()), BATCH)
        torch.npu.synchronize()
        yout = y.cpu().numpy()
        dcopy = float(np.abs(yout.astype(np.float32) - y_np.astype(np.float32)).max())
        copy_ok = dcopy == 0.0
        print(f"{rows:>5} {nbuf:>5} {rel:>10.4g} {('OK' if had_ok else 'FAIL'):>5} "
              f"{dcopy:>13.4g} {('OK' if copy_ok else 'FAIL'):>5}")
    print("CHECK256 DONE")


if __name__ == "__main__":
    main()
