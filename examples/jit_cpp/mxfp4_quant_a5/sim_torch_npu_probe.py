#!/usr/bin/env python3
"""Which plain torch_npu operations actually work under `msprof op simulator`?

Not kernel-specific: this probes the framework paths a runner depends on, because
the CA model implements a subset of them and the ones it does not implement fail
SILENTLY -- a copy that never lands leaves zeros, which then looks exactly like a
broken kernel. Discovered while debugging fast_hadamard_a5's padded-batch case.

Every probe prints and flushes immediately, so a hang or a timeout still leaves a
record of everything that worked up to that point. Tensors are deliberately tiny:
the CA model simulates every instruction of every operator it dispatches.

Run it the same way as any other kernel runner:
    msprof op simulator --soc-version=Ascend950PR_9599 --timeout=600 \
      --output=/tmp/probe python3 sim_torch_npu_probe.py --output-json /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
RESULTS: list[dict] = []


def probe(name: str, fn, want: np.ndarray | None = None) -> None:
    """Run one probe. `fn` returns a numpy array to compare against `want`.

    A raised exception is a LOUD failure and is recorded as "error"; a wrong
    result with no exception is the dangerous silent one and is recorded as
    "WRONG". Both are distinguished from "ok" in the report.
    """
    entry: dict = {"probe": name}
    try:
        got = fn()
        if want is None:
            entry["status"] = "ok"
            entry["note"] = "ran without error; no value check"
        elif np.array_equal(got, want):
            entry["status"] = "ok"
        else:
            entry["status"] = "WRONG"
            entry["detail"] = (
                f"got maxabs={float(np.abs(got).max())} "
                f"want maxabs={float(np.abs(want).max())}"
            )
    except Exception as exc:  # noqa: BLE001 - reporting every failure mode is the point
        entry["status"] = "error"
        entry["detail"] = f"{type(exc).__name__}: {exc}"
    RESULTS.append(entry)
    print(
        f"  {entry['status']:<5} {name}"
        + (f"  -- {entry.get('detail', '')}" if entry.get("detail") else "")
    )
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(DEVICE)
    print(f"[probe] device={DEVICE} torch={torch.__version__}")

    ones = np.ones((8, 4), dtype=np.float16)
    zeros = np.zeros((8, 4), dtype=np.float16)
    half_ones = zeros.copy()
    half_ones[:5] = 1.0

    # --- allocation and transfer -------------------------------------------------
    probe(
        "alloc: torch.zeros on device",
        lambda: torch.zeros((8, 4), device=DEVICE, dtype=torch.float16).cpu().numpy(),
        zeros,
    )
    probe(
        "copy: host -> device (whole tensor)",
        lambda: torch.from_numpy(ones.copy()).npu().cpu().numpy(),
        ones,
    )

    def d2d_whole():
        src = torch.from_numpy(ones.copy()).npu()
        dst = torch.zeros((8, 4), device=DEVICE, dtype=torch.float16)
        dst.copy_(src)
        torch.npu.synchronize()
        return dst.cpu().numpy()

    probe("copy: device -> device (.copy_, whole)", d2d_whole, ones)

    def d2d_slice():
        src = torch.from_numpy(ones.copy()).npu()
        dst = torch.zeros((8, 4), device=DEVICE, dtype=torch.float16)
        dst[:5] = src[:5]
        torch.npu.synchronize()
        return dst.cpu().numpy()

    probe("copy: device -> device (slice assign)", d2d_slice, half_ones)

    def h2d_slice():
        dst = torch.zeros((8, 4), device=DEVICE, dtype=torch.float16)
        dst[:5] = torch.from_numpy(ones[:5].copy())
        torch.npu.synchronize()
        return dst.cpu().numpy()

    probe("copy: host -> device (slice assign)", h2d_slice, half_ones)

    def host_pad_then_h2d():
        staged = zeros.copy()
        staged[:5] = ones[:5]
        return torch.from_numpy(staged).npu().cpu().numpy()

    probe("copy: pad on host, one h2d (workaround)", host_pad_then_h2d, half_ones)

    # --- compute ------------------------------------------------------------------
    def elementwise_add():
        a = torch.from_numpy(ones.copy()).npu()
        out = a + a
        torch.npu.synchronize()
        return out.cpu().numpy()

    probe("compute: elementwise add", elementwise_add, (ones * 2).astype(np.float16))

    def elementwise_mul_scalar():
        a = torch.from_numpy(ones.copy()).npu()
        out = a * 3.0
        torch.npu.synchronize()
        return out.cpu().numpy()

    probe(
        "compute: multiply by scalar",
        elementwise_mul_scalar,
        (ones * 3).astype(np.float16),
    )

    def reduction_sum():
        a = torch.from_numpy(ones.copy()).npu()
        out = a.sum()
        torch.npu.synchronize()
        return np.array(float(out.cpu()), dtype=np.float32)

    probe("compute: sum reduction", reduction_sum, np.array(32.0, dtype=np.float32))

    def dtype_cast():
        a = torch.from_numpy(ones.copy()).npu()
        out = a.to(torch.float32)
        torch.npu.synchronize()
        return out.cpu().numpy()

    probe("compute: fp16 -> fp32 cast", dtype_cast, ones.astype(np.float32))

    def matmul_small():
        rng = np.random.default_rng(0)
        a_np = rng.standard_normal((16, 16)).astype(np.float16)
        b_np = rng.standard_normal((16, 16)).astype(np.float16)
        a, b = torch.from_numpy(a_np).npu(), torch.from_numpy(b_np).npu()
        out = (a @ b).cpu().numpy().astype(np.float32)
        reference = a_np.astype(np.float32) @ b_np.astype(np.float32)
        # fp16 accumulation, so compare within a tolerance rather than exactly
        denom = float(np.abs(reference).max()) or 1.0
        close = float(np.abs(out - reference).max()) / denom < 0.05
        return np.array(close)

    probe("compute: matmul 16x16 (vs numpy, 5% tol)", matmul_small, np.array(True))

    def softmax_composite():
        rng = np.random.default_rng(1)
        a_np = rng.standard_normal((4, 8)).astype(np.float32)
        a = torch.from_numpy(a_np).npu()
        out = torch.softmax(a, dim=-1).cpu().numpy()
        exp = np.exp(a_np - a_np.max(axis=-1, keepdims=True))
        reference = exp / exp.sum(axis=-1, keepdims=True)
        return np.array(bool(np.abs(out - reference).max() < 1e-3))

    probe("compute: softmax (composite op)", softmax_composite, np.array(True))

    # --- runtime plumbing a ctypes runner needs ----------------------------------
    probe(
        "runtime: torch.npu.synchronize()", lambda: (torch.npu.synchronize(), None)[1]
    )
    probe(
        "runtime: current_stream pointer",
        lambda: np.array(int(torch.npu.current_stream().npu_stream) != 0),
        np.array(True),
    )
    probe(
        "runtime: get_device_properties",
        lambda: np.array(torch.npu.get_device_properties(DEVICE) is not None),
        np.array(True),
    )

    broken = [r for r in RESULTS if r["status"] != "ok"]
    payload = {
        "result": "PASS" if not broken else "PARTIAL",
        "ok": sum(r["status"] == "ok" for r in RESULTS),
        "broken": len(broken),
        "results": RESULTS,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
    # Exit 0 even when some probes fail: the point is the inventory, not a verdict.
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
