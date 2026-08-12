#!/usr/bin/env python3
"""Simulator smoke for mxfp4_quant_a5: run on the A5 CA model, verify in numpy.

The pytest suite compares against torch_npu.npu_dynamic_mx_quant, which needs
real A5 silicon. Under the CA model there is no vendor op, so this restates OCP
MX v1.0 6.3 Algorithm 1 (FLOOR) in numpy instead and compares bit-exactly.

The reference is exact, not approximate: the scale is a power of two, so the
kernel's bf16 multiply is exact, and float64 reproduces it before the E2M1
rounding step. Any mismatch is a real disagreement, not tolerance drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from jit_util_mxfp4_a5 import (  # noqa: E402
    MX_BLOCK,
    build_and_load,
    row_quantum,
    rows_for,
)

# E2M1 magnitude grid by 3-bit code, and the midpoints between neighbours.
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = (E2M1_GRID[:-1] + E2M1_GRID[1:]) / 2.0  # 0.25 0.75 1.25 1.75 2.5 3.5 5

BF16_ABS = 0x7FFF
BF16_MANT_BITS = 7
E8M0_BIAS_ADJ = -2  # scale byte = biased exponent - 2 (Algorithm 1, FLOOR)
RECIP_OFFSET = 256  # 1/X exponent field = 256 - biased exponent
B_MIN, B_MAX = 2, 254  # window where 1/X stays a finite, normal bf16


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    """The raw bf16 bit patterns, so the reference sees exactly what the DMA does."""
    return tensor.cpu().view(torch.uint16).numpy().astype(np.uint32)


def e2m1_encode(magnitude: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code for a magnitude, ties to even, saturating at 6.0.

    Round-to-nearest-even on a 1-bit mantissa: at a midpoint the odd-indexed
    boundaries round up and the even ones round down, which `index % 2` gives.
    """
    index = np.searchsorted(E2M1_MID, magnitude, side="left")
    tie = np.isin(magnitude, E2M1_MID)
    index = np.where(tie, index + (index % 2), index)
    return np.minimum(index, len(E2M1_GRID) - 1).astype(np.uint8)


def reference_quantize(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(nibbles, scales) from the spec, in numpy. Shapes match the kernel's."""
    bits = bf16_bits(tensor)
    batch, k = bits.shape
    blocks = k // MX_BLOCK

    magnitude = bits & BF16_ABS
    # sign cleared, so the max over bit patterns IS the magnitude max
    amax = magnitude.reshape(batch, blocks, MX_BLOCK).max(axis=-1)
    exponent = np.clip(amax >> BF16_MANT_BITS, B_MIN, B_MAX).astype(np.int64)
    scales = (exponent + E8M0_BIAS_ADJ).astype(np.uint8)

    # The reciprocal is 2^(129-b) exactly: field 256-b, value 2^(256-b-127).
    recip = np.exp2((RECIP_OFFSET - exponent) - 127).astype(np.float64)
    values = (bits << 16).astype(np.uint32).view(np.float32).astype(np.float64)
    scaled = values.reshape(batch, blocks, MX_BLOCK) * recip[:, :, None]

    codes = e2m1_encode(np.abs(scaled)).reshape(batch, k)
    # sign from the input's own bit 15, so -0.0 stays negative as the hardware does
    codes |= ((bits >> 15) << 3).astype(np.uint8)
    nibbles = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return nibbles, scales


def make_bf16(batch: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(batch, k, generator=generator, dtype=torch.float32)
    return values.to(torch.bfloat16)


# Values random N(0,1) never reaches: the clamp window, the saturation band, and
# every E2M1 midpoint, which is where a wrong rounding mode shows up.
ADVERSARIAL = {
    "clamp_window": [2.0**-15, 2.0**-14, 2.0**-13],
    "tiny_amax": [2.0**-20, 2.0**-24],
    "clip_to_six_band": [6.5, 7.0, 7.9],
    "e2m1_midpoints": [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    "all_zero": [0.0],
    "signed_zero": [-0.0],
    "huge_outlier": [1024.0],
    "near_bf16_max": [3.0e38],
}


def adversarial_tensor(k: int, rows: int) -> torch.Tensor:
    """One tensor carrying every adversarial family, each in its own block."""
    tensor = torch.zeros((rows, k), dtype=torch.bfloat16)
    block = 0
    for name in sorted(ADVERSARIAL):
        for value in ADVERSARIAL[name]:
            row, column = divmod(block * MX_BLOCK, k)
            if row >= rows:
                return tensor  # narrow k: took as many families as fit
            tensor[row, column] = value
            tensor[row, column + 1] = value / 2.0
            block += 1
    return tensor


def check(kernel, tensor: torch.Tensor, label: str) -> dict:
    """Launch once on the CA model and compare bit-exactly. Simulated cycles are
    expensive, so no repeat loop here -- the device suite covers sync races."""
    want_nibbles, want_scales = reference_quantize(tensor)
    nibbles, scales = kernel(tensor.npu())
    torch.npu.synchronize()
    got_nibbles, got_scales = nibbles.cpu().numpy(), scales.cpu().numpy()

    report = {"case": label, "batch": int(tensor.shape[0]), "k": int(tensor.shape[1])}
    for what, got, want in (
        ("scale", got_scales, want_scales),
        ("nibble", got_nibbles, want_nibbles),
    ):
        wrong = int((got != want).sum())
        report[f"{what}_mismatches"] = wrong
        if wrong:
            bad = np.argwhere(got != want)[:4]
            report["result"] = "FAIL"
            report[f"{what}_first_bad"] = [
                {
                    "at": [int(i) for i in index],
                    "got": int(got[tuple(index)]),
                    "want": int(want[tuple(index)]),
                }
                for index in bad
            ]
            print(f"  {label}: {what} differs in {wrong} of {want.size}")
            return report
    report["result"] = "PASS"
    print(f"  {label}: PASS ({report['batch']}x{report['k']})")
    return report


def cases(k: int, batch: int | None) -> list[tuple[str, torch.Tensor]]:
    rows, quantum = rows_for(k), row_quantum(k)
    if batch is not None:
        return [(f"batch{batch}", make_bf16(batch, k, batch))]
    # A whole tile, a batch that ends mid-tile (the kernel's tail path), and the
    # enumerated edge values. Kept to a few tiles: the CA model is cycle-level.
    tail = rows + quantum if rows > 1 else rows
    return [
        ("full_tile", make_bf16(rows, k, 1)),
        ("partial_tile", make_bf16(tail, k, 2)),
        ("adversarial", adversarial_tensor(k, rows)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # k=64 is the narrowest instantiation, so one tile is the least simulated
    # work that still exercises a whole tile. TILE_ELEMS is fixed, so a wider k
    # costs the same per tile but needs a bigger batch to fill one.
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--batch", type=int, help="one custom batch instead of the cases"
    )
    parser.add_argument("--block-dim", type=int, default=2, help="AIV cores to launch")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(os.environ.get("NPU_DEVICE", "npu:0"))

    kernel = build_and_load(vector_cores=args.block_dim, k=args.k, verbose=True)
    print(f"[sim] k={args.k} rows/tile={rows_for(args.k)} block_dim={args.block_dim}")

    reports = [
        check(kernel, tensor, label) for label, tensor in cases(args.k, args.batch)
    ]
    failed = [r for r in reports if r["result"] != "PASS"]
    payload = {
        "result": "FAIL" if failed else "PASS",
        "k": args.k,
        "block_dim": args.block_dim,
        "results": reports,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    # The simulator runtime aborts in its own teardown (std::bad_function_call)
    # AFTER the kernel and the comparison are done. That abort makes msprof
    # discard the profiling dump, so leave before the atexit handlers run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
