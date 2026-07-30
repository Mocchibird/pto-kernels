#!/usr/bin/env python3
"""Run the kernel and diff against the host reference, showing the permutation."""
import sys

import numpy as np
import torch
import torch_npu  # noqa

sys.path.insert(0, ".")
from jit_util_mxfp4_a5 import build, run  # noqa: E402
from mxfp4_ref import quantize  # noqa: E402

K = 256
BATCH = 16  # one tile


def main():
    fn = build(kwidth=K, rows_per_tile=16, verbose=True)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, K)).astype(np.float32)
    xt = torch.from_numpy(x).to(torch.bfloat16).npu()
    q_dev, s_dev = run(fn, xt, kwidth=K)
    q_dev = q_dev.cpu().numpy()
    s_dev = s_dev.cpu().numpy()
    q_ref, s_ref = quantize(x)

    print(f"\nscales  row0  dev: {s_dev[0].tolist()}")
    print(f"scales  row0  ref: {s_ref[0].tolist()}")
    print(
        f"scales match: {np.array_equal(s_dev, s_ref)}  "
        f"({int((s_dev == s_ref).sum())}/{s_dev.size})"
    )

    print(f"\npacked row0 dev[:24]: {' '.join('%02x' % v for v in q_dev[0][:24])}")
    print(f"packed row0 ref[:24]: {' '.join('%02x' % v for v in q_ref[0][:24])}")
    print(
        f"packed match: {np.array_equal(q_dev, q_ref)}  "
        f"({int((q_dev == q_ref).sum())}/{q_dev.size})"
    )

    nz = np.nonzero(q_dev[0])[0]
    print(
        f"\nrow0 dev nonzero byte positions: {nz[:16].tolist()}"
        f"{' ...' if len(nz) > 16 else ''}  (count {len(nz)}/{K // 2})"
    )
    if len(nz):
        d = np.diff(nz[:8])
        print(f"  stride between them: {d.tolist()}")
    print("DBG DONE")


if __name__ == "__main__":
    main()
