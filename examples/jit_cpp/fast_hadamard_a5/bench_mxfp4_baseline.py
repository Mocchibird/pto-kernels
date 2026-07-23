#!/usr/bin/env python3
"""MXFP4-quant-only baseline via torch_npu.npu_dynamic_mx_quant, on (batch,128).
This is the throughput target the fused Hadamard+MXFP4 kernel must approach."""
import sys, math
import torch, torch_npu

N = 128
BLK = 32
NBLK = N // BLK            # 4 scale blocks per row
READ_B = N * 2             # fp16 input, bytes/row
WRITE_B = N // 2 + NBLK    # 64 fp4 bytes + 4 e8m0 scale bytes/row
TOTAL_B = READ_B + WRITE_B # 324 bytes/row


def time_us(fn, warmup=10, repeats=100):
    torch.npu.synchronize()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) * 1e3 / repeats


def main():
    batches = [16384, 65536]
    dtype = torch.float16
    # pool of distinct input buffers to avoid pure-cache reuse
    print(f"MXFP4 baseline: npu_dynamic_mx_quant, N={N}, block={BLK}, in={dtype}")
    print(f"traffic/row: read={READ_B}B write={WRITE_B}B total={TOTAL_B}B\n")
    hdr = f"{'batch':>8}  {'op':>18}  {'dur_us':>10}  {'in_GB/s':>9}  {'tot_GB/s':>9}  {'Gelem/s':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    for batch in batches:
        POOL = 8
        pool = [torch.randn(batch, N, dtype=dtype).npu() for _ in range(POOL)]
        i = {"k": 0}

        def quant():
            b = pool[i["k"] % POOL]; i["k"] += 1
            return torch_npu.npu_dynamic_mx_quant(b, block_size=BLK, dst_type=296)

        us = time_us(quant)
        in_gbs = (batch * READ_B) / 1e9 / (us / 1e6)
        tot_gbs = (batch * TOTAL_B) / 1e9 / (us / 1e6)
        gelem = (batch * N) / 1e9 / (us / 1e6)
        print(f"{batch:>8}  {'mxfp4_quant':>18}  {us:>10.3f}  {in_gbs:>9.1f}  {tot_gbs:>9.1f}  {gelem:>9.2f}")
        # sanity: shapes
        q, sc = quant()
        if batch == batches[0]:
            print(f"          -> q{tuple(q.shape)}:{q.dtype}  scale{tuple(sc.shape)}:{sc.dtype}")
    print("-" * len(hdr))


if __name__ == "__main__":
    main()
