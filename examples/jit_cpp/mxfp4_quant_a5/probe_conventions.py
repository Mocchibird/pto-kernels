#!/usr/bin/env python3
"""Stage 0 probe: measure what the torch_npu MXFP4 baseline actually DOES.

Resolves the open questions that gate every later claim (PLAN.md section 7):
  R2  what is dst_type=296 -- 4-bit packed (K/2 bytes) or an 8-bit container?
      Until this is known, every bytes/row constant is a guess, including the
      WIP's TOTAL_B=324.
  R3  which scale rule -- FLOOR or RCEIL? A one-code difference is a 2x error.
  R5  which element rounding mode -- RNE, or round-half-away?
  --  nibble order: is element 0 the low or the high nibble?

Everything here is measurement. Nothing is asserted about the format; the
tabulated raw bytes are the evidence. Run on bz.39 with the BASE miniconda
python3 after sourcing set_env.sh (NOT the `ascend` conda env -- no torch there).
"""

import inspect

import numpy as np
import torch
import torch_npu  # noqa

BLK = 32
DST = 296  # the undocumented magic int the WIP passes


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def call(x, block_size=BLK, dst_type=DST):
    return torch_npu.npu_dynamic_mx_quant(x, block_size=block_size, dst_type=dst_type)


# ---------------------------------------------------------------- R2
def probe_dst_type():
    hr("R2 -- what does dst_type=296 produce?")
    fn = torch_npu.npu_dynamic_mx_quant
    try:
        print("signature:", inspect.signature(fn))
    except (TypeError, ValueError) as e:
        print("signature: unavailable:", e)
    doc = (getattr(fn, "__doc__", "") or "").strip()
    print("doc[:600]:", doc[:600] if doc else "(none)")

    for K in (256, 128):
        x = torch.zeros(64, K, dtype=torch.float16).npu()
        x[:, 0] = 1.0
        try:
            out = call(x)
        except Exception as e:  # noqa: BLE001 -- discovery probe
            print(f"K={K}: call FAILED: {type(e).__name__}: {e}")
            continue
        parts = out if isinstance(out, (tuple, list)) else (out,)
        print(f"\nK={K}: returned {len(parts)} tensor(s)")
        for i, t in enumerate(parts):
            n = t.numel() * t.element_size()
            per_row = n / 64
            print(
                f"  [{i}] dtype={t.dtype} shape={tuple(t.shape)} "
                f"elt_size={t.element_size()}B total={n}B per_row={per_row:g}B"
            )
        q = parts[0]
        last = int(q.shape[-1])
        if last == K // 2:
            print(f"  => VERDICT: q is 4-BIT PACKED (last dim {last} == K/2)")
        elif last == K:
            print(f"  => VERDICT: q is an 8-BIT CONTAINER (last dim {last} == K)")
        else:
            print(f"  => VERDICT: UNEXPECTED last dim {last} for K={K}")
        if len(parts) > 1:
            s = parts[1]
            exp = K // BLK
            print(
                f"  scale last dim={int(s.shape[-1])} (K/{BLK} = {exp}) "
                f"{'OK' if int(s.shape[-1]) == exp else 'MISMATCH'}"
            )

    # is 296 a known enum anywhere?
    hits = []
    for mod, name in ((torch, "torch"), (torch_npu, "torch_npu")):
        for attr in dir(mod):
            try:
                v = getattr(mod, attr)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, int) and v == DST:
                hits.append(f"{name}.{attr}")
    print(f"\nint-valued attrs equal to {DST}: {hits or '(none found)'}")


def raw_bytes(t):
    return t.cpu().numpy().tobytes()


def scale_of(parts, row=0, blk=0):
    s = parts[1].cpu().numpy()
    return int(s.reshape(s.shape[0], -1)[row, blk])


def q_bytes(parts, row=0, count=4):
    q = parts[0].cpu().numpy()
    return q.reshape(q.shape[0], -1)[row, :count].astype(np.uint8)


# ---------------------------------------------------------------- R3
def probe_scale_rule():
    hr("R3 -- scale rule: what E8M0 byte comes out for a controlled amax?")
    print("One block of 32; element 0 = A, rest 0. E8M0 byte s means 2^(s-127).")
    print("OCP FLOOR predicts s = floor(log2(A)) - 2 + 127 (emax_fp4 = 2).\n")
    print(
        f"{'A':>12} {'s':>5} {'2^(s-127)':>12} {'floor_pred':>11} "
        f"{'rceil_pred':>11} {'A/2^(s-127)':>12}"
    )
    print("-" * 70)
    As = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 0.5, 0.25, 12.0, 16.0, 96.0, 1024.0]
    for A in As:
        x = torch.zeros(64, BLK, dtype=torch.float16).npu()
        x[:, 0] = A
        try:
            parts = call(x)
        except Exception as e:  # noqa: BLE001
            print(f"{A:12g}  call failed: {e}")
            continue
        s = scale_of(parts)
        sc = 2.0 ** (s - 127)
        floor_pred = int(np.floor(np.log2(A))) - 2 + 127
        # RCEIL: scale chosen so A/scale <= 6 with ceil on the ratio's exponent
        rceil_pred = int(np.ceil(np.log2(A / 6.0))) + 127
        print(
            f"{A:12g} {s:5d} {sc:12g} {floor_pred:11d} {rceil_pred:11d} "
            f"{A / sc if sc else float('nan'):12.4f}"
        )


# ---------------------------------------------------------------- nibble order
def probe_nibble_order():
    hr("NIBBLE ORDER -- is element 0 the low or the high nibble?")
    # amax = 6.0 in the block, so the scale should map 6.0 -> code 0b111 (=6.0)
    x = torch.zeros(64, BLK, dtype=torch.float16).npu()
    x[:, 0] = 1.0  # expect code 0b010 = 0x2
    x[:, 1] = 2.0  # expect code 0b100 = 0x4
    x[:, 31] = 6.0  # fixes amax
    parts = call(x)
    s = scale_of(parts)
    b = q_bytes(parts, count=2)
    print(f"input: e0=1.0 e1=2.0 e31=6.0(amax)   scale_byte={s} (2^{s - 127})")
    print(f"first 2 output bytes: {[hex(v) for v in b]}")
    lo, hi = b[0] & 0xF, (b[0] >> 4) & 0xF
    print(f"byte0: low nibble=0x{lo:x}  high nibble=0x{hi:x}")
    if (lo, hi) == (0x2, 0x4):
        print("  => VERDICT: element 0 is the LOW nibble  (byte = c1<<4 | c0)")
    elif (lo, hi) == (0x4, 0x2):
        print("  => VERDICT: element 0 is the HIGH nibble (byte = c0<<4 | c1)")
    else:
        print("  => INCONCLUSIVE: codes are not the expected 0x2/0x4 pair;")
        print("     the scale rule may differ from the assumption -- read R3 first.")


# ---------------------------------------------------------------- R5
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # magnitude by field 0..7


def probe_rounding():
    hr("R5 -- element rounding at exact fp4 midpoints")
    mids = [
        (0.25, 0, 1),
        (0.75, 1, 2),
        (1.25, 2, 3),
        (1.75, 3, 4),
        (2.5, 4, 5),
        (3.5, 5, 6),
        (5.0, 6, 7),
    ]
    print("Each midpoint sits exactly between two fp4 codes. RNE breaks the tie")
    print("to the EVEN code field; round-half-away always takes the larger.\n")
    print(f"{'value':>8} {'lower':>14} {'upper':>14} {'got':>10} {'=> rule':>22}")
    print("-" * 74)
    for v, lo_f, hi_f in mids:
        x = torch.zeros(64, BLK, dtype=torch.float16).npu()
        x[:, 0] = v
        x[:, 31] = 6.0  # pin amax so scale is 1 (verify against R3 output)
        parts = call(x)
        s = scale_of(parts)
        if 2.0 ** (s - 127) != 1.0:
            note = f"(scale={2.0**(s-127):g}, not 1 -- interpret with care)"
        else:
            note = ""
        code = int(q_bytes(parts, count=1)[0] & 0xF)
        field = code & 0x7
        even = lo_f if lo_f % 2 == 0 else hi_f
        if field == lo_f:
            rule = "-> lower"
        elif field == hi_f:
            rule = "-> upper"
        else:
            rule = f"-> field {field}?!"
        tie = "RNE" if field == even else "half-away/other"
        print(
            f"{v:8g} {FP4[lo_f]:8g}(f{lo_f}) {FP4[hi_f]:8g}(f{hi_f}) "
            f"{FP4[field]:6g}(f{field}) {rule:>10} {tie:>12} {note}"
        )


def main():
    print(
        "torch",
        torch.__version__,
        "| torch_npu",
        getattr(torch_npu, "__version__", "?"),
    )
    print("npu available:", torch.npu.is_available())
    probe_dst_type()
    probe_scale_rule()
    probe_nibble_order()
    probe_rounding()
    print("\nPROBE DONE")


if __name__ == "__main__":
    main()
