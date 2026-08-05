"""Host reference for MXFP4 block quantization — numpy only, no device, no torch.

Models the *exact* chain the kernel runs, so the correctness gate is bit-exact
rather than a tolerance. Two independent references:

1. ``quantize`` — the bit chain: extract the biased exponent of ``|amax|``, clamp
   it, derive both the E8M0 byte and the bf16 reciprocal from the *clamped* value,
   multiply, then round to the E2M1 grid.
2. ``scale_bytes_from_spec`` — the same scale bytes computed independently in
   float64 from ``floor(log2(amax)) + 125``. It shares no arithmetic with (1), so
   it catches a bias-constant error that (1) would happily reproduce.

The bf16 input domain is deliberate: the only narrowing cast to fp4 on this device
takes bf16, so a bf16 input means a single rounding and an attainable bit-exact
gate. An fp16 path would go f16 -> bf16 -> fp4 and double-round.

Format facts are device-measured (see README.md); none are assumed here.
"""

import numpy as np

MX_BLOCK = 32  # elements sharing one E8M0 scale
E8M0_BIAS_ADJ = -2  # byte = b - 2, OCP MX v1.0 6.3 Algorithm 1 (FLOOR)
RECIP_OFFSET = 256  # 1/X exponent field = 256 - b
B_MIN, B_MAX = 2, 254  # b window keeping 1/X a finite, non-subnormal bf16
BF16_MANT_BITS = 7

# E2M1 magnitude grid, indexed by the 3-bit magnitude field.
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E2M1_MAX = 6.0


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round-to-nearest-even."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    # round-half-to-even on the 16 discarded bits
    lsb = (bits >> 16) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> exact float32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def e2m1_code(values: np.ndarray) -> np.ndarray:
    """Round |values| onto the E2M1 grid, RNE, saturating, and add the sign bit.

    Ties break to the neighbour with the even magnitude field, which puts
    0.25->0.0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0.
    """
    v = np.asarray(values, dtype=np.float64)
    sign = np.signbit(v)
    mag = np.abs(v)
    mag = np.where(np.isnan(mag), E2M1_MAX, mag)  # no NaN encoding; saturate

    # index of the first grid point >= mag
    upper = np.searchsorted(E2M1_GRID, mag, side="left").astype(np.int64)
    upper = np.clip(upper, 0, len(E2M1_GRID) - 1)
    lower = np.maximum(upper - 1, 0)
    lo_val, hi_val = E2M1_GRID[lower], E2M1_GRID[upper]

    d_lo, d_hi = mag - lo_val, hi_val - mag
    # strictly closer to one side, else the even magnitude field wins
    take_hi = d_hi < d_lo
    tie = d_hi == d_lo
    take_hi = np.where(tie, (upper % 2) == 0, take_hi)
    field = np.where(take_hi, upper, lower)
    field = np.where(mag >= E2M1_MAX, len(E2M1_GRID) - 1, field)  # saturate
    return (field.astype(np.uint8) | (sign.astype(np.uint8) << 3)).astype(np.uint8)


def _blocked(bits: np.ndarray) -> np.ndarray:
    batch, k = bits.shape
    if k % MX_BLOCK:
        raise ValueError(f"K must be a multiple of {MX_BLOCK}, got {k}")
    return bits.reshape(batch, k // MX_BLOCK, MX_BLOCK)


def quantize(x_bf16_bits: np.ndarray):
    """(q_bytes, scale_bytes) for a (batch, K) array of bf16 bit patterns.

    Reproduces the kernel step for step: magnitude mask, biased exponent, clamp,
    byte and reciprocal from the clamped b, bf16 multiply, E2M1 RNE, nibble pack.
    """
    blocks = _blocked(np.asarray(x_bf16_bits, dtype=np.uint16))
    mag_bits = blocks & np.uint16(0x7FFF)  # sign cleared

    # a signed max over sign-cleared bit patterns IS a magnitude max
    amax_bits = mag_bits.max(axis=2)
    b = (amax_bits >> BF16_MANT_BITS).astype(np.int32)
    b = np.clip(b, B_MIN, B_MAX)

    scale = (b + E8M0_BIAS_ADJ).astype(np.uint8)
    recip_bits = ((RECIP_OFFSET - b) << BF16_MANT_BITS).astype(np.uint16)

    # the reciprocal is an exact power of two, so this multiply is exact in bf16
    scaled = bf16_bits_to_f32(blocks) * bf16_bits_to_f32(recip_bits)[:, :, None]
    codes = e2m1_code(scaled)

    batch, nblk, _ = codes.shape
    flat = codes.reshape(batch, nblk * MX_BLOCK)
    lo, hi = flat[:, 0::2], flat[:, 1::2]
    # element 2j -> LOW nibble; device-measured, see README
    q = ((hi.astype(np.uint8) << 4) | lo.astype(np.uint8)).astype(np.uint8)
    return q, scale.reshape(batch, nblk)


def scale_bytes_from_spec(x_bf16_bits: np.ndarray) -> np.ndarray:
    """Scale bytes from the spec formula in float64, sharing no arithmetic with
    ``quantize`` — an independent check on the bias constant."""
    blocks = _blocked(np.asarray(x_bf16_bits, dtype=np.uint16))
    amax = np.abs(bf16_bits_to_f32(blocks)).max(axis=2).astype(np.float64)
    with np.errstate(divide="ignore"):
        shared = np.floor(np.log2(amax))
    shared = np.where(amax == 0, -np.inf, shared)
    byte = np.clip(shared + 125, B_MIN + E8M0_BIAS_ADJ, B_MAX + E8M0_BIAS_ADJ)
    return byte.astype(np.uint8)


def dequantize(q_bytes: np.ndarray, scale_bytes: np.ndarray) -> np.ndarray:
    """Reconstruct float64 values, for the RMSE/R-squared quality report."""
    q = np.asarray(q_bytes, dtype=np.uint8)
    lo, hi = q & np.uint8(0x0F), q >> 4
    codes = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    mag = E2M1_GRID[codes & np.uint8(0x07)]
    signed = np.where((codes & np.uint8(0x08)) != 0, -mag, mag)
    x = np.exp2(np.asarray(scale_bytes, dtype=np.float64) - 127.0)
    batch, nblk = q.shape[0], scale_bytes.shape[1]
    # tuple shapes, not varargs: pylint mis-infers np.where's return type and
    # reports too-many-function-args on the varargs form
    blocked = signed.reshape((batch, nblk, MX_BLOCK)) * x[:, :, None]
    return blocked.reshape((batch, nblk * MX_BLOCK))


def quality(original: np.ndarray, reconstructed: np.ndarray):
    """(rmse_relative, r_squared) — the skill's metric for outlier-heavy kernels.

    Max-error alone says only which value landed worst in a 16-level grid; these
    say whether the quantization is usable, and are comparable across kernels.
    """
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(reconstructed, dtype=np.float64).ravel()
    err = b - a
    rms = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sqrt(np.mean(a**2))) or 1.0
    var = float(np.var(a))
    r2 = 1.0 - float(np.mean(err**2)) / var if var > 0 else 1.0
    return rms / denom, r2
