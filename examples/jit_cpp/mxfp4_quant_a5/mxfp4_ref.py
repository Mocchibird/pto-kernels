#!/usr/bin/env python3
"""Host reference for MXFP4, written against the device-measured contract.

FLOOR scale rule, RNE elements, low-nibble-first packing -- all three measured in
PLAN.md section 0, none assumed. Pure numpy on bf16 bit patterns so it is exact.
"""
import numpy as np

QK = 32
FP4_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def bf16_bits(x_f32):
    """Round f32 -> bf16 (RNE) and return the uint16 bit pattern."""
    u = x_f32.astype(np.float32).view(np.uint32)
    lower = (u & 0xFFFF).astype(np.uint32)
    upper = (u >> 16).astype(np.uint32)
    round_up = (lower > 0x8000) | ((lower == 0x8000) & ((upper & 1) == 1))
    return (upper + round_up).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fp4_code(vals):
    """RNE to the nearest E2M1 magnitude, saturating at 6.0; returns 4-bit codes."""
    sign = (vals < 0) | ((vals == 0) & (np.signbit(vals)))
    a = np.abs(vals)
    # midpoints between consecutive representable magnitudes
    mids = (FP4_MAG[:-1] + FP4_MAG[1:]) / 2.0
    field = np.searchsorted(mids, a, side="left").astype(np.uint8)
    # exact ties go to the even field (RNE)
    for i, m in enumerate(mids):
        tie = a == m
        if tie.any():
            field[tie] = i if (i % 2 == 0) else i + 1
    field = np.minimum(field, 7).astype(np.uint8)
    return (field | (sign.astype(np.uint8) << 3)).astype(np.uint8)


def quantize(x_f32):
    """x (batch, K) f32 -> (packed uint8 (batch,K/2), e8m0 uint8 (batch,K/32))."""
    batch, k = x_f32.shape
    xb = bf16_bits(x_f32)
    x = bf16_to_f32(xb)  # exactly what the device sees
    blocks = x.reshape(batch, k // QK, QK)
    amax_bits = (xb.reshape(batch, k // QK, QK) & 0x7FFF).max(axis=2)
    b = ((amax_bits >> 7) & 0xFF).astype(np.int32)
    b = np.clip(b, 2, 254)
    scale_byte = (b - 2).astype(np.uint8)
    mult = np.exp2((129 - b).astype(np.float32))  # 1/X, exact power of two
    scaled = blocks * mult[:, :, None]
    codes = fp4_code(scaled.reshape(batch, k))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_byte


def dequantize(packed, scale_byte, k):
    lo = (packed & 0xF).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    codes = np.empty((packed.shape[0], k), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    mag = FP4_MAG[codes & 0x7]
    sign = np.where(codes & 0x8, -1.0, 1.0)
    x = (mag * sign).reshape(packed.shape[0], k // QK, QK)
    scale = np.exp2(scale_byte.astype(np.float32) - 127.0)
    return (x * scale[:, :, None]).reshape(packed.shape[0], k)
