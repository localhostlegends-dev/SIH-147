"""Small bit-level helpers shared by every stage of the pipeline.

Conventions used throughout the project
---------------------------------------
* Bits are ``numpy.uint8`` arrays holding 0/1, most-significant bit first.
* Soft bits are log-likelihood ratios (LLRs): ``llr = log P(b=0) / P(b=1)``.
  A positive LLR means "probably 0", a negative LLR means "probably 1",
  and the magnitude is the confidence. ``hard = (llr < 0)``.
"""
from __future__ import annotations

import numpy as np

# 48-bit channel preamble: a window of the length-127 m-sequence (x^7 + x^6 + 1)
# chosen for low autocorrelation sidelobes and no repeated symbols at its edges.
# It is transmitted before every frame, uncoded and not interleaved, and is
# used for burst synchronisation and for resolving the carrier-phase
# ambiguity of PSK/QAM. 48 bits divides evenly into 1, 2, 3 and 4 bits/symbol.
PREAMBLE_HEX = "C8B3A9F43893"

# 32-bit frame synchronisation word placed at the start of every *decoded*
# frame. This is the CCSDS Attached Sync Marker (ASM), a real-world standard.
FRAME_SYNC_HEX = "1ACFFC1D"


def hex_to_bits(h: str) -> np.ndarray:
    h = h.strip().replace(" ", "").replace("0x", "").replace("0X", "")
    if not h:
        return np.zeros(0, dtype=np.uint8)
    if len(h) % 2:
        h = "0" + h
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))


def bits_to_hex(bits: np.ndarray) -> str:
    return bits_to_bytes(bits).hex().upper()


def bytes_to_bits(b: bytes | np.ndarray) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(b), dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits).tobytes()  # zero-pads the last byte


def bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in np.asarray(bits, dtype=np.uint8):
        v = (v << 1) | int(b)
    return v


def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def gray(n):
    """Binary -> Gray code (works on ints and numpy arrays)."""
    return n ^ (n >> 1)


def bipolar(bits: np.ndarray) -> np.ndarray:
    """0/1 bits -> +1/-1 (0 -> +1), the same sign convention as LLRs."""
    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)


def hard(llr: np.ndarray) -> np.ndarray:
    return (np.asarray(llr) < 0).astype(np.uint8)


def printable(data: bytes) -> str:
    """Render bytes as text, replacing non-printable characters with '.'."""
    return "".join(chr(c) if 32 <= c < 127 else "." for c in data)
