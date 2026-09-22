"""Constellations, Gray bit mapping, pulse shaping and soft demapping.

Shared by the transmitter (synthetic generator) and the receiver.
"""
from __future__ import annotations

import numpy as np

from .utils import gray

LINEAR_MODS = ("BPSK", "QPSK", "8PSK", "16QAM")
FSK_MODS = ("2FSK", "4FSK", "8FSK")
ALL_MODS = LINEAR_MODS + FSK_MODS

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "2FSK": 1, "4FSK": 2, "8FSK": 3}
# Rotational symmetry of each constellation: the carrier-phase ambiguity a
# blind receiver cannot resolve without a known preamble.
PHASE_SYMMETRY = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4}
# Power used to strip the modulation for carrier estimation (s^M has a tone).
CARRIER_POWER = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4}


def constellation(mod: str) -> np.ndarray:
    """Points indexed by their bit label (int), unit average energy, Gray coded."""
    if mod == "BPSK":
        return np.array([1.0 + 0j, -1.0 + 0j])
    if mod == "QPSK":
        pts = np.zeros(4, dtype=complex)
        for v in range(4):
            b0, b1 = (v >> 1) & 1, v & 1
            pts[v] = ((1 - 2 * b0) + 1j * (1 - 2 * b1)) / np.sqrt(2)
        return pts
    if mod == "8PSK":
        pts = np.zeros(8, dtype=complex)
        for level in range(8):
            pts[gray(level)] = np.exp(1j * np.pi / 4 * level)
        return pts
    if mod == "16QAM":
        amp = np.array([-3, -1, 1, 3], dtype=float)
        pts = np.zeros(16, dtype=complex)
        for li in range(4):
            for lq in range(4):
                label = (gray(li) << 2) | gray(lq)
                pts[label] = amp[li] + 1j * amp[lq]
        return pts / np.sqrt(10)
    raise ValueError(mod)


def fsk_levels(M: int) -> np.ndarray:
    """Normalised FSK tone positions indexed by bit label: level l -> 2l-(M-1)."""
    lv = np.zeros(M)
    for level in range(M):
        lv[gray(level)] = 2 * level - (M - 1)
    return lv


def bits_to_labels(bits: np.ndarray, k: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int64).reshape(-1, k)
    w = 1 << np.arange(k - 1, -1, -1)
    return b @ w


def labels_to_bits(labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return ((labels[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)


def rrc_taps(sps: float, rolloff: float, span: int = 10) -> np.ndarray:
    """Root-raised-cosine impulse response. ``sps`` may be fractional."""
    n = int(np.ceil(span * sps)) | 1
    t = (np.arange(n) - (n - 1) / 2) / sps          # time in symbols
    a = rolloff
    h = np.zeros(n)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - a + 4 * a / np.pi
        elif a > 0 and abs(abs(ti) - 1 / (4 * a)) < 1e-9:
            h[i] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - a)) + 4 * a * ti * np.cos(np.pi * ti * (1 + a))) / (
                np.pi * ti * (1 - (4 * a * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def soft_demap(y: np.ndarray, points: np.ndarray, k: int, noise_var: float) -> np.ndarray:
    """Max-log LLRs. ``points`` indexed by label; works for complex or real y."""
    y = np.asarray(y)
    d = np.abs(y[:, None] - points[None, :]) ** 2          # (N, M)
    labels = np.arange(len(points))
    llr = np.zeros((len(y), k))
    for j in range(k):
        bit = (labels >> (k - 1 - j)) & 1
        d1 = d[:, bit == 1].min(axis=1)
        d0 = d[:, bit == 0].min(axis=1)
        llr[:, j] = (d1 - d0) / max(noise_var, 1e-6)
    return np.clip(llr, -50, 50).reshape(-1)
