"""Stage 6 - bitstream correlation: find headers, separate header from payload.

Two complementary views:

1. **Known sync word** (e.g. the CCSDS ASM 0x1ACFFC1D): cross-correlate it
   against each decoded frame. The peak marks the start of the header; the
   fields after it (counter, type, length) are parsed and the payload
   extracted.
2. **Blind structure**: stack the decoded frames and measure, for every bit
   position, how often it takes the same value (bit constancy). Headers are
   (almost) constant from frame to frame, counters change only in their low
   bits, payload looks random. A frame-to-frame similarity matrix shows the
   same thing at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .framing import constancy_profile, correlate_pattern
from .utils import bits_to_bytes, bits_to_hex, bits_to_int, printable

HEADER_FIELDS = [("counter", 16), ("type", 8), ("length", 8)]


@dataclass
class CorrelationResult:
    sync_hex: str | None
    sync_positions: list
    sync_peaks: list
    frames: list                               # parsed frame dicts
    message: str
    constancy: np.ndarray
    header_len_est: int
    similarity: np.ndarray = field(repr=False, default=None)
    sync_trace: np.ndarray = field(repr=False, default=None)


def segment_header(prof: np.ndarray, window: int = 16, thr: float = 0.8) -> int:
    """Header length estimate: where the smoothed bit constancy first drops."""
    if len(prof) < window:
        return 0
    sm = np.convolve(prof, np.ones(window) / window, mode="valid")
    below = np.nonzero(sm < thr)[0]
    return int(below[0] + window // 2) if len(below) else len(prof)


def analyse(info_frames: list[np.ndarray], sync_bits: np.ndarray | None) -> CorrelationResult:
    frames = [np.asarray(f, dtype=np.uint8) for f in info_frames]
    L = min(len(f) for f in frames) if frames else 0
    stack = np.array([f[:L] for f in frames]) if frames else np.zeros((0, 0), dtype=np.uint8)
    prof = constancy_profile(stack) if len(stack) > 1 else np.ones(L)
    sim = None
    if len(stack) > 1:
        b = 1.0 - 2.0 * stack
        sim = (b @ b.T) / L                       # +1 identical, 0 unrelated
    parsed, positions, peaks = [], [], []
    trace = None
    if sync_bits is not None and len(sync_bits) and L >= len(sync_bits):
        for i, f in enumerate(frames):
            c = correlate_pattern(f, sync_bits)
            if i == 0:
                trace = c
            p = int(np.argmax(c))
            positions.append(p)
            peaks.append(float(c[p]))
            rec = {"frame": i, "sync_pos": p, "sync_corr": round(float(c[p]), 3)}
            q = p + len(sync_bits)
            if c[p] >= 0.85 and q + 32 <= len(f):
                for name, w in HEADER_FIELDS:
                    rec[name] = bits_to_int(f[q:q + w])
                    q += w
                n = min(rec["length"], (len(f) - q) // 8)
                payload = bits_to_bytes(f[q:q + 8 * n])
                rec["payload_hex"] = payload.hex().upper()
                rec["payload_text"] = printable(payload)
                rec["header_hex"] = bits_to_hex(f[p:p + len(sync_bits) + 32])
            parsed.append(rec)
    ordered = sorted([r for r in parsed if "payload_text" in r], key=lambda r: r["counter"])
    seen, parts = set(), []
    for r in ordered:
        if r["counter"] not in seen:
            seen.add(r["counter"])
            parts.append(r["payload_text"])
    message = "".join(parts)
    return CorrelationResult(bits_to_hex(sync_bits) if sync_bits is not None and len(sync_bits) else None,
                             positions, peaks, parsed, message, prof, segment_header(prof), sim, trace)
