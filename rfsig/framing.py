"""Stage 4 - burst/frame synchronisation on the demodulated bitstream.

* Known preamble: slide the preamble over the bitstream (cross-correlation);
  peaks mark frame starts. The demodulator hypothesis (carrier rotation /
  spectral inversion) giving the cleanest peaks is the correct one.
* Blind mode (preamble unknown): the autocorrelation of the bitstream peaks at
  the frame period; folding the stream at that period and measuring how
  constant each bit column is reveals the repeated preamble.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate

from .utils import bipolar, hard


@dataclass
class FramingResult:
    hypothesis: str
    preamble_bits: np.ndarray
    starts: np.ndarray                        # bit index of each preamble
    peak_values: np.ndarray                   # normalised correlation at each start
    period: int                               # frame length in bits (preamble included)
    block_len: int                            # bits between preamble end and next frame
    blocks: list                              # LLR arrays, one per complete frame
    corr_trace: np.ndarray = field(repr=False, default=None)
    mode: str = "known preamble"
    hypothesis_scores: dict = field(default_factory=dict)
    constancy: np.ndarray | None = field(repr=False, default=None)


def correlate_pattern(bits_or_llr: np.ndarray, pattern_bits: np.ndarray) -> np.ndarray:
    """Normalised (+1 = perfect match) sliding correlation of a bit pattern."""
    x = np.sign(np.asarray(bits_or_llr, dtype=float)) if bits_or_llr.dtype != np.uint8 else bipolar(bits_or_llr)
    x[x == 0] = 1
    p = bipolar(pattern_bits)
    return correlate(x, p, mode="valid", method="fft") / len(p)


def _peaks(corr: np.ndarray, thr: float, min_dist: int) -> np.ndarray:
    cand = np.nonzero(corr >= thr)[0]
    out = []
    for c in cand[np.argsort(-corr[cand])]:
        if all(abs(c - o) >= min_dist for o in out):
            out.append(c)
    return np.array(sorted(out), dtype=int)


def frame_with_preamble(hypotheses, preamble_bits: np.ndarray, thr: float = 0.75) -> FramingResult | None:
    Lp = len(preamble_bits)
    best = None
    scores = {}
    for name, llr in hypotheses:
        corr = correlate_pattern(llr, preamble_bits)
        pk = _peaks(corr, thr, max(Lp, 64))
        score = float(np.sum(corr[pk])) if len(pk) else float(corr.max())
        scores[name] = score
        if best is None or score > best[0]:
            best = (score, name, llr, corr, pk)
    _, name, llr, corr, pk = best
    if len(pk) < 2:
        return None
    period = int(np.median(np.diff(pk)))
    # keep starts that sit on the period grid (reject spurious peaks)
    good = [pk[0]]
    for p in pk[1:]:
        if abs((p - good[-1]) - period) <= 2 or (p - good[-1]) % period in (0, 1, period - 1):
            good.append(p)
    starts = np.array(good)
    block_len = period - Lp
    blocks = [llr[s + Lp: s + Lp + block_len] for s in starts if s + Lp + block_len <= len(llr)]
    return FramingResult(name, preamble_bits, starts, corr[starts], period, block_len, blocks, corr,
                         "known preamble", scores)


def blind_period(bits: np.ndarray, min_period: int = 128, max_period: int = 20000) -> tuple[int, np.ndarray]:
    x = bipolar(bits)
    n = len(x)
    max_period = min(max_period, n // 3)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(X * np.conj(X), nfft)[: max_period + 1]
    ac = ac / np.maximum(n - np.arange(len(ac)), 1)
    seg = ac[min_period:]
    best = int(min_period + np.argmax(seg))
    # multiples of the true period score about as high: prefer the smallest divisor
    # of the best lag that still shows a clear peak
    for k in range(8, 1, -1):
        cand = int(round(best / k))
        if cand < min_period:
            continue
        lo, hi = max(cand - 2, 0), cand + 3
        j = lo + int(np.argmax(ac[lo:hi]))
        if ac[j] >= 0.6 * ac[best]:
            return j, ac
    return best, ac


def constancy_profile(frames: np.ndarray) -> np.ndarray:
    """Per bit position: fraction of frames agreeing with the majority (0.5 .. 1)."""
    frames = np.asarray(frames, dtype=float)
    m = frames.mean(axis=0)
    return np.maximum(m, 1 - m)


@dataclass
class BlindGeometry:
    period: int
    run_start: int                  # bit offset (within the period) of the constant run
    run_len: int
    constancy: np.ndarray           # rolled so the run starts at index 0
    candidates: list                # (preamble_offset, preamble_len) to try, most likely first


def _trim_repeats(pattern: np.ndarray, bps: int, min_rep: int = 4) -> tuple[int, int]:
    """Leading/trailing bits made of one symbol repeated >= min_rep times.

    Flush/pad regions (e.g. zeros of a convolutional interleaver) are constant
    across frames and stick to the preamble in the constancy run; they look like
    the same symbol over and over, which a well-designed preamble never does.
    """
    def lead(p):
        if len(p) < bps * min_rep:
            return 0
        sym = p[:bps]
        n = 0
        while (n + 1) * bps <= len(p) and np.array_equal(p[n * bps:(n + 1) * bps], sym):
            n += 1
        return n * bps if n >= min_rep else 0
    a = lead(pattern)
    b = lead(pattern[::-1])
    return a, b


def blind_geometry(bits: np.ndarray, min_period: int = 128, max_period: int = 20000,
                   const_thr: float = 0.9, max_gap: int = 2, bps: int = 1) -> BlindGeometry | None:
    """Frame period + location/length of the repeated preamble, without knowing it."""
    period, _ = blind_period(bits, min_period, max_period)
    n_frames = len(bits) // period
    if n_frames < 3:
        return None
    folded = bits[: n_frames * period].reshape(n_frames, period)
    rows = folded[1:-1] if n_frames > 4 else folded     # first/last rows hold lead-in noise
    prof = constancy_profile(rows)
    const = prof >= const_thr
    # longest circular run of constant columns, bridging isolated bit errors
    doubled = np.concatenate([const, const])
    best_len, best_start, i = 0, 0, 0
    N = len(doubled)
    while i < N:
        if not doubled[i]:
            i += 1
            continue
        j, gap, last = i, 0, i
        while j < N and j - i < period:
            if doubled[j]:
                gap, last = 0, j
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        L = last - i + 1
        if L > best_len:
            best_len, best_start = L, i
        i = last + 1
    if best_len < 16:
        return None
    off = best_start % period
    # The constant run may contain more than the preamble (a sync word sent in the
    # clear, constant pad/flush bits), so try the usual preamble lengths at every
    # offset inside the run, most common lengths first.
    preferred = [48, 32, 64, 16, 24, 40, 56, 72, 80, 96, 128]
    lengths = [L for L in preferred if L <= best_len + 4]
    lengths += [L for L in range(8, best_len + 5, 8) if L not in lengths and L >= 16]
    cands = []
    # first guess: the run with repeated-symbol flush/pad regions trimmed off
    maj = (rows.mean(axis=0) > 0.5).astype(np.uint8)
    run_bits = maj[(off + np.arange(best_len)) % period]
    a, b = _trim_repeats(run_bits, bps)
    core = best_len - a - b
    if core >= 16:
        for Lp in sorted({core, int(round(core / 8)) * 8} - {0}):
            cands.append(((off + a) % period, Lp))
    for Lp in lengths:
        span = max(best_len - Lp, 0)
        for d in [0, span] + list(range(1, span)):
            c = ((off + d) % period, Lp)
            if c not in cands:
                cands.append(c)
    return BlindGeometry(period, off, best_len, np.roll(prof, -off), cands[:48])


def blind_framing(hyp, geom: BlindGeometry, cand) -> FramingResult:
    name, llr = hyp
    off, Lp = cand
    P = geom.period
    starts = np.arange(off, len(llr) - P + 1, P)
    blocks = [llr[s + Lp: s + P] for s in starts]
    pre = hard(llr[starts[0]: starts[0] + Lp]) if len(starts) else np.zeros(0, dtype=np.uint8)
    return FramingResult(name, pre, starts, np.ones(len(starts)), P, P - Lp, blocks, None,
                         "blind (autocorrelation + bit constancy)", {}, geom.constancy)
