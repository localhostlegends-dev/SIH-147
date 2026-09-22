"""Stages 2-3 - modulation classification and demodulation.

Linear modulations (BPSK / QPSK / 8PSK / 16QAM)
    1. Root-raised-cosine matched filter.
    2. Symbol timing: Oerder & Meyr feed-forward estimator (the phase of the
       symbol-rate line in |y|^2), computed block-by-block so clock drift is tracked.
    3. Classification from higher-order statistics of the symbols:
       s^2 has a spectral line only for BPSK, s^4 for QPSK/16QAM, s^8 for 8PSK;
       QPSK vs 16QAM is decided by amplitude spread and the C42 cumulant.
    4. Carrier recovery: the frequency of the s^M line gives the carrier offset;
       a sliding Viterbi&Viterbi estimator removes the remaining phase.
    5. Soft demapping to LLRs (max-log).

FSK (2/4/8-FSK)
    1. Instantaneous frequency = d(phase)/dt.
    2. Symbol timing from the frequency-transition cyclic line.
    3. Integrate-and-dump of the frequency over each symbol.
    4. 1-D k-means finds the number of tones (M) and their frequencies.

Both paths return several *ambiguity hypotheses* (carrier-phase rotations,
spectral inversion) - resolved later using the preamble or FEC structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from .modem import (BITS_PER_SYMBOL, CARRIER_POWER, PHASE_SYMMETRY, constellation, rrc_taps,
                    soft_demap)
from .params import instantaneous_frequency, lowpass
from .utils import gray


@dataclass
class DemodResult:
    modulation: str
    family: str                                  # 'linear' | 'fsk'
    symbol_rate: float
    symbols: np.ndarray                          # complex (linear) or Hz (fsk), for plotting
    hypotheses: list                             # list of (name, llr array)
    noise_var: float = 0.0
    snr_db: float = 0.0
    freq_offset_hz: float = 0.0
    extra: dict = field(default_factory=dict)    # eye diagram, tone freqs, features...


# ======================================================================
# Linear front end
# ======================================================================
def _timing_phase(signal: np.ndarray, sps: float, block: int, n0: int = 0):
    """Block-wise cyclic-line phase -> smooth symbol phase function phi(n)."""
    N = len(signal)
    Xs, centres = [], []
    blk = max(int(block * sps), int(8 * sps))
    for start in range(0, N, blk):
        seg = signal[start:start + blk]
        if len(seg) < blk // 2 and Xs:
            break
        nn = np.arange(start, start + len(seg))
        Xs.append(np.sum(seg * np.exp(-2j * np.pi * nn / sps)))
        centres.append(start + len(seg) / 2)
    Xs = np.array(Xs)
    # Track the phase with a smoothing loop (forward + backward) instead of a raw
    # unwrap: a single noisy block would otherwise cause a symbol slip.
    w = np.clip(np.abs(Xs) / (np.mean(np.abs(Xs)) + 1e-12), 0.05, 1.0)
    raw = -np.angle(Xs)

    def track(order):
        est = np.zeros(len(raw))
        cur = np.angle(np.sum(Xs[order[: min(4, len(order))]].conj()))  # robust start
        for i in order:
            e = np.angle(np.exp(1j * (raw[i] - cur)))
            cur = cur + 0.3 * w[i] * e
            est[i] = cur
        return est

    fwd = track(np.arange(len(raw)))
    bwd = track(np.arange(len(raw))[::-1])
    # combine the two passes (align bwd to fwd modulo 2*pi first)
    bwd = bwd + 2 * np.pi * np.round((fwd - bwd) / (2 * np.pi))
    taus = 0.5 * (fwd + bwd) / (2 * np.pi) * sps
    n = np.arange(N)
    tau_n = np.interp(n, centres, taus)
    return (n - tau_n) / sps        # symbol index as a function of sample index


def linear_front_end(x: np.ndarray, fs: float, Rs: float, rolloff: float = 0.35, block_syms: int = 128):
    sps = fs / Rs
    h = rrc_taps(sps, rolloff, span=10)
    y = fftconvolve(x, h, mode="same")
    phi = _timing_phase(np.abs(y) ** 2, sps, block_syms)
    k0, k1 = int(np.ceil(phi[0])) + 1, int(np.floor(phi[-1])) - 1
    ks = np.arange(k0, k1)
    n = np.arange(len(y))
    t_k = np.interp(ks, phi, n)
    csr, csi = CubicSpline(n, y.real), CubicSpline(n, y.imag)
    sym = csr(t_k) + 1j * csi(t_k)
    # eye diagram: 2-symbol windows at 16 points/symbol
    n_eye = min(200, len(ks) - 4)
    eye_k = np.linspace(k0 + 2, k1 - 3, n_eye).astype(int)
    offs = np.linspace(-1, 1, 33)
    eye_t = np.interp(eye_k[:, None] + offs[None, :], phi, n)
    eye = csr(eye_t) + 1j * csi(eye_t)
    scale = np.sqrt(np.mean(np.abs(sym) ** 2)) or 1.0
    return sym / scale, {"eye": eye / scale, "eye_offsets": offs, "eye_idx": eye_k - k0}


def _power_line(s: np.ndarray, M: int):
    z = s ** M
    norm = np.mean(np.abs(s) ** M) + 1e-12
    nfft = int(2 ** np.ceil(np.log2(len(z)))) * 4
    Z = np.fft.fft(z, nfft) / len(z)
    k = int(np.argmax(np.abs(Z)))
    f = np.fft.fftfreq(nfft)[k]                    # cycles per symbol
    return float(np.abs(Z[k]) / norm), float(f)


def _cumulants(s: np.ndarray):
    s = s - np.mean(s)
    p = np.mean(np.abs(s) ** 2)
    s = s / np.sqrt(p)
    c20 = np.mean(s ** 2)
    c40 = np.mean(s ** 4) - 3 * c20 ** 2
    c42 = np.mean(np.abs(s) ** 4) - np.abs(c20) ** 2 - 2
    return complex(c40), float(np.real(c42))


def classify_linear(sym: np.ndarray, snr_db: float | None = None):
    m2, f2 = _power_line(sym, 2)
    m4, f4 = _power_line(sym, 4)
    m8, f8 = _power_line(sym, 8)
    feats = {"s^2 line": m2, "s^4 line": m4, "s^8 line": m8}
    if m2 > 0.4:
        mod, conf = "BPSK", m2
    elif m4 > 0.2:
        s1 = sym * np.exp(-2j * np.pi * f4 / 4 * np.arange(len(sym)))
        c40, c42 = _cumulants(s1)
        amp_cv = float(np.std(np.abs(s1)) / np.mean(np.abs(s1)))
        # additive noise pulls cumulants toward 0 and inflates amplitude spread:
        # correct both with the (spectral) SNR estimate
        snr_lin = 10 ** ((snr_db if snr_db is not None else 30.0) / 10)
        c42c = c42 * (1 + 1 / snr_lin) ** 2
        cvc = float(np.sqrt(max(amp_cv ** 2 - 1 / (2 * snr_lin), 0.0)))
        feats.update({"C42": c42, "C42 (noise-corrected)": c42c, "|C40|": abs(c40),
                      "amplitude CV": amp_cv, "amplitude CV (noise-corrected)": cvc})
        # QPSK: C42=-1, one amplitude ring. 16QAM: C42=-0.68, three rings (CV 0.33)
        is_qam = (c42c > -0.84) and (cvc > 0.12)
        mod = "16QAM" if is_qam else "QPSK"
        conf = min(1.0, m4 / 0.5 if is_qam else m4)
    elif m8 > 0.15:
        mod, conf = "8PSK", m8
    else:
        best = max((m2, "BPSK"), (m4 * 1.3, "QPSK"), (m8 * 2, "8PSK"))
        mod, conf = best[1], best[0] * 0.5
    return mod, float(min(conf, 1.0)), feats


def carrier_recover(sym: np.ndarray, mod: str, Rs: float, window: int = 64):
    pts = constellation(mod)
    M = CARRIER_POWER[mod]
    ref_angle = np.angle(np.mean(pts ** M))
    _, fM = _power_line(sym, M)
    k = np.arange(len(sym))
    s1 = sym * np.exp(-2j * np.pi * fM / M * k)
    z = s1 ** M
    zs = np.convolve(z, np.ones(window), mode="same")
    theta = (np.unwrap(np.angle(zs)) - ref_angle) / M
    s2 = s1 * np.exp(-1j * theta)
    # gain: least-squares fit to the nearest constellation points
    g = 1.0
    for _ in range(4):
        y = s2 * g
        c = pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]
        g = g * np.real(np.sum(c * np.conj(y))) / np.sum(np.abs(y) ** 2)
    y = s2 * g
    c = pts[np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)]
    nv = float(np.mean(np.abs(y - c) ** 2))
    return y, fM / M * Rs, nv


def linear_hypotheses(y: np.ndarray, mod: str, noise_var: float):
    pts = constellation(mod)
    k = BITS_PER_SYMBOL[mod]
    sym = PHASE_SYMMETRY[mod]
    hyps = []
    for conj in (False, True):
        base = np.conj(y) if conj else y
        for r in range(sym):
            rot = np.exp(-2j * np.pi * r / sym)
            name = f"rot {360 * r // sym} deg" + (" + conj" if conj else "")
            hyps.append((name, soft_demap(base * rot, pts, k, noise_var)))
    return hyps


def demod_linear(x, fs, Rs, rolloff=0.35, force_mod: str | None = None,
                 snr_db: float | None = None) -> DemodResult:
    sym, extra = linear_front_end(x, fs, Rs, rolloff)
    mod, conf, feats = classify_linear(sym, snr_db)
    if force_mod:
        mod = force_mod
    y, foff, nv = carrier_recover(sym, mod, Rs)
    hyps = linear_hypotheses(y, mod, nv)
    # de-rotate the eye-diagram traces with the recovered carrier phase
    rot = y[extra["eye_idx"]] / np.where(np.abs(sym[extra["eye_idx"]]) > 0, sym[extra["eye_idx"]], 1)
    extra["eye"] = extra["eye"] * (rot / np.maximum(np.abs(rot), 1e-12))[:, None] * np.mean(np.abs(rot))
    extra.update({"features": feats, "confidence": conf, "raw_symbols": sym})
    return DemodResult(mod, "linear", Rs, y, hyps, nv, float(10 * np.log10(1 / max(nv, 1e-9))), foff, extra)


# ======================================================================
# FSK
# ======================================================================
def _kmeans1d(v: np.ndarray, k: int, iters: int = 30):
    lo, hi = np.percentile(v, [1, 99])
    inits = [np.linspace(lo, hi, k), np.quantile(v, (np.arange(k) + 0.5) / k)]
    best = None
    for c in inits:
        c = c.astype(float).copy()
        for _ in range(iters):
            lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
            for j in range(k):
                if np.any(lab == j):
                    c[j] = v[lab == j].mean()
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        w = np.mean((v - c[lab]) ** 2)
        if best is None or w < best[0]:
            best = (w, c)
    c = np.sort(best[1])
    lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
    within = np.mean((v - c[lab]) ** 2)
    resid = within / (np.var(v) + 1e-12)
    # separation: smallest gap between tones relative to the spread of each tone
    sep = float(np.min(np.diff(c)) / (2 * np.sqrt(within) + 1e-12)) if k > 1 else 0.0
    return c, float(resid), sep


def fsk_front_end(x, fs, Rs, bw, block_syms: int = 64):
    sps = fs / Rs
    # keep the tones plus their keying sidebands (~Rs each side)
    y = lowpass(x, fs, max(bw * 0.6, bw / 2 + Rs))
    fi = instantaneous_frequency(y, fs)
    ks = max(int(sps / 8), 1)
    fs_s = np.convolve(fi, np.ones(ks) / ks, mode="same") if ks > 1 else fi
    d = np.abs(np.diff(fs_s))
    phi = _timing_phase(d, sps, block_syms)            # transitions at integer phi
    n = np.arange(len(d)) + 1.0
    k0, k1 = int(np.ceil(phi[0])) + 1, int(np.floor(phi[-1])) - 1
    kk = np.arange(k0, k1 + 1)
    t_edges = np.interp(kk, phi, n)
    C = np.concatenate([[0.0], np.cumsum(fi)])
    grid = np.arange(len(C))
    a = t_edges[:-1] + 0.2 * np.diff(t_edges)
    b = t_edges[1:] - 0.2 * np.diff(t_edges)
    vals = (np.interp(b, grid, C) - np.interp(a, grid, C)) / np.maximum(b - a, 1e-9)
    return vals, {"inst_freq": fi, "_y": y, "_edges": t_edges}


def _tone_amplitudes(y, fs, tones, t_edges):
    """Non-coherent filter bank: |correlation with each tone| over every symbol."""
    n = np.arange(len(y))
    dt = np.diff(t_edges)
    a = t_edges[:-1] + 0.05 * dt
    b = t_edges[1:] - 0.05 * dt
    grid = np.arange(len(y) + 1)
    out = []
    for f in tones:
        z = y * np.exp(-2j * np.pi * f * n / fs)
        C = np.concatenate([[0], np.cumsum(z)])
        s = (np.interp(b, grid, C.real) - np.interp(a, grid, C.real)) + 1j * (
            np.interp(b, grid, C.imag) - np.interp(a, grid, C.imag))
        out.append(np.abs(s))
    return np.stack(out, axis=1)


def _fsk_llr(A: np.ndarray, tone_labels: np.ndarray, k: int) -> np.ndarray:
    top = A.max(axis=1)
    S = np.median(top)
    rest = (np.sum(A ** 2, axis=1) - top ** 2) / max(A.shape[1] - 1, 1)
    noise = max(np.mean(rest), 1e-12)
    llr = np.zeros((len(A), k))
    for j in range(k):
        bit = (tone_labels >> (k - 1 - j)) & 1
        llr[:, j] = 2 * S * (A[:, bit == 0].max(axis=1) - A[:, bit == 1].max(axis=1)) / noise
    return np.clip(llr, -50, 50).reshape(-1)


def classify_fsk(vals: np.ndarray):
    """Number of tones = the largest k whose clusters are still well separated."""
    results = {k: _kmeans1d(vals, k) for k in (2, 4, 8)}
    M = 2
    for k in (4, 8):
        if results[k][2] >= 2.5 and results[k][2] > 0.6 * results[M][2]:
            M = k
    centers, resid, sep = results[M]
    feats = {f"tone separation k={k}": r[2] for k, r in results.items()}
    return M, centers, sep, feats


def demod_fsk(x, fs, Rs, bw, force_M: int | None = None) -> DemodResult:
    vals, extra = fsk_front_end(x, fs, Rs, bw)
    M, centers, sep, feats = classify_fsk(vals)
    if force_M:
        M = force_M
        centers, _, sep = _kmeans1d(vals, M)
    k = int(np.log2(M))
    spacing = float(np.mean(np.diff(centers))) if M > 1 else 0.0
    lab = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
    nv = float(np.mean((vals - centers[lab]) ** 2))
    scale = spacing / 2 if spacing else 1.0
    # symbol decisions from a non-coherent tone filter bank (much better than the
    # frequency discriminator at low SNR); tones come from the k-means centres
    A = _tone_amplitudes(extra.pop("_y"), fs, centers, extra.pop("_edges"))
    n_use = min(len(A), len(vals))
    A = A[:n_use]
    hyps = []
    for rev in (False, True):
        levels = np.arange(M)[::-1] if rev else np.arange(M)
        hyps.append(("normal" if not rev else "spectrum inverted", _fsk_llr(A, gray(levels), k)))
    snr = 10 * np.log10(max(scale ** 2 / max(nv, 1e-12), 1e-9))
    extra.update({"features": feats, "tones_hz": centers.tolist(), "tone_spacing_hz": spacing,
                  "mod_index_h": spacing / Rs if Rs else 0.0, "confidence": float(np.clip((sep - 1.5) / 2.0, 0.0, 1.0)), "separation": sep})
    return DemodResult(f"{M}FSK", "fsk", Rs, vals, hyps, nv, float(snr), float(np.mean(centers)), extra)


# ======================================================================
def envelope_cv(x: np.ndarray, fs: float, bw: float) -> float:
    y = lowpass(x, fs, bw * 0.55)
    a = np.abs(y[len(y) // 20: -len(y) // 20 or None])
    return float(np.std(a) / (np.mean(a) + 1e-12))
