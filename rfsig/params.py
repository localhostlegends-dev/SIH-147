"""Stage 1 - signal parameter extraction.

* Power spectral density (Welch) -> noise floor, occupied (99 %) bandwidth,
  centre-frequency offset and a spectral SNR estimate.
* Waterfall (spectrogram) for the GUI.
* Symbol-rate estimation using *cyclostationarity*: a digitally modulated
  signal repeats its statistics every symbol period T. Squaring the envelope
  of a pulse-shaped PSK/QAM signal (|x|^2), or looking at how often the
  instantaneous frequency of an FSK signal jumps, produces a spectral line
  exactly at the symbol rate Rs = 1/T.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, fftconvolve, spectrogram, welch


@dataclass
class SpectralInfo:
    fs: float
    n_samples: int
    duration_s: float
    psd_freqs: np.ndarray
    psd_db: np.ndarray
    noise_floor_db: float
    center_offset_hz: float
    bandwidth_hz: float
    band_lo_hz: float
    band_hi_hz: float
    snr_db: float


@dataclass
class SymbolRateInfo:
    symbol_rate: float
    method: str
    strength: float                   # peak / median of the cyclic spectrum
    spec_freqs: np.ndarray = field(repr=False, default=None)
    spec_mag: np.ndarray = field(repr=False, default=None)


def spectral_analysis(x: np.ndarray, fs: float) -> SpectralInfo:
    nper = int(min(4096, 2 ** np.floor(np.log2(max(len(x) // 8, 64)))))
    f, P = welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False, scaling="density")
    f, P = np.fft.fftshift(f), np.fft.fftshift(P)
    P = np.maximum(P, 1e-30)
    Ps = np.convolve(P, np.ones(9) / 9, mode="same")
    # an analytic signal (from a real recording) has an empty negative half:
    # estimate the noise floor only where there is any energy at all
    live = Ps > Ps.max() * 1e-7
    floor = np.percentile(Ps[live], 20)
    peak = Ps.max()
    # occupied band = bins clearly above the noise floor (threshold halfway in dB,
    # capped at 6 dB), then the 99 % power interval inside that region
    thr_db = min(6.0, 0.5 * 10 * np.log10(peak / floor))
    mask = Ps > floor * 10 ** (thr_db / 10)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmax(Ps))])
    # group bins into regions (gaps up to 3 % of the span are bridged) and keep
    # the strongest region plus any region carrying >= 10 % of its power
    gap = max(3, int(0.03 * len(f)))
    splits = np.nonzero(np.diff(idx) > gap)[0]
    regions = np.split(idx, splits + 1)
    powers = [np.sum(Ps[r] - floor) for r in regions]
    pmax = max(powers)
    keep = [r for r, p in zip(regions, powers) if p >= 0.1 * pmax]
    lo_i, hi_i = min(r[0] for r in keep), max(r[-1] for r in keep)
    sig = np.zeros_like(P)
    sig[lo_i:hi_i + 1] = np.maximum(P[lo_i:hi_i + 1] - floor, 0)
    if sig.sum() <= 0:
        sig[lo_i:hi_i + 1] = P[lo_i:hi_i + 1]
    c = np.cumsum(sig) / sig.sum()
    lo = f[np.searchsorted(c, 0.005)]
    hi = f[min(np.searchsorted(c, 0.995), len(f) - 1)]
    bw = max(hi - lo, 4 * fs / nper)
    center = (hi + lo) / 2
    band = (f >= lo) & (f <= hi)
    df = f[1] - f[0]
    p_sig = np.sum(np.maximum(P[band] - floor, 0)) * df
    p_noise = floor * bw
    snr = 10 * np.log10(max(p_sig, 1e-30) / max(p_noise, 1e-30))
    return SpectralInfo(fs, len(x), len(x) / fs, f, 10 * np.log10(P), 10 * np.log10(floor),
                        float(center), float(bw), float(lo), float(hi), float(snr))


def waterfall(x: np.ndarray, fs: float, max_time_bins: int = 300, nfft: int = 512):
    nfft = int(min(nfft, 2 ** np.floor(np.log2(max(len(x) // 4, 32)))))
    hop = max(nfft // 2, len(x) // max_time_bins)
    f, t, S = spectrogram(x, fs=fs, nperseg=nfft, noverlap=max(nfft - hop, 0), return_onesided=False,
                          detrend=False, mode="psd")
    S = np.fft.fftshift(S, axes=0)
    f = np.fft.fftshift(f)
    return f, t, 10 * np.log10(np.maximum(S, 1e-30))


def frequency_shift(x: np.ndarray, fs: float, f_hz: float) -> np.ndarray:
    n = np.arange(len(x))
    return x * np.exp(-2j * np.pi * f_hz * n / fs)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, ntaps: int = 257) -> np.ndarray:
    cutoff = min(max(cutoff_hz, fs * 0.002), fs * 0.49)
    h = firwin(ntaps, cutoff, fs=fs)
    return fftconvolve(x, h, mode="same")


def _line_search(sig: np.ndarray, fs: float, fmin: float, fmax: float):
    s = sig - np.mean(sig)
    nfft = int(2 ** np.ceil(np.log2(len(s)))) * 2
    nfft = min(nfft, 1 << 23)
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return None
    idx = np.nonzero(band)[0]
    k = idx[np.argmax(S[idx])]
    # parabolic interpolation for sub-bin accuracy
    if 0 < k < len(S) - 1:
        a, b, c = np.log(S[k - 1] + 1e-12), np.log(S[k] + 1e-12), np.log(S[k + 1] + 1e-12)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        d = 0.0
    f0 = (k + d) * fs / nfft
    med = np.median(S[idx]) + 1e-12
    return f0, float(S[k] / med), freqs, S, idx


def symbol_rate_linear(x: np.ndarray, fs: float, bw: float) -> SymbolRateInfo:
    """Rs line in the spectrum of |x|^2 (x already centred at 0 Hz)."""
    y = lowpass(x, fs, bw * 0.6)
    # For RRC pulses the occupied bandwidth is about (1 + rolloff) * Rs, so the
    # symbol-rate line must lie in roughly [0.45, 1.1] * bandwidth. Restricting the
    # search avoids low-frequency lines caused by the repeating frame preamble.
    r = _line_search(np.abs(y) ** 2, fs, max(bw * 0.45, fs / len(x) * 50), min(bw * 1.1, fs / 2))
    if r is None:
        return SymbolRateInfo(bw / 1.35, "bandwidth fallback", 0.0)
    f0, strength, freqs, S, idx = r
    return SymbolRateInfo(f0, "|x|^2 cyclic line", strength, freqs[idx], S[idx])


def instantaneous_frequency(x: np.ndarray, fs: float) -> np.ndarray:
    return np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)


def symbol_rate_fsk(x: np.ndarray, fs: float, bw: float) -> SymbolRateInfo:
    """Rs line in the spectrum of |d/dt f_inst| (frequency transitions)."""
    y = lowpass(x, fs, bw * 0.6)
    fi0 = instantaneous_frequency(y, fs)
    fmin, fmax = max(bw * 0.04, fs / len(x) * 50), min(bw * 1.1, fs / 2)
    best = None
    # The instantaneous frequency is noisy (FM "clicks"); smoothing helps as long
    # as the window stays shorter than a symbol. Try several windows and keep the
    # one giving the most prominent line.
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k > len(fi0) // 200:
            break
        fi = np.convolve(fi0, np.ones(k) / k, mode="same") if k > 1 else fi0
        d = np.abs(np.diff(fi))
        r = _line_search(d, fs, fmin, min(fmax, fs / (2 * k)))
        if r is not None and (best is None or r[1] > best[1]):
            best = r
    if best is None:
        return SymbolRateInfo(bw / 3, "bandwidth fallback", 0.0)
    f0, strength, freqs, S, idx = best
    return SymbolRateInfo(float(f0), "FSK transition cyclic line", strength, freqs[idx], S[idx])
