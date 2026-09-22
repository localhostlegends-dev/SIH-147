"""End-to-end analysis pipeline: raw recording -> decoded, parsed frames.

    load -> spectral analysis -> symbol rate -> modulation ID -> demodulation
         -> frame sync (preamble) -> interleaver + FEC identification
         -> de-interleave + decode -> bitstream correlation (header / payload)

Every stage can be left on "auto" or forced by the analyst (GUI overrides).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import demod as dm
from .correlate import CorrelationResult, analyse
from .fec import FEC_CODES
from .framing import FramingResult, blind_framing, blind_geometry, correlate_pattern, frame_with_preamble
from .identify import IdentifyResult, decode_frames, identify, inner_decode, refine_concat
from .interleave import Interleaver, conv_delay
from .io import SignalData
from .modem import BITS_PER_SYMBOL, FSK_MODS
from .params import (SpectralInfo, SymbolRateInfo, frequency_shift, spectral_analysis,
                     symbol_rate_fsk, symbol_rate_linear)
from .utils import FRAME_SYNC_HEX, PREAMBLE_HEX, bits_to_hex, hard, hex_to_bits


@dataclass
class AnalysisOptions:
    modulation: str = "auto"            # auto | BPSK | QPSK | 8PSK | 16QAM | 2FSK | 4FSK | 8FSK
    symbol_rate: float | None = None    # None = estimate
    rolloff: float = 0.35
    center_offset_hz: float | None = None
    preamble_hex: str = PREAMBLE_HEX    # "" = blind framing
    frame_sync_hex: str = FRAME_SYNC_HEX
    interleaver: str = "auto"           # auto | none | block | diagonal | conv | random
    il_params: dict = field(default_factory=dict)
    fec: str = "auto"                   # auto | none | conv_k3 | conv_k5 | conv_k7 | rs16 | rs32 | concat | ldpc
    seed_max: int = 256
    max_samples: int = 3_000_000


@dataclass
class AnalysisResult:
    signal: SignalData
    options: AnalysisOptions
    spectral: SpectralInfo | None = None
    rs_linear: SymbolRateInfo | None = None
    rs_fsk: SymbolRateInfo | None = None
    symbol_rate: float = 0.0
    family_note: str = ""
    demod: dm.DemodResult | None = None
    framing: FramingResult | None = None
    ident: IdentifyResult | None = None
    fec_name: str | None = None
    info_frames: list = field(default_factory=list)
    frame_stats: list = field(default_factory=list)
    corr: CorrelationResult | None = None
    log: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> dict:
        d = {
            "file": self.signal.name,
            "sampling rate (Hz)": self.signal.fs,
            "sampling rate source": self.signal.fs_source,
            "duration (s)": round(self.signal.duration, 3),
        }
        if self.spectral:
            d.update({"carrier offset (Hz)": round(self.spectral.center_offset_hz, 1),
                      "occupied bandwidth (Hz)": round(self.spectral.bandwidth_hz, 1),
                      "spectral SNR (dB)": round(self.spectral.snr_db, 1)})
        if self.demod:
            d.update({"symbol rate (Bd)": round(self.symbol_rate, 2),
                      "samples per symbol": round(self.signal.fs / self.symbol_rate, 2),
                      "modulation": self.demod.modulation,
                      "classifier confidence": round(self.demod.extra.get("confidence", 0), 2),
                      "decision SNR (dB)": round(self.demod.snr_db, 1)})
        if self.framing:
            d.update({"frames found": len(self.framing.starts), "frame length (bits)": self.framing.period,
                      "phase/ambiguity": self.framing.hypothesis})
        if self.ident and self.ident.best:
            d.update({"interleaver": self.ident.best.interleaver.describe(),
                      "FEC": FEC_CODES[self.fec_name].label if self.fec_name else "-",
                      "identification score": round(self.ident.best.score, 3)})
        if self.corr:
            ok = sum(1 for f in self.corr.frames if "payload_text" in f)
            d.update({"frames with valid header": f"{ok}/{len(self.corr.frames)}"})
        return d


def _log(res: AnalysisResult, stage: str, msg: str):
    res.log.append((stage, msg))


def run(sig: SignalData, opts: AnalysisOptions | None = None) -> AnalysisResult:
    opts = opts or AnalysisOptions()
    res = AnalysisResult(sig, opts)
    try:
        _run(sig, opts, res)
    except Exception as e:  # keep partial results for the GUI
        import traceback
        res.error = f"{type(e).__name__}: {e}"
        _log(res, "error", traceback.format_exc())
    return res


def _tick(res, name, t0):
    res.timings[name] = round(time.time() - t0, 3)
    return time.time()


def _run(sig: SignalData, opts: AnalysisOptions, res: AnalysisResult):
    t = time.time()
    x = sig.samples[: opts.max_samples]
    fs = sig.fs
    if len(sig.samples) > opts.max_samples:
        _log(res, "input", f"analysing the first {opts.max_samples} of {len(sig.samples)} samples")

    # ---------------- 1. spectral parameters ----------------
    sp = spectral_analysis(x, fs)
    res.spectral = sp
    center = opts.center_offset_hz if opts.center_offset_hz is not None else sp.center_offset_hz
    xc = frequency_shift(x, fs, center)
    _log(res, "spectrum", f"occupied bandwidth {sp.bandwidth_hz:.1f} Hz centred at {center:+.1f} Hz, "
                          f"spectral SNR {sp.snr_db:.1f} dB")
    t = _tick(res, "spectral analysis", t)

    # ---------------- 2. symbol rate + family ----------------
    res.rs_linear = symbol_rate_linear(xc, fs, sp.bandwidth_hz)
    res.rs_fsk = symbol_rate_fsk(xc, fs, sp.bandwidth_hz)
    forced = opts.modulation if opts.modulation != "auto" else None
    if forced:
        family = "fsk" if forced in FSK_MODS else "linear"
        res.family_note = f"modulation forced to {forced}"
    else:
        family = None
    t = _tick(res, "symbol rate", t)

    def pick_linear_rate():
        if opts.symbol_rate:
            return float(opts.symbol_rate)
        a, b = res.rs_linear, res.rs_fsk
        return b.symbol_rate if b.strength >= 2 * a.strength else a.symbol_rate

    fsk_try = None
    if family in (None, "fsk"):
        rs_f = float(opts.symbol_rate) if opts.symbol_rate else res.rs_fsk.symbol_rate
        fsk_try = dm.demod_fsk(xc, fs, rs_f, sp.bandwidth_hz,
                               force_M=int(forced[0]) if forced else None)
        if family is None:
            conf = fsk_try.extra["confidence"]
            if conf >= 0.5:
                family = "fsk"
                res.family_note = (f"FSK: instantaneous frequency forms {fsk_try.modulation[0]} clean clusters "
                                   f"(separation {fsk_try.extra['separation']:.1f}, confidence {conf:.2f})")
            else:
                family = "linear"
                res.family_note = (f"linear (PSK/QAM): no clean FSK tone clusters (FSK confidence {conf:.2f})")
    if family == "fsk":
        d = fsk_try
    else:
        d = dm.demod_linear(xc, fs, pick_linear_rate(), opts.rolloff, force_mod=forced, snr_db=sp.snr_db)
    res.demod = d
    res.symbol_rate = d.symbol_rate
    _log(res, "modulation", f"{res.family_note}; identified {d.modulation} at {d.symbol_rate:.2f} Bd, "
                            f"decision SNR {d.snr_db:.1f} dB")
    t = _tick(res, "demodulation", t)

    # ---------------- 3. framing ----------------
    bps = BITS_PER_SYMBOL[d.modulation]
    sync_bits = hex_to_bits(opts.frame_sync_hex) if opts.frame_sync_hex else None
    blind = not opts.preamble_hex
    fr = None
    if not blind:
        fr = frame_with_preamble(d.hypotheses, hex_to_bits(opts.preamble_hex))
        if fr is None:
            _log(res, "framing", "known preamble not found - falling back to blind framing")
            blind = True
    t = _tick(res, "framing", t)

    # ---------------- 4. interleaver + FEC identification ----------------
    kinds = ("none", "block", "diagonal", "conv", "random") if opts.interleaver == "auto" else (opts.interleaver,)
    forced_il = None
    if opts.interleaver != "auto" and opts.il_params:
        forced_il = Interleaver(opts.interleaver, {k: int(v) for k, v in opts.il_params.items()})
    fec_names = None
    if opts.fec != "auto":
        fec_names = ["conv_k7"] if opts.fec == "concat" else [opts.fec]

    def ident_for(framing):
        return identify(framing.blocks, bps, sync_bits, kinds, fec_names, range(0, opts.seed_max),
                        forced_interleaver=forced_il)

    if blind:
        geom = blind_geometry(hard(d.hypotheses[0][1]), bps=bps)
        if geom is None:
            _log(res, "framing", "blind framing failed: no repeating frame structure found")
            return
        _log(res, "framing", f"blind framing: frame period {geom.period} bits, constant run of "
                             f"{geom.run_len} bits (preamble candidate) at offset {geom.run_start}")
        best = None
        fast = tuple(k for k in kinds if k != "random") or kinds
        # quick search: 2 frames, no pseudo-random interleavers, stop at the first clear hit.
        # Coded hypotheses outrank "uncoded" (a sync word seen through a partly wrong
        # framing can fake an uncoded hit).
        def key(sc_idr):
            b = sc_idr.best
            return (b is not None and b.fec != "none", b.score if b else 0.0)

        def validate(f2, idr_):
            """With a known frame sync word, a correct framing must reveal it after decoding."""
            if sync_bits is None or idr_.best is None:
                return True
            c = idr_.best
            fec = c.fec
            if fec == "conv_k7" and opts.fec == "auto":
                fec = refine_concat(c, inner_decode(f2.blocks[:2], c))
            infos, _ = decode_frames(f2.blocks[:2], c, fec)
            for i in infos:
                if len(i) < len(sync_bits):
                    return False
                cp = correlate_pattern(i, sync_bits)
                # the sync word must open the decoded frame, not just appear somewhere
                if cp[: 9].max() < 0.9:
                    return False
            return True

        best_key, n_after_none, hit = None, 0, None
        for cand in geom.candidates:
            for hyp in d.hypotheses:
                f2 = blind_framing(hyp, geom, cand)
                idr = identify(f2.blocks, bps, sync_bits, fast, fec_names, range(0),
                               max_frames=2, forced_interleaver=forced_il)
                k = key(idr)
                if best_key is None or k > best_key:
                    best_key, best = k, (k[1], f2, idr)
                if k[0] and k[1] >= 0.75:
                    hit = (hyp, idr)
                    break
            if hit or best_key[1] >= 0.9 and (n_after_none := n_after_none + 1) > 6:
                break
        if hit and not validate(best[1], best[2]):
            # Structure found but framing is off (e.g. a stream interleaver hides the
            # exact block start): keep the code/interleaver, slide the framing.
            hyp0, idr0 = hit
            il0, fec0 = idr0.best.interleaver, [idr0.best.fec]
            # the identified code fixes the block length, hence the preamble length
            delay = conv_delay(il0.params["B"], il0.params["J"]) if il0.kind == "conv" else 0
            n_il = idr0.best.coded_len + delay
            Lp_implied = geom.period - n_il - idr0.best.pad
            # (a convolutional code fits any even length, so also try the usual sizes)
            lengths = [Lp_implied] + [L for L in (48, 32, 64, 16, 24, 40, 56, 72, 80, 96, 128)
                                      if L != Lp_implied]
            slide = [((geom.run_start + dd) % geom.period, L) for L in lengths if 0 < L < geom.period
                     for dd in range(-8, geom.run_len + 1)][:5000]
            for cand in slide:
                for sign in (1, -1):
                    f2 = blind_framing((hyp0[0], sign * hyp0[1]), geom, cand)
                    idr = identify(f2.blocks, bps, sync_bits, (il0.kind,), fec0, range(0),
                                   max_frames=2, forced_interleaver=il0)
                    if idr.best and idr.best.score >= 0.75 and validate(f2, idr):
                        best = (idr.best.score, f2, idr)
                        break
                else:
                    continue
                break
        if best[0] < 0.75 and "random" in kinds:
            # slower pass including pseudo-random interleavers, on the likeliest framings
            for cand in geom.candidates[:3]:
                for hyp in d.hypotheses:
                    f2 = blind_framing(hyp, geom, cand)
                    idr = identify(f2.blocks, bps, sync_bits, ("random",), fec_names,
                                   range(0, opts.seed_max), max_frames=2, forced_interleaver=forced_il)
                    sc = idr.best.score if idr.best else 0.0
                    if sc > best[0]:
                        best = (sc, f2, idr)
                    if sc >= 0.9:
                        break
                if best[0] >= 0.9:
                    break
        if best[1] is not None and best[2].best is not None:
            # final, full-strength identification on the chosen framing
            f2 = best[1]
            il = best[2].best.interleaver
            idr = identify(f2.blocks, bps, sync_bits, (il.kind,), fec_names, range(0, opts.seed_max),
                           forced_interleaver=forced_il)
            best = (best[0], f2, idr)
        _, fr, idr = best
        _log(res, "framing", f"blind framing chose preamble length {len(fr.preamble_bits)} bits "
                             f"(discovered preamble 0x{bits_to_hex(fr.preamble_bits)})")
    else:
        idr = ident_for(fr)
    res.framing = fr
    _log(res, "framing", f"{len(fr.starts)} frames of {fr.period} bits ({fr.mode}); "
                         f"ambiguity resolved as '{fr.hypothesis}'")
    res.ident = idr
    t = _tick(res, "coding identification", t)
    if not idr.best:
        _log(res, "coding", idr.note)
        return
    fec_name = idr.best.fec
    if opts.fec == "concat":
        fec_name = "concat"
    elif opts.fec == "auto" and fec_name == "conv_k7":
        fec_name = refine_concat(idr.best, inner_decode(fr.blocks, idr.best))
    res.fec_name = fec_name
    _log(res, "coding", f"interleaver: {idr.best.interleaver.describe()}; FEC: {FEC_CODES[fec_name].label} "
                        f"(score {idr.best.score:.3f}, {idr.n_tested} hypotheses tested)")

    # ---------------- 5. decode ----------------
    res.info_frames, res.frame_stats = decode_frames(fr.blocks, idr.best, fec_name)
    ber = [s.get("channel_ber") for s in res.frame_stats if s.get("channel_ber") is not None]
    if ber:
        _log(res, "decode", f"estimated channel BER before FEC: {np.mean(ber):.2e}")
    t = _tick(res, "decoding", t)

    # ---------------- 6. bitstream correlation ----------------
    res.corr = analyse(res.info_frames, sync_bits)
    ok = sum(1 for f in res.corr.frames if "payload_text" in f)
    _log(res, "correlation", f"sync word found in {ok}/{len(res.info_frames)} frames; "
                             f"estimated header length {res.corr.header_len_est} bits")
    _tick(res, "correlation", t)
