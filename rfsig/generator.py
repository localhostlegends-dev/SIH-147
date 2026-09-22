"""Synthetic transmitter: builds realistic test recordings with KNOWN ground truth.

Transmit chain (one frame)::

    info bits (512)                      ASM sync word | counter | type | length | payload (56 B)
      -> FEC encode                      conv / RS / concatenated / LDPC / none
      -> interleave                      block / diagonal / convolutional / pseudo-random / none
      -> pad to whole symbols
      -> prepend 48-bit preamble         (uncoded, used for burst sync + phase ambiguity)
    frames are concatenated, mapped to symbols and modulated:
      -> PSK/QAM: RRC pulse shaping      -> FSK: continuous-phase frequency shifting
      -> channel: clock drift (ppm), fractional delay, carrier offset, phase, AWGN
      -> saved as complex baseband (.iq / stereo .wav) or real passband (mono .wav)

Because we know exactly what went in, every receiver stage can be tested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from .fec import get_fec
from .interleave import Interleaver, default_block_dims
from .modem import BITS_PER_SYMBOL, FSK_MODS, bits_to_labels, constellation, fsk_levels, rrc_taps
from .utils import FRAME_SYNC_HEX, PREAMBLE_HEX, hex_to_bits, int_to_bits

PAYLOAD_BYTES = 56
FRAME_TYPE = 0x5A
DEFAULT_MESSAGE = (
    "SIH 2026 // RF SIGNAL INTELLIGENCE TEST TRANSMISSION. "
    "This payload was modulated, interleaved and FEC-encoded by the synthetic generator. "
    "If you can read this sentence, every stage of the receiver chain worked: parameter "
    "estimation, modulation identification, demodulation, de-interleaving, error correction "
    "and bitstream correlation. -- END OF MESSAGE -- "
)


@dataclass
class TxConfig:
    modulation: str = "QPSK"          # BPSK QPSK 8PSK 16QAM 2FSK 4FSK 8FSK
    symbol_rate: float = 2400.0
    fs: float = 48000.0
    rolloff: float = 0.35
    fsk_h: float = 1.0                # FSK modulation index: tone spacing = h * Rs
    fec: str = "conv_k7"              # none conv_k3 conv_k5 conv_k7 rs16 rs32 concat ldpc
    interleaver: str = "block"        # none block diagonal conv random
    il_params: dict = field(default_factory=dict)
    n_frames: int = 16
    message: str = DEFAULT_MESSAGE
    snr_db: float = 15.0              # Es/N0 in dB
    freq_offset_hz: float = 120.0
    phase_offset_rad: float = 0.7
    clock_ppm: float = 20.0
    lead_symbols: int = 64            # random symbols before/after the burst
    output: str = "iq"                # 'iq' complex baseband | 'real' passband
    carrier_hz: float = 1800.0        # carrier for 'real' output
    preamble_hex: str = PREAMBLE_HEX
    frame_sync_hex: str = FRAME_SYNC_HEX
    seed: int = 0


def build_info_frame(counter: int, payload: bytes, frame_sync_hex: str = FRAME_SYNC_HEX) -> np.ndarray:
    chunk = payload[:PAYLOAD_BYTES]
    body = chunk + b" " * (PAYLOAD_BYTES - len(chunk))
    header = np.concatenate([
        hex_to_bits(frame_sync_hex),
        int_to_bits(counter & 0xFFFF, 16),
        int_to_bits(FRAME_TYPE, 8),
        int_to_bits(len(chunk), 8),
    ])
    return np.concatenate([header, np.unpackbits(np.frombuffer(body, dtype=np.uint8))]).astype(np.uint8)


def resolve_interleaver(cfg: TxConfig, n_code: int) -> Interleaver:
    p = dict(cfg.il_params)
    if cfg.interleaver == "none":
        return Interleaver("none")
    if cfg.interleaver in ("block", "diagonal"):
        if "rows" in p and n_code % int(p["rows"]) == 0:
            r = int(p["rows"])
            return Interleaver(cfg.interleaver, {"rows": r, "cols": n_code // r})
        r, c = default_block_dims(n_code)
        return Interleaver(cfg.interleaver, {"rows": r, "cols": c})
    if cfg.interleaver == "conv":
        return Interleaver("conv", {"B": int(p.get("B", 8)), "J": int(p.get("J", 4))})
    if cfg.interleaver == "random":
        return Interleaver("random", {"seed": int(p.get("seed", 42))})
    raise ValueError(cfg.interleaver)


def build_bitstream(cfg: TxConfig):
    """Return (channel bits, ground-truth dict)."""
    fec = get_fec(cfg.fec)
    bps = BITS_PER_SYMBOL[cfg.modulation]
    pre = hex_to_bits(cfg.preamble_hex)
    msg = cfg.message.encode("ascii", "replace")
    chunks = [msg[i:i + PAYLOAD_BYTES] for i in range(0, len(msg), PAYLOAD_BYTES)] or [b""]
    frames, info_frames = [], []
    il = None
    for f in range(cfg.n_frames):
        info = build_info_frame(f, chunks[f % len(chunks)], cfg.frame_sync_hex)
        info_frames.append(info)
        coded = fec.encode(info)
        if il is None:
            il = resolve_interleaver(cfg, len(coded))
        tx = il.interleave(coded)
        pad = (-len(tx)) % bps
        tx = np.concatenate([tx, np.zeros(pad, dtype=np.uint8)])
        frames.append(np.concatenate([pre, tx]))
    rng = np.random.default_rng(cfg.seed + 7)
    lead = rng.integers(0, 2, cfg.lead_symbols * bps).astype(np.uint8)
    tail = rng.integers(0, 2, cfg.lead_symbols * bps).astype(np.uint8)
    bits = np.concatenate([lead] + frames + [tail])
    truth = {
        "modulation": cfg.modulation,
        "bits_per_symbol": bps,
        "symbol_rate": cfg.symbol_rate,
        "fs": cfg.fs,
        "fec": cfg.fec,
        "fec_label": fec.label,
        "interleaver": il.describe(),
        "interleaver_kind": il.kind,
        "interleaver_params": il.params,
        "coded_len": len(coded),
        "frame_len_bits": len(frames[0]),
        "n_frames": cfg.n_frames,
        "preamble_hex": cfg.preamble_hex,
        "frame_sync_hex": cfg.frame_sync_hex,
        "lead_bits": len(lead),
        "message": cfg.message,
        "snr_db": cfg.snr_db,
        "freq_offset_hz": cfg.freq_offset_hz,
    }
    return bits, truth, info_frames


def modulate(bits: np.ndarray, cfg: TxConfig, rng: np.random.Generator) -> np.ndarray:
    bps = BITS_PER_SYMBOL[cfg.modulation]
    labels = bits_to_labels(bits, bps)
    n_sym = len(labels)
    Rs, fs = cfg.symbol_rate, cfg.fs
    delay = rng.uniform(0, 1)                               # fractional symbol delay
    rate = Rs * (1 + cfg.clock_ppm * 1e-6)
    n_out = int((n_sym - 1) / rate * fs)
    t_sym = np.arange(n_out) / fs * rate + delay            # time in symbols
    t_sym = t_sym[t_sym < n_sym - 1]
    if cfg.modulation in FSK_MODS:
        M = 1 << bps
        lv = fsk_levels(M)[labels]
        f = lv[np.floor(t_sym).astype(int)] * cfg.fsk_h * Rs / 2
        phase = 2 * np.pi * np.cumsum(f) / fs
        return np.exp(1j * phase)
    pts = constellation(cfg.modulation)[labels]
    G = 16
    up = np.zeros(n_sym * G, dtype=complex)
    up[::G] = pts
    h = rrc_taps(G, cfg.rolloff, span=12)
    base = fftconvolve(up, h, mode="full")[(len(h) - 1) // 2:][: n_sym * G]
    grid = np.arange(n_sym * G) / G
    cs_r = CubicSpline(grid, base.real)
    cs_i = CubicSpline(grid, base.imag)
    return cs_r(t_sym) + 1j * cs_i(t_sym)


def generate(cfg: TxConfig):
    """Returns (samples, fs, truth). samples are complex for 'iq', real for 'real'."""
    rng = np.random.default_rng(cfg.seed)
    bits, truth, info_frames = build_bitstream(cfg)
    x = modulate(bits, cfg, rng)
    n = np.arange(len(x))
    x = x * np.exp(1j * (2 * np.pi * cfg.freq_offset_hz * n / cfg.fs + cfg.phase_offset_rad))
    # AWGN at the requested Es/N0
    P = np.mean(np.abs(x) ** 2)
    sps = cfg.fs / cfg.symbol_rate
    N0 = P * sps / 10 ** (cfg.snr_db / 10)
    x = x + np.sqrt(N0 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    truth["config"] = {k: v for k, v in asdict(cfg).items() if k != "message"}
    truth["info_frames_hex"] = [np.packbits(f).tobytes().hex() for f in info_frames]
    if cfg.output == "real":
        y = np.real(x * np.exp(1j * 2 * np.pi * cfg.carrier_hz * n / cfg.fs)) * np.sqrt(2)
        truth["carrier_hz"] = cfg.carrier_hz
        return y, cfg.fs, truth
    return x, cfg.fs, truth
