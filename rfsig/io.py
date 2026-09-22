"""Reading and writing .wav and raw .IQ recordings.

.wav files
    * 2 channels -> treated as I (left) and Q (right): complex baseband, the
      convention used by most SDR software (SDR#, HDSDR, SDRuno...).
    * 1 channel  -> a real (audio/passband) recording, e.g. the audio output of
      an HF receiver. It is converted to an *analytic* complex signal with the
      Hilbert transform; the carrier is found later by spectral analysis.
    The sampling frequency is read from the WAV header.

raw .IQ files
    Headerless interleaved I,Q samples. The sample format must be known
    (cf32 = GNU Radio ``.cfile``, ci16, ci8 = HackRF, cu8 = RTL-SDR). Because
    there is no header, the sampling rate comes from the user, a SigMF-style
    hint in the file name (e.g. ``capture_fs48000.iq`` or ``_2.4Msps``), or a
    default.
"""
from __future__ import annotations

import io as _io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

IQ_FORMATS = {
    "cf32": "complex float32 (GNU Radio .cfile)",
    "cf64": "complex float64",
    "ci16": "int16 interleaved I/Q",
    "ci8": "int8 interleaved I/Q (HackRF)",
    "cu8": "uint8 interleaved I/Q (RTL-SDR)",
}
IQ_EXTENSIONS = (".iq", ".cfile", ".cf32", ".bin", ".raw", ".dat", ".sigmf-data", ".ci16", ".cu8", ".ci8")


@dataclass
class SignalData:
    samples: np.ndarray              # complex128, analytic / complex baseband
    fs: float
    name: str = ""
    source: str = ""                 # 'wav-iq' | 'wav-real' | 'iq'
    fs_source: str = ""              # where the sampling rate came from
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.fs


def _to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x.astype(np.float64) - 127.5) / 127.5
    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float64) / 2147483648.0
    return x.astype(np.float64)


def load_wav(data: bytes | str | Path, name: str = "") -> SignalData:
    src = _io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    fs, x = wavfile.read(src)
    x = _to_float(x)
    if x.ndim == 2 and x.shape[1] >= 2:
        z = x[:, 0] + 1j * x[:, 1]
        return SignalData(z, float(fs), name, "wav-iq", "WAV header",
                          {"channels": x.shape[1], "note": "stereo WAV interpreted as I/Q"})
    if x.ndim == 2:
        x = x[:, 0]
    z = hilbert(x - np.mean(x))
    return SignalData(z, float(fs), name, "wav-real", "WAV header",
                      {"channels": 1, "note": "mono WAV: real passband -> analytic signal (Hilbert)"})


_FS_PATTERNS = [
    (re.compile(r"(?:fs|sr|rate)[_\-=]?(\d+(?:\.\d+)?)([kKmM]?)", re.I), None),
    (re.compile(r"(\d+(?:\.\d+)?)([kKmM]?)sps", re.I), None),
    (re.compile(r"(\d+(?:\.\d+)?)([kKmM]?)hz", re.I), None),
]


def guess_fs_from_name(name: str) -> float | None:
    stem = Path(name).stem
    for pat, _ in _FS_PATTERNS:
        m = pat.search(stem)
        if m:
            val = float(m.group(1))
            mult = {"": 1, "k": 1e3, "m": 1e6}[m.group(2).lower()]
            return val * mult
    return None


def load_iq(data: bytes | str | Path, fmt: str = "cf32", fs: float | None = None, name: str = "",
            swap_iq: bool = False, big_endian: bool = False) -> SignalData:
    raw = data if isinstance(data, (bytes, bytearray)) else Path(data).read_bytes()
    end = ">" if big_endian else "<"
    dt = {"cf32": end + "f4", "cf64": end + "f8", "ci16": end + "i2", "ci8": "i1", "cu8": "u1"}[fmt]
    x = np.frombuffer(raw, dtype=np.dtype(dt))
    x = x[: len(x) // 2 * 2]
    if fmt == "cu8":
        x = (x.astype(np.float64) - 127.5) / 127.5
    elif fmt == "ci8":
        x = x.astype(np.float64) / 128.0
    elif fmt == "ci16":
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    i, q = x[0::2], x[1::2]
    if swap_iq:
        i, q = q, i
    fs_src = "user"
    if fs is None:
        fs = guess_fs_from_name(name or str(data))
        fs_src = "file name" if fs else "default (unknown - please set)"
        fs = fs or 48000.0
    return SignalData(i + 1j * q, float(fs), name, "iq", fs_src, {"format": fmt, "note": IQ_FORMATS[fmt]})


def load_signal(name: str, data: bytes, iq_format: str = "cf32", fs: float | None = None,
                swap_iq: bool = False) -> SignalData:
    if name.lower().endswith(".wav"):
        sig = load_wav(data, name)
        if fs:  # explicit override
            sig.fs, sig.fs_source = float(fs), "user override"
        return sig
    return load_iq(data, iq_format, fs, name, swap_iq)


# ----------------------------------------------------------------------
def save_iq(path: str | Path, z: np.ndarray, fmt: str = "cf32") -> None:
    z = np.asarray(z)
    inter = np.empty(2 * len(z))
    inter[0::2], inter[1::2] = z.real, z.imag
    if fmt == "cf32":
        inter.astype("<f4").tofile(path)
    elif fmt == "ci16":
        peak = np.max(np.abs(inter)) or 1.0
        np.round(inter / peak * 32000).astype("<i2").tofile(path)
    elif fmt == "cu8":
        peak = np.max(np.abs(inter)) or 1.0
        np.clip(np.round(inter / peak * 127 + 127.5), 0, 255).astype("u1").tofile(path)
    elif fmt == "ci8":
        peak = np.max(np.abs(inter)) or 1.0
        np.round(inter / peak * 126).astype("i1").tofile(path)
    else:
        raise ValueError(fmt)


def save_wav_iq(path: str | Path, z: np.ndarray, fs: float) -> None:
    z = np.asarray(z)
    peak = max(np.max(np.abs(z.real)), np.max(np.abs(z.imag))) or 1.0
    st = np.stack([z.real, z.imag], axis=1) / peak * 0.9
    wavfile.write(str(path), int(fs), (st * 32767).astype(np.int16))


def save_wav_real(path: str | Path, x: np.ndarray, fs: float) -> None:
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) or 1.0
    wavfile.write(str(path), int(fs), (x / peak * 0.9 * 32767).astype(np.int16))
