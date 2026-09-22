"""Create a set of synthetic test recordings in ./samples with known ground truth.

Run from the project folder:
    python scripts/generate_samples.py

Each file uses a different modulation / FEC / interleaver / file format, so
together they exercise every stage of the receiver. The true settings are
written to samples/manifest.json so results can be checked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfsig.generator import TxConfig, generate  # noqa: E402
from rfsig.io import save_iq, save_wav_iq, save_wav_real  # noqa: E402

SAMPLES = [
    # name, file format, generator settings
    ("sig01_fs48000.iq", "cf32",
     dict(modulation="BPSK", fec="conv_k7", interleaver="block", symbol_rate=2400, fs=48000, snr_db=12)),
    ("sig02_fs96000.iq", "ci16",
     dict(modulation="QPSK", fec="concat", interleaver="conv", il_params={"B": 8, "J": 4},
          symbol_rate=4800, fs=96000, snr_db=14)),
    ("sig03_iq.wav", "wav-iq",
     dict(modulation="8PSK", fec="conv_k5", interleaver="diagonal", symbol_rate=9600, fs=96000, snr_db=18)),
    ("sig04_fs192000.iq", "cf32",
     dict(modulation="16QAM", fec="rs32", interleaver="random", il_params={"seed": 42},
          symbol_rate=19200, fs=192000, snr_db=22)),
    ("sig05_hf_audio.wav", "wav-real",
     dict(modulation="2FSK", fec="conv_k3", interleaver="block", symbol_rate=600, fs=48000,
          snr_db=15, output="real", carrier_hz=1800)),
    ("sig06_fs48000.iq", "cf32",
     dict(modulation="4FSK", fec="ldpc", interleaver="none", symbol_rate=1200, fs=48000, snr_db=16)),
    ("sig07_rtlsdr_fs48000.iq", "cu8",
     dict(modulation="QPSK", fec="none", interleaver="none", symbol_rate=2400, fs=48000, snr_db=16)),
]


def main() -> None:
    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)
    manifest = {}
    for i, (name, fmt, kw) in enumerate(SAMPLES):
        cfg = TxConfig(seed=100 + i, **kw)
        x, fs, truth = generate(cfg)
        path = out_dir / name
        if fmt == "wav-iq":
            save_wav_iq(path, x, fs)
        elif fmt == "wav-real":
            save_wav_real(path, x, fs)
        else:
            save_iq(path, x, fmt)
        truth.pop("info_frames_hex", None)
        truth["file_format"] = fmt
        manifest[name] = truth
        print(f"{name:26s} {kw['modulation']:6s} {truth['fec']:8s} {truth['interleaver']:36s} "
              f"{path.stat().st_size / 1e6:6.2f} MB")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote {len(SAMPLES)} files + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
