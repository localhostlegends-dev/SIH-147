"""Decode every file in ./samples and compare the result with the ground truth.

Run from the project folder (after scripts/generate_samples.py):
    python scripts/decode_samples.py              # all samples
    python scripts/decode_samples.py sig03_iq.wav # one file, with the stage log

A file PASSES when the modulation, interleaver and FEC were identified correctly
and the hidden text message was recovered.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfsig.io import load_signal  # noqa: E402
from rfsig.pipeline import run  # noqa: E402

IQ_FORMATS = ("cf32", "cf64", "ci16", "ci8", "cu8")


def check(name: str, truth: dict, verbose: bool = False) -> bool:
    fmt = truth.get("file_format", "cf32")
    data = (ROOT / "samples" / name).read_bytes()
    sig = load_signal(name, data, iq_format=fmt if fmt in IQ_FORMATS else "cf32")
    t0 = time.time()
    res = run(sig)
    dt = time.time() - t0

    got_mod = res.demod.modulation if res.demod else "-"
    got_il = res.ident.best.interleaver.describe() if res.ident and res.ident.best else "-"
    got_fec = res.fec_name or "-"
    msg = res.corr.message if res.corr else ""
    checks = {
        "modulation": got_mod == truth["modulation"],
        "interleaver": got_il == truth["interleaver"],
        "fec": got_fec == truth["fec"],
        "message": truth["message"][:120] in msg,
    }
    ok = all(checks.values())
    wrong = ", ".join(k for k, v in checks.items() if not v)
    print(f"{'PASS' if ok else 'FAIL'}  {name:26s} {got_mod:6s} {got_fec:8s} {got_il:36s} "
          f"{dt:5.1f}s  {'' if ok else 'wrong: ' + wrong}")
    if verbose or not ok:
        for stage, text in res.log:
            print(f"        [{stage}] {text}")
        if msg:
            print(f"        decoded message: {msg[:160]}...")
    return ok


def main() -> None:
    manifest_path = ROOT / "samples" / "manifest.json"
    if not manifest_path.exists():
        sys.exit("samples/manifest.json not found - run: python scripts/generate_samples.py")
    manifest = json.loads(manifest_path.read_text())
    names = sys.argv[1:] or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        sys.exit(f"not in manifest: {', '.join(unknown)}")
    print(f"{'':6s}{'file':26s} {'mod':6s} {'fec':8s} {'interleaver':36s} {'time':>6s}")
    passed = sum(check(n, manifest[n], verbose=len(names) == 1) for n in names)
    print(f"\n{passed}/{len(names)} passed")
    sys.exit(0 if passed == len(names) else 1)


if __name__ == "__main__":
    main()
