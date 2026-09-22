"""Stage 5 - blind identification of the interleaver and the FEC code.

Idea: *hypothesis testing on algebraic structure*.
For every interleaver that could produce a block of this length (all matrix
shapes, all (B, J) convolutional settings, a range of pseudo-random seeds) we
de-interleave the hard bits and ask every candidate code "do these bits look
like your codewords?":

* convolutional (rate 1/2): parity check  c1*g2 + c2*g1 = 0  (mod 2)
* Reed-Solomon: syndrome + Berlekamp-Massey LFSR length (= number of errors)
* LDPC: fraction of satisfied parity checks  H c = 0
* uncoded: the frame sync word appears in the clear

Random/wrong hypotheses score ~0, the right one scores ~1 (minus channel
errors), so the winner stands out clearly. All tests are cheap, so thousands
of hypotheses are checked in about a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fec import FEC_CODES, ConvCode, FecCode, RSCode
from .interleave import Interleaver, enumerate_candidates
from .utils import hard


@dataclass
class Candidate:
    interleaver: Interleaver
    fec: str
    pad: int
    score: float
    coded_len: int

    def as_row(self):
        return {"interleaver": self.interleaver.describe(), "fec": FEC_CODES[self.fec].label,
                "pad bits": self.pad, "coded length": self.coded_len, "score": round(self.score, 3)}


@dataclass
class IdentifyResult:
    best: Candidate | None
    ranked: list = field(default_factory=list)
    n_tested: int = 0
    note: str = ""


def sync_score(blocks: list[np.ndarray], sync_bits: np.ndarray | None) -> float:
    if sync_bits is None or len(sync_bits) == 0:
        return 0.0
    from .framing import correlate_pattern
    vals = []
    for b in blocks:
        if len(b) < len(sync_bits):
            return 0.0
        vals.append(correlate_pattern(b.astype(np.uint8), sync_bits).max())
    v = float(np.mean(vals))
    return max(0.0, (v - 0.8) / 0.2)


def identify(blocks_llr: list[np.ndarray], bits_per_symbol: int, sync_bits: np.ndarray | None,
             interleaver_kinds=("none", "block", "diagonal", "conv", "random"),
             fec_names=None, seed_range=range(0, 256), max_frames: int = 4,
             forced_interleaver: Interleaver | None = None) -> IdentifyResult:
    fec_names = list(fec_names or FEC_CODES.keys())
    blocks = [hard(b) for b in blocks_llr[:max_frames]]
    if not blocks:
        return IdentifyResult(None, [], 0, "no frames")
    n_total = len(blocks[0])
    results: list[Candidate] = []
    n_tested = 0
    for pad in range(bits_per_symbol):
        n_il = n_total - pad
        if n_il <= 0:
            continue
        cands = [forced_interleaver] if forced_interleaver else enumerate_candidates(
            n_il, interleaver_kinds, seed_range)
        cut = [b[:n_il] for b in blocks]
        for il in cands:
            if il.kind in ("block", "diagonal") and il.params["rows"] * il.params["cols"] != n_il:
                continue
            n_code = il.coded_len(n_il)
            if n_code <= 0:
                continue
            des = None
            for name in fec_names:
                code: FecCode = FEC_CODES[name]
                if name == "concat":
                    continue  # confirmed after decoding the inner code (see refine_concat)
                if name != "none" and not code.accepts(n_code):
                    continue
                if des is None:
                    des = [il.deinterleave(b) for b in cut]
                n_tested += 1
                scorer = (lambda bl: sync_score(bl, sync_bits)) if name == "none" else code.structure_score
                s = scorer(des[:1])                  # quick reject on one frame
                if s is None:
                    continue
                if s >= 0.25 and len(des) > 1:
                    s = scorer(des)
                results.append(Candidate(il, name, pad, float(s), n_code))
    # Evidence from code structure (thousands of parity relations) is far stronger
    # than a 32-bit sync word seen in the clear, so coded hypotheses rank first;
    # "uncoded" is only chosen when no code fits.
    coded_found = _any_coded(results)
    results.sort(key=lambda c: (c.fec == "none" and c.score >= 0.5 and coded_found,
                                -round(c.score, 6), c.pad, _complexity(c), _specificity(c)))
    best = results[0] if results and results[0].score >= 0.5 else None
    note = "" if best else "no hypothesis showed code structure (unknown code, or wrong framing)"
    return IdentifyResult(best, results[:25], n_tested, note)


def _any_coded(results) -> bool:
    return any(c.fec != "none" and c.score >= 0.5 for c in results)


def _specificity(c: Candidate) -> int:
    code = FEC_CODES[c.fec]
    # a codeword of RS(n, n-32) is also a codeword of RS(n, n-16): prefer the stronger code
    return -getattr(code, "nsym", 0)


def _complexity(c: Candidate) -> int:
    order = {"none": 0, "block": 1, "diagonal": 2, "conv": 3, "random": 4}
    return order.get(c.interleaver.kind, 9)


def refine_concat(best: Candidate, decoded_info_frames: list[np.ndarray]) -> str:
    """If the inner code is K=7 convolutional, test an outer RS code on its output."""
    if best.fec != "conv_k7":
        return best.fec
    rs: RSCode = FEC_CODES["rs32"]
    if not decoded_info_frames or not rs.accepts(len(decoded_info_frames[0])):
        return best.fec
    score = rs.structure_score([f.astype(np.uint8) for f in decoded_info_frames[:4]])
    return "concat" if score >= 0.5 else best.fec


def decode_frames(blocks_llr: list[np.ndarray], cand: Candidate, fec_name: str):
    """De-interleave and decode every frame with the identified configuration."""
    code = FEC_CODES[fec_name]
    infos, stats = [], []
    for b in blocks_llr:
        seg = b[: len(b) - cand.pad]
        des = cand.interleaver.deinterleave(seg)
        if fec_name == "none":
            info, st = hard(des), {"ok": True}
        else:
            info, st = code.decode(des)
        infos.append(np.asarray(info, dtype=np.uint8))
        stats.append(st)
    return infos, stats


def inner_decode(blocks_llr, cand: Candidate):
    code: ConvCode = FEC_CODES[cand.fec]
    outs = []
    for b in blocks_llr[:4]:
        seg = cand.interleaver.deinterleave(b[: len(b) - cand.pad])
        outs.append(code.decode(seg)[0])
    return outs
