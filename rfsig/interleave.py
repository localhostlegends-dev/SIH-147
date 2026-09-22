"""Interleavers and de-interleavers.

Interleaving shuffles coded bits before transmission so that a burst of
channel errors (fading, impulsive noise) is spread out into isolated errors
that the FEC decoder can fix. The receiver must apply the exact inverse
permutation ("de-interleaving") before decoding.

Four families are supported, matching the problem statement:

* **Block** (R x C): write row-by-row into a matrix, read column-by-column.
* **Diagonal** (R x C): write row-by-row, read along wrapped diagonals.
* **Convolutional** (Forney, B branches, J cells per step): branch ``i``
  delays its symbols by ``i*J`` commutator cycles. Works on a stream, so a
  block is flushed with ``D = B*(B-1)*J`` extra bits.
* **Pseudo-random**: a seeded random permutation (a real system would use an
  LFSR or a published table; here the seed is the secret).

All functions work on hard bits *and* on soft LLR arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Interleaver:
    kind: str = "none"                     # none | block | diagonal | conv | random
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        p = self.params
        if self.kind == "none":
            return "None"
        if self.kind in ("block", "diagonal"):
            return f"{self.kind.capitalize()} {p['rows']} x {p['cols']}"
        if self.kind == "conv":
            return f"Convolutional B={p['B']} J={p['J']} (delay {conv_delay(p['B'], p['J'])})"
        if self.kind == "random":
            return f"Pseudo-random (seed {p['seed']})"
        return self.kind

    def coded_len(self, n_interleaved: int) -> int:
        """Length of the de-interleaved (coded) block."""
        if self.kind == "conv":
            return n_interleaved - conv_delay(self.params["B"], self.params["J"])
        return n_interleaved

    # ------------------------------------------------------------------
    def interleave(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        k, p = self.kind, self.params
        if k == "none":
            return x.copy()
        if k == "block":
            return x.reshape(p["rows"], p["cols"]).T.reshape(-1)
        if k == "diagonal":
            return x[_diag_index(p["rows"], p["cols"])]
        if k == "conv":
            return _conv_interleave(x, p["B"], p["J"])
        if k == "random":
            return x[_rand_perm(len(x), p["seed"])]
        raise ValueError(k)

    def deinterleave(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        k, p = self.kind, self.params
        if k == "none":
            return y.copy()
        if k == "block":
            return y.reshape(p["cols"], p["rows"]).T.reshape(-1)
        if k == "diagonal":
            out = np.empty_like(y)
            out[_diag_index(p["rows"], p["cols"])] = y
            return out
        if k == "conv":
            return _conv_deinterleave(y, p["B"], p["J"])
        if k == "random":
            out = np.empty_like(y)
            out[_rand_perm(len(y), p["seed"])] = y
            return out
        raise ValueError(k)


# ----------------------------------------------------------------------
def conv_delay(B: int, J: int) -> int:
    return B * (B - 1) * J


def _diag_index(R: int, C: int) -> np.ndarray:
    # output position (j*R + i) takes matrix element (i, (i + j) mod C)
    i = np.tile(np.arange(R), C)
    j = np.repeat(np.arange(C), R)
    return i * C + (i + j) % C


_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _rand_perm(n: int, seed: int) -> np.ndarray:
    key = (n, seed)
    if key not in _PERM_CACHE:
        if len(_PERM_CACHE) > 4096:
            _PERM_CACHE.clear()
        _PERM_CACHE[key] = np.random.RandomState(seed).permutation(n)
    return _PERM_CACHE[key]


def _conv_interleave(x: np.ndarray, B: int, J: int) -> np.ndarray:
    """Forney interleaver on a block, flushed with D zeros: len(out) = len(x) + D."""
    D = conv_delay(B, J)
    L = len(x)
    out = np.zeros(L + D, dtype=x.dtype)
    n = np.arange(L)
    out[n + (n % B) * J * B] = x
    return out


def _conv_deinterleave(y: np.ndarray, B: int, J: int) -> np.ndarray:
    D = conv_delay(B, J)
    N = len(y)
    z = np.zeros(N + D, dtype=y.dtype)
    t = np.arange(N)
    z[t + (B - 1 - (t % B)) * J * B] = y
    return z[D:N]


# ----------------------------------------------------------------------
def divisor_pairs(n: int, min_dim: int = 4) -> list[tuple[int, int]]:
    out = []
    for r in range(min_dim, n // min_dim + 1):
        if n % r == 0:
            out.append((r, n // r))
    return out


def default_block_dims(n: int) -> tuple[int, int]:
    """Pick the most square R x C factorisation of n (used by the generator)."""
    pairs = divisor_pairs(n, 4)
    if not pairs:
        raise ValueError(f"cannot factor block length {n} into a matrix with both sides >= 4")
    return min(pairs, key=lambda rc: abs(rc[0] - rc[1]))


def enumerate_candidates(n: int, kinds=("none", "block", "diagonal", "conv", "random"),
                         seed_range=range(0, 256), max_B: int = 16, max_J: int = 16,
                         min_coded: int = 64) -> list[Interleaver]:
    """All interleaver hypotheses compatible with an interleaved block of n bits."""
    c: list[Interleaver] = []
    if "none" in kinds:
        c.append(Interleaver("none"))
    pairs = divisor_pairs(n, 4)
    if "block" in kinds:
        c += [Interleaver("block", {"rows": r, "cols": cc}) for r, cc in pairs]
    if "diagonal" in kinds:
        c += [Interleaver("diagonal", {"rows": r, "cols": cc}) for r, cc in pairs]
    if "conv" in kinds:
        for B in range(2, max_B + 1):
            for J in range(1, max_J + 1):
                if n - conv_delay(B, J) >= min_coded:
                    c.append(Interleaver("conv", {"B": B, "J": J}))
    if "random" in kinds:
        c += [Interleaver("random", {"seed": int(s)}) for s in seed_range]
    return c
