"""Forward Error Correction (FEC): encoders, decoders and blind "structure tests".

Every code exposes the same small interface (see :class:`FecCode`):

* ``encode(info_bits)``        -> coded bits (used by the synthetic generator)
* ``decode(llr)``              -> (info_bits, stats)   (used by the receiver)
* ``accepts(n)``               -> can a coded block of ``n`` bits belong to this code?
* ``structure_score(blocks)``  -> 0..1, how strongly the *hard* bits look like
                                  codewords of this code. This is what makes blind
                                  identification possible: we try every candidate
                                  de-interleaver + code and keep the one whose
                                  algebraic structure "clicks".

Codes implemented
-----------------
* Convolutional codes, rate 1/2, constraint lengths K=3, 5, 7 (the K=7
  (171,133) code is the NASA/CCSDS/Voyager standard), decoded with a
  soft-decision Viterbi decoder.
* Reed-Solomon RS(n, n-32) over GF(256) (shortened CCSDS-style), via ``reedsolo``.
* Concatenated: outer Reed-Solomon + inner K=7 convolutional (the classic
  deep-space concatenated scheme).
* LDPC: a (3,6)-regular rate-1/2 code with n=1024 built from a fixed seed,
  decoded with normalised min-sum belief propagation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import reedsolo

from .utils import bits_to_bytes, bytes_to_bits, hard

# --------------------------------------------------------------------------
# GF(256) arithmetic (same field as reedsolo defaults: prim=0x11d, generator=2)
# Used only for fast *identification* scoring; real decoding uses reedsolo.
# --------------------------------------------------------------------------
_GF_EXP = np.zeros(512, dtype=np.int64)
_GF_LOG = np.zeros(256, dtype=np.int64)
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
_GF_EXP[255:510] = _GF_EXP[:255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_GF_EXP[_GF_LOG[a] + _GF_LOG[b]])


def _gf_inv(a: int) -> int:
    return int(_GF_EXP[255 - _GF_LOG[a]])


def rs_syndromes(codeword: np.ndarray, nsym: int) -> np.ndarray:
    """S_j = c(alpha^j), j=0..nsym-1, vectorised (reedsolo convention, fcr=0)."""
    c = np.asarray(codeword, dtype=np.int64)
    n = len(c)
    nz = c != 0
    if not nz.any():
        return np.zeros(nsym, dtype=np.int64)
    powers = (n - 1 - np.arange(n))[nz]           # degree of each coefficient
    logs = _GF_LOG[c[nz]]
    j = np.arange(nsym)[:, None]
    terms = _GF_EXP[(logs[None, :] + j * powers[None, :]) % 255]
    return np.bitwise_xor.reduce(terms, axis=1)


def _berlekamp_massey(s: np.ndarray) -> int:
    """Length of the shortest LFSR generating the syndrome sequence.

    For a genuine RS codeword with e symbol errors this equals e (small);
    for random bytes it is about nsym/2. That contrast is our RS detector.
    """
    C = [1]
    B = [1]
    L, m, b = 0, 1, 1
    for n in range(len(s)):
        d = int(s[n])
        for i in range(1, L + 1):
            if i < len(C):
                d ^= _gf_mul(C[i], int(s[n - i]))
        if d == 0:
            m += 1
            continue
        coef = _gf_mul(d, _gf_inv(b))
        T = C[:]
        need = len(B) + m
        if len(C) < need:
            C = C + [0] * (need - len(C))
        for i, bi in enumerate(B):
            C[i + m] ^= _gf_mul(coef, bi)
        if 2 * L <= n:
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    return L


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class FecCode:
    name = "base"
    label = "base"

    def encode(self, info: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def decode(self, llr: np.ndarray):  # pragma: no cover
        raise NotImplementedError

    def accepts(self, n: int) -> bool:
        return True

    def structure_score(self, blocks: list[np.ndarray]) -> float | None:
        return None

    def describe(self) -> str:
        return self.label


class NoFec(FecCode):
    name = "none"
    label = "Uncoded"

    def encode(self, info):
        return np.asarray(info, dtype=np.uint8).copy()

    def decode(self, llr):
        return hard(llr), {"ok": True}


# --------------------------------------------------------------------------
# Convolutional code + Viterbi
# --------------------------------------------------------------------------
def _parity(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    out = np.zeros_like(x)
    while np.any(x):
        out ^= x & 1
        x >>= 1
    return out


class ConvCode(FecCode):
    """Rate-1/n feed-forward convolutional code, zero-tail terminated.

    ``gens`` are the usual octal generator polynomials, MSB = current input.
    """

    def __init__(self, K: int, gens: tuple[int, ...], name: str | None = None):
        self.K = K
        self.gens = tuple(gens)
        self.n_out = len(gens)
        self.name = name or f"conv_k{K}"
        self.label = f"Convolutional K={K} r=1/{self.n_out} ({', '.join(oct(g)[2:] for g in gens)})"
        S = 1 << (K - 1)
        self.S = S
        # Trellis tables. For next state ns, predecessors are (ns<<1 | p) & (S-1), p in {0,1};
        # the input bit that led to ns is its MSB.
        ns = np.arange(S)
        self.pred = np.stack([((ns << 1) & (S - 1)) | 0, ((ns << 1) & (S - 1)) | 1], axis=1)  # (S,2)
        self.in_bit = (ns >> (K - 2)) & 1 if K > 1 else ns
        reg = (self.in_bit[:, None] << (K - 1)) | self.pred          # (S,2) full register
        outs = np.stack([_parity(reg & g) for g in self.gens], axis=-1)  # (S,2,n)
        self.sign = (2 * outs - 1).astype(np.float64)                 # cost = sum sign*llr
        # generator coefficient arrays for polynomial arithmetic: g_arr[i] multiplies u[n-i]
        self.g_arr = [np.array([(g >> (K - 1 - i)) & 1 for i in range(K)], dtype=np.int64) for g in self.gens]

    # -- encoder ---------------------------------------------------------
    def encode(self, info):
        u = np.concatenate([np.asarray(info, dtype=np.int64), np.zeros(self.K - 1, dtype=np.int64)])
        outs = [np.convolve(u, g)[: len(u)] % 2 for g in self.g_arr]
        return np.stack(outs, axis=1).reshape(-1).astype(np.uint8)

    def accepts(self, n):
        return n % self.n_out == 0 and n // self.n_out > self.K + 8

    def info_len(self, n):
        return n // self.n_out - (self.K - 1)

    # -- decoder ---------------------------------------------------------
    def viterbi(self, llr: np.ndarray) -> np.ndarray:
        """Soft-decision Viterbi. Returns the full decoded sequence (incl. tail)."""
        r = np.asarray(llr, dtype=np.float64).reshape(-1, self.n_out)
        T = len(r)
        S = self.S
        pm = np.full(S, 1e18)
        pm[0] = 0.0
        surv = np.zeros((T, S), dtype=np.uint8)
        pred = self.pred
        sign = self.sign
        for t in range(T):
            bm = sign @ r[t]                        # (S,2)
            cand = pm[pred] + bm                     # (S,2)
            choice = np.argmin(cand, axis=1)
            surv[t] = choice
            pm = cand[np.arange(S), choice]
            pm -= pm.min()
        # zero-tail termination -> trace back from state 0
        s = 0
        out = np.zeros(T, dtype=np.uint8)
        for t in range(T - 1, -1, -1):
            out[t] = self.in_bit[s]
            s = pred[s, surv[t, s]]
        return out

    def decode(self, llr):
        full = self.viterbi(llr)
        info = full[: len(full) - (self.K - 1)]
        # re-encode and compare against the hard input: estimates channel BER
        re = self.encode(info)
        rx = hard(llr)[: len(re)]
        mism = int(np.count_nonzero(re != rx))
        return info, {"ok": True, "channel_bit_errors": mism, "channel_ber": mism / max(len(re), 1)}

    # -- blind identification ------------------------------------------
    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        """For rate 1/2: s = c1*g2 + c2*g1 (mod 2) is all-zero for any valid codeword."""
        b = np.asarray(bits, dtype=np.int64)
        c1, c2 = b[0::2], b[1::2]
        return (np.convolve(c1, self.g_arr[1]) + np.convolve(c2, self.g_arr[0])) % 2

    def structure_score(self, blocks):
        if self.n_out != 2:
            return None
        w = [self.syndrome(b).mean() for b in blocks]
        return float(max(0.0, 1.0 - 2.0 * np.mean(w)))


# --------------------------------------------------------------------------
# Reed-Solomon
# --------------------------------------------------------------------------
class RSCode(FecCode):
    def __init__(self, nsym: int = 32):
        self.nsym = nsym
        self.name = f"rs{nsym}"
        self.label = f"Reed-Solomon (n, n-{nsym}) GF(256), t={nsym // 2}"
        self.codec = reedsolo.RSCodec(nsym)

    def encode(self, info):
        return bytes_to_bits(bytes(self.codec.encode(bits_to_bytes(info))))

    def accepts(self, n):
        return n % 8 == 0 and self.nsym < n // 8 <= 255

    def decode_bytes(self, data: bytes):
        try:
            msg, _full, errata = self.codec.decode(data)
            return bytes(msg), {"ok": True, "corrected_symbols": len(errata)}
        except reedsolo.ReedSolomonError:
            return bytes(data[: len(data) - self.nsym]), {"ok": False, "corrected_symbols": None}

    def decode(self, llr):
        msg, st = self.decode_bytes(bits_to_bytes(hard(llr)))
        return bytes_to_bits(msg), st

    def block_score(self, bits: np.ndarray) -> float:
        cw = np.frombuffer(bits_to_bytes(bits), dtype=np.uint8)
        s = rs_syndromes(cw, self.nsym)
        if not s.any():
            return 1.0
        L = _berlekamp_massey(s)
        t = self.nsym // 2
        return float(max(0.0, 1.0 - L / t)) if L <= t else 0.0

    def structure_score(self, blocks):
        return float(np.mean([self.block_score(b) for b in blocks]))


# --------------------------------------------------------------------------
# Concatenated RS (outer) + convolutional (inner)
# --------------------------------------------------------------------------
class ConcatCode(FecCode):
    def __init__(self, outer: RSCode, inner: ConvCode):
        self.outer, self.inner = outer, inner
        self.name = f"concat_{outer.name}_{inner.name}"
        self.label = f"Concatenated: outer {outer.label} + inner {inner.label}"

    def encode(self, info):
        return self.inner.encode(self.outer.encode(info))

    def accepts(self, n):
        if not self.inner.accepts(n):
            return False
        return self.outer.accepts(self.inner.info_len(n))

    def decode(self, llr):
        mid, st_in = self.inner.decode(llr)
        info, st_out = self.outer.decode_bytes(bits_to_bytes(mid))
        return bytes_to_bits(info), {"ok": st_out["ok"], "inner": st_in, "outer": st_out,
                                     "channel_ber": st_in.get("channel_ber"),
                                     "corrected_symbols": st_out.get("corrected_symbols")}

    def structure_score(self, blocks):
        # Inner test only (cheap). The outer code is confirmed after decoding.
        return self.inner.structure_score(blocks)


# --------------------------------------------------------------------------
# LDPC
# --------------------------------------------------------------------------
def _gf2_rref(H: np.ndarray):
    A = H.copy().astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        rows = np.nonzero(A[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


@dataclass
class _LdpcTables:
    H: np.ndarray
    rows_e: np.ndarray   # (m, wr) edge ids grouped by check
    cols_e: np.ndarray   # (n, wc) edge ids grouped by variable
    edge_var: np.ndarray
    info_cols: np.ndarray
    parity_cols: np.ndarray
    P: np.ndarray        # parity = P @ info (mod 2)
    k: int
    n: int
    m: int = field(default=0)


@lru_cache(maxsize=4)
def _build_ldpc(n: int, wc: int, wr: int, seed: int) -> _LdpcTables:
    m = n * wc // wr
    rng = np.random.RandomState(seed)
    sockets = np.repeat(np.arange(m), wr)
    for _ in range(1000):
        rng.shuffle(sockets)
        cols = sockets.reshape(n, wc)
        bad = np.nonzero((cols[:, 0] == cols[:, 1]) | (cols[:, 0] == cols[:, 2]) | (cols[:, 1] == cols[:, 2]))[0] if wc == 3 else []
        tries = 0
        while len(bad) and tries < 10000:
            for v in bad:
                j = rng.randint(n)
                a, b = rng.randint(wc), rng.randint(wc)
                cols[v, a], cols[j, b] = cols[j, b], cols[v, a]
            bad = np.nonzero((cols[:, 0] == cols[:, 1]) | (cols[:, 0] == cols[:, 2]) | (cols[:, 1] == cols[:, 2]))[0]
            tries += 1
        if not len(bad):
            break
    H = np.zeros((m, n), dtype=np.uint8)
    H[cols.reshape(-1), np.repeat(np.arange(n), wc)] = 1
    # edges
    ci, vi = np.nonzero(H)                      # sorted by check (row-major)
    E = len(ci)
    rows_e = np.arange(E).reshape(m, wr)
    order_v = np.lexsort((ci, vi))
    cols_e = order_v.reshape(n, wc)
    R, piv = _gf2_rref(H)
    parity_cols = np.array(piv)
    info_cols = np.array(sorted(set(range(n)) - set(piv)))
    P = R[:, info_cols]
    return _LdpcTables(H=H, rows_e=rows_e, cols_e=cols_e, edge_var=vi, info_cols=info_cols,
                       parity_cols=parity_cols, P=P, k=len(info_cols), n=n, m=m)


class LdpcCode(FecCode):
    def __init__(self, n: int = 1024, wc: int = 3, wr: int = 6, seed: int = 2026, iters: int = 50):
        self.n, self.wc, self.wr, self.seed, self.iters = n, wc, wr, seed, iters
        self.name = "ldpc"
        self.label = f"LDPC ({wc},{wr})-regular n={n} r~1/2 (seed {seed}), min-sum BP"

    @property
    def t(self) -> _LdpcTables:
        return _build_ldpc(self.n, self.wc, self.wr, self.seed)

    @property
    def k(self):
        return self.t.k

    def encode(self, info):
        t = self.t
        u = np.zeros(t.k, dtype=np.uint8)
        info = np.asarray(info, dtype=np.uint8)
        if len(info) > t.k:
            raise ValueError(f"LDPC carries at most {t.k} info bits, got {len(info)}")
        u[: len(info)] = info
        c = np.zeros(t.n, dtype=np.uint8)
        c[t.info_cols] = u
        c[t.parity_cols] = (t.P.astype(np.int64) @ u) % 2
        return c

    def accepts(self, n):
        return n == self.n

    def decode(self, llr, n_info: int | None = None):
        t = self.t
        L = np.clip(np.asarray(llr, dtype=np.float64), -20, 20)
        q = L[t.edge_var].copy()
        alpha = 0.8
        ok = False
        it = 0
        c = hard(L)
        for it in range(1, self.iters + 1):
            qr = q[t.rows_e]                           # (m, wr)
            sgn = np.where(qr < 0, -1.0, 1.0)
            prod = np.prod(sgn, axis=1, keepdims=True)
            mag = np.abs(qr)
            i1 = np.argmin(mag, axis=1)
            m1 = mag[np.arange(t.m), i1]
            mag2 = mag.copy()
            mag2[np.arange(t.m), i1] = np.inf
            m2 = mag2.min(axis=1)
            mins = np.repeat(m1[:, None], self.wr, axis=1)
            mins[np.arange(t.m), i1] = m2
            r = np.empty_like(q)
            r[t.rows_e] = alpha * prod * sgn * mins
            tot = L + r[t.cols_e].sum(axis=1)
            c = hard(tot)
            if not ((t.H.astype(np.int64) @ c) % 2).any():
                ok = True
                break
            q = tot[t.edge_var] - r
        info = c[t.info_cols]
        return info, {"ok": ok, "iterations": it}

    def structure_score(self, blocks):
        H = self.t.H.astype(np.int64)
        w = [((H @ np.asarray(b, dtype=np.int64)) % 2).mean() for b in blocks]
        return float(max(0.0, 1.0 - 2.0 * np.mean(w)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
CONV_K3 = ConvCode(3, (0o7, 0o5))
CONV_K5 = ConvCode(5, (0o23, 0o35))
CONV_K7 = ConvCode(7, (0o171, 0o133))
RS32 = RSCode(32)
RS16 = RSCode(16)

FEC_CODES: dict[str, FecCode] = {
    "none": NoFec(),
    "conv_k3": CONV_K3,
    "conv_k5": CONV_K5,
    "conv_k7": CONV_K7,
    "rs16": RS16,
    "rs32": RS32,
    "concat": ConcatCode(RS32, CONV_K7),
    "ldpc": LdpcCode(),
}
FEC_CODES["concat"].name = "concat"

FEC_LABELS = {k: v.label for k, v in FEC_CODES.items()}


def get_fec(name: str) -> FecCode:
    return FEC_CODES[name]
