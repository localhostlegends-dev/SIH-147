"""One-click analyst report: the whole analysis as a printable PDF.

Everything is drawn straight from the numpy arrays with reportlab primitives -
no matplotlib, no browser, no image export step - so the only extra dependency
is reportlab itself and a report takes well under a second to build.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas as rl_canvas

from .fec import FEC_CODES
from .utils import bits_to_hex

PAGE_W, PAGE_H = A4
M = 40.0                       # page margin
CW = PAGE_W - 2 * M            # content width

BLUE = HexColor("#2a78d6")
ORANGE = HexColor("#eb6834")
INK = HexColor("#111111")
GREY = HexColor("#5a5a5a")
LIGHT = HexColor("#d8d8d8")
FILL = HexColor("#eef4fc")

TITLE = "RF Signal Analysis Report"
TOOL = "RF Signal Intelligence - SIH 2026, PS 26147"


# ---------------------------------------------------------------- primitives
def _ascii(s) -> str:
    """Helvetica's built-in encoding has no glyph for much beyond latin-1."""
    s = str(s)
    for a, b in (("≈", "~"), ("→", "->"), ("—", "-"), ("–", "-"),
                 ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return "".join(ch if 32 <= ord(ch) < 127 or ch in "·°±" else "?" for ch in s)


class Report:
    def __init__(self, res):
        self.res = res
        self.buf = io.BytesIO()
        self.c = rl_canvas.Canvas(self.buf, pagesize=A4)
        self.c.setTitle(f"{TITLE} - {res.signal.name}")
        self.c.setAuthor(TOOL)
        self.page = 0
        self.y = 0.0

    # -- page furniture ----------------------------------------------------
    def new_page(self, heading: str):
        if self.page:
            self._footer()
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillColor(BLUE)
        c.rect(0, PAGE_H - 26, PAGE_W, 26, stroke=0, fill=1)
        c.setFillColor(HexColor("#ffffff"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, PAGE_H - 18, _ascii(TITLE))
        c.setFont("Helvetica", 9)
        c.drawRightString(PAGE_W - M, PAGE_H - 18, _ascii(self.res.signal.name))
        self.y = PAGE_H - 52
        if heading:
            self.h1(heading)

    def _footer(self):
        c = self.c
        c.setStrokeColor(LIGHT)
        c.setLineWidth(0.6)
        c.line(M, 34, PAGE_W - M, 34)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY)
        c.drawString(M, 24, _ascii(f"{TOOL} - generated offline, no external service"))
        c.drawRightString(PAGE_W - M, 24, f"Page {self.page}")

    def finish(self) -> bytes:
        self._footer()
        self.c.save()
        return self.buf.getvalue()

    # -- text --------------------------------------------------------------
    def ensure(self, need: float):
        """Start a new page when the next block will not fit."""
        if self.y - need < 56:
            self.new_page("")

    def section(self, title: str, need: float):
        self.ensure(need + 26)
        self.h1(title)

    def h1(self, text: str):
        self.y -= 4
        self.c.setFont("Helvetica-Bold", 12.5)
        self.c.setFillColor(INK)
        self.c.drawString(M, self.y, _ascii(text))
        self.y -= 6
        self.c.setStrokeColor(BLUE)
        self.c.setLineWidth(1.4)
        self.c.line(M, self.y, M + CW, self.y)
        self.y -= 14

    def note(self, text: str, size: float = 8.2, gap: float = 4):
        self.c.setFont("Helvetica", size)
        self.c.setFillColor(GREY)
        for ln in simpleSplit(_ascii(text), "Helvetica", size, CW):
            self.c.drawString(M, self.y, ln)
            self.y -= size + 2.2
        self.y -= gap

    def kv_columns(self, items, cols: int = 2, row_h: float = 13.5):
        """Key/value pairs laid out in `cols` columns."""
        items = [(str(k), str(v)) for k, v in items]
        n = len(items)
        per = (n + cols - 1) // cols
        col_w = CW / cols
        top = self.y
        c = self.c
        for i, (k, v) in enumerate(items):
            col, row = divmod(i, per)
            x = M + col * col_w
            y = top - row * row_h
            if row % 2 == 0:
                c.setFillColor(FILL)
                c.rect(x, y - 3.5, col_w - 8, row_h, stroke=0, fill=1)
            c.setFont("Helvetica", 8.4)
            c.setFillColor(GREY)
            c.drawString(x + 4, y, _ascii(k))
            c.setFont("Helvetica-Bold", 8.4)
            c.setFillColor(INK)
            c.drawRightString(x + col_w - 14, y, _ascii(v))
        self.y = top - per * row_h - 8

    def table(self, headers, rows, widths, row_h: float = 12.0, size: float = 7.6):
        c = self.c
        total = sum(widths)
        widths = [w * CW / total for w in widths]
        rows = list(rows)
        while rows:
            fits = max(int((self.y - 60) / row_h) - 1, 1)
            self._table_chunk(headers, rows[:fits], widths, row_h, size)
            rows = rows[fits:]
            if rows:
                self.new_page("")

    def _table_chunk(self, headers, rows, widths, row_h, size):
        c = self.c
        y = self.y
        c.setFillColor(BLUE)
        c.rect(M, y - 3, CW, row_h, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", size)
        c.setFillColor(HexColor("#ffffff"))
        x = M
        for h, w in zip(headers, widths):
            c.drawString(x + 4, y + 0.5, _ascii(h))
            x += w
        y -= row_h
        for r, row in enumerate(rows):
            if r % 2:
                c.setFillColor(FILL)
                c.rect(M, y - 3, CW, row_h, stroke=0, fill=1)
            c.setFont("Helvetica", size)
            c.setFillColor(INK)
            x = M
            for cell, w in zip(row, widths):
                txt = _ascii(cell)
                while c.stringWidth(txt, "Helvetica", size) > w - 8 and len(txt) > 2:
                    txt = txt[:-2]
                c.drawString(x + 4, y + 0.5, txt)
                x += w
            y -= row_h
        self.y = y - 6

    # -- plots -------------------------------------------------------------
    def _frame(self, x, y, w, h, title, xlabel, ylabel):
        c = self.c
        c.setFont("Helvetica-Bold", 8.6)
        c.setFillColor(INK)
        c.drawString(x, y + h + 6, _ascii(title))
        c.setStrokeColor(LIGHT)
        c.setLineWidth(0.7)
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Helvetica", 7)
        c.setFillColor(GREY)
        c.drawString(x, y - 9, _ascii(xlabel))
        c.drawRightString(x + w, y - 9, _ascii(ylabel))

    def plot(self, x, y, w, h, xs, ys, title, xlabel, ylabel, *, band=None, hline=None,
             vline=None, ylim=None, kind="line", color=BLUE):
        """Line / scatter / bar plot drawn directly from arrays."""
        self._frame(x, y, w, h, title, xlabel, ylabel)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if not len(xs) or not len(ys):
            return
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = (ylim if ylim else (float(np.min(ys)), float(np.max(ys))))
        if x1 - x0 < 1e-12:
            x1 = x0 + 1
        if y1 - y0 < 1e-12:
            y1 = y0 + 1
        pad = (y1 - y0) * 0.06
        y0, y1 = y0 - pad, y1 + pad

        def px(v):
            return x + (np.clip(v, x0, x1) - x0) / (x1 - x0) * w

        def py(v):
            return y + (np.clip(v, y0, y1) - y0) / (y1 - y0) * h

        c = self.c
        if band is not None:
            c.setFillColor(HexColor("#dde9f9"))
            c.rect(px(band[0]), y + 1, max(px(band[1]) - px(band[0]), 1), h - 2, stroke=0, fill=1)
        c.setStrokeColor(color)
        if kind == "line":
            step = max(1, len(xs) // 1400)
            p = c.beginPath()
            p.moveTo(px(xs[0]), py(ys[0]))
            for i in range(step, len(xs), step):
                p.lineTo(px(xs[i]), py(ys[i]))
            c.setLineWidth(0.9)
            c.drawPath(p)
        elif kind == "scatter":
            c.setFillColor(color)
            c.setStrokeColor(color)
            for i in range(len(xs)):
                c.circle(px(xs[i]), py(ys[i]), 0.7, stroke=0, fill=1)
        elif kind == "bar":
            bw = max(w / max(len(xs), 1) * 0.7, 0.8)
            c.setFillColor(color)
            base = py(max(y0, 0))
            for i in range(len(xs)):
                c.rect(px(xs[i]) - bw / 2, base, bw, max(py(ys[i]) - base, 0.4), stroke=0, fill=1)
        if hline is not None:
            c.setStrokeColor(ORANGE)
            c.setLineWidth(0.9)
            c.setDash(3, 2)
            c.line(x, py(hline), x + w, py(hline))
            c.setDash()
        if vline is not None:
            c.setStrokeColor(ORANGE)
            c.setLineWidth(0.9)
            c.setDash(3, 2)
            c.line(px(vline), y, px(vline), y + h)
            c.setDash()

    def hbars(self, x, y, w, h, labels, values, title):
        """Ranked horizontal bars - the identification result."""
        self._frame(x, y, w, h, title, "structure score (1.0 = fits perfectly)", "")
        c = self.c
        n = max(len(labels), 1)
        row = h / n
        lab_w = w * 0.52
        for i, (lb, v) in enumerate(zip(labels, values)):
            yy = y + h - (i + 1) * row + row * 0.22
            c.setFont("Helvetica", 6.8)
            c.setFillColor(INK if i == 0 else GREY)
            txt = _ascii(lb)
            while c.stringWidth(txt, "Helvetica", 6.8) > lab_w - 8 and len(txt) > 4:
                txt = txt[:-2]
            c.drawString(x + 4, yy, txt)
            bw = max((w - lab_w - 34) * float(np.clip(v, 0, 1)), 0.8)
            c.setFillColor(BLUE if i == 0 else HexColor("#b7cdeb"))
            c.rect(x + lab_w, yy - 1.5, bw, row * 0.52, stroke=0, fill=1)
            c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 6.8)
            c.setFillColor(INK)
            c.drawString(x + lab_w + bw + 4, yy, f"{v:.2f}")


def _headline(r: Report, res, fec_label: str):
    """The answer, before any of the working: what this recording turned out to be."""
    d, ident, corr, fr = res.demod, res.ident, res.corr, res.framing
    cells = [("Modulation", d.modulation if d else "-"),
             ("Symbol rate", f"{res.symbol_rate:,.0f} Bd" if res.symbol_rate else "-"),
             ("Interleaver", ident.best.interleaver.describe() if ident and ident.best else "-"),
             ("Error correction", fec_label.split(" (")[0]),
             ("Frames", f"{len(fr.starts)}" if fr else "-"),
             ("Payload", f"{len(corr.message):,} chars" if corr and corr.message.strip() else "none")]
    h = 42
    c = r.c
    r.y -= h
    c.setFillColor(FILL)
    c.rect(M, r.y, CW, h, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.rect(M, r.y, 3.5, h, stroke=0, fill=1)
    weights = [0.13, 0.13, 0.17, 0.29, 0.09, 0.19]
    xs, acc = [], M + 14
    for wt in weights:
        xs.append(acc)
        acc += (CW - 20) * wt
    for i, (k, v) in enumerate(cells):
        x = xs[i]
        w = (CW - 20) * weights[i]
        c.setFont("Helvetica", 7)
        c.setFillColor(GREY)
        c.drawString(x, r.y + h - 15, _ascii(k.upper()))
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(INK)
        txt = _ascii(v)
        while c.stringWidth(txt, "Helvetica-Bold", 9) > w - 10 and len(txt) > 4:
            txt = txt[:-2]
        c.drawString(x, r.y + 12, txt)
    r.y -= 16


# ------------------------------------------------------------------- content
def build_pdf(res) -> bytes:
    """Render a complete analyst report for one analysis result."""
    r = Report(res)
    sig, sp, d = res.signal, res.spectral, res.demod
    fr, ident, corr = res.framing, res.ident, res.corr
    fec_label = FEC_CODES[res.fec_name].label if res.fec_name else "-"

    # ---------------- page 1: what the signal is --------------------------
    r.new_page("")
    r.c.setFont("Helvetica-Bold", 15)
    r.c.setFillColor(INK)
    r.c.drawString(M, r.y, _ascii(sig.name))
    r.y -= 15
    r.c.setFont("Helvetica", 8.6)
    r.c.setFillColor(GREY)
    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    r.c.drawString(M, r.y, _ascii(f"Analysed {stamp} - {sig.duration:.2f} s of signal, "
                                  f"{len(sig.samples):,} samples, sampling rate from {sig.fs_source}"))
    r.y -= 16

    _headline(r, res, fec_label)

    if res.error:
        r.h1("Analysis did not complete")
        r.note(res.error)

    r.h1("Signal parameters")
    r.kv_columns(list(res.summary().items()), cols=2)
    r.note("Every value above was measured from the samples themselves. The only thing supplied by "
           "the analyst is the sampling rate, and then only for a raw IQ file that carries no header.")

    if sp is not None:
        h = 132
        r.y -= h + 12
        r.plot(M, r.y, CW, h, sp.psd_freqs, sp.psd_db, "Power spectrum", "frequency (Hz)", "power (dB)",
               band=(sp.band_lo_hz, sp.band_hi_hz), hline=sp.noise_floor_db)
        r.y -= 24
        r.note(f"Shaded: the occupied band, {sp.bandwidth_hz:,.0f} Hz wide, centred "
               f"{sp.center_offset_hz:+,.0f} Hz from the middle of the recording. "
               f"Dashed: the noise floor. Spectral SNR {sp.snr_db:.1f} dB.")

    if d is not None:
        h = 140
        half = CW / 2 - 10
        r.y -= h + 12
        pts = d.symbols
        if np.iscomplexobj(pts):
            idx = np.linspace(0, len(pts) - 1, min(2500, len(pts))).astype(int)
            p = pts[idx]
            lim = float(np.percentile(np.abs(p), 99.5)) * 1.2 or 1.0
            r.plot(M, r.y, half, h, np.real(p), np.imag(p),
                   f"Constellation after carrier recovery - {d.modulation}", "in-phase (I)",
                   "quadrature (Q)", kind="scatter", ylim=(-lim, lim))
        else:
            hist, edges = np.histogram(np.asarray(pts, dtype=float), bins=90)
            centres = (edges[:-1] + edges[1:]) / 2
            r.plot(M, r.y, half, h, centres, hist.astype(float),
                   f"Tone clusters - {d.modulation}", "frequency (Hz)", "symbols", kind="bar")
        x2 = M + half + 20
        r._frame(x2, r.y, half, h, "Decision summary", "", "")
        yy = r.y + h - 16
        rows = [("Modulation", d.modulation),
                ("Symbol rate", f"{res.symbol_rate:,.1f} Bd"),
                ("Decision SNR", f"{d.snr_db:.1f} dB"),
                ("Classifier confidence", f"{d.extra.get('confidence', 0):.2f}"),
                ("Residual carrier offset", f"{d.freq_offset_hz:+.1f} Hz"),
                ("Frames found", f"{len(fr.starts)}" if fr else "-"),
                ("Frame length", f"{fr.period} bits" if fr else "-"),
                ("Framing mode", fr.mode if fr else "-")]
        for k, v in rows:
            r.c.setFont("Helvetica", 8)
            r.c.setFillColor(GREY)
            r.c.drawString(x2 + 8, yy, _ascii(k))
            r.c.setFont("Helvetica-Bold", 8)
            r.c.setFillColor(INK)
            r.c.drawRightString(x2 + half - 8, yy, _ascii(v))
            yy -= 14
        r.y -= 24
        if np.iscomplexobj(d.symbols):
            r.note("Left: the received symbols once the carrier has been recovered. Tight clusters on "
                   "the ideal constellation points mean the signal is being read correctly.")
        else:
            tones = d.extra.get("tones_hz", [])
            r.note("Left: how often each instantaneous frequency occurs. Separated peaks are the keying "
                   "tones" + (" at " + ", ".join(f"{t:,.0f} Hz" for t in tones) if len(tones) else "")
                   + " - a clean split means the symbols are being read correctly.")

    # ---------------- page 2: how it was coded ----------------------------
    r.section("Interleaver and error-correction identification", 400)
    if ident and ident.best:
        r.kv_columns([("Interleaver", ident.best.interleaver.describe()),
                      ("Error-correcting code", fec_label),
                      ("Structure score", f"{ident.best.score:.3f}"),
                      ("Runner-up score", f"{ident.ranked[1].score:.3f}" if len(ident.ranked) > 1 else "-"),
                      ("Hypotheses tested", f"{ident.n_tested:,}"),
                      ("Coded block", f"{ident.best.coded_len} bits ({ident.best.pad} pad)")], cols=2)
        top = ident.ranked[:7]
        h = 108
        r.y -= h + 12
        r.hbars(M, r.y, CW, h,
                [f'{c_.interleaver.describe()} - {FEC_CODES[c_.fec].label.split(" (")[0]}' for c_ in top],
                [c_.score for c_ in top], "Ranked hypotheses")
        r.y -= 24
        r.note("Every interleaver shape, convolutional setting and pseudo-random seed is applied, and "
               "each result is scored against every candidate code using algebraic structure: the "
               "parity relation of a convolutional code, Reed-Solomon syndromes and the Berlekamp-Massey "
               "error-locator length, unsatisfied LDPC parity checks. A wrong guess scores near zero, so "
               "the winner is decided by the margin rather than by a threshold we chose.")
        r.table(["Interleaver", "Error-correcting code", "Coded bits", "Pad", "Score"],
                [[c_.interleaver.describe(), FEC_CODES[c_.fec].label, str(c_.coded_len), str(c_.pad),
                  f"{c_.score:.3f}"] for c_ in ident.ranked[:8]],
                widths=[26, 42, 12, 8, 12])
    else:
        r.note("No interleaver / code hypothesis was confirmed for this recording.")

    bers = [s.get("channel_ber") for s in res.frame_stats if s.get("channel_ber") is not None]
    corrected = [s.get("corrected_symbols") for s in res.frame_stats if s.get("corrected_symbols") is not None]
    iters = [s.get("iterations") for s in res.frame_stats if s.get("iterations") is not None]
    ok_frames = sum(1 for s in res.frame_stats if s.get("ok"))
    r.section("Decode quality", 150)
    r.kv_columns([("Frames decoded", f"{ok_frames}/{len(res.frame_stats)}"),
                  ("Channel BER before correction", f"{np.mean(bers):.2e}" if bers else "-"),
                  ("RS symbols corrected", f"{int(np.sum(corrected))}" if corrected else "-"),
                  ("LDPC iterations (mean)", f"{np.mean(iters):.1f}" if iters else "-")], cols=2)
    series = bers or corrected or iters
    if series and float(np.max(series)) > 0:
        name = "channel BER" if bers else ("RS symbols corrected" if corrected else "LDPC iterations")
        h = 86
        r.ensure(h + 46)
        r.y -= h + 12
        r.plot(M, r.y, CW, h, np.arange(len(series)), np.asarray(series, dtype=float),
               f"Per frame: {name}", "frame", name, kind="bar")
        r.y -= 24
        r.note("What the channel damaged, and what the decoder repaired. The message below is what "
               "survived that repair.")
    elif bers:
        r.note("The channel introduced no bit errors the decoder had to repair at this signal-to-noise "
               "ratio: every frame re-encoded to exactly what was received.")

    # ---------------- page 3: header, payload, message --------------------
    r.section("Header and payload", 300)
    if corr is not None and len(corr.frames):
        h = 96
        r.y -= h + 12
        prof = np.asarray(corr.constancy, dtype=float)
        r.plot(M, r.y, CW, h, np.arange(len(prof)), prof, "Bit constancy across frames",
               "bit position in the decoded frame", "fraction agreeing",
               vline=corr.header_len_est, ylim=(0.4, 1.02))
        r.y -= 24
        r.note(f"Stacking the decoded frames and asking how often each bit position repeats separates "
               f"the header (constant) from the payload (random). That puts the boundary near bit "
               f"{corr.header_len_est} without knowing the frame format. The sync word "
               f"{corr.sync_hex} was then located by cross-correlation in every frame.")
        rows = [[str(f.get("frame", i)), str(f.get("sync_pos", "-")),
                 f'{f.get("sync_corr", 0):.2f}', str(f.get("counter", "-")),
                 f'0x{f["type"]:02X}' if "type" in f else "-", str(f.get("length", "-")),
                 (f.get("payload_text", "") or "")[:40]]
                for i, f in enumerate(corr.frames[:12])]
        r.table(["Frame", "Sync at bit", "Match", "Counter", "Type", "Length", "Payload"],
                rows, widths=[8, 12, 9, 11, 8, 9, 43])
        if len(corr.frames) > 12:
            r.note(f"First 12 of {len(corr.frames)} frames; the rest parsed identically.", gap=2)
    else:
        r.note("No decoded frames were available to correlate.")

    r.section("Recovered message", 150)
    msg = (corr.message if corr else "") or ""
    if msg.strip():
        r.c.setFont("Helvetica", 8.6)
        r.c.setFillColor(INK)
        for ln in simpleSplit(_ascii(msg), "Helvetica", 8.6, CW)[:26]:
            r.c.drawString(M, r.y, ln)
            r.y -= 11.5
        r.y -= 6
        r.note(f"{len(msg):,} characters recovered from {len(corr.frames)} frames. Every stage - "
               f"parameter estimation, modulation identification, demodulation, de-interleaving, "
               f"error correction and bitstream correlation - had to be right for this to be readable.")
    else:
        r.note("No readable payload was recovered; the decoded bits are still available from the "
               "export tab of the application.")

    # ---------------- page 4: audit trail ---------------------------------
    r.section("Analysis log", 150)
    r.note("Each stage in the order it ran, with what it found. This is the audit trail behind every "
           "number in this report.")
    r.table(["Stage", "Detail"], [[s, t] for s, t in res.log][:42], widths=[16, 84], row_h=11.4, size=7.2)
    detail = [(k, f"{v:.3f} s") for k, v in res.timings.items()]
    if res.timings:
        detail.append(("total", f"{sum(res.timings.values()):.3f} s"))
    if fr is not None:
        detail += [("framing mode", fr.mode), ("frames", str(len(fr.starts))),
                   ("frame length", f"{fr.period} bits"), ("block length", f"{fr.block_len} bits"),
                   ("phase / ambiguity", str(fr.hypothesis)), ("preamble", bits_to_hex(fr.preamble_bits))]
    if detail:
        r.section("Run detail - stage timings and framing", 76)
        r.kv_columns(detail, cols=4, row_h=12.5)
    return r.finish()
