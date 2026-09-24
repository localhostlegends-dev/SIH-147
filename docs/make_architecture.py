#!/usr/bin/env python3
"""Layered system-architecture diagram — slide version (one headline + one line per box)."""

W = 1217           # matches the reference image exactly
H_TARGET = 831
LANE_L, LANE_R = 12, W - 12
BAND_L, BAND_R = 22, W - 22
LBL_X = 42
CX0 = 176
CX1 = 1178
GAP = 18
MARGIN = 11.5
BOX_H = 80
BAND_PAD = 10
FS_T = 16          # headline
FS_S = 13.5        # supporting line

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def wid(s, fs=12.5, bold=False):
    narrow = set("iljItf.,;:'`|! ")
    wide = set("MWmw@%")
    t = 0.0
    for ch in s:
        if ch in narrow:
            t += 3.9
        elif ch in wide:
            t += 9.6
        elif ch.isupper() or ch.isdigit():
            t += 7.3
        else:
            t += 6.5
    return t * (fs / 12.5) * (1.05 if bold else 1.0)

# (weight, headline, supporting line)
LAYERS = [
    dict(id="L1", name=["ANALYST"], bg="#e7f0fc", bd="#1f3f7a", boxes=[
        (1.0, "Analyst", "any laptop, any browser — fully offline"),
        (1.0, "Streamlit web app", "pick a sample or upload a .wav / .IQ file"),
    ]),
    dict(id="L2", name=["APP"], bg="#f4eafc", bd="#7b2fa0", boxes=[
        (1.0, "GUI shell (app.py)", "file and format controls, parameter row, log"),
        (1.0, "Plot tabs", "spectrum · waterfall · constellation · eye"),
        (1.0, "Decode tabs", "hypotheses · header/payload · message"),
    ]),
    dict(id="L3", name=["INPUT"], bg="#e2f6f4", bd="#12766e", boxes=[
        (1.0, "File readers (io.py)", ".wav stereo and mono · raw .iq, five formats"),
        (1.0, "Sampling rate", "WAV header · file name · set by the analyst"),
    ]),
    dict(id="L4", name=["SIGNAL", "FRONT END"], bg="#fdf4dd", bd="#8a6d1f", boxes=[
        (1.0, "Signal parameters", "bandwidth · carrier offset · SNR · symbol rate"),
        (1.0, "Modulation ID", "BPSK · QPSK · 8PSK · 16QAM · 2/4/8-FSK"),
        (1.0, "Demodulation", "matched filter, timing and carrier recovery"),
    ]),
    dict(id="L5", name=["BLIND", "IDENTIFICATION"], bg="#e5f5ea", bd="#14795a", boxes=[
        (1.0, "Frame sync", "preamble correlation, or blind framing"),
        (1.0, "Interleaver", "block · diagonal · Forney · pseudo-random"),
        (1.0, "Error-correcting code", "ranked by algebraic structure, in ~1 s"),
    ]),
    dict(id="L6", name=["DECODE", "&amp; OUTPUT"], bg="#fdeadf", bd="#a34a12", boxes=[
        (1.0, "Decode", "Viterbi · Reed-Solomon · concatenated · LDPC"),
        (1.0, "Correlate", "sync word, header and payload split"),
        (1.0, "Output", "decoded message, bits, JSON report"),
    ]),
    dict(id="L7", name=["TEST DATA", "(offline)"], bg="#fce3ed", bd="#9b1f52", boxes=[
        (1.0, "Synthetic transmitter", "known ground truth, noise and clock drift"),
        (1.0, "7 sample recordings", "every modulation, code and interleaver"),
        (1.0, "Regression", "7 / 7 samples · 42 / 42 configurations"),
    ]),
]

y = MARGIN
for L in LAYERS:
    L["h"] = BOX_H + 2 * BAND_PAD
    L["y"] = y
    y += L["h"] + GAP
H = y - GAP + MARGIN

parts = [
    f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    '<defs>'
    '<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    '<path d="M0 0 L10 5 L0 10 z" fill="#e8730c"/></marker>'
    '<marker id="ap" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    '<path d="M0 0 L10 5 L0 10 z" fill="#7b2fa0"/></marker>'
    '</defs>',
]

warn = []
for L in LAYERS:
    parts.append(f'<rect x="{BAND_L}" y="{L["y"]}" width="{BAND_R-BAND_L}" height="{L["h"]}" rx="7" fill="{L["bg"]}" stroke="{L["bd"]}" stroke-width="1.2" stroke-opacity="0.35"/>')
    ly = L["y"] + L["h"] / 2 - (len(L["name"]) - 1) * 9
    parts.append(f'<text x="{LBL_X}" y="{ly - 7}" font-family="Helvetica,Arial,sans-serif" font-size="15.5" font-weight="700" fill="#1b1b1b">{L["id"]}</text>')
    for i, nm in enumerate(L["name"]):
        parts.append(f'<text x="{LBL_X}" y="{ly + 11 + i * 16}" font-family="Helvetica,Arial,sans-serif" font-size="13" font-weight="700" fill="#1b1b1b">{nm}</text>')
    tot = sum(b[0] for b in L["boxes"])
    n = len(L["boxes"])
    avail = (CX1 - CX0) - 12 * (n - 1)
    x = CX0
    for wgt, title, sub in L["boxes"]:
        bw = avail * wgt / tot
        by = L["y"] + BAND_PAD
        parts.append(f'<rect x="{x:.1f}" y="{by}" width="{bw:.1f}" height="{BOX_H}" rx="5" fill="#ffffff" stroke="{L["bd"]}" stroke-width="1.6"/>')
        cx = x + bw / 2
        for txt, fs, bold, dy, col in ((title, FS_T, True, 34, "#111"), (sub, FS_S, False, 59, "#4a4a4a")):
            if wid(txt, fs, bold) > bw - 20:
                warn.append(f'{L["id"]}: "{txt}" {wid(txt, fs, bold):.0f}px in {bw:.0f}px box')
            wgt_attr = ' font-weight="700"' if bold else ''
            parts.append(f'<text x="{cx:.1f}" y="{by + dy}" font-family="Helvetica,Arial,sans-serif" font-size="{fs}"{wgt_attr} fill="{col}" text-anchor="middle">{esc(txt)}</text>')
        x += bw + 12

def down(a, b, fracs):
    y0 = LAYERS[a]["y"] + LAYERS[a]["h"]
    y1 = LAYERS[b]["y"]
    for f in fracs:
        xx = CX0 + (CX1 - CX0) * f
        parts.append(f'<path d="M{xx:.0f} {y0+1} V{y1-3}" stroke="#e8730c" stroke-width="2" fill="none" marker-end="url(#ao)"/>')

for a in range(5):
    down(a, a + 1, [0.22, 0.52, 0.82])

y7 = LAYERS[6]["y"] + LAYERS[6]["h"] / 2
y3 = LAYERS[2]["y"] + LAYERS[2]["h"] / 2
parts.append(f'<path d="M{BAND_R} {y7:.0f} H{LANE_R} V{y3:.0f} H{BAND_R}" stroke="#e8730c" stroke-width="2" fill="none" marker-end="url(#ao)"/>')
parts.append(f'<text x="{LANE_R}" y="{(y7+y3)/2:.0f}" font-family="Helvetica,Arial,sans-serif" font-size="11" font-weight="700" fill="#e8730c" text-anchor="middle" transform="rotate(-90 {LANE_R} {(y7+y3)/2:.0f})">sample recordings</text>')

y6 = LAYERS[5]["y"] + LAYERS[5]["h"] / 2
y2 = LAYERS[1]["y"] + LAYERS[1]["h"] / 2
parts.append(f'<path d="M{BAND_L} {y6:.0f} H{LANE_L} V{y2:.0f} H{BAND_L}" stroke="#7b2fa0" stroke-width="2" fill="none" stroke-dasharray="7 5" marker-end="url(#ap)"/>')
parts.append(f'<text x="{LANE_L}" y="{(y6+y2)/2:.0f}" font-family="Helvetica,Arial,sans-serif" font-size="11" font-weight="700" fill="#7b2fa0" text-anchor="middle" transform="rotate(-90 {LANE_L} {(y6+y2)/2:.0f})">results to the GUI</text>')

parts.append('</svg>')

open("/home/claude/proj/docs/architecture.html", "w").write(
    '<!doctype html><html><head><meta charset="utf-8">'
    '<style>html,body{margin:0;padding:0;background:#fff}svg{display:block}</style>'
    '</head><body>' + "\n".join(parts) + '</body></html>')
print("size", W, H, "aspect %.2f" % (W / H), "target", H_TARGET)
assert abs(H - H_TARGET) < 0.51, H
for w in warn:
    print("OVERFLOW", w)
if not warn:
    print("no overflow")
