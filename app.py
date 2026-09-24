"""RF Signal Intelligence - Streamlit GUI.

Run from the project folder:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from rfsig.io import IQ_FORMATS, guess_fs_from_name, load_signal
from rfsig.modem import constellation
from rfsig.params import waterfall
from rfsig.fec import FEC_CODES
from rfsig.pipeline import AnalysisOptions, run
from rfsig.utils import bits_to_hex

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
UPLOAD_TYPES = ["wav", "iq", "cfile", "cf32", "bin", "raw", "dat", "cu8", "ci16", "ci8"]

# chart colours (validated reference palette: slot 1 blue, slot 2 orange)
C_MAIN = "#2a78d6"
C_ACCENT = "#eb6834"
C_MUTED = "rgba(128,128,128,0.35)"

st.set_page_config(page_title="RF Signal Intelligence", page_icon="📡", layout="wide")


# ----------------------------------------------------------------------------
# analysis (cached per file + settings)
# ----------------------------------------------------------------------------
@st.cache_resource(max_entries=8, show_spinner=False)
def analyse(name: str, data: bytes, fmt: str, fs: float | None, swap: bool):
    sig = load_signal(name, data, iq_format=fmt, fs=fs, swap_iq=swap)
    res = run(sig, AnalysisOptions())
    wf = waterfall(sig.samples[: res.options.max_samples], sig.fs)
    return res, wf


def base_layout(fig: go.Figure, x: str, y: str, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x, yaxis_title=y,
                      showlegend=True, legend=dict(orientation="h", y=1.08, x=0), hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor=C_MUTED, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=C_MUTED, zeroline=False)
    return fig


def fmt_hz(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.3f} MHz"
    if a >= 1e3:
        return f"{v / 1e3:.2f} kHz"
    return f"{v:.1f} Hz"


# ----------------------------------------------------------------------------
# sidebar: input selection
# ----------------------------------------------------------------------------
st.sidebar.title("📡 Input")
manifest = {}
if (SAMPLES / "manifest.json").exists():
    manifest = json.loads((SAMPLES / "manifest.json").read_text())
sample_files = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in (".iq", ".wav")) \
    if SAMPLES.exists() else []

source = st.sidebar.radio("Source", ["Sample file", "Upload a recording"], horizontal=False)
name, data = None, None
if source == "Sample file":
    if not sample_files:
        st.sidebar.warning("No samples found. Run:\n\n`python scripts/generate_samples.py`")
    else:
        name = st.sidebar.selectbox("Recording", sample_files)
        data = (SAMPLES / name).read_bytes()
else:
    up = st.sidebar.file_uploader("Choose a .wav or raw .iq file", type=UPLOAD_TYPES)
    if up is not None:
        name, data = up.name, up.getvalue()

fmt, fs, swap = "cf32", None, False
if name and not name.lower().endswith(".wav"):
    st.sidebar.subheader("Raw IQ settings")
    st.sidebar.caption("Raw .iq files have no header, so the sample format and rate must be given.")
    default_fmt = manifest.get(name, {}).get("file_format", "cu8" if "rtl" in name.lower() else "cf32")
    keys = list(IQ_FORMATS)
    fmt = st.sidebar.selectbox("Sample format", keys, index=keys.index(default_fmt) if default_fmt in keys else 0,
                               format_func=lambda k: f"{k} - {IQ_FORMATS[k]}")
    guess = guess_fs_from_name(name)
    fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=1000.0, value=float(guess or 48000.0),
                                 step=1000.0, format="%.0f",
                                 help="Pre-filled from the file name when it contains e.g. fs48000.")
    swap = st.sidebar.checkbox("Swap I and Q", value=False)
elif name:
    st.sidebar.caption("WAV: sampling rate is read from the header. Stereo = I/Q, mono = real audio.")

if name and name in manifest:
    t = manifest[name]
    with st.sidebar.expander("Ground truth (for testing)"):
        st.write({"modulation": t["modulation"], "symbol rate": t["symbol_rate"],
                  "FEC": t["fec"], "interleaver": t["interleaver"], "Es/N0 (dB)": round(t["snr_db"], 1)})

# ----------------------------------------------------------------------------
# main area
# ----------------------------------------------------------------------------
st.title("RF Signal Intelligence")
st.caption("Signal parameter extraction · demodulation · de-interleaving · FEC decoding · bitstream correlation")

if not name:
    st.info("Pick a sample file or upload a recording in the sidebar to start.")
    st.stop()

with st.spinner(f"Analysing {name} ..."):
    res, (wf_f, wf_t, wf_db) = analyse(name, data, fmt, fs, swap)

if res.error:
    st.error(f"Analysis stopped: {res.error}")

sig, sp, d = res.signal, res.spectral, res.demod

# ---- key parameters ----------------------------------------------------------
st.subheader("Signal parameters")
c = st.columns(4)
c[0].metric("Sampling rate", fmt_hz(sig.fs), help=f"Source: {sig.fs_source}")
c[1].metric("Symbol rate", f"{res.symbol_rate:,.1f} Bd" if d else "-",
            help="Estimated from the cyclic spectral line (see the 'Symbol rate' tab)")
c[2].metric("Modulation", d.modulation if d else "-",
            help=f"Classifier confidence {d.extra.get('confidence', 0):.2f}" if d else None)
c[3].metric("Decision SNR", f"{d.snr_db:.1f} dB" if d else "-", help="From the spread of the symbols around their ideal points")
c = st.columns(4)
c[0].metric("Occupied bandwidth", fmt_hz(sp.bandwidth_hz) if sp else "-", help="99 % power bandwidth")
c[1].metric("Carrier offset", fmt_hz(sp.center_offset_hz) if sp else "-", help="Centre of the signal relative to 0 Hz")
c[2].metric("Samples / symbol", f"{sig.fs / res.symbol_rate:.2f}" if d else "-")
c[3].metric("Duration", f"{sig.duration:.2f} s", help=f"{len(sig.samples):,} samples · {sig.source}")

with st.expander("Analysis log (what each stage found)"):
    for stage, text in res.log:
        st.markdown(f"**{stage}** — {text}")
    if res.timings:
        st.caption(" · ".join(f"{k}: {v:.2f}s" for k, v in res.timings.items()))

# ---- plots ---------------------------------------------------------------------
is_fsk = d is not None and d.family == "fsk"
tabs = st.tabs(["Spectrum", "Waterfall", "Tone clusters" if is_fsk else "Constellation",
                "Instantaneous frequency" if is_fsk else "Eye diagram", "Symbol rate"])

with tabs[0]:
    fig = go.Figure(go.Scatter(x=sp.psd_freqs, y=sp.psd_db, mode="lines", line=dict(color=C_MAIN, width=2),
                               name="Power spectral density", hovertemplate="%{x:.1f} Hz<br>%{y:.1f} dB<extra></extra>"))
    fig.add_vrect(x0=sp.band_lo_hz, x1=sp.band_hi_hz, fillcolor=C_MAIN, opacity=0.08, line_width=0,
                  annotation_text="occupied band", annotation_position="top left")
    fig.add_hline(y=sp.noise_floor_db, line=dict(color=C_ACCENT, width=2, dash="dash"),
                  annotation_text="noise floor", annotation_position="bottom right")
    st.plotly_chart(base_layout(fig, "Frequency (Hz)", "Power (dB)"), width="stretch")
    st.caption(f"Signal occupies {fmt_hz(sp.bandwidth_hz)} centred at {fmt_hz(sp.center_offset_hz)}; "
               f"spectral SNR ≈ {sp.snr_db:.1f} dB.")

with tabs[1]:
    fig = go.Figure(go.Heatmap(x=wf_f, y=wf_t, z=wf_db.T, colorscale="Blues", reversescale=False,
                               zmin=float(np.percentile(wf_db, 5)), zmax=float(np.percentile(wf_db, 99.5)),
                               colorbar=dict(title="dB"),
                               hovertemplate="%{x:.1f} Hz<br>t = %{y:.3f} s<br>%{z:.1f} dB<extra></extra>"))
    fig = base_layout(fig, "Frequency (Hz)", "Time (s)", 480)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Waterfall: frequency (x) against time (y, top = start). Colour = power.")

with tabs[2]:
    if d is None:
        st.info("No demodulation result.")
    elif not is_fsk:
        left, right = st.columns(2)
        raw = d.extra.get("raw_symbols", d.symbols)
        idx = np.linspace(0, len(d.symbols) - 1, min(4000, len(d.symbols))).astype(int)
        for col, pts, title in ((left, raw, "Before carrier recovery"), (right, d.symbols, "After carrier recovery")):
            col.markdown(f"**{title}**")
            fig = go.Figure(go.Scattergl(x=np.real(pts[idx]), y=np.imag(pts[idx]), mode="markers",
                                         marker=dict(size=4, color=C_MAIN, opacity=0.35), name="received symbols",
                                         hovertemplate="I %{x:.2f}<br>Q %{y:.2f}<extra></extra>"))
            if title.startswith("After"):
                ref = constellation(d.modulation)
                fig.add_trace(go.Scattergl(x=ref.real, y=ref.imag, mode="markers", name=f"ideal {d.modulation}",
                                         marker=dict(size=11, color=C_ACCENT, symbol="x", line=dict(width=2))))
            lim = float(np.percentile(np.abs(pts[idx]), 99.5)) * 1.15
            fig = base_layout(fig, "In-phase (I)", "Quadrature (Q)", 420)
            fig.update_xaxes(range=[-lim, lim])
            fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1)
            col.plotly_chart(fig, width="stretch")
        st.caption(f"Left: symbols after matched filter + timing recovery (still spinning from the "
                   f"{d.freq_offset_hz:+.1f} Hz residual carrier offset). Right: after carrier recovery, "
                   f"clustering on the {d.modulation} points.")
    else:
        tones = d.extra.get("tones_hz", [])
        fig = go.Figure(go.Histogram(x=d.symbols, nbinsx=120, marker_color=C_MAIN, name="symbols",
                                     hovertemplate="%{x:.0f} Hz: %{y} symbols<extra></extra>"))
        for i, tf in enumerate(tones):
            fig.add_vline(x=tf, line=dict(color=C_ACCENT, width=2, dash="dash"),
                          annotation_text=f"tone {i}", annotation_position="top")
        fig = base_layout(fig, "Average frequency over one symbol (Hz)", "Symbols")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(tones)} tones found, spacing {d.extra.get('tone_spacing_hz', 0):.1f} Hz, "
                   f"modulation index h ≈ {d.extra.get('mod_index_h', 0):.2f}.")

with tabs[3]:
    if d is None:
        st.info("No demodulation result.")
    elif not is_fsk:
        eye = d.extra.get("eye")
        offs = d.extra.get("eye_offsets")
        if eye is not None:
            xs = np.concatenate([np.append(offs, np.nan) for _ in range(len(eye))])
            ys = np.concatenate([np.append(e.real, np.nan) for e in eye])
            fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=C_MAIN, width=1),
                                         opacity=0.35, name="I channel traces", hoverinfo="skip"))
            fig.add_vline(x=0, line=dict(color=C_ACCENT, width=2, dash="dash"),
                          annotation_text="sampling instant", annotation_position="top")
            st.plotly_chart(base_layout(fig, "Time (symbols)", "Amplitude (I)", 420), width="stretch")
            st.caption("Eye diagram: many 2-symbol slices overlaid. A wide-open 'eye' at the sampling instant "
                       "means timing recovery found the right moment to sample.")
    else:
        fi = d.extra.get("inst_freq")
        if fi is not None:
            sps = sig.fs / res.symbol_rate
            n = min(len(fi), int(40 * sps))
            k = max(int(sps / 4), 1)                      # light smoothing (noise "clicks")
            fs_s = np.convolve(fi[:n + k], np.ones(k) / k, mode="same")[:n]
            t = np.arange(n) / sig.fs * 1e3
            fig = go.Figure(go.Scattergl(x=t, y=fs_s, mode="lines", line=dict(color=C_MAIN, width=1.5),
                                         name="instantaneous frequency (smoothed)"))
            tones = d.extra.get("tones_hz", [])
            for tf in tones:
                fig.add_hline(y=tf, line=dict(color=C_ACCENT, width=1.5, dash="dot"))
            fig = base_layout(fig, "Time (ms)", "Frequency (Hz)", 420)
            if tones:
                pad = max(d.extra.get("tone_spacing_hz", 0), 1.0)
                fig.update_yaxes(range=[min(tones) - pad, max(tones) + pad])
            st.plotly_chart(fig, width="stretch")
            st.caption("The first 40 symbols: the frequency jumps between the tones (dotted lines) once per symbol.")

with tabs[4]:
    r = res.rs_fsk if is_fsk else res.rs_linear
    if r is not None and r.spec_freqs is not None:
        mag = r.spec_mag / (np.max(r.spec_mag) or 1)
        fig = go.Figure(go.Scatter(x=r.spec_freqs, y=mag, mode="lines", line=dict(color=C_MAIN, width=2),
                                   name=r.method, hovertemplate="%{x:.1f} Hz<extra></extra>"))
        fig.add_vline(x=res.symbol_rate, line=dict(color=C_ACCENT, width=2, dash="dash"),
                      annotation_text=f"Rs = {res.symbol_rate:,.1f} Bd", annotation_position="top")
        st.plotly_chart(base_layout(fig, "Frequency (Hz)", "Relative magnitude"), width="stretch")
        st.caption("A digital signal repeats its statistics once per symbol, so a non-linear transform "
                   + ("(frequency jumps)" if is_fsk else "(|x|², the squared envelope)")
                   + " produces a sharp spectral line exactly at the symbol rate.")

st.divider()

# ============================================================================
# Decoding results
# ============================================================================
st.subheader("Decoding")

fr, ident, corr = res.framing, res.ident, res.corr
if fr is None:
    st.warning("No frames were found, so nothing could be decoded. "
               "Try setting the modulation or symbol rate by hand in the sidebar.")
    st.stop()

fec_label = FEC_CODES[res.fec_name].label if res.fec_name else "-"
c = st.columns(4)
c[0].metric("Frames found", len(fr.starts), help=f"{fr.period} bits per frame · {fr.mode}")
c[1].metric("Interleaver", ident.best.interleaver.describe() if ident and ident.best else "-")
c[2].metric("Error correction", res.fec_name or "-", help=fec_label)
c[3].metric("Message recovered", f"{len(corr.message)} chars" if corr else "-")

dtabs = st.tabs(["Frame sync", "Interleaver + FEC", "Decode quality", "Header / payload",
                 "Message", "Export"])

# ---- frame sync ------------------------------------------------------------
with dtabs[0]:
    if fr.corr_trace is not None:
        step = max(len(fr.corr_trace) // 40000, 1)
        x = np.arange(0, len(fr.corr_trace), step)
        fig = go.Figure(go.Scattergl(x=x, y=fr.corr_trace[::step], mode="lines",
                                     line=dict(color=C_MAIN, width=1), name="preamble correlation",
                                     hovertemplate="bit %{x}<br>%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=fr.starts, y=fr.peak_values, mode="markers", name="frame start",
                                   marker=dict(size=9, color=C_ACCENT, symbol="circle-open",
                                               line=dict(width=2))))
        fig.add_hline(y=0.75, line=dict(color=C_ACCENT, width=1.5, dash="dot"),
                      annotation_text="detection threshold", annotation_position="bottom right")
        st.plotly_chart(base_layout(fig, "Bit position in the demodulated stream", "Correlation"),
                        width="stretch")
        st.caption(f"The 48-bit preamble is slid along the bitstream. Each peak is one frame: "
                   f"{len(fr.starts)} found, every {fr.period} bits. Carrier-phase ambiguity resolved "
                   f"as '{fr.hypothesis}'.")
    elif fr.constancy is not None:
        fig = go.Figure(go.Scattergl(x=np.arange(len(fr.constancy)), y=fr.constancy, mode="lines",
                                     line=dict(color=C_MAIN, width=1), name="bit constancy"))
        st.plotly_chart(base_layout(fig, "Bit position in the frame", "Fraction of frames agreeing"),
                        width="stretch")
        st.caption("Blind framing: the frame period comes from the autocorrelation of the bitstream, "
                   "and the flat region at the start is the repeated preamble.")
    if fr.hypothesis_scores:
        with st.expander("Why this phase rotation"):
            rows = [{"hypothesis": k, "score": round(v, 2)} for k, v in
                    sorted(fr.hypothesis_scores.items(), key=lambda kv: -kv[1])]
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("A PSK/QAM receiver cannot tell which constellation point is 'north'. Every "
                       "rotation (and a spectrum flip) is tried; the preamble picks the winner.")

# ---- interleaver + FEC identification --------------------------------------
with dtabs[1]:
    if not (ident and ident.best):
        st.warning(ident.note if ident else "Identification did not run.")
    else:
        c = st.columns(3)
        c[0].metric("Identification score", f"{ident.best.score:.3f}", help="1.0 = perfect code structure")
        c[1].metric("Hypotheses tested", f"{ident.n_tested:,}")
        c[2].metric("Coded block", f"{ident.best.coded_len} bits",
                    help=f"{ident.best.pad} pad bits at the end of each frame")
        st.markdown(f"**Winner:** {ident.best.interleaver.describe()} · {fec_label}")
        rows = [c_.as_row() for c_ in ident.ranked[:10]]
        top = rows[:8][::-1]
        fig = go.Figure(go.Bar(x=[r["score"] for r in top],
                               y=[f'{r["interleaver"]} · {r["fec"].split(" (")[0]}' for r in top],
                               orientation="h", marker_color=C_MAIN, name="structure score",
                               text=[f'{r["score"]:.2f}' for r in top], textposition="outside",
                               hovertemplate="%{y}<br>score %{x:.3f}<extra></extra>"))
        fig = base_layout(fig, "Structure score (1.0 = fits perfectly)", "", 420)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=40, t=30, b=10))
        fig.update_xaxes(range=[0, 1.12])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Every interleaver shape, convolutional setting and pseudo-random seed is tried "
                   "against every candidate code. A wrong guess scores near 0 because the parity "
                   "relations of a real code only hold for the right one, so the winner is unambiguous.")

# ---- decode quality --------------------------------------------------------
with dtabs[2]:
    bers = [s.get("channel_ber") for s in res.frame_stats if s.get("channel_ber") is not None]
    corrected = [s.get("corrected_symbols") for s in res.frame_stats
                 if s.get("corrected_symbols") is not None]
    iters = [s.get("iterations") for s in res.frame_stats if s.get("iterations") is not None]
    ok_frames = sum(1 for s in res.frame_stats if s.get("ok"))
    c = st.columns(4)
    c[0].metric("Frames decoded", f"{ok_frames}/{len(res.frame_stats)}")
    c[1].metric("Channel BER before FEC", f"{np.mean(bers):.2e}" if bers else "-",
                help="Estimated by re-encoding the decoded bits and comparing with what was received")
    c[2].metric("RS symbols corrected", f"{int(np.sum(corrected))}" if corrected else "-",
                help="Byte errors repaired by the Reed-Solomon decoder")
    c[3].metric("LDPC iterations", f"{np.mean(iters):.1f}" if iters else "-")
    series = bers or corrected or iters
    if series:
        name = "channel BER" if bers else ("RS symbols corrected" if corrected else "LDPC iterations")
        fig = go.Figure(go.Bar(x=np.arange(len(series)), y=series, marker_color=C_MAIN, name=name,
                               hovertemplate="frame %{x}<br>%{y}<extra></extra>"))
        fig = base_layout(fig, "Frame", name, 340)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    st.caption("Errors the channel introduced, and what the error-correction decoder repaired. "
               "The message below is what survived that repair.")

# ---- header / payload ------------------------------------------------------
with dtabs[3]:
    if corr is None or not corr.frames:
        st.info("No decoded frames to correlate.")
    else:
        prof = corr.constancy
        fig = go.Figure(go.Scattergl(x=np.arange(len(prof)), y=prof, mode="lines",
                                     line=dict(color=C_MAIN, width=1.5), name="bit constancy",
                                     hovertemplate="bit %{x}<br>%{y:.2f}<extra></extra>"))
        fig.add_vline(x=corr.header_len_est, line=dict(color=C_ACCENT, width=2, dash="dash"),
                      annotation_text="header ends", annotation_position="top")
        fig.add_hline(y=0.5, line=dict(color=C_MUTED, width=1, dash="dot"))
        fig = base_layout(fig, "Bit position in the decoded frame", "Fraction of frames agreeing", 360)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0.4, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Stack the decoded frames and ask how often each bit position repeats. The header "
                   f"stays constant (sync word, type, length), the frame counter changes slowly and the "
                   f"payload looks random, which puts the boundary near bit {corr.header_len_est} "
                   f"without knowing the frame format.")
        rows = []
        for f in corr.frames:
            rows.append({"frame": f["frame"], "sync at bit": f["sync_pos"], "sync match": f["sync_corr"],
                         "counter": f.get("counter", "-"),
                         "type": f"0x{f['type']:02X}" if "type" in f else "-",
                         "length": f.get("length", "-"),
                         "payload": (f.get("payload_text", "") or "")[:44]})
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"Sync word {corr.sync_hex} located by cross-correlation in every frame, then the "
                   f"header fields are read straight off the decoded bits.")

# ---- message ---------------------------------------------------------------
with dtabs[4]:
    if corr and corr.message.strip():
        st.success(f"{len(corr.message)} characters recovered from {len(corr.frames)} frames")
        st.code(corr.message, language=None, wrap_lines=True)
    else:
        st.info("No readable payload was recovered. The bits are still available in the Export tab.")

# ---- export ----------------------------------------------------------------
with dtabs[5]:
    bits = np.concatenate(res.info_frames) if res.info_frames else np.zeros(0, dtype=np.uint8)
    report = {
        "summary": res.summary(),
        "log": [{"stage": s, "detail": t} for s, t in res.log],
        "timings_s": res.timings,
        "framing": {"mode": fr.mode, "frames": int(len(fr.starts)), "frame_bits": int(fr.period),
                    "block_bits": int(fr.block_len), "ambiguity": fr.hypothesis,
                    "preamble_hex": bits_to_hex(fr.preamble_bits)},
        "identification": {"best": ident.best.as_row() if ident and ident.best else None,
                           "hypotheses_tested": ident.n_tested if ident else 0,
                           "ranked": [c_.as_row() for c_ in (ident.ranked[:10] if ident else [])]},
        "frames": corr.frames if corr else [],
        "message": corr.message if corr else "",
    }
    stem = Path(name).stem
    c = st.columns(3)
    c[0].download_button("Decoded bits (.txt)", "".join(map(str, bits.tolist())),
                         file_name=f"{stem}_bits.txt", width="stretch")
    payload = "".join(f.get("payload_text", "") for f in (corr.frames if corr else []))
    c[1].download_button("Payload text (.txt)", payload or corr.message if corr else "",
                         file_name=f"{stem}_payload.txt", width="stretch")
    c[2].download_button("Analysis report (.json)", json.dumps(report, indent=2, default=str),
                         file_name=f"{stem}_report.json", width="stretch")
    st.caption(f"{len(bits):,} decoded information bits across {len(res.info_frames)} frames. "
               f"The JSON report holds every parameter, the ranked hypotheses and the parsed frames.")
