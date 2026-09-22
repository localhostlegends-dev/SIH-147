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
from rfsig.pipeline import AnalysisOptions, run

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
st.caption("Decoding results (interleaver, FEC, header / payload, decoded message) are added in the next step.")
