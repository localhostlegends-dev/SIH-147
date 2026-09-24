# RF Signal Intelligence

A GUI tool that takes a raw radio recording (`.wav` or `.IQ`), works out how the signal
was built, and decodes the message inside it — sampling rate, modulation, interleaving
and error-correction code are all identified automatically.

Runs entirely offline on a laptop. Pure Python: no GPU, no internet, no licensed software.

---

## Quick start (Windows)

Needs Python 3.10 or newer ([python.org](https://www.python.org/downloads/); tick
"Add Python to PATH" during install).

```
cd C:\Users\<you>\...\26147
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python scripts\generate_samples.py     # writes 7 test recordings into samples\
streamlit run app.py                   # opens the GUI in your browser
```

Run `venv\Scripts\activate` in every new Command Prompt. Stop the GUI with Ctrl+C.

To check the whole pipeline from the command line instead:

```
python scripts\decode_samples.py                  # all 7 samples, PASS / FAIL each
python scripts\decode_samples.py sig05_hf_audio.wav   # one file, stage by stage
```

---

## What it does, against the problem statement

| Required | Status | Where |
| --- | --- | --- |
| Accept `.IQ` or `.wav` input | Yes — stereo WAV as I/Q, mono WAV as real passband, raw IQ as cf32 / ci16 / ci8 / cu8 (RTL-SDR) | `rfsig/io.py` |
| Identify sampling frequency | From the WAV header, a hint in the file name, or the analyst for headerless IQ | `rfsig/io.py` |
| Identify modulation | BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, 8FSK | `rfsig/demod.py` |
| Identify FEC | Convolutional (K=3/5/7), Reed-Solomon, concatenated RS + convolutional, LDPC, or uncoded | `rfsig/identify.py`, `rfsig/fec.py` |
| Identify interleaving | Block, diagonal, convolutional (Forney), pseudo-random — including the exact size or seed | `rfsig/identify.py`, `rfsig/interleave.py` |
| Demodulate FSK / PSK / QAM | Matched filtering, timing recovery, carrier recovery, soft-decision output | `rfsig/demod.py` |
| De-interleave | All four families above | `rfsig/interleave.py` |
| FEC decoding | Viterbi (soft decision), Reed-Solomon, concatenated, LDPC min-sum belief propagation | `rfsig/fec.py` |
| Bitstream correlation for header / payload | Sync-word cross-correlation plus a blind bit-constancy method | `rfsig/correlate.py` |
| Spectral features in a GUI | Spectrum, waterfall, constellation, eye diagram, tone clusters, symbol-rate line | `app.py` |

---

## The sample recordings

`python scripts\generate_samples.py` writes seven files that between them exercise every
stage. The true settings are recorded in `samples\manifest.json`.

| File | Modulation | Error correction | Interleaver | Format |
| --- | --- | --- | --- | --- |
| sig01 | BPSK | Convolutional K=7 | Block 28×37 | .iq float32 |
| sig02 | QPSK | Reed-Solomon + convolutional | Convolutional B=8 J=4 | .iq int16 |
| sig03 | 8PSK | Convolutional K=5 | Diagonal 24×43 | stereo .wav (I/Q) |
| sig04 | 16QAM | Reed-Solomon | Pseudo-random, seed 42 | .iq float32 |
| sig05 | 2FSK | Convolutional K=3 | Block 4×257 | mono .wav (HF audio) |
| sig06 | 4FSK | LDPC | none | .iq float32 |
| sig07 | QPSK | none | none | .iq uint8 (RTL-SDR) |

Each carries the same hidden text message, split across 16 frames. Recovering that text
is the end-to-end proof that every stage worked.

---

## Using the GUI

**Sidebar** — pick a sample or upload your own recording. Raw `.iq` files have no header,
so the sample format and sampling rate are set here (pre-filled from the file name when it
contains something like `fs48000`). WAV files read their own header. "Ground truth" shows
the real settings of a sample so you can compare.

**Signal parameters** — sampling rate, symbol rate, modulation, decision SNR, occupied
bandwidth, carrier offset, samples per symbol, duration. The analysis log below lists what
each stage found, in order.

**Plot tabs**

| Tab | Shows |
| --- | --- |
| Spectrum | Power against frequency, with the occupied band and noise floor marked |
| Waterfall | Frequency against time, colour = power |
| Constellation | Symbols before carrier recovery (a spinning ring) and after (tight clusters on the ideal points) |
| Eye diagram | Overlaid symbol traces; an open eye means the timing is right |
| Tone clusters / Instantaneous frequency | For FSK: the separate tones, and the signal hopping between them |
| Symbol rate | The spectral line the symbol rate was measured from |

**Decoding tabs**

| Tab | Shows |
| --- | --- |
| Frame sync | Preamble correlation along the bitstream, one spike per frame, and which phase rotation was correct |
| Interleaver + FEC | The winning hypothesis, how many were tested, and a ranked chart of the runners-up |
| Decode quality | Bit error rate before correction, Reed-Solomon bytes repaired, LDPC iterations |
| Header / payload | Bit constancy across frames, the estimated header boundary, and parsed header fields per frame |
| Message | The recovered text |
| Export | Decoded bits, payload, and a JSON report |

---

## How it works

1. **Load.** Samples are converted to complex baseband. A mono WAV is treated as a real
   passband recording and turned into an analytic signal with the Hilbert transform.
2. **Measure.** Welch power spectral density gives the noise floor, the 99 % occupied
   bandwidth, the carrier offset and an SNR estimate.
3. **Symbol rate.** A digitally modulated signal repeats its statistics once per symbol,
   so a non-linear transform produces a spectral line exactly at the symbol rate: the
   squared envelope |x|² for PSK/QAM, the rate of frequency jumps for FSK.
4. **Modulation.** For linear modulations, raising the symbols to the power M leaves a
   tone only for the matching scheme (M=2 for BPSK, 4 for QPSK and 16QAM, 8 for 8PSK);
   QPSK and 16QAM are separated by the noise-corrected C42 cumulant and amplitude spread.
   FSK is identified by clustering the instantaneous frequency into equally spaced tones.
5. **Demodulate.** Root-raised-cosine matched filter, Oerder & Meyr feed-forward timing
   recovery (block-wise, so clock drift is tracked), carrier recovery from the M-th power
   line plus a Viterbi & Viterbi phase estimator, then soft (log-likelihood) demapping.
   FSK uses a non-coherent tone filter bank.
6. **Frame sync.** The 48-bit preamble is cross-correlated along the bitstream. Because a
   PSK/QAM receiver cannot tell which constellation point is "north", every rotation and a
   spectrum flip are tried and the preamble picks the winner. With no known preamble the
   tool finds the frame period from the bitstream autocorrelation and the preamble from
   bit positions that stay constant across frames.
7. **Identify interleaver and code.** Every plausible interleaver (matrix shapes,
   convolutional B/J settings, pseudo-random seeds) is applied, and each result is scored
   against every candidate code using algebraic structure: the rate-1/2 parity relation
   c1·g2 + c2·g1 = 0 for convolutional codes, syndromes and Berlekamp-Massey error-locator
   length for Reed-Solomon, unsatisfied parity checks for LDPC. Wrong guesses score near
   zero, so the right one stands out. Thousands of hypotheses are checked in about a second.
8. **Decode.** Soft-decision Viterbi, Reed-Solomon, concatenated (inner Viterbi then outer
   RS) or LDPC min-sum belief propagation.
9. **Correlate.** The frame sync word is located by cross-correlation, the header fields
   parsed, and the payload extracted. Independently, stacking the frames and measuring how
   often each bit position repeats reveals the header/payload boundary without knowing the
   format at all.

---

## Test results

| Test | Result |
| --- | --- |
| The 7 sample recordings (`scripts\decode_samples.py`) | 7 / 7 — modulation, interleaver, code and message all correct |
| 42 random configurations across every modulation, code and interleaver, 9–25 dB SNR, including 7 in blind mode | 42 / 42 |
| Blind mode (unknown preamble), 9 targeted cases | 9 / 9 |

Typical analysis time is 1–5 seconds per recording on a laptop.

---

## Limits and honest notes

- The tool recognises the code families listed above with the parameters it knows. A real
  deployment would carry a larger library (other rates, puncturing, DVB/802.11 LDPC tables).
  Adding a code means adding an encoder, a decoder and a structure test in `rfsig/fec.py`.
- A raw `.IQ` file carries no sampling rate. The tool takes it from the file name or from
  the analyst; everything downstream is measured relative to it.
- Blind mode (unknown preamble) is slower because it must also guess where frames begin.
  A pseudo-random interleaver in blind mode can take minutes.
- Very low SNR (below about 8 dB for BPSK, more for higher-order schemes) breaks
  identification, as it would for any receiver.
- Signals are analysed offline from a file. There is no live SDR capture path.

---

## Project layout

```
app.py                      Streamlit GUI
requirements.txt            Python dependencies
rfsig/                      the analysis library
  io.py                     file loading (.wav, raw .iq)
  params.py                 spectrum, bandwidth, SNR, symbol rate
  modem.py                  constellations, Gray mapping, pulse shaping
  demod.py                  modulation ID and demodulation
  framing.py                frame synchronisation
  interleave.py             the four interleaver families
  fec.py                    convolutional/Viterbi, Reed-Solomon, concatenated, LDPC
  identify.py               blind interleaver + code identification
  correlate.py              sync word, header/payload
  pipeline.py               runs the stages end to end
  generator.py              synthetic transmitter used to make test signals
  utils.py                  bit helpers and constants
scripts/
  generate_samples.py       writes the 7 sample recordings
  decode_samples.py         decodes them and checks against the ground truth
docs/
  system-architecture.html  interactive architecture diagram
  analyst-workflow.html     interactive user-workflow diagram
samples/                    generated recordings + manifest.json (ground truth)
```

Dependencies: numpy, scipy, reedsolo, streamlit, plotly. Everything else — Viterbi
decoding, LDPC, the interleavers, the identification search — is implemented in this
repository.

---

## Demo script (about 3 minutes)

1. **Open `sig01_fs48000.iq`.** Say: this is a raw recording, no metadata at all. Point at
   the parameter row — bandwidth, symbol rate and modulation were all measured from the
   samples.
2. **Constellation tab.** Left plot is a ring: the receiver has not locked the carrier yet.
   Right plot snaps onto two BPSK points. That is demodulation working.
3. **Interleaver + FEC tab.** 1,859 hypotheses tested. The right answer scores 1.00; the
   next best scores 0.24. Nobody told the tool the code — it proved it.
4. **Message tab.** The text comes out. Every stage had to be right for this to read.
5. **Switch to `sig04`** (pseudo-random interleaver, seed 42) and say: this one was
   scrambled with a secret seed; the tool searched 256 of them and found it in about a
   second.
6. **Switch to `sig05_hf_audio.wav`** and say: this is not I/Q at all, it is audio from an
   HF receiver, and the same pipeline decodes it.
