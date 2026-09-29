# RF Signal Intelligence — progress

## How to run

```
venv\Scripts\activate
python scripts\generate_samples.py     # once: writes 7 test recordings into samples\
python scripts\decode_samples.py       # command line: decodes all 7, prints PASS/FAIL
streamlit run app.py                   # the GUI, opens in your browser
```

## Files

| File | Stage | What it does |
| --- | --- | --- |
| `app.py` | GUI | Streamlit app: signal parameters, five plots, and the full decoding results |
| `rfsig/utils.py` | shared | Bit helpers, preamble (`C8B3A9F43893`) and frame sync word (CCSDS ASM `1ACFFC1D`) |
| `rfsig/io.py` | input | Loads `.wav` (stereo = I/Q, mono = real passband) and raw `.iq` (cf32/ci16/ci8/cu8) |
| `rfsig/generator.py` | test data | Synthetic transmitter with known ground truth: FEC, interleaving, modulation, noise, drift |
| `rfsig/params.py` | 1. parameters | Spectrum, bandwidth, carrier offset, SNR, waterfall, symbol-rate estimation |
| `rfsig/modem.py` | shared | Constellations, Gray mapping, RRC pulse shaping, soft demapping |
| `rfsig/demod.py` | 2–3. modulation ID + demod | BPSK/QPSK/8PSK/16QAM and 2/4/8-FSK |
| `rfsig/framing.py` | 4. frame sync | Preamble correlation; blind framing (period + bit constancy) |
| `rfsig/interleave.py` | 5. de-interleaving | Block, diagonal, convolutional, pseudo-random |
| `rfsig/fec.py` | 5. FEC | Convolutional + Viterbi (K=3/5/7), Reed-Solomon, concatenated, LDPC |
| `rfsig/identify.py` | 5. blind ID | Scores every interleaver + code hypothesis by algebraic structure |
| `rfsig/correlate.py` | 6. correlation | Sync-word search, header parsing, header/payload split |
| `rfsig/pipeline.py` | all | Runs the stages end to end |
| `rfsig/report.py` | output | One-click PDF analyst report (reportlab, drawn from the arrays) |
| `docs/system-architecture.html` | docs | Interactive system architecture diagram |
| `docs/analyst-workflow.html` | docs | Interactive user-level workflow diagram |

## Status

Done:

1. Project scaffold, synthetic signal generator, sample files
2. Command-line regression check — 7/7 samples pass on this laptop
3. GUI part 1: signal parameters, spectrum, waterfall, constellation / tone clusters, eye diagram, symbol rate
4. GUI part 2: frame sync, interleaver + FEC identification with ranked hypotheses, decode quality, header/payload, decoded message, export
5. Architecture and workflow diagrams (`docs/`)

6. Blind-mode validation fix and `README.md` (setup, requirement mapping, how it works, demo script)

All five build steps are complete.

7. One-click PDF analyst report on the Export tab - parameters, spectrum, constellation,
   ranked hypotheses, header/payload, message and the stage log, in three pages.

## Test status

- 42 of 42 random configurations (every modulation, code and interleaver, 9-25 dB SNR, 7 of them in blind mode) decoded the full message end to end.
- Blind mode (unknown preamble), 9 targeted cases: 9 of 9. It is slower, and a pseudo-random interleaver in blind mode can take minutes.
- The 7 sample recordings: 7 of 7.
