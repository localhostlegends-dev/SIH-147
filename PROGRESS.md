# RF Signal Intelligence — progress

## Done (in `rfsig/`)

| File | Stage | What it does |
| --- | --- | --- |
| `utils.py` | shared | Bit helpers, preamble (`C8B3A9F43893`) and frame sync word (CCSDS ASM `1ACFFC1D`) |
| `io.py` | input | Load `.wav` (stereo = I/Q, mono = real passband) and raw `.iq` (cf32/ci16/ci8/cu8), sampling rate from header or file name |
| `generator.py` | test data | Synthetic transmitter with known ground truth: FEC, interleaving, modulation, noise, carrier offset, clock drift |
| `params.py` | 1. parameters | Spectrum, bandwidth, carrier offset, SNR, waterfall, symbol-rate estimation |
| `modem.py` | shared | Constellations, Gray mapping, RRC pulse shaping, soft demapping |
| `demod.py` | 2–3. modulation ID + demod | BPSK/QPSK/8PSK/16QAM (matched filter, timing, carrier recovery) and 2/4/8-FSK (tone filter bank) |
| `framing.py` | 4. frame sync | Preamble correlation; blind framing (period + bit constancy) |
| `interleave.py` | 5. de-interleaving | Block, diagonal, convolutional, pseudo-random |
| `fec.py` | 5. FEC | Convolutional + Viterbi (K=3/5/7), Reed-Solomon, concatenated RS+conv, LDPC (min-sum) |
| `identify.py` | 5. blind ID | Tries every interleaver + code and scores algebraic structure |
| `correlate.py` | 6. correlation | Sync-word search, header parsing, bit-constancy header/payload split |
| `pipeline.py` | all | Runs the stages end to end |

## Test status

- Known-preamble mode: 41 of 42 random configurations (all modulations × FEC × interleavers) decoded the full message. The only failure was an impossible test setting: an 8FSK signal wider than its sampling rate.
- Blind mode (unknown preamble): 7 of 8 cases passed. It's slower, and pseudo-random interleavers take about 5 minutes.

## Next steps

1. Tighten blind-mode validation (the sync word must open the decoded frame). This edit was stopped midway and has not been applied.
2. Streamlit GUI (`app.py`): upload, spectrum/waterfall/constellation/eye plots, results tables.
3. Sample files script + tests + README with Windows setup.
