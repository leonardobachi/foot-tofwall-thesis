# Optimization of the TOF-Wall detector for charged particle detection in the FOOT experiment

Analysis code of the Master's thesis of **Leonardo Bachi** (University of Pisa, 2026).

The code studies the TOF-Wall (TW) of the FOOT experiment with the data of the
CNAO 2025 campaign (¹²C beam at 200 MeV/u) and of the 2026 TW scans. It follows
the two lines of the thesis:

- **[`waveform_analysis/`](waveform_analysis)** — an independent reconstruction
  from the digitised waveforms of the Start Counter (ST) and of the TW: baseline,
  signal identification, CFD timing, charge, bar hits, charge identification and
  a template fit of the TW waveforms (also on artificially saturated waveforms).
- **[`tofwall_optimization/`](tofwall_optimization)** — studies on the SHOE
  results (SHOE is the FOOT reconstruction software): calibration of the position
  along the bar, charge equalisation along the bar and time resolution of the TW.

Each folder has its own README with the list of the scripts, the method, the
commands in order and the outputs. The header of every script describes its
input, output and method, and every script prints its options with `-h`.

Neither the data nor SHOE are included: both belong to the FOOT collaboration.

## Requirements

Python ≥ 3.9 with

```
pip install -r requirements.txt     # numpy, scipy, matplotlib, uproot, awkward
```

`uproot` and `awkward` are only needed by `tofwall_optimization/extract_shoe.py`.
The scripts of each folder import each other by module name, so they are run
from inside their folder.
