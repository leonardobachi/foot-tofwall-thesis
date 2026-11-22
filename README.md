# Optimization of the TOF-Wall detector for charged particle detection in the FOOT experiment

Analysis code of the master's thesis of Leonardo Bachi (University of Pisa, Academic Year 2025–2026).

The code analyzes the data of the TOF-Wall (TW) and of the Start Counter (ST) of the FOOT experiment,
acquired at CNAO in 2025 with a 12C beam at 200 MeV/u, together with dedicated scan data of the TW.

## Contents

The repository contains Python scripts and shell scripts for:

- **Waveform analysis**: reading of the raw waveforms of the WaveDAQ system, baseline subtraction,
  signal identification, charge and arrival time (CFD algorithms), following the same algorithms
  as SHOE, the reconstruction software of FOOT;
- **Hit and point reconstruction**: combination of the two ends of each bar into hits, TOF and
  energy calibrations, charge identification and matching of the two layers into points;
- **Template fit**: construction of the waveform templates and fit of the waveforms to measure
  their arrival time and charge, also for saturated waveforms;
- **Time resolution**: comparison of the timing methods and study of the time resolution of the TW
  as a function of the velocity and of the energy loss of the fragments;
- **Calibrations**: calibration of the hit position along the bars and equalization of the charge
  along the bars;
- **Running SHOE**: shell scripts to run the reconstruction on the CNAO2025 runs and on the scan data.

The data and the SHOE software belong to the FOOT collaboration and are not included in this repository.

## Requirements

Python 3 with numpy, scipy, matplotlib and uproot.
