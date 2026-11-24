# Optimization of the TOF-Wall detector for charged particle detection in the FOOT experiment

Analysis code of the Master's thesis of **Leonardo Bachi** (University of Pisa, 2026),
supervisors Prof. Matteo Morrocchi and Dr. Aafke Christine Kraan.

The code studies the TOF-Wall (TW) of the FOOT experiment with the data of the
CNAO 2025 campaign (¹²C beam at 200 MeV/u) and of the 2026 TW scans. It follows
the two lines of the thesis:

- **`waveform_analysis/`** — an independent reconstruction from the digitised
  waveforms of the Start Counter (ST) and of the TW: baseline, signal
  identification, CFD timing, charge, bar hits, charge identification and a
  template fit of the TW waveforms (also on artificially saturated waveforms).
- **`tofwall_optimization/`** — studies on the output of SHOE, the FOOT
  reconstruction software: calibration of the position along the bar, charge
  equalisation along the bar and time resolution of the TW.

Neither the data nor SHOE are included: both belong to the FOOT collaboration.

## Requirements

Python ≥ 3.9 with

```
pip install -r requirements.txt     # numpy, scipy, matplotlib, uproot, awkward
```

`uproot` and `awkward` are only needed by `tofwall_optimization/extract_shoe.py`.
Every script prints its options with `-h`; the header of each file describes
input, output and method.

## Repository structure

```
waveform_analysis/
├── read_wf.py            reader of the binary waveform files (wf_raw*.bin)
├── baseline_wf.py        baseline subtraction, noise, pedestal
├── amplitude_wf.py       amplitude, signal threshold (8 σ_noise), ST pile-up rejection
├── cfd_wf.py             CFD times (simple, tangent, bipolar), rise time, ST sum waveform
├── charge_wf.py          charge (trapezoidal integral minus pedestal)
├── tw_map.py             WaveDAQ channel -> TW bar map
├── bar_analysis.py       bar charge, time and TOF on a single file (checks)
├── template.py           template construction and template fit
├── analyze.py            full waveform analysis of one run -> results_<run>.pkl
├── run_all.sh            analyze.py on all the runs
├── results.py            bars from results_<run>.pkl, calibrations, Z, final plots
├── tw_calibration.py     charge -> energy (Birks), TOF offsets, position along the bar
├── particle_id.py        charge identification from (TOF, E_loss)
├── chi2_by_bar.py        chi2 of the template fit for groups of bars
└── plot_scatter.py       summary plots per bar

tofwall_optimization/
├── run_shoe.sh               SHOE reconstruction (DecodeGlb) of a list of runs
├── extract_shoe.py           TW hits and points from the SHOE .root files -> .pkl
├── deltatime_calibration.py  position calibration along the bar (v, offset per bar)
├── plot_deltatime.py         comparison of position calibrations
├── charge_equalization.py    charge equalisation along the bar
├── equalization_check.py     check of the equalisation (bar ends vs centre)
├── hitmap_protons.py         hitmap of the points on the 400 bar intersections
├── tw_resolution.py          time resolution per bar intersection
├── tw_resolution_vs_beta.py  time resolution in bins of β or of energy loss
└── plot_tof_vs_beta.py       resolution vs β / energy loss, with fit
```

The scripts of each folder import each other by module name, so they are run
from inside their folder.

## 1. Waveform analysis

### Input

The `wf_raw*.bin` files contain, event by event, all the ST and TW waveforms
(1024 samples, DRS4 at ~3 GS/s). They are written by a modified version of the
SHOE WaveDAQ reader, not included here; the format is described in `read_wf.py`.
The expected layout is one folder per run: `data/<run>/*.bin`.

### Reconstruction steps (per waveform and per event)

| Step | Module | Method |
|---|---|---|
| Baseline and noise | `baseline_wf.py` | mean of samples 5–205; σ_noise = their standard deviation |
| Signal | `amplitude_wf.py` | TW: amplitude > 8 σ_noise; ST: events with a channel above 600 mV rejected (pile-up) |
| ST time | `cfd_wf.py` | sum of the ST channels resampled at 0.2 ns, CFD on the sum |
| CFD | `cfd_wf.py` | simple, tangent and bipolar CFD, fraction 0.2 |
| Charge | `charge_wf.py` | trapezoidal integral (last 50 samples excluded) minus pedestal |
| Bar | `tw_map.py`, `results.py` | Q = √(Q_L·Q_R), t = (t_L + t_R)/2, TOF = t − t_ST |
| Template fit | `template.py` | 50 ps grid, χ² with σ_noise, analytic amplitude, Brent minimisation in the arrival time |

The TW template is built once from the waveforms of bar 9, aligned with the
simple CFD time, and used for all bars and runs. With `--saturation 200` every
waveform is also fitted after clipping it at 200 mV, excluding the saturated
samples from the χ².

### Running

```bash
cd waveform_analysis

# 1. build the TW template on one run
python3 analyze.py data/8102 results/ --run 8102 --save-template template_tw.pkl

# 2. analyse all the runs (resumable: runs already processed are skipped)
bash run_all.sh data/ results/ template_tw.pkl --saturation 200 --cores 8

# 3. bars, calibrations, Z and plots
python3 results.py results/ --tw-method fit_sat --save-bars bars_fit_sat.pkl
python3 results.py --load-bars bars_fit_sat.pkl --bars 9 --q-min 10000 --q-max 13500
python3 results.py --load-bars bars_fit_sat.pkl --coincidence 9 29
python3 results.py --load-bars bars_fit_sat.pkl --bars 9 --plots tof-vs-eloss charge-fit chi2
```

- `--tw-method`: `simple`, `tangent`, `bipolar`, `fit` or `fit_sat`.
- `--st-method`: `simple`, `tangent` or `bipolar`.
- Charge selections are in mV·ns (carbon ≈ 10000–13500).
- `--charge-eq` applies the equalisation file of `charge_equalization.py`.

The single modules can also be run on one `.bin` file to produce the check
plots of each step, for example `python3 cfd_wf.py file.bin --event 87`.

### Output of `analyze.py`

`results_<run>.pkl` is a list of events. Each event holds:

- the ST channels (amplitude, noise, CFD times);
- the ST sum;
- the TW waveforms with signal, each with bar, end, amplitude, noise, charge,
  rise time, the three CFD times and the results of the normal and saturated
  template fits (time, charge, χ², number of points).

The full list of fields is in `process_file`.

## 2. TOF-Wall studies on the SHOE reconstruction

```bash
cd tofwall_optimization

# 1. SHOE reconstruction (from build/Reconstruction of SHOE; runs and campaign set in the script)
./run_shoe.sh <data_dir> <out_dir> 4

# 2. hits and points from the .root files (--calib for the scan files)
python3 extract_shoe.py "reco/*.root" --hits hits.pkl --hits-dt hits_dt.pkl --points points.pkl --workers 8

# 3. position calibration along the bar
python3 deltatime_calibration.py --hits hits_dt.pkl --points points.pkl --z 2 --tag physics_z2 --output-dir calib/
python3 plot_deltatime.py --calib SHOE TATWdeltaTimeMap.map --calib "physics (Z=2)" calib/deltatime_calibration_physics_z2.txt

# 4. charge equalisation along the bar and its check
python3 charge_equalization.py hits.pkl --output-dir calib/ --plot-bar 7 --plot-xmin
python3 equalization_check.py hits.pkl --bar 6 --z 1 2

# 5. time resolution
python3 tw_resolution.py --points points.pkl --all
python3 tw_resolution_vs_beta.py --points points.pkl --bins 20 40 80 160 320 --output-dir res/
python3 plot_tof_vs_beta.py res/tof_resolution_vs_main_eloss_gauss0.2_allZ.txt
python3 hitmap_protons.py --points points.pkl --output-dir maps/ --z 1
```

- **Position calibration.** For every bar, the reference position of a hit is
  the centre of the crossing bar of the other layer. The fit has two stages:
  1. a gaussian fit of t_R − t_L at each nominal position;
  2. a weighted linear fit without the two outermost positions, giving
     v = 2/A and offset = −B/A.

  The output also comes in the format of SHOE's `TATWdeltaTimeMap.map`.
- **Charge equalisation.** For every bar, a parabolic fit of the charge vs
  position, with outlier rejection, normalised to the minimum of the parabola:
  Q_eq = Q / f(x).
- **Time resolution.**
  - σ of t_front − t_rear at the bar intersections.
  - The resolution of the TW time (mean of the two bars) is σ/2.
  - The binning in energy loss uses a gaussian selection in each bin.
  - The resolution vs energy loss is fitted with
    σ(E) = √((a/√E)² + (b/E)² + c²).

## Calibration constants

Some constants of the CNAO 2025 configuration are written in the code:

| Constants | Origin | File |
|---|---|---|
| Channel map | SHOE configuration | `tw_map.py` |
| Birks parameters, TOF offsets, position calibration | TW calibration of the collaboration | `tw_calibration.py` |
| Bethe-Bloch parametrisation | SHOE | `particle_id.py`, `tw_resolution_vs_beta.py` |
| Detector geometry | SHOE | `tw_resolution_vs_beta.py` |

Each constant is commented with the SHOE file it comes from. Other campaigns
need the corresponding values.

## Citation

```bibtex
@misc{bachi2026code,
  author       = {Bachi, Leonardo},
  title        = {Analysis code for the optimization of the {TOF-Wall} detector of the {FOOT} experiment},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/leonardobachi/foot-tofwall-thesis}
}
```
