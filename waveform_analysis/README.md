# Waveform analysis

Independent reconstruction of the TOF-Wall (TW) and Start Counter (ST) signals
from the digitised waveforms.

## Scripts

```
read_wf.py            reader of the binary waveform files (wf_raw*.bin)
baseline_wf.py        baseline subtraction, noise, pedestal
amplitude_wf.py       amplitude, signal threshold (8 σ_noise), ST pile-up rejection
cfd_wf.py             CFD times (simple, tangent, bipolar), rise time, ST sum waveform
charge_wf.py          charge (trapezoidal integral minus pedestal)
tw_map.py             WaveDAQ channel -> TW bar map
bar_analysis.py       bar charge, time and TOF on a single file (checks)
template.py           template construction and template fit
analyze.py            full waveform analysis of one run -> results_<run>.pkl
run_all.sh            analyze.py on all the runs
results.py            bars from results_<run>.pkl, calibrations, Z, final plots
tw_calibration.py     charge -> energy (Birks), TOF offsets, position along the bar
particle_id.py        charge identification from (TOF, E_loss)
chi2_by_bar.py        chi2 of the template fit for groups of bars
plot_scatter.py       summary plots per bar
```

## Input

The `wf_raw*.bin` files contain, event by event, all the ST and TW waveforms
(1024 samples, DRS4 at ~3 GS/s). They are written by a modified version of the
SHOE WaveDAQ reader, not included here; the format is described in `read_wf.py`.
The expected layout is one folder per run: `data/<run>/*.bin`.

## Reconstruction steps

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

## Running

```bash
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
- `--charge-eq` applies the equalisation file of `../tofwall_optimization/charge_equalization.py`.

The single modules can also be run on one `.bin` file to produce the check
plots of each step, for example `python3 cfd_wf.py file.bin --event 87`.

## Output of `analyze.py`

`results_<run>.pkl` is a list of events. Each event holds:

- the ST channels (amplitude, noise, CFD times);
- the ST sum;
- the TW waveforms with signal, each with bar, end, amplitude, noise, charge,
  rise time, the three CFD times and the results of the normal and saturated
  template fits (time, charge, χ², number of points).

The full list of fields is in `process_file`.

## Calibration constants

Some constants of the CNAO 2025 configuration are written in the code, each
commented with its origin; other campaigns need the corresponding values.

| Constants | Origin | File |
|---|---|---|
| Channel map | SHOE configuration | `tw_map.py` |
| Birks parameters, TOF offsets, position calibration | TW calibration of the collaboration | `tw_calibration.py` |
| Bethe-Bloch parametrisation | SHOE | `particle_id.py` |
