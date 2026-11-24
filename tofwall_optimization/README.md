# TOF-Wall studies on the SHOE results

Calibration of the position along the bar, charge equalisation along the bar
and time resolution of the TOF-Wall (TW), from the hits and points reconstructed
by SHOE, the FOOT reconstruction software.

## Scripts

```
run_shoe.sh               SHOE reconstruction (DecodeGlb) of a list of runs
extract_shoe.py           TW hits and points from the SHOE .root files -> .pkl
deltatime_calibration.py  position calibration along the bar (v, offset per bar)
plot_deltatime.py         comparison of position calibrations
charge_equalization.py    charge equalisation along the bar
equalization_check.py     check of the equalisation (bar ends vs centre)
hitmap_protons.py         hitmap of the points on the 400 bar intersections
tw_resolution.py          time resolution per bar intersection
tw_resolution_vs_beta.py  time resolution in bins of β or of energy loss
plot_tof_vs_beta.py       resolution vs β / energy loss, with fit
```

## Running

```bash
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

## Methods

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

Some constants of the CNAO 2025 configuration are written in the code, each
commented with its origin; other campaigns need the corresponding values.

| Constants | Origin | File |
|---|---|---|
| Bethe-Bloch parametrisation | SHOE | `tw_resolution_vs_beta.py` |
| Detector geometry | SHOE | `tw_resolution_vs_beta.py`, `deltatime_calibration.py` |
