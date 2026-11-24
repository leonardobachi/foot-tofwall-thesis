#!/usr/bin/env python3
"""
plot_deltatime.py — Comparison of delta-time -> position calibrations (v and offset per bar).

Each calibration file can be the SHOE map (TATWdeltaTimeMap.map, without
errors) or a file of deltatime_calibration.py (with errors); the format is
recognised from the number of columns:
    "id:  layer  barShoe  v  off"                (SHOE)
    "id:  layer  barShoe  v  dv  off  doff"      (deltatime_calibration.py)

Four figures (v and offset, rear and front) with a residual panel with respect
to the first calibration (the reference): v in % (or normalised), offset in mm.
A table of dv/v (%) and d(offset) (cm) per bar, with mean and rms, is printed
and optionally saved (--table-out).

With --allz HITS POINTS BAR: for one bar, mu_dt vs nominal position for each Z
separately, with the residuals from the weighted mean of the Z.

Usage:
    python3 plot_deltatime.py --calib SHOE TATWdeltaTimeMap.map \\
                              --calib scan deltatime_calibration_scan.txt \\
                              --calib "physics (Z=2)" deltatime_calibration_physics_z2.txt \\
                              [--table-out comparison.txt]
    python3 plot_deltatime.py --allz hits_dt.pkl points.pkl 9
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

COLORS  = ["black", "#0000ff", "#ff0000", "#00a000", "#ff9900", "#8000ff"]
MARKERS = ["s", "o", "^", "D", "v", "P"]

X_SHIFT = 0.15          # horizontal shift between calibrations, to avoid overlapping points
OFFSET_YLIM = None      # e.g. (-1, 2) to zoom on the offsets
FIGSIZE = (12, 8)
PAD = 1.5
RES_YLIM = None         # e.g. (-10, 10) for the normalised residuals
RES_LABELSIZE = 16
RES_V_MODE = "percent"  # "norm" = normalised residuals, "percent" = (v - v_ref)/v_ref in %

ALLZ_MIN_POINTS = 100   # minimum entries per position for the gaussian fit (--allz)


def read_calib(path):
    """{layer: {"bar", "v", "dv", "off", "doff"}} (dv/doff = None for the SHOE format)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()[1:]])   # skip "id:"
    data = np.array(rows)
    has_err = data.shape[1] >= 6
    out = {}
    for layer in (0, 1):
        d = data[data[:, 0] == layer]
        if has_err:
            out[layer] = {"bar": d[:, 1], "v": d[:, 2], "dv": d[:, 3], "off": d[:, 4], "doff": d[:, 5]}
        else:
            out[layer] = {"bar": d[:, 1], "v": d[:, 2], "dv": None, "off": d[:, 3], "doff": None}
    return out


def _ref_values(d, ref, key):
    """Values of the reference calibration for the bars of d (NaN for bars missing in the reference)."""
    lookup = dict(zip(ref["bar"], ref[key]))
    return np.array([lookup.get(b, np.nan) for b in d["bar"]])


def print_percent_table(calibs, table_out=None):
    """dv/v_ref (%) and d(offset) (cm) of each calibration with respect to the first one."""
    ref_label, ref_cal = calibs[0][0], calibs[0][1]
    others = [(lab, cal) for lab, cal, _, _ in calibs[1:]]

    lines = []
    for layer, name in ((0, "rear"), (1, "front")):
        ref = ref_cal[layer]
        lines.append(f"\n=== {name} — reference: {ref_label} ===")
        lines.append(f"{'bar':>5}" + "".join(f"{lab + ' Δv%':>22}{lab + ' Δoff[cm]':>26}" for lab, _ in others))
        # values of each calibration on the bars of the reference (NaN if missing)
        pct, doff = {}, {}
        for lab, cal in others:
            v = _ref_values(ref, cal[layer], "v")
            o = _ref_values(ref, cal[layer], "off")
            pct[lab] = 100 * (v - ref["v"]) / ref["v"]
            doff[lab] = o - ref["off"]
        for k, bar in enumerate(ref["bar"]):
            lines.append(f"{int(bar):>5}" + "".join(f"{pct[lab][k]:>22.2f}{doff[lab][k]:>26.3f}" for lab, _ in others))
        lines.append(f"{'mean':>5}" + "".join(f"{np.nanmean(pct[lab]):>22.2f}{np.nanmean(doff[lab]):>26.3f}" for lab, _ in others))
        lines.append(f"{'rms':>5}" + "".join(f"{np.nanstd(pct[lab]):>22.2f}{np.nanstd(doff[lab]):>26.3f}" for lab, _ in others))

    print("\n".join(lines), flush=True)

    if table_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(table_out)), exist_ok=True)
        with open(table_out, "w") as f:
            f.write("\n".join(lines).lstrip("\n") + "\n")
        print(f"\n[INFO] Table saved to: {table_out}", flush=True)


def plot_quantity(calibs, layer, qty, style):
    """One figure: qty ("v" or "off") vs bar, with the residuals from the reference below."""
    n = len(calibs)
    ref = calibs[0][1][layer]
    err_key = "d" + qty

    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    bar_offset = 20 if layer == 1 else 0   # front: bars 20-39

    for i, (label, cal, color, marker) in enumerate(calibs):
        d = cal[layer]
        x = d["bar"] + bar_offset + X_SHIFT * (i - (n - 1) / 2)
        ax.errorbar(x, d[qty], yerr=d[err_key], fmt=marker, color=color, label=label, **style)

        # residuals only for the calibrations with errors
        if i == 0 or d[err_key] is None:
            continue
        ref_q = _ref_values(d, ref, qty)
        if qty == "v" and RES_V_MODE == "percent":
            res = 100 * (d["v"] - ref_q) / ref_q
            yerr = 100 * d["dv"] / ref_q
        elif qty == "off":
            res = 10 * (d["off"] - ref_q)     # cm -> mm
            yerr = 10 * d["doff"]
        else:
            res = (d[qty] - ref_q) / d[err_key]
            yerr = 1.0
        ax_r.errorbar(x, res, yerr=yerr, fmt=marker, color=color, **style)

    if qty == "v":
        ax.set_ylabel("v (cm/ns)")
        res_label = "Δv/v (%)" if RES_V_MODE == "percent" else "res. norm."
    else:
        ax.set_ylabel("offset (cm)")
        if OFFSET_YLIM is not None:
            ax.set_ylim(OFFSET_YLIM)
        res_label = "Δoffset (mm)"
    ax.legend(ncol=1, loc="upper left")

    ax_r.axhline(0, color="black", lw=1.5, linestyle="--")
    ax_r.set_ylabel(res_label, fontsize=RES_LABELSIZE)
    if RES_YLIM is not None and res_label == "res. norm.":
        ax_r.set_ylim(RES_YLIM)
    ax_r.set_xlabel("Bar Id")
    ax_r.set_xticks(range(bar_offset, bar_offset + 20))

    for a in (ax, ax_r):
        a.grid(True, alpha=0.3)
    fig.tight_layout(pad=PAD)


def plot_deltatime_allZ(hits_path, points_path, bar, min_points):
    """
    For one bar: mu_dt vs nominal position, one colour per Z, with the residuals
    normalised to the weighted mean of the Z (same pairing as deltatime_calibration.py).
    """
    from deltatime_calibration import (load_pkl, build_hit_index, fit_dt_distribution_per_position,
                                       LAYER0_X, LAYER1_Y)

    hits = load_pkl(hits_path, "hits")
    points = load_pkl(points_path, "points")
    hit_idx = build_hit_index(hits)

    bar_usage = defaultdict(int)
    for p in points:
        bar_usage[(p["run_id"], p["eventN"], p["column"])] += 1
        bar_usage[(p["run_id"], p["eventN"], p["row"] + 20)] += 1

    data_by_z = defaultdict(list)
    for p in points:
        run_id, evN = p["run_id"], p["eventN"]
        col_id, row_id = p["column"], p["row"] + 20
        if bar not in (col_id, row_id):
            continue
        if bar_usage[(run_id, evN, col_id)] > 1 or bar_usage[(run_id, evN, row_id)] > 1:
            continue
        h_col = hit_idx.get((run_id, evN, col_id))
        h_row = hit_idx.get((run_id, evN, row_id))
        if h_col is None or h_row is None or h_col["chargeZ"] != h_row["chargeZ"]:
            continue
        z = h_col["chargeZ"]
        if bar == col_id:      # rear bar: reference position from the front bar
            data_by_z[z].append((h_col["t_R"] - h_col["t_L"], LAYER1_Y[p["row"]]))
        else:                  # front bar: reference position from the rear bar
            data_by_z[z].append((h_row["t_R"] - h_row["t_L"], LAYER0_X[p["column"]]))

    print("\nEntries per Z:", {z: len(v) for z, v in sorted(data_by_z.items())})

    label = "rear" if bar < 20 else "front"
    Z_COLORS = {1: "#0000ff", 2: "#00a000", 3: "#ff9900", 4: "#8b4513", 5: "#8000ff", 6: "#ff0000"}
    EXTRA_COLORS = ["#000000", "#00b8b8", "#808080"]
    color_for = lambda z: Z_COLORS.get(z, EXTRA_COLORS[z % len(EXTRA_COLORS)])

    results = {}
    for z in sorted(data_by_z.keys()):
        pos_results = fit_dt_distribution_per_position(data_by_z[z], min_points=min_points)
        if not pos_results:
            print(f"Z={z}: no position with enough entries, skipped")
            continue
        results[z] = (np.array([r["pos"] for r in pos_results]),
                      np.array([r["mu"] for r in pos_results]),
                      np.array([r["dmu"] for r in pos_results]))

    # weighted mean (1/dmu^2) over the Z, position by position
    by_pos = defaultdict(list)
    for z, (pos, mu, dmu) in results.items():
        for p, m, d in zip(pos, mu, dmu):
            by_pos[p].append((m, d))
    wmean = {}
    for p, lst in by_pos.items():
        if len(lst) < 2:       # with a single Z the residual would be 0 by construction
            continue
        w = np.array([1.0 / d**2 for _, d in lst])
        m = np.array([m for m, _ in lst])
        wmean[p] = np.sum(w * m) / np.sum(w)

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    for i, z in enumerate(results):
        pos, mu, dmu = results[z]
        shift = 0.12 * (i - (len(results) - 1) / 2)
        ax.errorbar(pos + shift, mu, yerr=dmu, fmt="o", color=color_for(z),
                    capsize=3, markersize=6, elinewidth=1.5, capthick=1.5, label=f"Z={z}")
        sel = np.array([p in wmean for p in pos])
        if sel.any():
            res = (mu[sel] - np.array([wmean[p] for p in pos[sel]])) / dmu[sel]
            ax_res.errorbar(pos[sel] + shift, res, yerr=1.0, fmt="o", color=color_for(z),
                            capsize=3, markersize=6, elinewidth=1.5, capthick=1.5)

    ax.set_ylabel(r"$\mu_{\Delta t}$ (ns)")
    ax.set_title(f"Delta-time calibration — all Z — Bar {bar} ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax_res.axhline(0, color="black", lw=1.5, linestyle="--")
    ax_res.set_xlabel("Nominal position (cm)")
    ax_res.set_ylabel("res. norm.")
    ax_res.grid(True, alpha=0.3)
    fig.tight_layout(pad=PAD)


def main():
    parser = argparse.ArgumentParser(description="Comparison of delta-time calibrations")
    parser.add_argument("--calib", nargs=2, action="append", metavar=("LABEL", "FILE"), default=[],
                        help="Calibration to plot (repeatable); the first one is the reference")
    parser.add_argument("--table-out", default=None, help="Save the comparison table to this file")
    parser.add_argument("--allz", nargs=3, metavar=("HITS", "POINTS", "BAR"), default=None,
                        help="mu_dt vs position for each Z, for one bar")
    args = parser.parse_args()
    if not args.calib and not args.allz:
        parser.error("give at least one --calib or --allz")

    if args.calib:
        calibs = [(label, read_calib(path), COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)])
                  for i, (label, path) in enumerate(args.calib)]
        if len(calibs) > 1:
            print_percent_table(calibs, args.table_out)
        style = dict(markersize=7, capsize=3, elinewidth=1.5, capthick=1.5)
        for layer in (0, 1):
            for qty in ("v", "off"):
                plot_quantity(calibs, layer, qty, style)

    if args.allz:
        plot_deltatime_allZ(args.allz[0], args.allz[1], int(args.allz[2]), ALLZ_MIN_POINTS)

    plt.show()


if __name__ == "__main__":
    main()
