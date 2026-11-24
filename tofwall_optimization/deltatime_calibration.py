#!/usr/bin/env python3
"""
deltatime_calibration.py — Calibration of the position along the bar from the
time difference of the two ends, dt = t_R - t_L: velocity v and offset per bar,

    x = v * dt / 2 + offset

For each bar, the reference position of a hit is the centre of the crossing
bar of the other layer. Two-stage fit:
    1. for each nominal position, gaussian fit of the dt distribution -> mu, dmu
       (positions with fewer than MIN_POINTS_PER_POSITION entries are skipped)
    2. weighted linear fit mu(x) = A*x + B, excluding the two outermost
       positions (bar edges) -> v = 2/A, offset = -B/A, with the full error
       propagation (covariance of A and B included)

Matching between the two layers:
    with --points  through the SHOE points (row, column); the two hits must have
                   the same Z; bars shared by more points in the same event are
                   discarded (--event-filter)
    hits only      events with exactly one rear hit (bar_id 0-19) and one front
                   hit (bar_id 20-39); no Z check

Output in --output-dir:
    deltatime_calibration_<tag>.txt   v, dv, offset, doffset for every fitted bar
    deltatime_calibration_<tag>.map   same format as TATWdeltaTimeMap.map
                                      (written only if all 40 bars are fitted)

Usage:
    python3 deltatime_calibration.py --hits hits.pkl [--points points.pkl] --output-dir DIR
                                     [--tag physics_z2] [--z 2] [--plot-bar 7 30]
"""

import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PLOT_PER_POSITION = True   # with --plot-bar, also show the dt histogram of every position

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

# =============================================================================
# BAR CENTRES (cm), TATWdetector.geo (CNAO2025)
# =============================================================================

# layer 0 (rear, vertical bars, column): x
LAYER0_X = {i: 19.0 - 2.0 * i for i in range(20)}
# layer 1 (front, horizontal bars, row): y
LAYER1_Y = {i: 19.0 - 2.0 * i for i in range(20)}

# =============================================================================
# INPUT
# =============================================================================

def load_pkl(path, key):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    items = payload[key]
    print(f"{key.capitalize()} loaded: {len(items)}")
    print(f"  runs={payload.get('runs', '?')}")
    return items


# =============================================================================
# HIT INDEX
# =============================================================================

def build_hit_index(hits):
    """{(run_id, eventN, bar_id): hit}. SHOE has one hit per bar per event; duplicates are reported."""
    idx = {}
    n_collisions = 0
    for h in hits:
        key = (h["run_id"], h["eventN"], h["bar_id"])
        if key in idx:
            n_collisions += 1
            continue
        idx[key] = h

    if n_collisions > 0:
        print(f"[WARNING] {n_collisions} hits with the same run/event/bar_id (only the first is used)")
    else:
        print(f"Hit index: {len(idx)} unique keys.")
    return idx


# =============================================================================
# MATCHING THROUGH THE POINTS
# =============================================================================

def build_calibration_data_from_points(points, hits, z_filter=None, event_filter="bar-reuse"):
    """
    {bar_id: [(dt, reference position), ...]} from the points, requiring the
    same Z for the two hits (and Z = z_filter, if given).
    event_filter: "bar-reuse" discards the points with a bar shared by another
    point of the same event; "single-point" discards the events with more than
    one point; "none" applies no ambiguity selection.
    """
    hit_idx = build_hit_index(hits)
    data = {bid: [] for bid in range(40)}

    bar_usage_count = defaultdict(int)
    event_point_count = defaultdict(int)
    for p in points:
        bar_usage_count[(p["run_id"], p["eventN"], p["column"])] += 1
        bar_usage_count[(p["run_id"], p["eventN"], p["row"] + 20)] += 1
        event_point_count[(p["run_id"], p["eventN"])] += 1

    n_missing = 0
    n_z_mismatch = 0
    n_reused = 0
    n_z_filtered = 0
    for p in points:
        run_id, evN = p["run_id"], p["eventN"]
        col_bar_id = p["column"]
        row_bar_id = p["row"] + 20

        if event_filter == "bar-reuse":
            if bar_usage_count[(run_id, evN, col_bar_id)] > 1 or bar_usage_count[(run_id, evN, row_bar_id)] > 1:
                n_reused += 1
                continue
        elif event_filter == "single-point":
            if event_point_count[(run_id, evN)] > 1:
                n_reused += 1
                continue

        h_col = hit_idx.get((run_id, evN, col_bar_id))
        h_row = hit_idx.get((run_id, evN, row_bar_id))
        if h_col is None or h_row is None:
            n_missing += 1
            continue

        if h_col["chargeZ"] != h_row["chargeZ"]:
            n_z_mismatch += 1
            continue

        if z_filter is not None and h_col["chargeZ"] != z_filter:
            n_z_filtered += 1
            continue

        dt_col = h_col["t_R"] - h_col["t_L"]
        data[col_bar_id].append((dt_col, LAYER1_Y[p["row"]]))

        dt_row = h_row["t_R"] - h_row["t_L"]
        data[row_bar_id].append((dt_row, LAYER0_X[p["column"]]))

    print(f"Points: {len(points)}, missing hit: {n_missing}, "
          f"Z mismatch: {n_z_mismatch}, other Z (--z={z_filter}): {n_z_filtered}, "
          f"ambiguous ({event_filter}): {n_reused}")
    return data


# =============================================================================
# MATCHING ON THE HITS ONLY
# =============================================================================

def build_calibration_data_from_hits_only(hits):
    """
    {bar_id: [(dt, reference position), ...]} from the events with exactly one
    rear hit (bar_id 0-19) and one front hit (bar_id 20-39). No Z check.
    """
    by_event = defaultdict(lambda: {"rear": [], "front": []})
    for h in hits:
        key = (h["run_id"], h["eventN"])
        side = "rear" if h["bar_id"] < 20 else "front"
        by_event[key][side].append(h)

    data = {bid: [] for bid in range(40)}

    n_events = len(by_event)
    n_zero_side = 0
    n_ambiguous = 0
    n_paired = 0

    for (run_id, evN), sides in by_event.items():
        rear_hits, front_hits = sides["rear"], sides["front"]

        if len(rear_hits) == 0 or len(front_hits) == 0:
            n_zero_side += 1
            continue
        if len(rear_hits) > 1 or len(front_hits) > 1:
            n_ambiguous += 1
            continue

        h_col = rear_hits[0]
        h_row = front_hits[0]

        col_local = h_col["bar_id"]
        row_local = h_row["bar_id"] - 20

        dt_col = h_col["t_R"] - h_col["t_L"]
        data[h_col["bar_id"]].append((dt_col, LAYER1_Y[row_local]))

        dt_row = h_row["t_R"] - h_row["t_L"]
        data[h_row["bar_id"]].append((dt_row, LAYER0_X[col_local]))

        n_paired += 1

    print(f"Events: {n_events}; without hits on one layer: {n_zero_side}; "
          f"ambiguous (>1 hit on one layer): {n_ambiguous}; paired: {n_paired}")
    return data


# =============================================================================
# CHECK PLOT
# =============================================================================

def plot_deltatime_scatter_raw(bar_data, bar_id, label="", z_filter=None):
    """All the (dt, reference position) pairs of one bar."""
    dt_arr = np.array([x[0] for x in bar_data])
    pos_arr = np.array([x[1] for x in bar_data])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dt_arr, pos_arr, s=4, alpha=0.3, color="steelblue")
    ax.set_xlabel(r"$\Delta$t (ns)", fontsize=14)
    ax.set_ylabel("Nominal position (cm)", fontsize=14)
    z_str = f"Z={z_filter}" if z_filter is not None else ""
    ax.set_title(f"Scatter plot — Bar {bar_id} ({label}) — {z_str} — {len(bar_data)} entries", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STAGE 1 — gaussian fit of the dt distribution at each nominal position
# =============================================================================

# histogram binning: bin width = sigma/BINS_PER_SIGMA, but at least
# MIN_COUNTS_PER_BIN entries per bin on average
BINS_PER_SIGMA = 4
MIN_COUNTS_PER_BIN = 12
N_SIGMA_RANGE = 4.0
N_SIGMA_FIT = 2.0

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _fit_gauss_dt(dt_values, bins_per_sigma=BINS_PER_SIGMA, min_counts_per_bin=MIN_COUNTS_PER_BIN,
                   n_sigma_range=N_SIGMA_RANGE, n_sigma_fit=N_SIGMA_FIT):
    """
    Gaussian fit of the histogram of dt_values within ±N_SIGMA_FIT sigma of the
    peak; sigma first estimated from the MAD, binning adapted to both sigma and
    the number of entries. Returns {"mu", "dmu", "sigma", "n", "popt", "rng", "bins"} or None.
    """
    dt_values = np.asarray(dt_values)
    dt_values = dt_values[np.isfinite(dt_values)]
    if len(dt_values) < 30:
        return None

    mu0 = np.median(dt_values)
    mad = np.median(np.abs(dt_values - mu0))
    sigma0 = 1.4826 * mad
    if sigma0 <= 0 or not np.isfinite(sigma0):
        return None

    rng = (mu0 - n_sigma_range * sigma0, mu0 + n_sigma_range * sigma0)

    n_bins_shape = int(round((rng[1] - rng[0]) / (sigma0 / bins_per_sigma)))
    n_bins_stat = int(len(dt_values) / min_counts_per_bin)
    n_bins = int(np.clip(min(n_bins_shape, n_bins_stat), 10, 150))

    counts, edges = np.histogram(dt_values, bins=n_bins, range=rng)
    centers = (edges[:-1] + edges[1:]) / 2

    mu0_refined = centers[np.argmax(counts)]
    mask = (centers >= mu0_refined - n_sigma_fit * sigma0) & (centers <= mu0_refined + n_sigma_fit * sigma0)
    if mask.sum() < 4:
        mask = np.ones_like(centers, dtype=bool)
    x_fit, y_fit = centers[mask], counts[mask]

    try:
        sigma_y = np.sqrt(np.maximum(y_fit, 1))
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=[max(y_fit), mu0_refined, sigma0],
                               sigma=sigma_y, absolute_sigma=True, maxfev=10000)
        dmu = np.sqrt(max(pcov[1, 1], 0))
        return {"mu": popt[1], "dmu": dmu, "sigma": abs(popt[2]), "n": len(dt_values),
                "popt": popt, "rng": rng, "bins": n_bins}
    except Exception:
        return None


MIN_POINTS_PER_POSITION = 100

def fit_dt_distribution_per_position(bar_data, min_points=MIN_POINTS_PER_POSITION,
                                      plot=False, bar_id=None, label="", z_filter=None):
    """
    Gaussian fit of the dt distribution at each nominal position (positions with
    fewer than min_points entries are skipped).
    Returns [{"pos", "mu", "dmu", "sigma", "n"}, ...].
    """
    by_pos = defaultdict(list)
    for dt, pos in bar_data:
        by_pos[round(pos, 3)].append(dt)

    pos_results = []
    for pos in sorted(by_pos.keys()):
        dt_values = np.array(by_pos[pos])

        if len(dt_values) < min_points:
            print(f"    pos={pos:+.1f} cm: {len(dt_values)} entries (< {min_points}), skipped")
            continue

        fit = _fit_gauss_dt(dt_values)
        if fit is None:
            print(f"    pos={pos:+.1f} cm: gaussian fit failed, skipped")
            continue

        print(f"    pos={pos:+.1f} cm: mu_dt={fit['mu']:.4f}±{fit['dmu']:.4f} ns, "
              f"sigma_dt={fit['sigma']*1e3:.1f} ps, n={fit['n']}")

        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(dt_values, bins=fit["bins"], range=fit["rng"],
                    histtype="step", color="seagreen", lw=1.8)
            x_plot = np.linspace(fit["rng"][0], fit["rng"][1], 500)
            ax.plot(x_plot, gauss(x_plot, *fit["popt"]), color="black", lw=2, zorder=2,
                    label=fr"$\mu_{{\Delta t}} = {fit['mu']:.3f} \pm {fit['dmu']:.3f}$ ns")
            ax.set_xlabel(r"$\Delta t$ (ns)")
            ax.set_ylabel("Counts")
            ax.set_title(fr"Bar {bar_id} ({label}) — $x_{{\mathrm{{nom}}}}$ = {pos:+.1f} cm — {fit['n']} entries")
            ax.tick_params(axis='both')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        pos_results.append({"pos": pos, "mu": fit["mu"], "dmu": fit["dmu"],
                            "sigma": fit["sigma"], "n": fit["n"]})

    return pos_results


# =============================================================================
# STAGE 2 — weighted linear fit mu(pos) = A*pos + B -> v, offset
# =============================================================================

def fit_v_offset_from_positions(pos_results):
    """
    Weighted linear fit of the stage-1 points, without the two outermost
    positions; v = 2/A, offset = -B/A with the covariance of A and B in the
    error propagation.
    """
    pos_results_sorted = sorted(pos_results, key=lambda p: p["pos"])
    pos_arr = np.array([p["pos"] for p in pos_results_sorted])[1:-1]
    mu_arr = np.array([p["mu"] for p in pos_results_sorted])[1:-1]
    dmu_arr = np.array([p["dmu"] for p in pos_results_sorted])[1:-1]

    def linfunc(pos, A, B):
        return A * pos + B

    try:
        popt, pcov = curve_fit(linfunc, pos_arr, mu_arr, sigma=dmu_arr,
                               absolute_sigma=True, p0=[2 / 15.5, 0.0])
    except Exception as e:
        print(f"    Stage-2 fit failed: {e}")
        return None

    A, B = popt
    dA, dB = np.sqrt(np.diag(pcov))
    cov_AB = pcov[0, 1]

    v = 2 / A
    dv = abs(2 / A**2) * dA

    offset = -B / A
    doffset = np.sqrt((B / A**2)**2 * dA**2 + (1 / A)**2 * dB**2
                      - 2 * (B / A**2) * (1 / A) * cov_AB)

    residuals = mu_arr - linfunc(pos_arr, *popt)
    chi2 = float(np.sum((residuals / dmu_arr)**2))
    ndof = len(pos_arr) - 2

    return {"v": v, "dv": dv, "offset": offset, "doffset": doffset,
            "n_positions": len(pos_arr), "chi2": chi2, "ndof": ndof,
            "pos_arr": pos_arr, "mu_arr": mu_arr, "dmu_arr": dmu_arr, "linfunc": linfunc, "popt": popt}


def plot_v_offset_summary(fit2, bar_id, label="", z_filter=None):
    """Stage-2 fit of one bar with the normalised residuals."""
    pos_arr, mu_arr, dmu_arr = fit2["pos_arr"], fit2["mu_arr"], fit2["dmu_arr"]
    linfunc = fit2["linfunc"]

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})

    ax.errorbar(pos_arr, mu_arr, yerr=dmu_arr, fmt='o', color="steelblue",
                ecolor="steelblue", capsize=4, markersize=6)
    pos_fit = np.linspace(pos_arr.min(), pos_arr.max(), 200)
    ax.plot(pos_fit, linfunc(pos_fit, *fit2["popt"]), color="black", lw=2,
            label=f"v = {fit2['v']:.3f}±{fit2['dv']:.3f} cm/ns,  offset = {fit2['offset']:.3f}±{fit2['doffset']:.3f} cm")
    ax.set_ylabel(fr"$\mu_{{\Delta t}}$ (ns)")
    z_str = f" — $Z={z_filter}$" if z_filter is not None else ""
    ax.set_title(fr"Bar {bar_id} ({label}){z_str} — {fit2['n_positions']} positions")
    ax.tick_params(axis='both')
    ax.legend()
    ax.grid(True, alpha=0.3)

    residuals = mu_arr - linfunc(pos_arr, *fit2["popt"])
    residuals_norm = residuals / dmu_arr

    ax_res.plot(pos_arr, residuals_norm, 'o--', color="black", markersize=6, lw=1.2)
    ax_res.axhline(0, color="black", lw=1.0, linestyle=":")
    ax_res.set_xlabel(r"$x_{{\mathrm{{nom}}}}$ (cm)")
    ax_res.set_ylabel("norm. res.")
    ax_res.tick_params(axis='both')
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# OUTPUT
# =============================================================================

def write_calibration_file(results, out_path):
    """Same columns as TATWdeltaTimeMap.map, plus the errors on v and offset."""
    lines = ["#barId  layer   barId(Shoe)     v [cm/ns]   dv [cm/ns]      dx|y_AB [cm]   d(dx|y_AB) [cm]"]
    for bar_id in sorted(results.keys()):
        layer = 0 if bar_id < 20 else 1
        bar_shoe = bar_id if bar_id < 20 else bar_id - 20
        r = results[bar_id]
        lines.append(f"{bar_id}:    {layer}    {bar_shoe}    {r['v']:.6f}    {r['dv']:.6f}    "
                    f"{r['offset']:.6f}    {r['doffset']:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] Calibration saved to: {out_path}")

def write_shoe_map(results, out_path):
    """File in the format of TATWdeltaTimeMap.map (SHOE): 40 lines 'barId: layer barId(Shoe) v offset'."""
    missing = [b for b in range(40) if b not in results]
    if missing:
        print(f"\n[WARNING] Bars without fit {missing}: .map not written (SHOE needs all 40 bars).")
        return

    lines = ["#barId  layer   barId(Shoe)     v [cm/ns]       dx|y_AB [cm]"]
    for bar_id in range(40):
        layer = 0 if bar_id < 20 else 1
        bar_shoe = bar_id if bar_id < 20 else bar_id - 20
        r = results[bar_id]
        lines.append(f"{bar_id}:    {layer}    {bar_shoe}    {r['v']:.6f}    {r['offset']:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] SHOE map saved to: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta-time -> position calibration (two-stage fit)")
    parser.add_argument("--hits", required=True, help="hits.pkl from extract_shoe.py (--hits or --hits-dt)")
    parser.add_argument("--points", default=None,
                        help="points.pkl from extract_shoe.py; without it the hits are paired directly")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--tag", default="physics", help="Suffix of the output files")
    parser.add_argument("--plot-bar", type=int, nargs="+", default=None,
                        help="Bars for the check plots (scatter, dt per position, stage-2 fit)")
    parser.add_argument("--z", type=int, default=None, help="Only hits with this Z (with --points)")
    parser.add_argument("--event-filter", choices=["bar-reuse", "single-point", "none"], default="bar-reuse",
                        help="Ambiguity selection with --points (default: bar-reuse)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hits = load_pkl(args.hits, "hits")

    if args.points:
        points = load_pkl(args.points, "points")
        print(f"\n[INFO] Matching through the points (Z check, ambiguity selection: {args.event_filter})")
        data = build_calibration_data_from_points(points, hits, z_filter=args.z, event_filter=args.event_filter)
    else:
        print("\n[INFO] Matching on the hits only (no Z check)")
        data = build_calibration_data_from_hits_only(hits)

    plot_bars = set(args.plot_bar) if args.plot_bar else set()
    results = {}

    for bar_id in range(40):
        bar_data = data[bar_id]
        if not bar_data:
            continue

        label = "rear" if bar_id < 20 else "front"
        print(f"\n{'─'*55}\nBar {bar_id} ({label}) — {len(bar_data)} (dt, position) pairs")

        if bar_id in plot_bars:
            plot_deltatime_scatter_raw(bar_data, bar_id, label, z_filter=args.z)

        pos_results = fit_dt_distribution_per_position(
            bar_data, plot=(PLOT_PER_POSITION and bar_id in plot_bars), bar_id=bar_id, label=label, z_filter=args.z)

        fit2 = fit_v_offset_from_positions(pos_results)
        if fit2 is None:
            print("  Stage-2 fit failed, bar skipped.")
            continue

        print(f"  RESULT: v={fit2['v']:.4f}±{fit2['dv']:.4f} cm/ns, "
              f"offset={fit2['offset']:.4f}±{fit2['doffset']:.4f} cm, "
              f"positions={fit2['n_positions']}, chi2/ndof={fit2['chi2']:.1f}/{fit2['ndof']}")

        results[bar_id] = fit2

        if bar_id in plot_bars:
            plot_v_offset_summary(fit2, bar_id, label, z_filter=args.z)

    if not results:
        print("\n[ERROR] No bar fitted, nothing saved.")
        sys.exit(1)

    write_calibration_file(results, os.path.join(args.output_dir, f"deltatime_calibration_{args.tag}.txt"))
    write_shoe_map(results, os.path.join(args.output_dir, f"deltatime_calibration_{args.tag}.map"))


if __name__ == "__main__":
    main()
