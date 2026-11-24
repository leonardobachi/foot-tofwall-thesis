#!/usr/bin/env python3
"""
tw_resolution.py — TW time resolution from the time difference of the two
crossing bars, dt = t_front - t_rear, for each row/column intersection.

Input (at least one):
    --points  dt = tof1 - tof2 of the SHOE points (row/front minus column/rear)
    --hits    without --points: events with exactly one rear hit (bar_id 0-19)
              and one front hit (bar_id 20-39), dt = tof_front - tof_rear;
              needs the full hits of extract_shoe.py --hits (field "tof")

Modes (can be combined):
    --row N --column M    one intersection: fit and histogram
    --all                 all the 400 intersections, table on the terminal
    --save-txt B1 B2 ...  for each bar_id (0-39), a .txt with all its
                          intersections (needs --output-dir)
--z selects one reconstructed Z.

sigma(dt) is the resolution of the difference; the single-bar resolution is
sigma/sqrt(2) and the resolution of the TW time (mean of the two bars) sigma/2.

Usage:
    python3 tw_resolution.py --points points.pkl --row 5 --column 12
    python3 tw_resolution.py --points points.pkl --all --z 6
    python3 tw_resolution.py --points points.pkl --save-txt 7 8 30 --output-dir out/
"""

import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# =============================================================================
# INPUT
# =============================================================================

def load_pkl(path, key):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    items = payload[key]
    print(f"{key.capitalize()} loaded: {len(items)}  (runs={payload.get('runs', '?')})")
    return items


def build_tof_data_from_points(points, z_filter=None):
    """{(row, column): [dt = tof1 - tof2, ...]}, optionally for one Z."""
    data = defaultdict(list)
    n_filtered = 0
    for p in points:
        if z_filter is not None and p["chargeZ"] != z_filter:
            n_filtered += 1
            continue
        data[(p["row"], p["column"])].append(p["tof1"] - p["tof2"])

    print(f"Points: {len(points)}, other Z (--z={z_filter}): {n_filtered}")
    return data


def build_tof_data_from_hits(hits, z_filter=None):
    """
    {(row, column): [dt = tof_front - tof_rear, ...]} from the events with exactly
    one rear and one front hit (same pairing as deltatime_calibration.py).
    """
    if len(hits) > 0 and "tof" not in hits[0]:
        print("[ERROR] The hits have no 'tof' field: use the full hits of extract_shoe.py --hits.")
        sys.exit(1)

    by_event = defaultdict(lambda: {"rear": [], "front": []})
    for h in hits:
        side = "rear" if h["bar_id"] < 20 else "front"
        by_event[(h["run_id"], h["eventN"])][side].append(h)

    data = defaultdict(list)
    n_zero_side = n_ambiguous = n_filtered = n_paired = 0

    for (run_id, evN), sides in by_event.items():
        rear_hits, front_hits = sides["rear"], sides["front"]
        if len(rear_hits) == 0 or len(front_hits) == 0:
            n_zero_side += 1
            continue
        if len(rear_hits) > 1 or len(front_hits) > 1:
            n_ambiguous += 1
            continue

        h_col, h_row = rear_hits[0], front_hits[0]
        if z_filter is not None and (h_col["chargeZ"] != z_filter or h_row["chargeZ"] != z_filter):
            n_filtered += 1
            continue

        data[(h_row["bar_id"] - 20, h_col["bar_id"])].append(h_row["tof"] - h_col["tof"])
        n_paired += 1

    print(f"Events: {len(by_event)}, without hits on one layer: {n_zero_side}, "
          f"ambiguous (>1 hit on one layer): {n_ambiguous}, other Z: {n_filtered}, paired: {n_paired}")
    return data


# =============================================================================
# GAUSSIAN FIT
# =============================================================================

MIN_ENTRIES_PER_CROSSING = 150   # intersections with fewer entries are skipped

# histogram binning: bin width = sigma/BINS_PER_SIGMA, but at least
# MIN_COUNTS_PER_BIN entries per bin on average
BINS_PER_SIGMA = 8
MIN_COUNTS_PER_BIN = 8
N_SIGMA_RANGE = 4.0              # histogram range: median ± N_SIGMA_RANGE * sigma
N_SIGMA_FIT = 2.0                # fit range: peak ± N_SIGMA_FIT * sigma


def fit_sigma_tof(dt_values, bins_per_sigma=BINS_PER_SIGMA, min_counts_per_bin=MIN_COUNTS_PER_BIN,
                  n_sigma_range=N_SIGMA_RANGE, n_sigma_fit=N_SIGMA_FIT):
    """
    Gaussian fit around the peak, with sigma first estimated from the MAD and a
    binning adapted to sigma and to the number of entries (10-150 bins).
    Returns {"mu", "dmu", "sigma", "dsigma", "n", "popt", "rng", "bins"} (ns) or None.
    """
    dt_values = np.asarray(dt_values)
    dt_values = dt_values[np.isfinite(dt_values)]
    if len(dt_values) < MIN_ENTRIES_PER_CROSSING:
        return None

    mu0 = np.median(dt_values)
    sigma0 = 1.4826 * np.median(np.abs(dt_values - mu0))
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
        dsigma = np.sqrt(max(pcov[2, 2], 0))
        return {"mu": popt[1], "dmu": dmu, "sigma": abs(popt[2]), "dsigma": dsigma,
                "n": len(dt_values), "popt": popt, "rng": rng, "bins": n_bins}
    except Exception:
        return None


def plot_tof_crossing(dt_values, fit, row, column, z=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(dt_values, bins=fit["bins"], range=fit["rng"], color="seagreen", alpha=0.7)
    x_plot = np.linspace(fit["rng"][0], fit["rng"][1], 500)
    ax.plot(x_plot, gauss(x_plot, *fit["popt"]), color="black", lw=2,
            label=f"sigma={fit['sigma']*1e3:.1f}±{fit['dsigma']*1e3:.1f} ps")
    ax.set_xlabel(r"$t_{\mathrm{front}} - t_{\mathrm{rear}}$ (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    z_str = f", Z={z}" if z is not None else ""
    ax.set_title(f"row={row}, column={column}{z_str} — {fit['n']} entries", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def write_bar_crossings_file(data, bar_id, out_path, z_filter=None):
    """All the intersections of one bar (bar_id < 20: fixed column, else fixed row)."""
    is_rear = bar_id < 20
    lines = [f"# TW resolution (front - rear) for all the intersections of bar {bar_id} "
             f"({'rear/column' if is_rear else 'front/row'})"]
    if z_filter is not None:
        lines.append(f"# Z: {z_filter}")
    lines.append("# row  column  mu[ns]  sigma[ps]  dsigma[ps]  n")

    n_written = 0
    for other in range(20):
        row, column = (other, bar_id) if is_rear else (bar_id - 20, other)
        dt_values = data.get((row, column))
        if not dt_values:
            continue
        fit = fit_sigma_tof(dt_values)
        if fit is None:
            continue
        lines.append(f"{row}  {column}  {fit['mu']:.4f}  {fit['sigma']*1e3:.2f}  "
                     f"{fit['dsigma']*1e3:.2f}  {fit['n']}")
        n_written += 1

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Bar {bar_id}: {n_written}/20 intersections saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="TW time resolution (front - rear) per intersection")
    parser.add_argument("--points", default=None, help="points.pkl from extract_shoe.py")
    parser.add_argument("--hits", default=None, help="full hits.pkl from extract_shoe.py --hits")
    parser.add_argument("--row", type=int, default=None, help="Row (front) index of one intersection")
    parser.add_argument("--column", type=int, default=None, help="Column (rear) index of one intersection")
    parser.add_argument("--all", action="store_true", help="All the 400 intersections, table only")
    parser.add_argument("--save-txt", type=int, nargs="+", default=None,
                        help="bar_id (0-39): one .txt per bar with all its intersections")
    parser.add_argument("--output-dir", default=None, help="Output folder (with --save-txt)")
    parser.add_argument("--z", type=int, default=None, help="Only this reconstructed Z")
    args = parser.parse_args()

    if not args.points and not args.hits:
        parser.error("give at least one of --points and --hits")
    if args.save_txt and not args.output_dir:
        parser.error("--save-txt needs --output-dir")
    if not (args.row is not None and args.column is not None) and not args.all and not args.save_txt:
        parser.error("give --row/--column, --all or --save-txt")

    if args.points:
        print("\n[INFO] dt from the points (tof1 - tof2)")
        data = build_tof_data_from_points(load_pkl(args.points, "points"), z_filter=args.z)
    else:
        print("\n[INFO] dt from the paired hits")
        data = build_tof_data_from_hits(load_pkl(args.hits, "hits"), z_filter=args.z)

    if args.row is not None and args.column is not None:
        dt_values = data.get((args.row, args.column), [])
        fit = fit_sigma_tof(dt_values)
        if fit is None:
            print(f"\nrow={args.row}, column={args.column}: too few entries ({len(dt_values)}) or fit failed")
        else:
            print(f"\nrow={args.row}, column={args.column}: mu={fit['mu']:.4f}±{fit['dmu']:.4f} ns, "
                  f"sigma={fit['sigma']*1e3:.1f}±{fit['dsigma']*1e3:.1f} ps, n={fit['n']}")
            plot_tof_crossing(dt_values, fit, args.row, args.column, z=args.z)

    if args.all:
        print(f"\n{'row':>4} {'column':>7} {'mu[ns]':>10} {'sigma[ps]':>11} {'dsigma[ps]':>12} {'n':>8}")
        n_ok = 0
        for row in range(20):
            for column in range(20):
                dt_values = data.get((row, column))
                if not dt_values:
                    continue
                fit = fit_sigma_tof(dt_values)
                if fit is None:
                    continue
                print(f"{row:>4} {column:>7} {fit['mu']:>10.4f} {fit['sigma']*1e3:>11.2f} "
                      f"{fit['dsigma']*1e3:>12.2f} {fit['n']:>8}")
                n_ok += 1
        print(f"\n[INFO] Intersections fitted: {n_ok}/400")

    if args.save_txt:
        os.makedirs(args.output_dir, exist_ok=True)
        for bar_id in args.save_txt:
            write_bar_crossings_file(data, bar_id, os.path.join(args.output_dir, f"tof_resolution_bar{bar_id}.txt"),
                                     z_filter=args.z)


if __name__ == "__main__":
    main()
