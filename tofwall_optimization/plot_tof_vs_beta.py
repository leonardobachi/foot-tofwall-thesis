#!/usr/bin/env python3
"""
plot_tof_vs_beta.py — TW time resolution vs beta or vs energy loss, from the
tables of tw_resolution_vs_beta.py (one file per curve, e.g. one per CFD method
or per Z).

The plotted resolution is sigma(t_front - t_rear) / 2, the resolution of the TW
time (mean of the two crossing bars). For the energy loss the points are fitted
with the three terms added in quadrature

    sigma(E) = sqrt( (a/sqrt(E))^2 + (b/E)^2 + c^2 )

x axis (X_MODE):
    "center"  bin centre (= centre of the selection gaussian); x error = sigma of
              the gaussian, or bin width / sqrt(12) for the box selection
    "mean"    mean of the variable in the selected events, x error = their std

Usage:
    python3 plot_tof_vs_beta.py FILE [FILE ...] [--labels L1 L2 ...]
Example:
    python3 plot_tof_vs_beta.py tof_resolution_vs_main_eloss_gauss0.2_simple_allZ.txt \\
                                tof_resolution_vs_main_eloss_gauss0.2_tangent_allZ.txt \\
                                --labels "simple CFD" "tangent CFD"
"""

import os
import re
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

VAR_LABELS = {"beta": r"$\beta$", "main_eloss": r"E$_{loss}$ (MeV)"}
COLORS  = ["black", "red", "blue", "green", "orange", "purple"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

SHOW_XERR = True
X_MODE = "center"             # "center" or "mean"
LOG_X = None                  # None = log for main_eloss, linear for beta

FIT_ENABLED = True            # fit sigma(E) (main_eloss only)
FIT_XRANGE = (None, None)     # (min, max) of E in the fit, MeV


def read_table(path):
    """
    Columns: bin lo hi mu[ns] sigma[ps] dsigma[ps] n [x_mean x_std]; the header
    gives the selection (box/gauss and its sigma fraction) and the variable.
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        return None

    gauss_frac, var = None, "beta"
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"(?:selection|selezione):\s*gauss,\s*sigma\s*=\s*([0-9.eE+-]+)", line)
            if m:
                gauss_frac = float(m.group(1))
            m = re.search(r"(?:variable|variabile):\s*(\w+)", line)
            if m:
                var = m.group(1)

    t = {"var": var, "lo": data[:, 1], "hi": data[:, 2],
         "sigma": data[:, 4], "dsigma": data[:, 5], "n": data[:, 6].astype(int)}

    if X_MODE == "mean" and data.shape[1] >= 9:
        t["x"], t["xerr"] = data[:, 7], data[:, 8]
    else:
        width = t["hi"] - t["lo"]
        t["x"] = 0.5 * (t["lo"] + t["hi"])
        t["xerr"] = gauss_frac * width if gauss_frac is not None else width / np.sqrt(12)
    return t


def fit_sigma_vs_eloss(E, sigma, dsigma, p0=(200.0, 10.0, 25.0)):
    """
    Weighted fit sigma(E) = sqrt((a/sqrt(E))^2 + (b/E)^2 + c^2) in FIT_XRANGE,
    with a, b, c >= 0. Returns (model, popt, perr, chi2, ndof) or None.
    """
    def model(E, a, b, c):
        return np.sqrt((a / np.sqrt(E))**2 + (b / E)**2 + c**2)

    E, sigma, dsigma = map(np.asarray, (E, sigma, dsigma))
    mask = np.isfinite(E) & np.isfinite(sigma) & (dsigma > 0)
    if FIT_XRANGE[0] is not None:
        mask &= E >= FIT_XRANGE[0]
    if FIT_XRANGE[1] is not None:
        mask &= E <= FIT_XRANGE[1]
    if mask.sum() <= 3:
        return None

    popt, pcov = curve_fit(model, E[mask], sigma[mask], p0=p0, sigma=dsigma[mask],
                           absolute_sigma=True, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((sigma[mask] - model(E[mask], *popt)) / dsigma[mask]) ** 2))
    ndof = int(mask.sum()) - 3
    return model, popt, perr, chi2, ndof


def main():
    parser = argparse.ArgumentParser(description="TW resolution vs beta or energy loss")
    parser.add_argument("files", nargs="+", help="Tables of tw_resolution_vs_beta.py")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels (default: file names)")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.files):
        parser.error("give one label per file")

    tables = []
    for i, f in enumerate(args.files):
        t = read_table(f)
        if t is None:
            print(f"[WARNING] {f}: empty, skipped")
            continue
        t["label"] = args.labels[i] if args.labels else os.path.splitext(os.path.basename(f))[0]
        tables.append(t)
    if not tables:
        sys.exit(1)

    var = tables[0]["var"]
    if any(t["var"] != var for t in tables):
        print("[ERROR] The files do not have the same bin variable")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, t in enumerate(tables):
        color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        sigma_tw, dsigma_tw = t["sigma"] / 2, t["dsigma"] / 2
        ax.errorbar(t["x"], sigma_tw, yerr=dsigma_tw, xerr=t["xerr"] if SHOW_XERR else None,
                    fmt=marker, color=color, capsize=3, markersize=6,
                    elinewidth=1.5, capthick=1.5, label=t["label"])

        if FIT_ENABLED and var == "main_eloss":
            res = fit_sigma_vs_eloss(t["x"], sigma_tw, dsigma_tw)
            if res is None:
                print(f"  [FIT] {t['label']}: too few points in the range, skipped")
                continue
            model, popt, perr, chi2, ndof = res
            print(f"  [FIT] {t['label']}: a = {popt[0]:.2f} ± {perr[0]:.2f} ps*MeV^1/2, "
                  f"b = {popt[1]:.2f} ± {perr[1]:.2f} ps*MeV, c = {popt[2]:.2f} ± {perr[2]:.2f} ps, "
                  f"chi2/ndof = {chi2:.1f}/{ndof}")
            e_plot = np.linspace(np.nanmin(t["x"]), np.nanmax(t["x"]), 300)
            ax.plot(e_plot, model(e_plot, *popt), "--", color=color, lw=1)

    ax.set_xlabel(VAR_LABELS.get(var, var))
    if LOG_X if LOG_X is not None else var == "main_eloss":
        ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_{TW}$ (ps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
