#!/usr/bin/env python3
"""
chi2_by_bar.py — chi2 of the TW template fit for groups of bars, at equal waveform charge.

Input: a .npz file with one entry per TW waveform and the arrays
    bar_id, amplitude (V), noise (V), charge (V*ns), chi2_fit, npts_fit
(extracted from the results_<run>.pkl files of analyze.py).

The variable is chi2/npts (~1 for a waveform described by the template within
the noise). The shape mismatch between waveform and template, relative to
the peak, is estimated as
    eps^2 = (sigma_noise / A)^2 * (chi2/npts - c)
with c the median chi2/npts of the 10-20 mV waveforms (noise only).

Output:
    1. one figure per charge interval: chi2/npts of the groups, each histogram
       normalised to its maximum; table with n, median, KS distance from the
       first group and eps = sqrt(median eps^2)
    2. one figure for BAR_FIXED: chi2/npts in the different charge intervals

Usage:
    python3 chi2_by_bar.py <chi2.npz>
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

GROUPS = {                              # name: bars; the first one is the KS reference
    "bar 9":        [9],
    "bar 29":       [29],
    "central":      [8, 10, 28, 30],
    "intermediate": [5, 6, 7, 11, 12, 13, 14, 25, 26, 27, 31, 32, 33, 34],
    "outer":        [0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39],
}
COLORS = ["black", "blue", "red", "green", "orange", "purple", "gray", "brown", "cyan"]

CHARGE_BINS = [0.5, 1., 2., 4., 6., 8., 10., 13.5]   # V*ns; None = from the percentiles
BAR_FIXED   = 9                                      # None = skip figure 2
LOG_X       = True                                   # histograms of log10(chi2/npts)
N_BINS      = 160

XLABEL = r"$\chi^2/n_{\mathrm{pts}}$"


def hist_edges(values):
    v = np.log10(values) if LOG_X else values
    return np.linspace(*np.percentile(v, [0.5, 99.9]), N_BINS + 1)


def draw_hist(x, edges, color, label):
    """Histogram normalised to its maximum."""
    v = np.log10(x) if LOG_X else x
    h, _ = np.histogram(v, bins=edges)
    plt.stairs(h / h.max(), edges, color=color, lw=2, label=label)


def finish():
    plt.xlabel(r"$\log_{10}$" + XLABEL if LOG_X else XLABEL)
    plt.ylabel("Normalized counts")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    d = np.load(sys.argv[1])
    bar    = d["bar_id"]
    amp    = d["amplitude"] * 1e3       # mV
    noise  = d["noise"]                 # V
    charge = d["charge"]                # V*ns
    chi2   = d["chi2_fit"] / d["npts_fit"]

    good = np.isfinite(chi2) & (chi2 > 0) & np.isfinite(charge) & (charge > 0)
    print(f"waveforms: {len(chi2)}, with a valid chi2: {good.sum()}")
    print(f"charge (V*ns), percentiles 1/5/25/50/75/95/99: "
          f"{np.round(np.percentile(charge[good], [1, 5, 25, 50, 75, 95, 99]), 4)}")
    for name, bars in GROUPS.items():
        sel = good & np.isin(bar, bars)
        print(f"  {name}: {sel.sum()} waveforms, median noise {np.median(noise[sel]) * 1e3:.3f} mV")

    if CHARGE_BINS is None:
        CHARGE_BINS = [float(x) for x in np.round(np.percentile(charge[good], [5, 25, 50, 75, 95]), 4)]
    print(f"charge intervals (V*ns): {CHARGE_BINS}")

    c = np.median(chi2[good & (amp >= 10) & (amp < 20)])
    print(f"c = median chi2/npts at 10-20 mV = {c:.3f}")
    eps2 = (noise / (amp / 1e3))**2 * (chi2 - c)     # can be negative for small waveforms

    # --- 1. one figure per charge interval ---
    print(f"\n{'charge':>14} {'group':>13} {'n':>9} {'median':>10} {'KS D':>7} {'eps (%)':>8}")
    for lo, hi in zip(CHARGE_BINS[:-1], CHARGE_BINS[1:]):
        in_bin = good & (charge >= lo) & (charge < hi)
        ref = chi2[in_bin & np.isin(bar, list(GROUPS.values())[0])]
        if len(ref) == 0:
            continue
        edges = hist_edges(chi2[in_bin])

        plt.figure(figsize=(9, 6))
        for (name, bars), color in zip(GROUPS.items(), COLORS):
            sel = in_bin & np.isin(bar, bars)
            x = chi2[sel]
            if len(x) == 0:
                continue
            D = ks_2samp(x, ref).statistic
            eps = np.sqrt(max(np.median(eps2[sel]), 0)) * 100
            print(f"{f'{lo:g}-{hi:g}':>14} {name:>13} {len(x):>9} {np.median(x):>10.4g} {D:>7.3f} {eps:>8.2f}")
            draw_hist(x, edges, color, name)
        finish()

    # --- 2. fixed bar, different charge intervals ---
    if BAR_FIXED is not None:
        sel_bar = good & (bar == BAR_FIXED)
        edges = hist_edges(chi2[sel_bar])

        print(f"\nbar {BAR_FIXED}:")
        plt.figure(figsize=(9, 6))
        for (lo, hi), color in zip(zip(CHARGE_BINS[:-1], CHARGE_BINS[1:]), COLORS):
            x = chi2[sel_bar & (charge >= lo) & (charge < hi)]
            if len(x) == 0:
                continue
            print(f"  {lo:g}-{hi:g} V·ns: n={len(x)}, median {np.median(x):.4g}")
            draw_hist(x, edges, color, f"{lo:g}–{hi:g} V·ns ({len(x)})")
        finish()
