"""
plot_scatter.py — Summary plots per bar: TW time resolution (CFD vs template fit)
and resolution per fragment, and charge difference fit vs measured.

Input text files: blocks introduced by a line "# <key>" followed by one line
with one value per bar ("nan" where missing).
    --tw FILE          keys p_cfd, dp_cfd, p_fit, dp_fit: sigma(t_front - t_rear)
                       per front bar (20-39), simple CFD and template fit (ps)
    --charge FILE      keys mu_<s>, dmu_<s>, sigma_<s>, dsigma_<s> for
                       s = p, he, li, be, b, c: relative difference
                       (Q_fit - Q_orig)/Q_orig per rear bar (0-19), in %
    --resolution FILE  one block "## a-b ##" per bar intersection (a = 0..19),
                       followed by a line of (sigma, dsigma) pairs in ps for
                       p, He, Li, Be, B, C in this order (heavier fragments may
                       be missing on the outer bars); sigma of t_front - t_rear

Usage:
    python3 plot_scatter.py [--tw FILE] [--charge FILE] [--resolution FILE]
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

SPECIES = ["p", "He", "Li", "Be", "B", "C"]


# =============================================================================
# INPUT
# =============================================================================

def _load_block(path, key):
    """Values of the line after "# key"."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"# {key}":
            return np.array([float(x) if x != "nan" else np.nan for x in lines[i+1].split()])
    raise KeyError(f"Key '{key}' not found in {path}")


def load_charge_data(path):
    keys = [f"{q}_{s.lower()}" for s in SPECIES for q in ("mu", "dmu", "sigma", "dsigma")]
    return {k: _load_block(path, k) for k in keys}


def load_tw_data(path):
    return {k: _load_block(path, k) for k in ["p_cfd", "dp_cfd", "p_fit", "dp_fit"]}


def load_tw_resolution(path):
    """Returns two dicts {species: array(20)} with sigma and dsigma (ps), NaN where missing."""
    sigma  = {s: np.full(20, np.nan) for s in SPECIES}
    dsigma = {s: np.full(20, np.nan) for s in SPECIES}

    current_idx = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^##\s*(\d+)-(\d+)\s*##$", line)
        if m:
            current_idx = int(m.group(1))
            continue
        if line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        for i in range(len(vals) // 2):
            sigma[SPECIES[i]][current_idx]  = vals[2*i]
            dsigma[SPECIES[i]][current_idx] = vals[2*i + 1]

    return sigma, dsigma


# =============================================================================
# PLOTS
# =============================================================================

def weighted_mean(y, dy):
    """Weighted mean (1/dy^2) ignoring NaN, and its error."""
    weights  = 1 / dy**2
    mean_y   = np.nansum(weights * y) / np.nansum(weights)
    err_mean = 1 / np.sqrt(np.nansum(weights))
    return mean_y, err_mean


def scatter_plot(x, ys, dys, labels=None, colors=None, markers=None,
                 xlabel="x", ylabel="y", title="", xticklabels=None, unit="ps"):
    """Points with error bars for several populations with a common x, and the weighted mean of each (dashed line)."""
    if labels is None:
        labels = [f"Pop {i+1}" for i in range(len(ys))]
    if colors is None:
        colors = ['royalblue', 'tomato', 'mediumseagreen', 'mediumpurple', 'goldenrod', 'teal']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(len(ys)):
        ax.errorbar(x, ys[i], yerr=dys[i], fmt=markers[i], color=colors[i], ecolor=colors[i],
                    capsize=4, markersize=8)
        mean_y, err_mean = weighted_mean(ys[i], dys[i])
        ax.axhline(mean_y, color=colors[i], lw=2, linestyle='--', alpha=0.7,
                   label=f"{labels[i]}: {mean_y:.0f} ± {err_mean:.0f} {unit}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=12)
    else:
        ax.set_xticks(np.arange(int(np.nanmin(x)), int(np.nanmax(x)) + 1))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def species_mean_plot(species_labels, means, errs, ylabel="y", title=""):
    """One point per species (weighted mean over the bars)."""
    x = np.arange(len(species_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(species_labels)):
        ax.errorbar(x[i], means[i], yerr=errs[i], fmt='o', color='black', ecolor='black',
                    capsize=4, markersize=8,
                    label=f"{species_labels[i]}: {means[i]:.2f}±{errs[i]:.2f} ps")

    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=16, loc="upper right")
    ax.grid(True, alpha=0.4)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summary plots per bar")
    parser.add_argument("--tw", default=None, help="Resolution per front bar, CFD vs fit")
    parser.add_argument("--charge", default=None, help="Charge difference fit vs measured, per rear bar")
    parser.add_argument("--resolution", default=None, help="Resolution per intersection and species")
    args = parser.parse_args()
    if not (args.tw or args.charge or args.resolution):
        parser.error("give at least one of --tw, --charge, --resolution")

    bars_rear  = np.arange(20)
    bars_front = np.arange(20, 40)

    if args.charge:
        ch = load_charge_data(args.charge)
        s = [x.lower() for x in SPECIES]
        scatter_plot(x=bars_rear,
                     ys=[ch[f"mu_{k}"] for k in s], dys=[ch[f"sigma_{k}"] for k in s],
                     labels=SPECIES, xlabel="Bar Id", ylabel=r"$\mu \pm \sigma$ (%)", unit="%",
                     title=r"Relative charge difference: $(Q_{fit} - Q_{orig}) / Q_{orig}$")
        scatter_plot(x=bars_rear,
                     ys=[ch[f"sigma_{k}"] for k in s], dys=[ch[f"dsigma_{k}"] for k in s],
                     labels=SPECIES, xlabel="Bar Id", ylabel=r"$\sigma \pm d\sigma$ (%)", unit="%",
                     title=r"Relative charge difference: $(Q_{fit} - Q_{orig}) / Q_{orig}$")

    if args.tw:
        tw = load_tw_data(args.tw)
        scatter_plot(x=bars_front,
                     ys=[tw["p_cfd"], tw["p_fit"]], dys=[tw["dp_cfd"], tw["dp_fit"]],
                     labels=["simple CFD", "fit time"], colors=["seagreen", "firebrick"],
                     xlabel="Bar Id", ylabel=r"$\sigma (t_{\mathrm{front}}-t_{\mathrm{rear}})$ (ps)")

    if args.resolution:
        sigma_tw, dsigma_tw = load_tw_resolution(args.resolution)
        # resolution of the TW time (mean of the two crossing bars) = sigma(t_front - t_rear) / 2
        for s in sigma_tw:
            sigma_tw[s]  = sigma_tw[s]  / 2
            dsigma_tw[s] = dsigma_tw[s] / 2
        bars_intersection = np.arange(20)   # rear bar id; the front bar is +20

        scatter_plot(x=bars_intersection,
                     ys=[sigma_tw[s] for s in SPECIES], dys=[dsigma_tw[s] for s in SPECIES],
                     labels=SPECIES, xlabel="Bar intersections", ylabel=r"$\sigma_{TW}$ (ps)",
                     title="TOF-Wall Time resolution",
                     xticklabels=[f"{i}-{i+20}" for i in range(20)])

        means, errs = zip(*(weighted_mean(sigma_tw[s], dsigma_tw[s]) for s in SPECIES))
        species_mean_plot(SPECIES, means, errs, ylabel=r"$\sigma_{TW}$ (ps)", title="TOF-Wall Time resolution")
