#!/usr/bin/env python3
"""
tw_resolution_vs_beta.py — TW time resolution sigma(t_front - t_rear) of the
SHOE points in bins of beta or of the energy loss (BIN_VAR).

beta of each point:
    path length = distance target -> intersection (row, column)
    time        = tof - t0, with t0 the time of the primaries from ST to target
    beta        = path length / time / c
Geometry (FOOT_8028.geo, run without magnetic field): target at (0, 0, 0), TW
origin at (-1, -1, 169.55) cm; intersection at X = -1 + LAYER0_X[column],
Y = -1 + LAYER1_Y[row], Z = 169.55. Primaries: 12C at 201 MeV/u, beta = 0.5688.

Bin variable (BIN_VAR):
    "beta"        beta as above
    "main_eloss"  energy released in the main layer (MeV); points with
                  main_eloss <= 0 are discarded

Event selection in each bin (SELECTION_MODE):
    "box"    all the points with the variable in [lo, hi)
    "gauss"  accept-reject: the selected points follow A*G(x), a gaussian
             centred in the bin with sigma = GAUSS_SIGMA_FRACTION * bin width,
             with A = counts of the central sub-bin (where the distribution is
             below the gaussian all the points are kept)

sigma in each bin: gaussian fit (SIGMA_METHOD = "fit") or rms within
median ± RMS_NSIGMA * sigma_MAD ("rms").

Without --bins: only the distribution of the variable (to choose the bins).
With --output-dir the table is saved as
    tof_resolution_vs_<BIN_VAR>[_gauss<f>][_rms][_easym<c>]_<Z<n>|allZ>.txt
read by plot_tof_vs_beta.py.

Usage:
    python3 tw_resolution_vs_beta.py --points points.pkl
    python3 tw_resolution_vs_beta.py --points points.pkl --bins 20 40 80 160 320 --output-dir out/
    python3 tw_resolution_vs_beta.py --points points.pkl --bins 0.3 0.4 0.5 --z 6 --plot-bin 0 1
"""

import os
import pickle
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

# =============================================================================
# PARAMETERS
# =============================================================================

C_LIGHT = 29.9792458        # cm/ns

ST_TARGET_DIST = 47.15      # cm, |StartPosZ - TargetPosZ| (FOOT_8028.geo)
TW_OFFSET_X = -1.0          # cm, TW position (FOOT_8028.geo, no magnetic field)
TW_OFFSET_Y = -1.0          # cm
TW_OFFSET_Z = 169.55        # cm, target -> TW

# bar centres (cm), TATWdetector.geo (CNAO2025)
LAYER0_X = {i: 19.0 - 2.0 * i for i in range(20)}   # column/rear
LAYER1_Y = {i: 19.0 - 2.0 * i for i in range(20)}   # row/front

BETA_PRIMARY = 0.568800     # 12C at 201 MeV/u
GAMMA_PRIMARY = 1.215842

# optional cut on the Z=6 points: tof > TOF_CUT_Z6 or main_eloss > ELOSS_CUT_Z6 removed
CUT_Z6_ENABLED = False
TOF_CUT_Z6 = 13.5           # ns
ELOSS_CUT_Z6 = 70.0         # MeV

BIN_VAR = "main_eloss"      # "beta" or "main_eloss"
VAR_LABELS = {"beta": "beta", "main_eloss": r"E$_{loss}$ (MeV)"}

# distribution of the variable
DIST_RANGE = {"beta": (0.25, 0.7), "main_eloss": None}   # None = from the DIST_PCT percentiles
DIST_PCT = (0.01, 99.99)
DIST_NBINS = 600
DIST_LOGY = True

# selection in each bin
SELECTION_MODE = "gauss"        # "box" or "gauss"
GAUSS_SIGMA_FRACTION = 0.2      # sigma of the selection gaussian / bin width
GAUSS_RESTRICT_TO_BIN = False   # True: also require the variable inside the bin
GAUSS_SEED = 12345              # random seed (reproducible selection)
GAUSS_N_SUBBINS = 41            # odd number of sub-bins over ±4 sigma

# sigma in each bin
SIGMA_METHOD = "fit"            # "fit" or "rms"
RMS_NSIGMA = 4.0                # rms within median ± RMS_NSIGMA * sigma_MAD; None = all

PLOT_SELECTION = True           # distribution of the selected points in each bin
PLOT_DISTRIBUTION = True        # distribution of the variable, with the bin edges
PLOT_DT_VS_VAR = False          # scatter dt vs BIN_VAR
PLOT_DT_VS_VAR_3D = False       # 2D histogram dt vs BIN_VAR

HIST3D_NBINS = (50, 50)
HIST3D_STYLE = "2d"             # "surface", "bars" or "2d"
HIST3D_LOGZ = False
HIST3D_CMAP = "viridis"

# optional cut on the front/rear energy asymmetry |eloss1 - eloss2| / eloss1
ELOSS_ASYM_CUT_ENABLED = False
ELOSS_ASYM_CUT = 0.05

# gaussian fit of dt
MIN_ENTRIES_PER_BIN = 100
BINS_PER_SIGMA = 10             # histogram bin width = sigma/BINS_PER_SIGMA,
MIN_COUNTS_PER_BIN = 30         # but at least MIN_COUNTS_PER_BIN entries per bin on average
N_SIGMA_RANGE = 4.0
N_SIGMA_FIT = 2.0

MAX_POINTS_SCATTER = 200000     # random subsample per Z in the scatter plots (None = all)
SCATTER_PCT = (0.1, 99.9)       # axis limits from these percentiles
Z_COLORS = {1: "#0000ff", 2: "#00a000", 3: "#ff9900",
            4: "#8b4513", 5: "#8000ff", 6: "#ff0000"}

# parametrised Bethe-Bloch of SHOE (TATW_BBparameters_12C_200.cfg):
# Eloss = Norm * Z^2 * (Const * tof^2 - 1), Z -> (Norm, Const)
BB_PARAMS_LAYERY = {   # row/front, with tof1
    1: (0.28632, 0.034845), 2: (0.30184, 0.033549), 3: (0.33114, 0.031323),
    4: (0.42850, 0.025726), 5: (0.49078, 0.023228), 6: (0.65024, 0.019050),
}
BB_PARAMS_LAYERX = {   # column/rear, with tof2
    1: (0.23786, 0.040868), 2: (0.24850, 0.039624), 3: (0.27403, 0.036649),
    4: (0.28090, 0.036119), 5: (0.23377, 0.042326), 6: (0.17973, 0.053624),
}


# =============================================================================
# INPUT AND CUTS
# =============================================================================

def load_pkl(path, key):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    items = payload[key]
    print(f"{key.capitalize()} loaded: {len(items)}")
    return items


def apply_cut_z6(points):
    """Remove the Z=6 points with tof > TOF_CUT_Z6 or main_eloss > ELOSS_CUT_Z6."""
    if not CUT_Z6_ENABLED:
        return points
    kept = [p for p in points
            if not (p["chargeZ"] == 6 and (p["tof"] > TOF_CUT_Z6 or p["main_eloss"] > ELOSS_CUT_Z6))]
    print(f"[INFO] Z=6 cut (tof > {TOF_CUT_Z6} ns or main_eloss > {ELOSS_CUT_Z6} MeV): "
          f"{len(points) - len(kept)} points removed, {len(kept)} left")
    return kept


def apply_cut_eloss_asym(points):
    """Remove the points with |eloss1 - eloss2| / eloss1 > ELOSS_ASYM_CUT (or eloss1 <= 0)."""
    if not ELOSS_ASYM_CUT_ENABLED:
        return points
    kept = [p for p in points
            if p["eloss1"] > 0 and abs(p["eloss1"] - p["eloss2"]) / p["eloss1"] <= ELOSS_ASYM_CUT]
    print(f"[INFO] Front/rear energy cut (|r| > {100 * ELOSS_ASYM_CUT:g}%): "
          f"{len(points) - len(kept)} points removed, {len(kept)} left")
    return kept


# =============================================================================
# GEOMETRY, BETA AND ENERGY LOSS
# =============================================================================

def compute_t0():
    """Time of the primaries from the ST to the target (ns)."""
    return ST_TARGET_DIST / (BETA_PRIMARY * C_LIGHT)


def compute_path_length(row, column):
    """Distance target -> intersection (row, column), cm."""
    dx = TW_OFFSET_X + LAYER0_X[column]
    dy = TW_OFFSET_Y + LAYER1_Y[row]
    dz = TW_OFFSET_Z
    return np.sqrt(dx**2 + dy**2 + dz**2)


def predict_eloss(tof_layer, z, params_table):
    if z not in params_table:
        return None
    norm, const = params_table[z]
    return norm * z**2 * (const * tof_layer**2 - 1)


def build_beta_data(points, z_filter=None):
    """
    One record per point: beta, dt = tof1 - tof2, main_eloss, predicted energy
    loss in the two layers (row with tof1, column with tof2), row, column.
    """
    t0 = compute_t0()
    print(f"t0 (primaries, ST -> target) = {t0:.4f} ns")

    records = []
    n_filtered_z = 0
    n_bad_time = 0

    for p in points:
        if z_filter is not None and p["chargeZ"] != z_filter:
            n_filtered_z += 1
            continue

        path_length = compute_path_length(p["row"], p["column"])
        time = p["tof"] - t0
        if time <= 0 or not np.isfinite(time):
            n_bad_time += 1
            continue

        beta = path_length / time / C_LIGHT

        records.append({
            "beta": beta, "dt": p["tof1"] - p["tof2"], "chargeZ": p["chargeZ"],
            "main_eloss": p["main_eloss"],
            "eloss_pred_row": predict_eloss(p["tof1"], p["chargeZ"], BB_PARAMS_LAYERY),
            "eloss_pred_col": predict_eloss(p["tof2"], p["chargeZ"], BB_PARAMS_LAYERX),
            "row": p["row"], "column": p["column"],
        })

    print(f"Points: {len(points)}, other Z (--z={z_filter}): {n_filtered_z}, "
          f"invalid time: {n_bad_time}, left: {len(records)}")
    return records


# =============================================================================
# SIGMA OF dt
# =============================================================================

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_sigma_tof(dt_values, bins_per_sigma=BINS_PER_SIGMA, min_counts_per_bin=MIN_COUNTS_PER_BIN,
                  n_sigma_range=N_SIGMA_RANGE, n_sigma_fit=N_SIGMA_FIT):
    """Gaussian fit around the peak (same method as tw_resolution.py). Values in ns."""
    dt_values = np.asarray(dt_values)
    dt_values = dt_values[np.isfinite(dt_values)]
    if len(dt_values) < MIN_ENTRIES_PER_BIN:
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


def rms_sigma_tof(dt_values):
    """
    sigma of dt as the rms (n-1) of the selected values within median ±
    RMS_NSIGMA * sigma_MAD. Same keys as fit_sigma_tof (popt/rng/bins only for the plot).
    """
    dt = np.asarray(dt_values)
    dt = dt[np.isfinite(dt)]
    if len(dt) < MIN_ENTRIES_PER_BIN:
        return None
    if RMS_NSIGMA is not None:
        med = np.median(dt)
        s_mad = 1.4826 * np.median(np.abs(dt - med))
        if s_mad > 0:
            dt = dt[np.abs(dt - med) <= RMS_NSIGMA * s_mad]
    n = len(dt)
    if n < 2:
        return None
    mu, rms = dt.mean(), dt.std(ddof=1)
    rng = (mu - N_SIGMA_RANGE * rms, mu + N_SIGMA_RANGE * rms)
    bins = 100
    amp = n * (rng[1] - rng[0]) / bins / (rms * np.sqrt(2 * np.pi))   # gaussian with the same area
    return {"mu": mu, "dmu": rms / np.sqrt(n), "sigma": rms, "dsigma": rms / np.sqrt(2 * (n - 1)),
            "n": n, "popt": [amp, mu, rms], "rng": rng, "bins": bins}


def select_bin(var_arr, lo, hi, rng_gen):
    """
    Boolean mask of the points used for the bin [lo, hi) (SELECTION_MODE).
    "gauss": a point is kept with probability min(1, A*G(x)/f), with f the
    counts of its sub-bin and A those of the central sub-bin.
    """
    in_bin = (var_arr >= lo) & (var_arr < hi)
    if SELECTION_MODE == "box":
        return in_bin

    center = 0.5 * (lo + hi)
    sigma_sel = GAUSS_SIGMA_FRACTION * (hi - lo)
    sub_edges = np.linspace(center - 4 * sigma_sel, center + 4 * sigma_sel, GAUSS_N_SUBBINS + 1)
    counts, _ = np.histogram(var_arr, bins=sub_edges)
    A = counts[GAUSS_N_SUBBINS // 2]

    k = np.digitize(var_arr, sub_edges) - 1
    in_win = (k >= 0) & (k < GAUSS_N_SUBBINS)
    prob = np.zeros(var_arr.size)
    g = np.exp(-0.5 * ((var_arr[in_win] - center) / sigma_sel) ** 2)
    f = counts[k[in_win]]
    prob[in_win] = np.minimum(1.0, A * g / f)          # f >= 1 by construction

    keep = rng_gen.random(var_arr.size) < prob
    if GAUSS_RESTRICT_TO_BIN:
        keep &= in_bin
    return keep


# =============================================================================
# PLOTS
# =============================================================================

def plot_var_distribution(var_arr, z_arr, edges, z_filter=None):
    """Distribution of BIN_VAR (all Z and, without --z, each Z), with the bin edges."""
    rng = DIST_RANGE.get(BIN_VAR)
    if rng is None:
        rng = tuple(np.percentile(var_arr, DIST_PCT))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(var_arr, bins=DIST_NBINS, range=rng, color="gold", histtype="step", lw=2,
            label=f"all ({var_arr.size})")
    if z_filter is None:
        for z in sorted(set(z_arr.tolist())):
            ax.hist(var_arr[z_arr == z], bins=DIST_NBINS, range=rng, color=Z_COLORS.get(z, "black"),
                    histtype="step", lw=1.5)
    for e in edges or []:
        ax.axvline(e, color="black", linestyle="--", lw=1.5)
    ax.set_xlim(rng)
    ax.set_xlabel(VAR_LABELS[BIN_VAR], fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    if DIST_LOGY:
        ax.set_yscale("log")
    z_str = f"Z={z_filter}" if z_filter is not None else "all Z"
    ax.set_title(f"{BIN_VAR} distribution — {z_str} — {var_arr.size} entries", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_selection(var_arr, edges, selections):
    """Distribution of the variable for all the points and for the points selected in each bin."""
    lo = edges[0] - 0.5 * (edges[1] - edges[0])
    hi = edges[-1] + 0.5 * (edges[-1] - edges[-2])
    rng = (lo, hi)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(var_arr, bins=DIST_NBINS, range=rng, histtype="step", color="gray", lw=1.5, label="all")
    for idx, sel in selections.items():
        ax.hist(var_arr[sel], bins=DIST_NBINS, range=rng, histtype="step", lw=2,
                label=f"bin {idx} ({sel.sum()})")
    for e in edges:
        ax.axvline(e, color="black", linestyle="--", lw=1)
    ax.set_xlim(rng)
    ax.set_xlabel(VAR_LABELS[BIN_VAR])
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_var_bin(dt_values, fit, lo, hi, z_filter=None):
    """dt histogram of one bin with the fitted gaussian."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(dt_values, bins=fit["bins"], range=fit["rng"], color="#1b7a3d", histtype="step", lw=2)
    x_plot = np.linspace(fit["rng"][0], fit["rng"][1], 500)
    ax.plot(x_plot, gauss(x_plot, *fit["popt"]), color="black", lw=2,
            label=f"sigma={fit['sigma']*1e3:.1f}±{fit['dsigma']*1e3:.1f} ps")
    ax.set_xlabel("dt = tof1 - tof2 (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    z_str = f"Z={z_filter}" if z_filter is not None else ""
    if SELECTION_MODE == "gauss":
        sel_str = f"{BIN_VAR} ~ gauss({0.5*(lo+hi):.4g}, {GAUSS_SIGMA_FRACTION*(hi-lo):.3g})"
    else:
        sel_str = f"{BIN_VAR} in [{lo:.4g}, {hi:.4g}]"
    ax.set_title(f"sigma_tof — {z_str} — {sel_str} — {fit['n']} entries", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dt_vs_var(records):
    """Scatter dt (front - rear) vs BIN_VAR: one figure per Z and one with all the Z."""
    rng_gen = np.random.default_rng(0)
    x = np.array([r[BIN_VAR] for r in records])
    dt = np.array([r["dt"] for r in records])
    zs = np.array([r["chargeZ"] for r in records])

    ok = np.isfinite(x) & np.isfinite(dt)
    x, dt, zs = x[ok], dt[ok], zs[ok]
    xlim = np.percentile(x, SCATTER_PCT)
    ylim = np.percentile(dt, SCATTER_PCT)

    def subsample(idx):
        if MAX_POINTS_SCATTER is not None and idx.size > MAX_POINTS_SCATTER:
            return rng_gen.choice(idx, MAX_POINTS_SCATTER, replace=False)
        return idx

    def finish(ax, title, legend=False):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(VAR_LABELS[BIN_VAR], fontsize=14)
        ax.set_ylabel(r"$\Delta t = t_{front} - t_{rear}$ (ns)", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=13, markerscale=8)
        plt.tight_layout()

    z_list = sorted(set(zs.tolist()))

    for z in z_list:
        idx = subsample(np.flatnonzero(zs == z))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(x[idx], dt[idx], s=1.5, alpha=0.5, color=Z_COLORS.get(z, "black"), rasterized=True)
        n_tot = int((zs == z).sum())
        finish(ax, rf"$\Delta t$ vs {BIN_VAR} — Z={z} — {n_tot} points" + (f" ({idx.size} shown)" if idx.size < n_tot else ""))
        plt.show()

    if len(z_list) > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        for z in z_list:
            idx = subsample(np.flatnonzero(zs == z))
            ax.scatter(x[idx], dt[idx], s=1, alpha=0.3, color=Z_COLORS.get(z, "black"),
                       rasterized=True, label=f"Z={z}")
        finish(ax, rf"$\Delta t$ vs {BIN_VAR} — all Z — {zs.size} points", legend=True)
        plt.show()


def plot_dt_vs_var_3d(records):
    """2D histogram of the counts in (BIN_VAR, dt), as a colour map or in 3D: one figure per Z and one with all."""
    x = np.array([r[BIN_VAR] for r in records], dtype=float)
    dt = np.array([r["dt"] for r in records], dtype=float)
    zs = np.array([r["chargeZ"] for r in records])
    ok = np.isfinite(x) & np.isfinite(dt)
    x, dt, zs = x[ok], dt[ok], zs[ok]

    def one_figure(sel, title):
        xs, ys = x[sel], dt[sel]
        xr = tuple(np.percentile(xs, SCATTER_PCT))
        yr = tuple(np.percentile(ys, SCATTER_PCT))
        H, xe, ye = np.histogram2d(xs, ys, bins=HIST3D_NBINS, range=[xr, yr])
        Hz = np.log10(H + 1) if HIST3D_LOGZ else H
        zlabel = r"log$_{10}$(counts + 1)" if HIST3D_LOGZ else "Counts"
        xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])

        if HIST3D_STYLE == "2d":
            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.pcolormesh(xe, ye, Hz.T, cmap=HIST3D_CMAP, shading="flat")
            fig.colorbar(im, ax=ax, label=zlabel)
        else:
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(projection="3d")
            if HIST3D_STYLE == "bars":
                X, Y = np.meshgrid(xe[:-1], ye[:-1], indexing="ij")
                dz = Hz.ravel()
                colors = plt.get_cmap(HIST3D_CMAP)(dz / dz.max() if dz.max() > 0 else dz)
                ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz),
                         np.diff(xe)[0], np.diff(ye)[0], dz, color=colors, shade=True)
            else:
                X, Y = np.meshgrid(xc, yc, indexing="ij")
                ax.plot_surface(X, Y, Hz, cmap=HIST3D_CMAP, rstride=1, cstride=1, linewidth=0)
            ax.set_zlabel(zlabel, labelpad=10)
        ax.set_xlabel(VAR_LABELS[BIN_VAR], labelpad=10)
        ax.set_ylabel(r"$\Delta t = t_{front} - t_{rear}$ (ns)", labelpad=10)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    for z in sorted(set(zs.tolist())):
        sel = zs == z
        one_figure(sel, rf"$\Delta t$ vs {BIN_VAR} — Z={z} — {sel.sum()} points")
    if len(set(zs.tolist())) > 1:
        one_figure(np.ones_like(zs, dtype=bool), rf"$\Delta t$ vs {BIN_VAR} — all Z — {zs.size} points")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TW resolution in bins of beta or of the energy loss")
    parser.add_argument("--points", required=True, help="points.pkl from extract_shoe.py")
    parser.add_argument("--bins", type=float, nargs="+", default=None,
                        help="Bin edges of BIN_VAR (N+1 values for N bins); without: distribution only")
    parser.add_argument("--z", type=int, default=None, help="Only this reconstructed Z")
    parser.add_argument("--plot-bin", type=int, nargs="+", default=None,
                        help="Bins (0-based) for which the dt histogram is shown")
    parser.add_argument("--output-dir", default=None, help="Folder for the output table")
    args = parser.parse_args()

    if args.bins is not None and len(args.bins) < 2:
        parser.error("--bins needs at least 2 values")
    if BIN_VAR not in VAR_LABELS:
        parser.error(f"BIN_VAR = {BIN_VAR!r} not valid, use 'beta' or 'main_eloss'")

    points = load_pkl(args.points, "points")
    points = apply_cut_z6(points)
    points = apply_cut_eloss_asym(points)
    print(f"\n[INFO] Bins in {BIN_VAR}")
    records = build_beta_data(points, z_filter=args.z)

    var_arr = np.array([r[BIN_VAR] for r in records], dtype=float)
    dt_arr = np.array([r["dt"] for r in records])
    z_arr = np.array([r["chargeZ"] for r in records])

    ok = np.isfinite(var_arr)
    if BIN_VAR == "main_eloss":
        ok &= var_arr > 0
    if (~ok).any():
        print(f"[WARNING] {(~ok).sum()} points with invalid {BIN_VAR} discarded")
    var_arr, dt_arr, z_arr = var_arr[ok], dt_arr[ok], z_arr[ok]
    records = [r for r, k in zip(records, ok) if k]

    edges = sorted(args.bins) if args.bins else None
    plot_bins = set(args.plot_bin) if args.plot_bin else set()

    if PLOT_DISTRIBUTION:
        plot_var_distribution(var_arr, z_arr, edges, z_filter=args.z)
    if PLOT_DT_VS_VAR:
        plot_dt_vs_var(records)
    if PLOT_DT_VS_VAR_3D:
        plot_dt_vs_var_3d(records)

    if edges is None:
        print("\n[INFO] No --bins: distribution only.")
        return

    rng_gen = np.random.default_rng(GAUSS_SEED)
    selections = {idx: select_bin(var_arr, edges[idx], edges[idx + 1], rng_gen)
                  for idx in range(len(edges) - 1)}
    if SELECTION_MODE == "gauss":
        print(f"\n[INFO] Gaussian selection: sigma = {GAUSS_SIGMA_FRACTION} x bin width, "
              f"restrict_to_bin={GAUSS_RESTRICT_TO_BIN}, seed={GAUSS_SEED}")
        if PLOT_SELECTION:
            plot_selection(var_arr, edges, selections)
    else:
        print("\n[INFO] Box selection: all the points in the bin")

    print(f"\n{'bin':>4} {'lo':>9} {'hi':>9} {'x_mean':>10} {'x_std':>9} {'mu[ns]':>10} "
          f"{'sigma[ps]':>11} {'dsigma[ps]':>12} {'n':>8}")
    # the header lines are read by plot_tof_vs_beta.py
    results_lines = [f"# selection: {SELECTION_MODE}"
                     + (f", sigma = {GAUSS_SIGMA_FRACTION} x bin, restrict_to_bin={GAUSS_RESTRICT_TO_BIN}, "
                        f"seed={GAUSS_SEED}" if SELECTION_MODE == "gauss" else ""),
                     f"# variable: {BIN_VAR}",
                     "#bin  lo  hi  mu[ns]  sigma[ps]  dsigma[ps]  n  x_mean  x_std",
                     f"# sigma estimate: {SIGMA_METHOD}"
                     + (f", window median ± {RMS_NSIGMA} sigma_MAD" if SIGMA_METHOD == "rms" and RMS_NSIGMA is not None
                        else (", all selected" if SIGMA_METHOD == "rms" else "")),
                     f"# eloss_asym cut: {ELOSS_ASYM_CUT if ELOSS_ASYM_CUT_ENABLED else 'off'}"]

    for idx in range(len(edges) - 1):
        sel = selections[idx]
        dt_values = dt_arr[sel]
        fit = rms_sigma_tof(dt_values) if SIGMA_METHOD == "rms" else fit_sigma_tof(dt_values)
        lo, hi = edges[idx], edges[idx + 1]
        x_in_bin = var_arr[sel]
        x_mean = x_in_bin.mean() if x_in_bin.size else float("nan")
        x_std = x_in_bin.std() if x_in_bin.size else float("nan")
        if fit is None:
            print(f"{idx:>4} {lo:>9.4g} {hi:>9.4g}   too few entries ({len(dt_values)}) or fit failed")
            continue
        print(f"{idx:>4} {lo:>9.4g} {hi:>9.4g} {x_mean:>10.5g} {x_std:>9.4g} {fit['mu']:>10.4f} "
              f"{fit['sigma']*1e3:>11.2f} {fit['dsigma']*1e3:>12.2f} {fit['n']:>8}")
        results_lines.append(f"{idx}  {lo:.6g}  {hi:.6g}  {fit['mu']:.4f}  "
                             f"{fit['sigma']*1e3:.2f}  {fit['dsigma']*1e3:.2f}  {fit['n']}  "
                             f"{x_mean:.6g}  {x_std:.6g}")

        if idx in plot_bins:
            plot_var_bin(dt_values, fit, lo, hi, z_filter=args.z)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        z_tag = f"Z{args.z}" if args.z is not None else "allZ"
        sel_tag = f"_gauss{GAUSS_SIGMA_FRACTION:g}" if SELECTION_MODE == "gauss" else ""
        rms_tag = "_rms" if SIGMA_METHOD == "rms" else ""
        cut_tag = f"_easym{100 * ELOSS_ASYM_CUT:g}" if ELOSS_ASYM_CUT_ENABLED else ""
        out_path = os.path.join(args.output_dir, f"tof_resolution_vs_{BIN_VAR}{sel_tag}{rms_tag}{cut_tag}_{z_tag}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(results_lines) + "\n")
        print(f"\n[INFO] Table saved to: {out_path}")


if __name__ == "__main__":
    main()
