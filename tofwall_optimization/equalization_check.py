#!/usr/bin/env python3
"""
equalization_check.py — Check of the charge equalisation along the bar.

For one bar and the chosen Z, compares the energy-loss distribution of the
SHOE hits ("energy", the quantity corrected by the equalisation) at the centre
of the bar (|x| < 5 cm) and at the ends (15 < |x| < 20 cm). Each distribution
is fitted with a gaussian around its peak:
    1. mu0 = most populated bin (fine histogram smoothed over 3 bins)
    2. first fit within mu0 ± N_SIGMA_FIT * sigma0 (sigma0 from the MAD)
    3. second fit centred on mu and sigma of the first one
and the ratio ends/centre of the fitted means is printed with its error:
> 1 without equalisation (higher signal at the ends), ~ 1 with it.

--save stores energies and fit results in a small .npz, for combined figures.

Usage:
    python3 equalization_check.py hits.pkl --bar 6 --z 1 2 [--save bar6_noeq.npz]
"""

import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})

CENTER = (-5.0, 5.0)    # cm
EDGE   = (15.0, 20.0)   # cm, in absolute value: (-20,-15) U (15,20)
N_SIGMA_FIT  = 1.5      # fit within mu ± N_SIGMA_FIT * sigma
N_SIGMA_PLOT = 4.0      # the gaussian is drawn within mu ± N_SIGMA_PLOT * sigma
PLOT_BIN_FACTOR = 3.0   # plot bins PLOT_BIN_FACTOR times wider than the fit bins


def gauss(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def find_mode(e, s0):
    """Peak of the distribution: most populated bin of a fine histogram smoothed over 3 bins."""
    med = np.median(e)
    lo, hi = med - 4 * s0, med + 4 * s0
    nb = max(20, int((hi - lo) / (s0 / 5)))
    counts, edges = np.histogram(e, bins=nb, range=(lo, hi))
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[np.argmax(smooth)]


def _fit_window(e, mu0, s0, bins):
    """Gaussian fit of the histogram of e within mu0 ± N_SIGMA_FIT * s0 (Poisson errors)."""
    lo, hi = mu0 - N_SIGMA_FIT * s0, mu0 + N_SIGMA_FIT * s0
    counts, edges = np.histogram(e, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(np.maximum(counts, 1))
    popt, pcov = curve_fit(gauss, centers, counts, p0=[counts.max(), mu0, s0],
                           sigma=err, absolute_sigma=True, maxfev=10000)
    return popt, pcov, (lo, hi)


def robust_gauss_fit(e, bins):
    """Peak search + two gaussian fits; returns a, mu, dmu, sigma, bin width, range and n."""
    med = np.median(e)
    s0 = 1.4826 * np.median(np.abs(e - med))
    mu0 = find_mode(e, s0)
    popt, pcov, rng = _fit_window(e, mu0, s0, bins)
    popt, pcov, rng = _fit_window(e, popt[1], abs(popt[2]), bins)
    a, mu, s = popt
    return dict(a=a, mu=mu, dmu=np.sqrt(pcov[1, 1]), sigma=abs(s),
                bw=(rng[1] - rng[0]) / bins, rng=rng, n=e.size)


def main():
    parser = argparse.ArgumentParser(description="Check of the charge equalisation along the bar")
    parser.add_argument("pkl", help="hits.pkl (extract_shoe.py --hits)")
    parser.add_argument("--bar", type=int, required=True, help="bar_id (0-39)")
    parser.add_argument("--z", type=int, nargs="+", required=True, help="one or more Z, e.g. --z 1 2")
    parser.add_argument("--bins", type=int, default=40, help="number of bins in the fit window")
    parser.add_argument("--save", default=None, help="save energies and fit results to this .npz")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        hits = pickle.load(f)["hits"]

    data = {z: {"center": [], "edge": []} for z in args.z}
    for h in hits:
        if h["bar_id"] != args.bar or h["chargeZ"] not in data or h["energy"] <= 0:
            continue
        x = h["coordinate"]
        if CENTER[0] < x < CENTER[1]:
            data[h["chargeZ"]]["center"].append(h["energy"])
        elif EDGE[0] < abs(x) < EDGE[1]:
            data[h["chargeZ"]]["edge"].append(h["energy"])
    del hits

    fig, axes = plt.subplots(1, len(args.z), figsize=(7 * len(args.z), 5), squeeze=False)
    to_save = {"bar": args.bar}
    for ax, z in zip(axes[0], args.z):
        ec, ee = np.array(data[z]["center"]), np.array(data[z]["edge"])
        if ec.size < 50 or ee.size < 50:
            print(f"Z={z}: too few hits (centre {ec.size}, ends {ee.size})")
            continue
        try:
            fc = robust_gauss_fit(ec, args.bins)
            fe = robust_gauss_fit(ee, args.bins)
        except RuntimeError:
            print(f"Z={z}: fit did not converge")
            continue

        r = fe["mu"] / fc["mu"]
        dr = r * np.hypot(fe["dmu"] / fe["mu"], fc["dmu"] / fc["mu"])
        print(f"Z={z}: centre mu = {fc['mu']:.3f} ± {fc['dmu']:.3f} MeV (sigma {fc['sigma']:.3f}, n={fc['n']}) | "
              f"ends mu = {fe['mu']:.3f} ± {fe['dmu']:.3f} MeV (sigma {fe['sigma']:.3f}, n={fe['n']}) | "
              f"ends/centre = {r:.3f} ± {dr:.3f}")

        # histograms and gaussians scaled so that the peak of the fitted gaussian is 1
        all_e = np.concatenate([ec, ee])
        s_all = 1.4826 * np.median(np.abs(all_e - np.median(all_e)))
        lo = min(fc["mu"], fe["mu"]) - 4 * s_all
        hi = max(fc["mu"], fe["mu"]) + 6 * s_all
        nb = max(20, int((hi - lo) / (PLOT_BIN_FACTOR * fc["bw"])))
        for e, f, col, lab in [(ec, fc, "tab:blue", "center"),
                               (ee, fe, "tab:red", "ends")]:
            h, edges = np.histogram(e, bins=nb, range=(lo, hi), density=True)
            a_dens = f["a"] / (f["n"] * f["bw"])   # fit amplitude in density units
            norm = 1.0 / a_dens
            ax.stairs(h * norm, edges, lw=1.5, color=col, alpha=0.6)
            xf = np.linspace(f["mu"] - N_SIGMA_PLOT * f["sigma"],
                             f["mu"] + N_SIGMA_PLOT * f["sigma"], 300)
            ax.plot(xf, gauss(xf, a_dens * norm, f["mu"], f["sigma"]), color=col, lw=2.5,
                    label=f"{lab}: $\\mu$ = {f['mu']:.3f} ± {f['dmu']:.3f} MeV")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$E_{\mathrm{loss}}$ (MeV)")
        ax.set_ylabel("Normalized counts")
        ax.set_title(f"Bar {args.bar} — Z={z} — ends/center = {r:.3f} ± {dr:.3f}")
        ax.legend(loc="upper right")

        to_save.update({
            f"center_z{z}": ec, f"edge_z{z}": ee,
            f"fit_center_z{z}": np.array([fc["a"], fc["mu"], fc["dmu"], fc["sigma"], fc["bw"], fc["n"]]),
            f"fit_edge_z{z}":   np.array([fe["a"], fe["mu"], fe["dmu"], fe["sigma"], fe["bw"], fe["n"]]),
            f"ratio_z{z}": np.array([r, dr]),
        })

    plt.tight_layout()
    plt.show()

    if args.save:
        np.savez(args.save, **to_save)
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()