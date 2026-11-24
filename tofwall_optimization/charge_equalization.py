#!/usr/bin/env python3
"""
charge_equalization.py — Equalisation of the TW charge along the bar.

For every bar the charge of the SHOE hits vs the position along the bar is
fitted with a parabola f(x) = A*x^2 + B*x + C, in two steps: a first fit, then
the points whose normalised residual exceeds OUTLIER_THRESH are excluded and
the fit is repeated. The parameters are normalised to the minimum of the
parabola (C if A is not significant), so that the equalised charge

    Q_eq = Q_raw / f(x)

keeps the scale of the bar centre and can be applied to all the fragments.
The outliers are excluded from the fit only; the correction applies to all hits.

Output in --output-dir:
    charge_equalization.txt   bar_id  A  dA  B  dB  C  dC  (normalised parameters)
    eq_xmin.pdf               position of the minimum per bar (with --plot-xmin)

Usage:
    python3 charge_equalization.py <hits.pkl> --output-dir DIR [--plot-bar 7 29] [--plot-xmin]
"""

import os
import sys
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
    "legend.fontsize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTLIER_THRESH = 0.15        # maximum |Q_norm - f(x)/Q_min| in the final fit
CHARGE_RESOLUTION = 0.035    # relative charge resolution used in the chi2/ndof
MIN_HITS = 100               # minimum number of hits per bar


def load_hits(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    hits = payload["hits"]
    print(f"Hits loaded: {len(hits)}  (runs={payload.get('runs', '?')})")
    return hits


def _parabola(x, A, B, C):
    return A * x**2 + B * x + C


def fit_smile_bar(x_vals, q, outlier_thresh=OUTLIER_THRESH, charge_resolution=CHARGE_RESOLUTION):
    """
    Two-step parabolic fit of the charge q (mV*ns) vs position x_vals (cm).
    Returns a dict with the raw and normalised parameters, the value and the
    position of the minimum (with error from the covariance), the chi2/ndof
    computed with a relative resolution charge_resolution and the outlier
    mask; None if there are too few points or a fit fails.
    """
    if len(x_vals) < MIN_HITS:
        return None

    def _fit(xf, qf):
        popt, pcov = curve_fit(_parabola, xf, qf, p0=[100., 0., 1800.], maxfev=10000)
        perr = np.sqrt(pcov.diagonal())
        A, B, C = popt
        # minimum of the parabola, if A is significant
        fmv = C - B**2 / (4 * A) if abs(A) > 3 * perr[0] else C
        return popt, perr, fmv, pcov

    try:
        popt1, perr1, fit_min_val0, _ = _fit(x_vals, q)
    except Exception as e:
        print(f"    First fit failed: {e}")
        return None

    resid_norm_fit = (q / fit_min_val0) - (_parabola(x_vals, *popt1) / fit_min_val0)
    keep_out = np.abs(resid_norm_fit) <= outlier_thresh
    outlier_mask = ~keep_out
    n_outliers = int(outlier_mask.sum())

    try:
        popt, perr, fit_min_val, pcov = _fit(x_vals[keep_out], q[keep_out])
    except Exception as e:
        print(f"    Final fit failed: {e}")
        return None

    A, B, C = popt
    if abs(A) > 3 * perr[0]:
        x_min = -B / (2 * A)
        dxdA = B / (2 * A**2)
        dxdB = -1 / (2 * A)
        dx_min = float(np.sqrt(dxdA**2 * pcov[0, 0] + dxdB**2 * pcov[1, 1]
                               + 2 * dxdA * dxdB * pcov[0, 1]))
    else:
        x_min = float("nan")
        dx_min = float("nan")

    A_n = A / fit_min_val
    B_n = B / fit_min_val
    C_n = C / fit_min_val
    perr_n = perr / fit_min_val

    # chi2/ndof on the points used in the fit
    q_norm_diag = q[keep_out] / fit_min_val
    pred_norm = _parabola(x_vals[keep_out], *popt) / fit_min_val
    res = (q_norm_diag - pred_norm) / (charge_resolution * q_norm_diag)
    chi2_val = float(np.sum(res**2))
    ndof = int(keep_out.sum()) - 3
    chi2_ndof = chi2_val / ndof if ndof > 0 else float("nan")

    return {
        "A": A, "B": B, "C": C, "dA": perr[0], "dB": perr[1], "dC": perr[2],
        "A_n": A_n, "B_n": B_n, "C_n": C_n, "dA_n": perr_n[0], "dB_n": perr_n[1], "dC_n": perr_n[2],
        "fit_min_val": fit_min_val, "x_min": x_min, "dx_min": dx_min,
        "chi2": chi2_val, "ndof": ndof, "chi2_ndof": chi2_ndof,
        "outlier_mask": outlier_mask, "n_outliers": n_outliers, "n_total": len(x_vals),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_xmin(results, save=None):
    """Position of the minimum x_min = -B/(2A) for every bar, with error (covariance included)."""
    bars  = np.array(sorted(results.keys()))
    x_min = np.array([results[b]["x_min"] for b in bars])
    dx    = np.array([results[b]["dx_min"] for b in bars])
    ok    = np.isfinite(x_min)
    rear  = ok & (bars < 20)
    front = ok & (bars >= 20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.errorbar(bars[rear], x_min[rear], yerr=dx[rear], fmt="o", ms=6,
                color="steelblue", capsize=3, label="rear layer")
    ax.errorbar(bars[front], x_min[front], yerr=dx[front], fmt="s", ms=6,
                color="crimson", capsize=3, label="front layer")
    ax.set_xlabel("Bar ID")
    ax.set_ylabel(r"$x_{\mathrm{min}}$ (cm)")
    ax.set_xticks(np.arange(0, 40, 5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"\n[x_min] bars without a defined minimum (A not significant): {int((~ok).sum())}")
    print(f"[x_min] median |x_min| = {np.median(np.abs(x_min[ok])):.2f} cm, "
          f"max |x_min| = {np.max(np.abs(x_min[ok])):.2f} cm "
          f"(bar {bars[ok][np.argmax(np.abs(x_min[ok]))]})")

    if save:
        fig.savefig(save)
        print(f"Saved: {save}")
    plt.show()


def plot_smile_before(x_vals, q, fit, bar_id, label=""):
    """Normalised charge vs position with the parabolic fit, and the residuals Q_norm - f(x)/Q_min."""
    fmv = fit["fit_min_val"]
    q_norm = q / fmv
    mask = fit["outlier_mask"]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    ax.axhline(1, color="black", lw=1.5, linestyle="--")
    ax.scatter(x_vals[~mask], q_norm[~mask], s=6, alpha=0.5, color="steelblue",
               label="fit data", rasterized=True)
    ax.scatter(x_vals[mask], q_norm[mask], s=6, alpha=0.5, color="red",
               label="outliers", rasterized=True)

    x_fit = np.linspace(x_vals.min(), x_vals.max(), 2000)
    ax.plot(x_fit, _parabola(x_fit, fit["A"], fit["B"], fit["C"]) / fmv, color="black", lw=2, label="fit")

    ax.set_ylabel(r"$Q_{\mathrm{norm}}$")
    ax.set_title(f"Bar {bar_id} ({label}) — Before correction — {fit['n_total']} entries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    resid_n = q_norm - _parabola(x_vals, fit["A"], fit["B"], fit["C"]) / fmv
    axr.axhline(0, color="black", lw=1.5, linestyle="--")
    axr.scatter(x_vals[~mask], resid_n[~mask], s=6, alpha=0.5, color="steelblue", rasterized=True)
    axr.scatter(x_vals[mask], resid_n[mask], s=6, alpha=0.5, color="red", rasterized=True)
    axr.set_xlabel(r"$x$ (cm)")
    axr.set_ylabel("residual")
    axr.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.4)
    axr.set_ylim(-0.2, 0.2)

    plt.tight_layout()
    plt.show()


def plot_smile_after(x_vals, q, fit, bar_id, label=""):
    """Normalised charge vs position after the correction (applied to all hits, outliers included)."""
    fmv = fit["fit_min_val"]
    q_corr_norm = q / _parabola(x_vals, fit["A"], fit["B"], fit["C"])
    mask = fit["outlier_mask"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1, color="black", lw=1.5, linestyle="--")
    ax.scatter(x_vals[~mask], q_corr_norm[~mask], s=6, alpha=0.5, color="seagreen",
               label="fit data", rasterized=True)
    ax.scatter(x_vals[mask], q_corr_norm[mask], s=6, alpha=0.5, color="red",
               label="outliers", rasterized=True)

    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$Q_{\mathrm{norm}}$")
    ax.set_title(f"Bar {bar_id} ({label}) — After correction — {fit['n_total']} entries")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# OUTPUT
# =============================================================================

def write_calibration_file(results, out_path):
    lines = [
        "# TW charge equalisation along the bar: Q_eq = Q_raw / f(x), f(x) = A*x**2 + B*x + C,",
        "# with x the position of the hit along the bar (cm).",
        "# A, B, C are normalised to the minimum of the parabola (fitted on carbon ions), so they",
        "# can be applied directly to all the fragments; the errors are given for completeness.",
        "# bar_id 0-19: layer 0 (rear), 20-39: layer 1 (front).",
        "# bar_id  A  dA  B  dB  C  dC",
    ]
    for bar_id in sorted(results.keys()):
        r = results[bar_id]
        lines.append(f"{bar_id}  {r['A_n']}  {r['dA_n']}  {r['B_n']}  {r['dB_n']}  {r['C_n']}  {r['dC_n']}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] Equalisation file saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="TW charge equalisation along the bar")
    parser.add_argument("hits_pkl", help="hits.pkl from extract_shoe.py --hits")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--plot-bar", type=int, nargs="+", default=None,
                        help="Bars for the before/after plots")
    parser.add_argument("--plot-xmin", action="store_true",
                        help="Position of the minimum of the parabola for all the bars")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hits = load_hits(args.hits_pkl)
    plot_bars = set(args.plot_bar) if args.plot_bar else set()
    results = {}

    for bar_id in sorted({h["bar_id"] for h in hits}):
        label = "rear" if bar_id < 20 else "front"
        print(f"\n{'─'*55}\nBar {bar_id} ({label})")

        sel = [h for h in hits
               if h["bar_id"] == bar_id
               and h.get("charge_bar") is not None
               and h.get("coordinate") is not None]

        x_vals = np.array([h["coordinate"] for h in sel])
        q = np.array([h["charge_bar"] for h in sel]) * 1e3   # V*ns -> mV*ns

        fit = fit_smile_bar(x_vals, q)
        if fit is None:
            print(f"  Too few hits ({len(sel)}) or fit failed, bar skipped.")
            continue

        print(f"  Hits: {fit['n_total']}, outliers excluded from the fit: {fit['n_outliers']} "
              f"({100*fit['n_outliers']/fit['n_total']:.1f}%)")
        print(f"  A_n*1e3={fit['A_n']*1e3:.4f}±{fit['dA_n']*1e3:.4f}, "
              f"B_n*1e3={fit['B_n']*1e3:.4f}±{fit['dB_n']*1e3:.4f}, "
              f"C_n={fit['C_n']:.4f}±{fit['dC_n']:.4f}")
        print(f"  chi2/ndof = {fit['chi2']:.0f}/{fit['ndof']} = {fit['chi2_ndof']:.2f}")
        print(f"  Minimum at x_min={fit['x_min']:.3f} cm")

        results[bar_id] = fit

        if bar_id in plot_bars:
            plot_smile_before(x_vals, q, fit, bar_id, label)
            plot_smile_after(x_vals, q, fit, bar_id, label)

    if not results:
        print("\n[ERROR] No bar fitted, nothing saved.")
        sys.exit(1)

    write_calibration_file(results, os.path.join(args.output_dir, "charge_equalization.txt"))

    if args.plot_xmin:
        plot_xmin(results, save=os.path.join(args.output_dir, "eq_xmin.pdf"))


if __name__ == "__main__":
    main()
