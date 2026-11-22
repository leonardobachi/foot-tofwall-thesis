#!/usr/bin/env python3
# =============================================================================
# CHARGE_EQUALIZATION.PY — Correzione "effetto smile" (carica vs posizione lungo
# barra), standalone: da shoe_hits.pkl a file di calibrazione .txt, con plot di
# debug opzionali prima/dopo la correzione.
# =============================================================================
#
# Uso:
#   python3 charge_equalization.py <hits.pkl> --output-dir <cartella_output>
#       [--plot-bar N1 N2 ...]
#
# Esempio:
#   python3 charge_equalization.py /home/ubuntu_leo/FOOT/bars/shoe_twscan_hits.pkl --output-dir /home/ubuntu_leo/FOOT/calib --plot-bar 7
# =============================================================================

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

# =============================================================================
# CARICAMENTO
# =============================================================================

def load_hits(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    hits = payload["hits"]
    print(f"Hit caricati: {len(hits)}")
    print(f"  runs={payload.get('runs', '?')}")
    return hits


# =============================================================================
# FIT PARABOLICO
# =============================================================================

def _parabola(x, A, B, C):
    return A * x**2 + B * x + C


OUTLIER_THRESH = 0.15        # soglia di rigetto outlier su q_norm
CHARGE_RESOLUTION = 0.035    # risoluzione relativa in carica, per il chi2/ndof

def fit_smile_bar(x_vals, q, outlier_thresh=OUTLIER_THRESH, charge_resolution=CHARGE_RESOLUTION):
    """
    Fit parabolico Q(x)=A*x^2+B*x+C, con rigetto outlier in due passaggi
    (fit preliminare -> scarta punti con |q_norm - fit1| > outlier_thresh ->
    rifit finale). Gli outlier vengono esclusi SOLO dal fit — la correzione
    finale (Q_equalized = Q_raw / f(x)) viene comunque applicata anche a loro,
    usando i parametri A,B,C stimati sui punti buoni.
    Ritorna un dict con tutti i risultati, o None se il fit fallisce o non ci
    sono abbastanza punti.
    """
    if len(x_vals) < 100:
        return None

    def _fit(xf, qf):
        popt, pcov = curve_fit(_parabola, xf, qf, p0=[100., 0., 1800.], maxfev=10000)
        perr = np.sqrt(pcov.diagonal())
        A, B, C = popt
        if abs(A) > 3 * perr[0]:   # se A è troppo piccolo, siccome è al denominatore rischia di sballare il minimo
            fmv = C - B**2 / (4 * A)   # minimo analitico della parabola
        else:
            fmv = C
        return popt, perr, fmv, pcov

    try:
        popt1, perr1, fit_min_val0, _ = _fit(x_vals, q)
    except Exception as e:
        print(f"    Fit preliminare fallito: {e}")
        return None

    resid_norm_fit = (q / fit_min_val0) - (_parabola(x_vals, *popt1) / fit_min_val0)
    keep_out = np.abs(resid_norm_fit) <= outlier_thresh
    outlier_mask = ~keep_out
    n_outliers = int(outlier_mask.sum())

    try:
        popt, perr, fit_min_val, pcov = _fit(x_vals[keep_out], q[keep_out])
    except Exception as e:
        print(f"    Fit finale fallito: {e}")
        return None

    A, B, C = popt
    if abs(A) > 3 * perr[0]:
        x_min = -B / (2 * A)   # posizione lungo x del minimo
        # propagazione con la covarianza: x_min = -B/(2A)
        dxdA = B / (2 * A**2)
        dxdB = -1 / (2 * A)
        dx_min = float(np.sqrt(dxdA**2 * pcov[0, 0] + dxdB**2 * pcov[1, 1]
                               + 2 * dxdA * dxdB * pcov[0, 1]))
    else:
        x_min = float("nan")
        dx_min = float("nan")

    # normalizzo i parametri da mettere nel .txt
    A_n = A / fit_min_val
    B_n = B / fit_min_val
    C_n = C / fit_min_val
    perr_n = perr / fit_min_val

    # --- chi2/ndof (diag)nostico  ---

    # escludo come ne fit gli outliers
    x_diag = x_vals[keep_out]
    q_diag = q[keep_out]
    q_norm_diag = q_diag / fit_min_val
    pred_norm = _parabola(x_diag, *popt) / fit_min_val
    sigma_norm = charge_resolution * q_norm_diag
    res = (q_norm_diag - pred_norm) / sigma_norm
    chi2_val = float(np.sum(res**2))
    ndof = len(x_diag) - 3
    chi2_ndof = chi2_val / ndof if ndof > 0 else float("nan")

    return {
        "A": A, "B": B, "C": C, "dA": perr[0], "dB": perr[1], "dC": perr[2],
        "A_n": A_n, "B_n": B_n, "C_n": C_n, "dA_n": perr_n[0], "dB_n": perr_n[1], "dC_n": perr_n[2],
        "fit_min_val": fit_min_val, "x_min": x_min, "dx_min": dx_min,
        "chi2": chi2_val, "ndof": ndof, "chi2_ndof": chi2_ndof,
        "outlier_mask": outlier_mask, "n_outliers": n_outliers, "n_total": len(x_vals),
    }


def plot_xmin(results, save=None):
    """
    Posizione del minimo della parabola, x_min = -B/(2A), per ogni barra,
    con errore propagato da A e B (covarianza inclusa).
    Barre 0-19 rear, 20-39 front.
    """
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

    n_nan = int((~ok).sum())
    print(f"\n[x_min] barre con minimo non definito (A non significativo): {n_nan}")
    print(f"[x_min] mediana |x_min| = {np.median(np.abs(x_min[ok])):.2f} cm, "
          f"max |x_min| = {np.max(np.abs(x_min[ok])):.2f} cm "
          f"(barra {bars[ok][np.argmax(np.abs(x_min[ok]))]})")

    if save:
        fig.savefig(save)
        print(f"Salvato: {save}")
    plt.show()

# =============================================================================
# PLOT DI DEBUG (prima vs dopo)
# =============================================================================

def plot_smile_before(x_vals, q, fit, bar_id, label="", charge_resolution=CHARGE_RESOLUTION):
    """
    Carica normalizzata vs posizione con il fit parabolico (pannello sopra)
    e i residui normalizzati (pannello sotto), definiti come nel chi2/ndof
    di fit_smile_bar: (Q_norm - f(x)/Q0) / (CHARGE_RESOLUTION * Q_norm).
    I punti sono rasterizzati: salvando in PDF restano vettoriali solo assi,
    curve e scritte.
    """
    fmv = fit["fit_min_val"]
    q_norm = q / fmv
    mask = fit["outlier_mask"]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    # --- pannello sopra: dati e fit ---
    ax.axhline(1, color="black", lw=1.5, linestyle="--")
    ax.scatter(x_vals[~mask], q_norm[~mask], s=6, alpha=0.5, color="steelblue",
               label="fit data", rasterized=True)
    ax.scatter(x_vals[mask], q_norm[mask], s=6, alpha=0.5, color="red",
               label="outliers", rasterized=True)

    x_fit = np.linspace(x_vals.min(), x_vals.max(), 2000)
    y_fit = _parabola(x_fit, fit["A"], fit["B"], fit["C"]) / fmv
    ax.plot(x_fit, y_fit, color="black", lw=2, label="fit")

    ax.set_ylabel(r"$Q_{\mathrm{norm}}$")
    ax.set_title(f"Bar {bar_id} ({label}) — Before correction — {fit['n_total']} entries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- pannello sotto: residui normalizzati, come nel chi2 di fit_smile_bar ---
    # (Q_norm - f(x)/Q0) / sigma, con sigma = CHARGE_RESOLUTION * Q_norm
    pred_norm = _parabola(x_vals, fit["A"], fit["B"], fit["C"]) / fmv
    resid_n = (q_norm - pred_norm)
    axr.axhline(0, color="black", lw=1.5, linestyle="--")
    axr.scatter(x_vals[~mask], resid_n[~mask], s=6, alpha=0.5, color="steelblue",
                rasterized=True)
    axr.scatter(x_vals[mask], resid_n[mask], s=6, alpha=0.5, color="red",
                rasterized=True)
    axr.set_xlabel(r"$x$ (cm)")
    axr.set_ylabel("norm. res.")
    axr.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.4)
    axr.set_ylim(-0.2, 0.2)

    plt.tight_layout()
    plt.show()


def plot_smile_after(x_vals, q, fit, bar_id, label=""):
    fmv = fit["fit_min_val"]
    # correzione applicata a TUTTI i punti, outlier inclusi — usando A,B,C
    # stimati sui soli punti "buoni" (gli outlier sono stati esclusi solo dal fit)
    q_corr = q * fmv / _parabola(x_vals, fit["A"], fit["B"], fit["C"])
    q_corr_norm = q_corr / fmv

    mask = fit["outlier_mask"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1, color="black", lw=1.5, linestyle="--")
    ax.scatter(x_vals[~mask], q_corr_norm[~mask], s=6, alpha=0.5, color="seagreen",
               label="fit data", rasterized=True)
    ax.scatter(x_vals[mask], q_corr_norm[mask], s=6, alpha=0.5, color="red",
               label="outliers", rasterized=True)

    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$Q_{\mathrm{norm}}$")
    ax.set_title(f"Bar {bar_id} ({label}) — After correction — entries 6922")
    ax.tick_params(axis='both')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# OUTPUT — file di calibrazione .txt (header identico all'originale)
# =============================================================================

def write_calibration_file(results, out_path):
    lines = []
    lines.append("# File di calibrazione-correzione (EnergyEqualizer) della carica lungo la posizione della barra (\"effetto smile\"):")
    lines.append("# Q_equlized=Q_raw/f(x), con f(x)=A*x**2+B*x+C, con x posizione dell'hit lungo da barra")
    lines.append("# Nota: i parametri A dA B dB C dC (sono riportati per completezza, anche se non utilizzati, anche gli errori su A, B e C)")
    lines.append("# sono già stati normalizzati al minimo della carica per la calibrazione fatta con carboni in modo da poter essere utilizzati direttamente")
    lines.append("# su tutti gli ioni mediante la formula riportata sopra")
    lines.append("# Nota2: l'indice della barre riportato va da 0 a 39, con le barre 0-19 appartenti al layer 0 (rear) e quelle da 20-39 al layer 1 (front)")
    lines.append("# bar_id  A  dA  B  dB  C  dC")

    for bar_id in sorted(results.keys()):
        r = results[bar_id]
        lines.append(f"{bar_id}  {r['A_n']}  {r['dA_n']}  {r['B_n']}  {r['dB_n']}  {r['C_n']}  {r['dC_n']}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] File di calibrazione salvato in: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Correzione effetto smile, standalone")
    parser.add_argument("hits_pkl", help="Percorso al file hits.pkl (chiave 'hits')")
    parser.add_argument("--output-dir", required=True, help="Cartella dove salvare il file di calibrazione .txt")
    parser.add_argument("--plot-bar", type=int, nargs="+", default=None,
                        help="Lista di bar_id per cui mostrare i plot di debug (prima/dopo)")
    parser.add_argument("--plot-xmin", action="store_true",
                        help="Plot della posizione del minimo della parabola per tutte le barre")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hits = load_hits(args.hits_pkl)

    bar_ids = sorted({h["bar_id"] for h in hits})
    plot_bars = set(args.plot_bar) if args.plot_bar else set()

    results = {}

    for bar_id in bar_ids:
        label = "rear" if bar_id < 20 else "front"
        print(f"\n{'─'*55}\nBarra {bar_id} ({label})")

        sel = [h for h in hits
               if h["bar_id"] == bar_id
               and h.get("charge_bar") is not None
               and h.get("coordinate") is not None]

        if len(sel) < 30:
            print(f"  Troppo pochi dati ({len(sel)}), barra saltata.")
            continue

        x_vals = np.array([h["coordinate"] for h in sel])
        q = np.array([h["charge_bar"] for h in sel]) * 1e3   # V*ns -> mV*ns

        fit = fit_smile_bar(x_vals, q)
        if fit is None:
            print("  Fit fallito, barra saltata.")
            continue

        print(f"  Entrate totali: {fit['n_total']}, outlier scartati dal fit: {fit['n_outliers']} "
              f"({100*fit['n_outliers']/fit['n_total']:.1f}%) — comunque corretti nell'output")
        print(f"  A_n*10^3={fit['A_n']*1e3:.4f}±{fit['dA_n']*1e3:.4f}, "
              f"B_n*10^3={fit['B_n']*1e3:.4f}±{fit['dB_n']*1e3:.4f}, "
              f"C_n={fit['C_n']:.4f}±{fit['dC_n']:.4f}")
        print(f"  chi2/ndof = {fit['chi2']:.0f}/{fit['ndof']} = {fit['chi2_ndof']:.2f}")
        print(f"  Posizione minimo: x_min={fit['x_min']:.3f} cm")

        results[bar_id] = fit

        if bar_id in plot_bars:
            plot_smile_before(x_vals, q, fit, bar_id, label)
            plot_smile_after(x_vals, q, fit, bar_id, label)

    if not results:
        print("\n[ERRORE] Nessuna barra processata con successo, nessun file salvato.")
        sys.exit(1)

    out_path = os.path.join(args.output_dir, "tw_position_calibration_final.txt")
    write_calibration_file(results, out_path)

    if args.plot_xmin:
        plot_xmin(results, save=os.path.join(args.output_dir, "eq_xmin.pdf"))


if __name__ == "__main__":
    main()