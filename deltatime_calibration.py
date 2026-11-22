#!/usr/bin/env python3
# =============================================================================
# DELTATIME_CALIBRATION.PY — Calibrazione dt->posizione (v, offset per barra),
# standalone: da hits.pkl (+ opzionalmente points.pkl) al file .txt di output.
#
# Fit a DUE STADI, per barra:
#   1. per ciascuna delle 20 posizioni nominali, fit gaussiano della
#      distribuzione dei dt (mu, dmu) — saltata se ha meno di
#      MIN_POINTS_PER_POSITION eventi
#   2. fit lineare PESATO mu(pos) = A*pos + B sui punti (pos, mu, dmu) del
#      passo 1 -> v = 2/A, offset = -B/A
#
# --hits è sempre obbligatorio. --points è opzionale:
#   - CON --points: matching preciso via point SHOE (row/column), con i due
#     filtri di sicurezza per escludere — barre riusate nello stesso evento,
#     E Z mismatch tra row e column (qui lo Z è affidabile perché viene dal point,
#     tenuto per sicurezza).
#   - SENZA --points: pairing diretto sugli hit per evento — un evento
#     contribuisce solo se ha ESATTAMENTE un hit rear (bar_id 0-19) e un hit
#     front (bar_id 20-39), altrimenti è ambiguo e viene scartato. NESSUN
#     controllo di coerenza Z in questa modalità (non richiesto).
# =============================================================================
#
# Uso:
#   python3 deltatime_calibration.py --hits hits.pkl [--points points.pkl] \
#       --output-dir <cartella_output> [--plot-bar N1 N2 ...]
#   python3 deltatime_calibration.py --hits /home/ubuntu_leo/FOOT/bars/shoe_twscan_hits.pkl
#       --output-dir /home/ubuntu_leo/FOOT/calib --plot-bar 7 30
# =============================================================================

import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PLOT_PER_POSITION = True   # True: con --plot-bar mostra anche l'istogramma dt di ogni posizione

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

# =============================================================================
# GEOMETRIA DELLE BARRE - da TATWdetector.geo (CNAO2025)
# =============================================================================

# Layer 0 (vertical/column/rear): PositionX = "indirizzo" fisso della barra
LAYER0_X = {i: 19.0 - 2.0 * i for i in range(20)}
# Layer 1 (horizontal/row/front): PositionY = "indirizzo" fisso della barra
LAYER1_Y = {i: 19.0 - 2.0 * i for i in range(20)}

# =============================================================================
# CARICAMENTO
# =============================================================================

def load_pkl(path, key):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    items = payload[key]
    print(f"{key.capitalize()} caricati: {len(items)}")
    print(f"  runs={payload.get('runs', '?')}")
    return items


# =============================================================================
# INDICE HIT — per lookup O(1) nel matching via point
# =============================================================================

def build_hit_index(hits):
    """Costruisce un dizionario: Chiave -> indice (run_id, eventN, bar_id), Valore -> hit.
    Un solo hit per barra per evento per costruzione (vedi TATWactNtuHit) — verifica comunque."""
    idx = {}
    n_collisions = 0
    for h in hits:
        key = (h["run_id"], h["eventN"], h["bar_id"])
        if key in idx:
            n_collisions += 1
            continue
        idx[key] = h

    if n_collisions > 0:
        print(f"[ATTENZIONE] {n_collisions} collisioni nell'indice hit "
              f"(stessa run/evento/bar_id per più hit) — verifica il dataset.")
    else:
        print(f"Indice hit costruito: {len(idx)} chiavi uniche, nessuna collisione.")
    return idx


# =============================================================================
# MODALITÀ 1 — match via point (hits + points) — con controllo Z
# =============================================================================

def build_calibration_data_from_points(points, hits, z_filter=None, event_filter="bar-reuse"):
    """
    ...
    event_filter: 'bar-reuse' (default) scarta solo le barre condivise da più
    point nello stesso evento (il più chirurgico); 'single-point' scarta
    l'intero evento se ha più di un point (più severo, butta via anche dati
    non ambigui); 'none' non applica nessun filtro di ambiguità.
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
        # event_filter == "none": nessun controllo di ambiguità

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

    print(f"Point processati: {len(points)}, mancava hit: {n_missing}, "
          f"Z mismatch: {n_z_mismatch}, Z filtrati (--z={z_filter}): {n_z_filtered}, "
          f"scartati per ambiguità ({event_filter}): {n_reused}")
    return data


# =============================================================================
# MODALITÀ 2 — match diretto sugli hit (senza points) — SENZA controllo Z
# =============================================================================

def build_calibration_data_from_hits_only(hits):
    """
    Raggruppa gli hit per (run_id, eventN); tiene solo gli eventi con
    ESATTAMENTE un hit rear (bar_id 0-19) e un hit front (bar_id 20-39) —
    nessun tentativo di disambiguazione se ce ne sono di più per lato.
    Nessun controllo di coerenza Z (non disponibile/non richiesto qui).
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

        h_col = rear_hits[0]   # column = rear
        h_row = front_hits[0]  # row = front

        col_local = h_col["bar_id"]            # 0-19
        row_local = h_row["bar_id"] - 20       # 0-19

        dt_col = h_col["t_R"] - h_col["t_L"]
        data[h_col["bar_id"]].append((dt_col, LAYER1_Y[row_local]))

        dt_row = h_row["t_R"] - h_row["t_L"]
        data[h_row["bar_id"]].append((dt_row, LAYER0_X[col_local]))

        n_paired += 1

    print(f"Eventi totali: {n_events}; senza hit su un lato: {n_zero_side}; "
          f"ambigui (>1 hit su un lato): {n_ambiguous}; accoppiati con successo: {n_paired}")
    return data


# =============================================================================
# PLOT PRELIMINARE — scatter grezzo (dt, posizione_nota), nessun fit
# =============================================================================

def plot_deltatime_scatter_raw(bar_data, bar_id, label="", z_filter=None):
    """
    Scatter grezzo di tutte le coppie (dt, posizione_nota) di una barra,
    senza nessun fit sovrapposto — controllo visivo preliminare
    """
    dt_arr = np.array([x[0] for x in bar_data])
    pos_arr = np.array([x[1] for x in bar_data])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dt_arr, pos_arr, s=4, alpha=0.3, color="steelblue")
    ax.set_xlabel(r"$\Delta$t (ns)", fontsize=14)
    ax.set_ylabel("Posizione nominale (cm)", fontsize=14)
    z_str = f"Z={z_filter}" if z_filter is not None else ""
    ax.set_title(f"Scatter plot — Bar {bar_id} ({label}) — {z_str} — {len(bar_data)} entries", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STADIO 1 — fit gaussiano della distribuzione dt per ciascuna posizione nominale
# =============================================================================

# binning robusto per il fit gaussiano di ogni posizione 
BINS_PER_SIGMA = 4        # quando ho molta statistica: da aumentare se vedo la forma a "pettine"
MIN_COUNTS_PER_BIN = 12   # quando ho poca statistica: da aumentare per avere bin più larghi
N_SIGMA_RANGE = 4.0
N_SIGMA_FIT = 2.0

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _fit_gauss_dt(dt_values, bins_per_sigma=BINS_PER_SIGMA, min_counts_per_bin=MIN_COUNTS_PER_BIN,
                   n_sigma_range=N_SIGMA_RANGE, n_sigma_fit=N_SIGMA_FIT):
    """
    Fit gaussiano su un array di dt, con binning che si adatta sia alla vera
    scala della distribuzione (sigma, stimata via MAD) sia alla statistica
    disponibile — stessa logica già validata per sigma_TOF.
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


MIN_POINTS_PER_POSITION = 100   # sotto questa statistica, la posizione viene saltata (stadio 1)

def fit_dt_distribution_per_position(bar_data, min_points=MIN_POINTS_PER_POSITION,
                                      plot=False, bar_id=None, label="", z_filter=None):
    """
    Raggruppa le coppie (dt, pos_nota) per posizione nominale, fitta la
    distribuzione dei dt per ciascuna posizione con una gaussiana (via
    _fit_gauss_dt). Posizioni con meno di min_points eventi vengono saltate.
    Ritorna la lista dei fit riusciti: [{pos, mu, dmu, sigma, n}, ...]
    mu(pos) = A·pos + B, v=2/A, offset=-B/A, con propagazione dell'errore completa,
    inclusa la covarianza tra A e B, necessaria perché non sono stime indipendenti
    """
    by_pos = defaultdict(list)
    for dt, pos in bar_data:
        by_pos[round(pos, 3)].append(dt)

    pos_results = []
    for pos in sorted(by_pos.keys()):
        dt_values = np.array(by_pos[pos])

        if len(dt_values) < min_points:
            print(f"    pos={pos:+.1f} cm: solo {len(dt_values)} punti (< {min_points}), skip")
            continue

        fit = _fit_gauss_dt(dt_values)
        if fit is None:
            print(f"    pos={pos:+.1f} cm: fit gaussiano fallito, skip")
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
            z_str = f"Z={z_filter}" if z_filter is not None else ""
            ax.set_title(fr"Bar {bar_id} ({label}) — $x_{{\mathrm{{nom}}}}$ = {pos:+.1f} cm — {fit['n']} entries")
            ax.tick_params(axis='both')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            #plt.close(fig)
            plt.show()

        pos_results.append({"pos": pos, "mu": fit["mu"], "dmu": fit["dmu"],
                            "sigma": fit["sigma"], "n": fit["n"]})

    return pos_results


# =============================================================================
# STADIO 2 — fit lineare pesato mu(pos) = A*pos + B -> v, offset
# =============================================================================

def fit_v_offset_from_positions(pos_results):
    """
    Fit lineare PESATO (con dmu come errore sulla y) sui punti (pos, mu, dmu)
    prodotti dallo stadio 1. Converte A,B -> v=2/A, offset=-B/A, con
    propagazione dell'errore (inclusa la covarianza tra A e B).
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
        print(f"    Fit finale (stadio 2) fallito: {e}")
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
    z_str = f"Z={z_filter}" if z_filter is not None else ""
    ax.set_title(fr"Bar {bar_id} ({label}) — Physics data ($Z=2$) — {fit2['n_positions']} positions")
    ax.tick_params(axis='both')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- residui NORMALIZZATI (pull = residuo/dmu), neri, uniti da linea tratteggiata nera ---
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
# OUTPUT — file .txt, formato SHOE (bar_id, layer, barId(Shoe), v, offset)
# esteso con gli errori su v e offset, stessa precisione numerica (.6f)
# =============================================================================

def write_calibration_file(results, out_path):
    lines = ["#barId  layer   barId(Shoe)     v [cm/ns]   dv [cm/ns]      dx|y_AB [cm]   d(dx|y_AB) [cm]"]
    for bar_id in sorted(results.keys()):
        layer = 0 if bar_id < 20 else 1
        bar_shoe = bar_id if bar_id < 20 else bar_id - 20
        r = results[bar_id]
        lines.append(f"{bar_id}:    {layer}    {bar_shoe}    {r['v']:.6f}    {r['dv']:.6f}    "
                    f"{r['offset']:.6f}    {r['doffset']:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] File di calibrazione salvato in: {out_path}")

def write_shoe_map(results, out_path):
    """File nel formato di TATWdeltaTimeMap.map di SHOE: 40 righe
    'barId:  layer  barId(Shoe)  v  offset', senza errori, stessa precisione (.6f)."""
    missing = [b for b in range(40) if b not in results]
    if missing:
        print(f"\n[ERRORE] Barre senza fit {missing}: .map NON scritto (SHOE richiede tutte e 40 le barre).")
        return

    lines = ["#barId  layer   barId(Shoe)     v [cm/ns]       dx|y_AB [cm]"]
    for bar_id in range(40):
        layer = 0 if bar_id < 20 else 1
        bar_shoe = bar_id if bar_id < 20 else bar_id - 20
        r = results[bar_id]
        lines.append(f"{bar_id}:    {layer}    {bar_shoe}    {r['v']:.6f}    {r['offset']:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Mappa in formato SHOE salvata in: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrazione dt->posizione (fit a due stadi), standalone")
    parser.add_argument("--hits", required=True, help="Percorso al file hits.pkl (chiave 'hits')")
    parser.add_argument("--points", default=None,
                        help="Percorso al file points.pkl (opzionale, chiave 'points') — "
                             "se assente, il matching viene fatto direttamente sugli hit, senza controllo Z")
    parser.add_argument("--output-dir", required=True, help="Cartella dove salvare il file .txt")
    parser.add_argument("--plot-bar", type=int, nargs="+", default=None,
                        help="Lista di bar_id per cui mostrare i plot di debug "
                             "(scatter grezzo + istogramma dt per ogni posizione + riepilogo v/offset)")
    parser.add_argument("--z", type=int, default=None,
                    help="Filtra per uno specifico Z ricostruito (solo con --points; default: nessun filtro, prende tutti gli Z)")
    parser.add_argument("--event-filter", choices=["bar-reuse", "single-point", "none"], default="bar-reuse",
                    help="Filtro di ambiguità nel matching via point: 'bar-reuse' (default, scarta solo "
                         "le barre condivise), 'single-point' (scarta l'intero evento se ha più di un "
                         "point), 'none' (nessun filtro di ambiguità)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hits = load_pkl(args.hits, "hits")

    if args.points:
        points = load_pkl(args.points, "points")
        print(f"\n[INFO] Modalità: matching via point (con controllo Z, filtro ambiguità: {args.event_filter})")
        data = build_calibration_data_from_points(points, hits, z_filter=args.z, event_filter=args.event_filter)
    else:
        print("\n[INFO] Modalità: matching diretto sugli hit (nessun controllo Z)")
        data = build_calibration_data_from_hits_only(hits)

    plot_bars = set(args.plot_bar) if args.plot_bar else set()
    results = {}

    for bar_id in range(40):
        bar_data = data[bar_id]
        if not bar_data:
            continue

        label = "rear" if bar_id < 20 else "front"
        print(f"\n{'─'*55}\nBar {bar_id} ({label}) — {len(bar_data)} coppie (dt, pos) totali")

        # --- plot preliminare: scatter grezzo, nessun fit ---
        if bar_id in plot_bars:
            plot_deltatime_scatter_raw(bar_data, bar_id, label, z_filter=args.z)

        # --- stadio 1: fit gaussiano per ciascuna posizione ---
        pos_results = fit_dt_distribution_per_position(
            bar_data, plot=(PLOT_PER_POSITION and bar_id in plot_bars), bar_id=bar_id, label=label, z_filter=args.z)

        # --- stadio 2: fit lineare pesato sui (pos, mu, dmu) ---
        fit2 = fit_v_offset_from_positions(pos_results)
        if fit2 is None:
            print("  Fit finale (stadio 2) fallito, barra saltata.")
            continue

        print(f"  RISULTATO: v={fit2['v']:.4f}±{fit2['dv']:.4f} cm/ns, "
              f"offset={fit2['offset']:.4f}±{fit2['doffset']:.4f} cm, "
              f"posizioni usate={fit2['n_positions']}, chi2/ndof={fit2['chi2']:.1f}/{fit2['ndof']}")

        results[bar_id] = fit2

        if bar_id in plot_bars:
            plot_v_offset_summary(fit2, bar_id, label, z_filter=args.z)

    if not results:
        print("\n[ERRORE] Nessuna barra processata con successo, nessun file salvato.")
        sys.exit(1)

    out_path = os.path.join(args.output_dir, "deltatime_calibration_physics_z2.txt")
    write_calibration_file(results, out_path)

    map_path = os.path.join(args.output_dir, "deltatime_calibration_phyisics_z2.map")
    write_shoe_map(results, map_path)


if __name__ == "__main__":
    main()
