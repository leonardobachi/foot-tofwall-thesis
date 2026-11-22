"""
results.py — Aggrega i risultati e produce plot/analisi.

Uso:
    python3 results.py <results_dir/> [opzioni]

Esempi:
    # tutto
    python3 results.py ~/CNAO2025_results/

    # solo alcune run
    python3 results.py ~/CNAO2025_results/ --runs 8050 8051

    # scegli metodo CFD per TW e ST
    python3 results.py ~/CNAO2025_results/ --st-method simple --tw-method fit

    # solo certe barre
    python3 results.py ~/CNAO2025_results/ --bars 9 29 (se non lo metto analizza tutte le barre)

    # range in carica
    python3 results.py ~/CNAO2025_results/ --q-min 10000 --q-max 13500

    # eventi in coincidenza
    python3 results.py ~/CNAO2025_results/ --coincidence 0 29

    # salva array barre in un .pkl per uso successivo
    python3 results.py ~/CNAO2025_results/ --save-bars ~/bars.pkl

    # carica le barre già processate invece di ricostruirle ogni volta
    python3 results.py ~/CNAO2025_results/ --load-hits ~/bars.pkl --coincidence 9 29

    # seleziona eventi con una sola barra per layer
    python3 results.py ~/CNAO2025_results/ --load-hits ~/bars.pkl --single-hit

"""

import sys
import os
import glob     # per trovare tutti i file .pkl nella cartella
import pickle   # per caricare i file .pkl
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from tw_calibration import apply_energy_calibration, apply_tof_calibration, add_coordinate, LAYER0_X, LAYER1_Y
from particle_id import apply_particle_id

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
# PARAMETRI DEFAULT
# =============================================================================

CFD_MAX_TW = 200.0   # ns


# =============================================================================
# 1. CARICAMENTO 
# =============================================================================

def load_results(folder, runs=None):
    """
    Carica tutti i results_*.pkl dalla cartella.
    Se runs è specificato, carica solo quelli.
    Restituisce lista flat di tutti gli eventi.
    """
    if runs:
        files = [os.path.join(folder, f"results_{r}.pkl") for r in runs]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(folder, "results_*.pkl")))  # sorted garantisce ordine alfabetico/numerico

    if not files:
        print(f"Nessun file results_*.pkl trovato in {folder}")
        sys.exit(1)

    print(f"\nFile .pkl trovati: {len(files)}")

    all_events = []   # lista piatta di tutti gli eventi di tutte le run selezionate
    for f in files:
        with open(f, "rb") as fh:
            evs = pickle.load(fh)   # Carica ogni .pkl, che è una lista di dict evento, e la appende alla lista globale
        print(f"  {os.path.basename(f)}: {len(evs)} eventi")
        all_events.extend(evs)

    print(f"Eventi totali caricati: {len(all_events)}\n")
    return all_events


def load_payload(path, key):
    """Carica un pickle generico (bars/points/hits) ed estrae la lista sotto 'key'."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    items = payload[key]
    print(f"{key.capitalize()} caricati: {len(items)}")
    #print(f"  runs={payload.get('runs')}")
    return items


# =============================================================================
# 2. COSTRUZIONE BARRE
# =============================================================================

def build_bars(events, tw_method="simple", st_method="simple", coincidence=None, single_hit = False):
    """
    Ricostruisce le barre TW dagli eventi.
    Per ogni (run_id, file_idx, eventN, bar_id) trova L e R,
    calcola charge_bar, time_bar, tof.

    tw_method: metodo per il tempo TW — "simple"|"tangent"|"bipolar"|"fit"
    st_method: metodo per il tempo ST — "simple"|"tangent"|"bipolar"|"fit"|"fit_single"

    coincidence: tupla (bar_id_1, bar_id_2) — se specificata restituisce solo
                 gli eventi in cui ENTRAMBE le barre sono ricostruite (AND logico),
                 aggiungendo il campo "dt" = time_bar_1 - time_bar_2.

    Restituisce lista di dict (una per barra per evento).
    """
    t_tw_key = "t_fit_sat" if tw_method == "fit_sat" else f"t_{tw_method}"
    t_st_key = f"t_{st_method}" if st_method != "fit_single" else "t_fit_single"

    bars = []

    n_wf_excluded = 0
    n_bars_incomplete = 0
    n_bars_cfd = 0
    n_bars_both = 0

    for ev in events:
        # escludo gli eventi costruito da 0 wfs ST (0.07%)
        if ev.get("ST_sum") is None:
            n_tw = sum(1 for wf in ev.get("TW", []) if wf.get("signal"))
            #print(f"  ST_sum=None: nST={ev['nST']} run={ev['run_id']} file={ev['file_idx']} ev={ev['eventN']} TW segnale={n_tw}")
            continue

        ev_n     = ev["eventN"]
        run_id   = ev["run_id"]
        file_idx = ev["file_idx"]

        # tempo ST di riferimento (scelto) dalla ST_sum
        st_sum = ev.get("ST_sum")
        t_st   = st_sum.get(t_st_key) if st_sum else None
        # Se ST_sum è None (evento senza wf ST valide) t_st = None e il TOF non verrà calcolato

        # raggruppa wf TW per bar_id
        bar_wfs = {}
        for wf in ev.get("TW", []):
            bid = wf.get("bar_id")
            if bid is None:   #  Le wf non mappate (bar_id=None) vengono saltate
                continue
            bar_wfs.setdefault(bid, {"L": None, "R": None})   # setdefault crea la entry se non esiste ancora.
            end = wf.get("end")
            if end in ("L", "R"):
                bar_wfs[bid][end] = wf   # Per ogni barra tiene un dict {"L": wf_L, "R": wf_R}
        
        ev_bars = {}
        # Per ogni barra: se manca uno dei due poli la barra è incompleta → skip
        for bid, ends in bar_wfs.items():
            wf_L = ends["L"]
            wf_R = ends["R"]

            if wf_L is None or wf_R is None:
                continue   # barra incompleta
            n_bars_both += 1

            # tempo TW: deve essere valido e entro CFD_MAX_TW
            t_L = wf_L.get(t_tw_key)
            t_R = wf_R.get(t_tw_key)
            if t_L is None or t_R is None:
                end_mancante = "L" if wf_L is None else "R"
                #print(f"  POLO MANCANTE: run={run_id} file={file_idx} ev={ev_n} bar={bid} manca={end_mancante}")
                n_bars_incomplete += 1
                continue
            if t_L > CFD_MAX_TW or t_R > CFD_MAX_TW:
                tipo = "anomalo" if t_L < 150 or t_R < 150 else "tardivo"
                #print(f"  TAGLIATA [{tipo}]: run={run_id} file={file_idx} ev={ev_n} bar={bid} t_L={t_L:.2f} t_R={t_R:.2f}")
                n_bars_cfd += 1
                if t_L is not None and t_L > CFD_MAX_TW: n_wf_excluded += 1
                if t_R is not None and t_R > CFD_MAX_TW: n_wf_excluded += 1
                continue
                        
            # carica barre
            ch_L = wf_L.get("charge")
            ch_R = wf_R.get("charge")
            if ch_L is None or ch_R is None or ch_L <= 0 or ch_R <= 0:
                charge_bar = None   # la barra viene comunque salvata ma senza carica, quindi non passerà il filtro carica in filter_bars
            else:
                charge_bar = float(np.sqrt(ch_L * ch_R))

            # tempo barre
            time_bar = (t_L + t_R) / 2.0
            # TOF
            tof      = (time_bar - t_st) if t_st is not None else None

            # carica e tempo dal fit su wf saturate
            cfs_L = wf_L.get("charge_fit_sat")
            cfs_R = wf_R.get("charge_fit_sat")
            charge_fit_sat_bar = float(np.sqrt(cfs_L * cfs_R)) \
                if (cfs_L and cfs_R and cfs_L > 0 and cfs_R > 0) else None

            t_fit_sat_L = wf_L.get("t_fit_sat")
            t_fit_sat_R = wf_R.get("t_fit_sat")
            time_bar_fit_sat = (t_fit_sat_L + t_fit_sat_R) / 2.0 \
                if (t_fit_sat_L is not None and t_fit_sat_R is not None) else None
            
            # chi2 della barra come massimo del chi2 delle due wfs: taglio per barre con chi2 troppo alto
            chi2_L = wf_L.get("chi2_fit_sat")
            chi2_R = wf_R.get("chi2_fit_sat")
            chi2_bar = max(chi2_L, chi2_R) \
                if (chi2_L is not None and chi2_R is not None) else None
            
            tof_fit_sat = (time_bar_fit_sat - t_st) \
                if (time_bar_fit_sat is not None and t_st is not None) else None

            ev_bars[bid] = {
                "run_id":     run_id,
                "file_idx":   file_idx,
                "eventN":     ev_n,
                "bar_id":     bid,
                "layer":      wf_L.get("layer"),
                "amp_st_sum":  ev["ST_sum"]["amplitude"] if ev.get("ST_sum") else None,
                "amp_L":      wf_L.get("amplitude"),
                "amp_R":      wf_R.get("amplitude"),
                "charge_L":   ch_L,
                "charge_R":   ch_R,
                "charge_bar": charge_bar,
                "t_st":       t_st,
                "t_L":        t_L,
                "t_R":        t_R,
                "time_bar":   time_bar,
                "tof":        tof,
                "dt":         None,
                "charge_fit_sat":  charge_fit_sat_bar,
                #"t_fit_sat_L":     t_fit_sat_L,
                #"t_fit_sat_R":     t_fit_sat_R,
                "time_bar_fit_sat": time_bar_fit_sat,
                "chi2_fit_sat":    chi2_bar,
            }

        if single_hit:
            # conta barre per layer — scarta eventi con più di una barra per layer
            layer0 = [bid for bid in ev_bars if ev_bars[bid].get("layer") == 0]
            layer1 = [bid for bid in ev_bars if ev_bars[bid].get("layer") == 1]
            if len(layer0) > 1 or len(layer1) > 1:
                continue

        # filtro coincidenza — scarta l'evento se una delle due barre manca
        if coincidence is not None:
            b1, b2 = coincidence
            if b1 not in ev_bars or b2 not in ev_bars:
                continue
            dt = ev_bars[b1]["time_bar"] - ev_bars[b2]["time_bar"]
            ev_bars[b1]["dt"] = dt
            ev_bars[b2]["dt"] = dt

        bars.extend(ev_bars.values())

    # numero barre attive negli eventi (non ricostruite)
    n_bars_total = sum(len(set(wf["bar_id"] for wf in ev["TW"] if wf.get("bar_id") is not None)) for ev in events)
    print(f"\nWf TW escluse per CFD_MAX_TW: {n_wf_excluded}")
    print(f"Barre escluse per CFD_MAX_TW: {n_bars_cfd} ({100*n_bars_cfd/max(n_bars_total,1):.2f}%)")
    print(f"Barre escluse per tempo mancante (metodo {tw_method} fallito): {n_bars_incomplete}/{n_bars_both} "
          f"({100*n_bars_incomplete/max(n_bars_both,1):.2f}%)")
    print(f"\nBarre ricostruite: {len(bars)} (ST:{st_method}, TW:{tw_method})\n")

    return bars

def build_bars_streaming(folder, runs=None, tw_method="simple", 
                          st_method="simple", coincidence=None, single_hit=False):
    """
    Costruisce le barre processando un .pkl alla volta — 
    non carica mai più di un file in memoria contemporaneamente.
    """
    if runs:
        files = [os.path.join(folder, f"results_{r}.pkl") for r in runs]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(folder, "results_*.pkl")))

    print(f"File .pkl trovati: {len(files)}")

    total_events = 0
    all_bars = []

    for f in files:
        print(f"\n  {os.path.basename(f)}...", end=" ", flush=True)
        with open(f, "rb") as fh:
            events = pickle.load(fh)
        total_events += len(events)
        bars = build_bars(events, tw_method=tw_method,
                          st_method=st_method, coincidence=coincidence,
                          single_hit=single_hit)
        all_bars.extend(bars)
        del events   # libera memoria subito

    print(f"\nEventi totali: {total_events}")
    print(f"Barre totali: {len(all_bars)}")
    return all_bars

# =============================================================================
# 2.1 FILTRI
# =============================================================================

def filter_bars(bars, bar_ids=None, q_min=None, q_max=None, chi2_max=None, amp_st_min=None, z_select=None):
    """Filtra per bar_id (se le barre sono associate correttamente), Z (chargeZ) e finestra di carica (se carica None, scarta)."""
    result = bars
    if bar_ids is not None:
        result = [b for b in result if b["bar_id"] in bar_ids]
    if z_select is not None:
        z_set = {z_select} if isinstance(z_select, int) else set(z_select)
        result = [b for b in result if b.get("chargeZ") in z_set]
    if q_min is not None or q_max is not None or chi2_max is not None or amp_st_min is not None:
        result = [b for b in result
                  if b["charge_bar"] is not None
                  and (q_min is None or b["charge_bar"] * 1e3 >= q_min)
                  and (q_max is None or b["charge_bar"] * 1e3 <= q_max)
                  and (chi2_max is None or (b.get("chi2_fit_sat") is not None and b["chi2_fit_sat"] <= chi2_max))
                  and (amp_st_min is None or (b.get("amp_st_sum") is not None and b["amp_st_sum"] * 1e3 >= amp_st_min))] # Taglio ST_sum < 1100 mV·ns
    return result

def plot_amp_st_sum(bars, amp_st_min=1100, bins=500, save=None):
    """Distribuzione dell'ampiezza della ST_sum (una entry per barra) con il taglio amp_st_min (mV)."""
    amp = np.array([b["amp_st_sum"] for b in bars if b.get("amp_st_sum") is not None]) * 1e3   # V -> mV
    n_cut = int(np.sum(amp < amp_st_min))
    print(f"\nTaglio amp_st_sum >= {amp_st_min} mV: escluse {n_cut}/{len(amp)} barre "
          f"({100 * n_cut / max(len(amp), 1):.2f}%)")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(amp, bins=bins, range=(0, amp.max()), color="steelblue", alpha=0.8)
    ax.axvspan(0, amp_st_min, color="red", alpha=0.12, label="rejected")
    ax.axvline(amp_st_min, color="red", linestyle="--", lw=2, label=f"cut: {amp_st_min} mV")
    ax.set_yscale("log")
    ax.set_xlabel("ST_sum amplitude (mV)")
    ax.set_ylabel("Counts")
    #ax.set_title(f"ST_sum amplitude — {len(amp)} bars")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    plt.show()


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# =============================================================================
# 3. CARICA
# =============================================================================

def plot_charge(bars, bar_id, label="", fit_windows=None, compare_fit_sat=None):
    """
    fit_windows: lista di tuple (q_center, q_half_width) in mV·ns
    """
    q = np.array([b["charge_bar"] for b in bars
                  if b["bar_id"] == bar_id 
                  and b["charge_bar"] is not None]) * 1e3
    if len(q) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    q_sat = np.array([]) if not compare_fit_sat else np.array(
        [b["charge_fit_sat"] for b in bars
         if b["bar_id"] == bar_id and b.get("charge_fit_sat") is not None]) * 1e3

    vmin = min(q.min(), q_sat.min()) if len(q_sat) else q.min()
    vmax = max(q.max(), q_sat.max()) if len(q_sat) else q.max()

    counts_all, edges, _ = ax.hist(q, bins=500, range=(vmin, 20000), color="darkorange",
                                    histtype="step", lw=2.5, label="charge_original")
    centers_all = (edges[:-1] + edges[1:]) / 2

    if compare_fit_sat and len(q_sat) > 0:
        ax.hist(q_sat, bins=500, range=(vmin, vmax), color="steelblue",
                histtype="step", lw=2.5, label="charge_fit_sat")
    
    if fit_windows is not None:
        for i, qc in enumerate(fit_windows):
            qhw=500
            mask = (centers_all >= qc - qhw) & (centers_all <= qc + qhw)
            if mask.sum() < 5:
                continue
            x_fit = centers_all[mask]
            y_fit = counts_all[mask]
            try:
                popt, _ = curve_fit(gauss, x_fit, y_fit, p0=[max(y_fit), qc, qhw/3], maxfev=10000)
                mu = popt[1]
                sigma = np.abs(popt[2])
                x_plot = np.linspace(mu - 2*sigma, mu + 2*sigma, 2000)
                ax.plot(x_plot, gauss(x_plot, *popt), color="black", lw=2)
                ax.axvline(mu - 2*sigma, color="black", lw=1.2, linestyle="--")
                ax.axvline(mu + 2*sigma, color="black", lw=1.2, linestyle="--")
                print(f"  Picco {i+1}: μ={mu:.0f} mV·ns, σ={abs(sigma):.0f} mV·ns, ris.en.={abs(sigma/mu)*100:.1f}%, range=[{mu-2*abs(sigma):.0f},{mu+2*abs(sigma):.0f}]")
            except Exception as e:
                print(f"  Fit carica picco {i} fallito: {e}")

    for q_cut in (10500, 13500):
        ax.axvline(q_cut, color="black", lw=1.5, linestyle="--")
    
    ax.set_xlabel("Charge (mV·ns)")
    ax.set_ylabel("Counts")
    ax.set_xlim(right=15000)
    ax.set_title(f"Charge — Bar {bar_id} — {len(q)} entries")
    ax.set_yscale("log")
    #ax.legend(fontsize=16)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. TOF
# =============================================================================

def _fit_tof_sigma(tofs, window_ns=None, n_sigma_range=4.0, n_sigma_fit=2.0,
                    bins_per_sigma=6, min_counts_per_bin=10, min_entries=100):
    """
    Fit gaussiano sull'istogramma della distribuzione TOF, con binning che
    si adatta sia alla scala fisica della distribuzione (sigma, stimata via
    MAD) sia alla statistica disponibile — non solo al numero di eventi.
    """
    tofs = np.asarray(tofs)
    tofs = tofs[np.isfinite(tofs)]

    if len(tofs) < min_entries:
        return None

    # --- stime robuste iniziali: mediana e MAD (Median Absolute Deviation) ---
    mu0 = np.median(tofs)
    mad = np.median(np.abs(tofs - mu0))
    sigma0 = 1.4826 * mad
    if sigma0 <= 0 or not np.isfinite(sigma0):
        return None

    # --- range del fit: centrato su mu0, ampio n_sigma_range volte sigma0 ---
    if window_ns is not None:
        rng = (mu0 - window_ns/2, mu0 + window_ns/2)
    else:
        rng = (mu0 - n_sigma_range*sigma0, mu0 + n_sigma_range*sigma0)

    # --- numero di bin ---
    # bins_per_sigma: larghezza del singolo bin, espressa come frazione di σ0
    n_bins_shape = int(round((rng[1] - rng[0]) / (sigma0 / bins_per_sigma))) # non dipende dal numero di eventi
    # Quanti bin puoi permetterti se vuoi che, in media, ogni bin contenga almeno min_counts_per_bin conteggi
    n_bins_stat  = int(len(tofs) / min_counts_per_bin)
    # Prende il più piccolo dei due candidati + (mai sotto 10 bin, mai sopra 150)
    n_bins = int(np.clip(min(n_bins_shape, n_bins_stat), 10, 150))

    counts, edges = np.histogram(tofs, bins=n_bins, range=rng)
    centers = (edges[:-1] + edges[1:]) / 2

    # --- raffina la stima di mu dal bin più popolato, poi restringe il fit ---
    mu0_refined = centers[np.argmax(counts)]
    mask = (centers >= mu0_refined - n_sigma_fit*sigma0) & (centers <= mu0_refined + n_sigma_fit*sigma0)
    if mask.sum() < 4:
        mask = np.ones_like(centers, dtype=bool)
    x_fit, y_fit = centers[mask], counts[mask]

    try:
        sigma_y = np.sqrt(np.maximum(y_fit, 1))   # errore di Poisson per bin
        popt, pcov = curve_fit(gauss, x_fit, y_fit,
                               p0=[max(y_fit), mu0_refined, sigma0],
                               sigma=sigma_y, absolute_sigma=True, maxfev=10000)
        sigma_ps = abs(popt[2]) * 1e3
        dsigma   = np.sqrt(max(pcov[2,2], 0)) * 1e3
        return {"mu": popt[1], "sigma_ps": sigma_ps, "dsigma_ps": dsigma,
                "n": len(tofs), "popt": popt, "rng": rng, "bins": n_bins}
    except Exception:
        return None

def plot_tof(bars, bar_id, label="", window_ns=None, pct_low=1, pct_high=99, bins=100):
    tofs = np.array([b["tof"] for b in bars
                     if b["bar_id"] == bar_id and b["tof"] is not None])
    if len(tofs) < 50:
        print(f"  Barra {bar_id}: troppo pochi eventi ({len(tofs)}), skip.")
        return None

    mu0_init = np.median(tofs)

    if window_ns is not None:
        rng = (mu0_init - window_ns/2, mu0_init + window_ns/2)
    else:
        rng = (np.percentile(tofs, pct_low), np.percentile(tofs, pct_high))

    counts, edges = np.histogram(tofs, bins=bins, range=rng)
    centers = (edges[:-1] + edges[1:]) / 2

    mu0    = centers[np.argmax(counts)]   # centro del bin più alto
    sigma0 = (rng[1] - rng[0]) / 6

    mask = (centers >= mu0 - sigma0) & (centers <= mu0 + sigma0)
    x_fit = centers[mask]
    y_fit = counts[mask]

    try:
        popt, pcov = curve_fit(gauss, x_fit, y_fit,
                               p0=[max(y_fit), mu0, sigma0], maxfev=10000)
        sigma_ps = abs(popt[2]) * 1e3
        dsigma   = np.sqrt(max(pcov[2,2], 0)) * 1e3
        print(f"\n  TOF : "
              f"mu={popt[1]:.2f} ns,  "
              f"σ={sigma_ps:.1f} ± {dsigma:.1f} ps  "
              f"({len(tofs)} entries)")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(tofs, bins=bins, range=rng, color="seagreen", alpha=0.7)
        x_fit = np.linspace(popt[1]-3*abs(popt[2]),
                             popt[1]+3*abs(popt[2]), 1000)
        ax.plot(x_fit, gauss(x_fit, *popt), color="black", lw=2,
                label=f"σ = {sigma_ps:.1f} ps")
        ax.set_xlabel("t_front-t_rear (ns)", fontsize=18)
        ax.set_ylabel("Counts", fontsize=18)
        ax.set_title(f"{label} — {len(tofs)} entries",
                     fontsize=18)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
        return {"mu": popt[1], "sigma_ps": sigma_ps, "dsigma_ps": dsigma,
                "n": len(tofs)}
    except Exception as e:
        print(f"  Fit fallito barra {bar_id}: {e}")
        return None


def plot_tof_point(points, row, column, z=None, label="", window_ns=None, n_sigma_range=4.0, n_sigma_fit=2.0):
    sel = [p for p in points
           if p["row"] == row and p["column"] == column
           and (z is None or p["chargeZ"] in z)
           and p["tof"] is not None]
    dts = np.array([p["tof1"] - p["tof2"] for p in sel])   # front - rear

    fit = _fit_tof_sigma(dts, window_ns, n_sigma_range, n_sigma_fit)
    if fit is None:
        print(f"  row={row}, col={column}, Z={z}: fit fallito o troppo pochi eventi ({len(dts)})")
        return None

    print(f"\n  TOF front-rear (row={row}, col={column}, Z={z}): mu={fit['mu']:.3f} ns,  "
          f"σ={fit['sigma_ps']:.1f} ± {fit['dsigma_ps']:.1f} ps  ({fit['n']} entries)")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(dts, bins=fit["bins"], range=fit["rng"], color="seagreen", alpha=0.7)
    x_plot = np.linspace(fit["popt"][1]-3*abs(fit["popt"][2]), fit["popt"][1]+3*abs(fit["popt"][2]), 1000)
    ax.plot(x_plot, gauss(x_plot, *fit["popt"]), color="black", lw=2, label=f"σ = {fit['sigma_ps']:.1f} ps")
    ax.set_xlabel("TOF front - rear (ns)", fontsize=18)
    ax.set_ylabel("Counts", fontsize=18)
    ax.set_title(f"{label} — row={row}, col={column}, Z={z} — {fit['n']} entries", fontsize=18)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    return fit


# =============================================================================
# 4.1 COINCIDENZA
# =============================================================================

def apply_coincidence(bars, b1, b2):
    """
    Applica il filtro coincidenza su barre già costruite.
    Raggruppa per (run_id, file_idx, eventN), seleziona solo gli eventi
    con entrambe le barre, aggiunge dt = time_bar_b1 - time_bar_b2.
    """
    from collections import defaultdict
    ev_index = defaultdict(dict)
    for b in bars:
        key = (b["run_id"], b["file_idx"], b["eventN"])
        ev_index[key][b["bar_id"]] = b

    result = []
    for key, bids in ev_index.items():
        if b1 not in bids or b2 not in bids:
            continue
        dt = bids[b1]["tof"] - bids[b2]["tof"]
        for bid in (b1, b2):
            entry = dict(bids[bid])   # copia
            entry["dt"] = dt
            result.append(entry)

    return result

# =============================================================================
# 5. PLOT TOF vs CARICA (identificazione particelle)
# =============================================================================

from matplotlib.colors import LogNorm

def plot_tof_vs_charge(bars, bar_id=None, tof_range=None, label="", show_bb_curves=True, max_points=100000):
    """
    Scatter TOF vs energia, punto per punto, colorato per chargeZ.
    Se i punti superano max_points, ne seleziona un campione casuale.
    """
    sel = [b for b in bars
           if (bar_id is None or b["bar_id"] == bar_id)
           and b["tof"] is not None
           and b["energy"] is not None
           and b.get("chargeZ") is not None]
    if not sel:
        return
    if len(sel) > max_points:
        idx = np.random.choice(len(sel), max_points, replace=False)
        sel = [sel[i] for i in idx]

    tofs = np.array([b["tof"]    for b in sel])
    q    = np.array([b["energy"] for b in sel])
    zs   = np.array([b["chargeZ"] for b in sel])

    if tof_range is None:
        tof_range = (np.percentile(tofs, 0.01), np.percentile(tofs, 99.9))

    Z_COLORS = {0: "black", 1: "red", 2: "blue", 3: "green",
                4: "orange", 5: "purple", 6: "gold"}

    fig, ax = plt.subplots(figsize=(12, 9))
    '''
    h = ax.hist2d(tofs, q, bins=(300, 300),
              range=[list(tof_range),
                     [np.percentile(q, 0.), np.percentile(q, 99.5)]],
              cmap="jet", norm=LogNorm())
    plt.colorbar(h[3], ax=ax, label="Counts (log)")
    '''
    for z_val in sorted(set(zs)):
        mask = zs == z_val
        ax.scatter(tofs[mask], q[mask], color=Z_COLORS.get(z_val, "gray"),
                   s=6, alpha=0.8, label=f"Z={z_val}", rasterized=True)
    ax.legend(fontsize=16, markerscale=4, loc="upper right")
    

    if show_bb_curves:
        from particle_id import BB_PARAMS, TOF_MIN, TOF_MAX
        layer = sel[0]["layer"]
        #bb_colors = plt.cm.tab10(np.linspace(0, 1, 6))

        x_curve = np.linspace(TOF_MIN, TOF_MAX, 2000)
        for Z, p in BB_PARAMS.get(layer, {}).items():
            y_curve = p["norm"] * Z**2 * (p["const"] * (x_curve)**2 - 1)
            ax.plot(x_curve, y_curve, color='black', lw=1.8, alpha=0.9, label=f"Z={Z}")   # bb_colors[Z-1]
        #ax.legend(fontsize=9, loc="upper left")

    ax.set_xlim(tof_range)
    ax.set_ylim(np.percentile(q, 0.), np.percentile(q, 99.5))
    ax.set_ylabel(r"E$_{\mathrm{loss}}$ (MeV)", fontsize=18)
    ax.set_xlabel("TOF (ns)", fontsize=18)
    ax.set_title(f"TOF vs Energy — bar {bar_id} — {len(sel)} entries", fontsize=18)
    plt.tight_layout()
    fig.savefig(f"tof_vs_eloss_bar{bar_id}.pdf", dpi=300)
    plt.show()


# =============================================================================
# 6. "EFFETTO SMILE" (carica vs dt_LR)
# =============================================================================

def plot_charge_vs_dt_LR(bars, bar_id, label="", x_field="dt_LR", normalize_to_fit_min=True,
                          outlier_thresh=None, charge_resolution=0.035):
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b["charge_bar"] is not None
           and b["t_L"] is not None
           and b["t_R"] is not None
           and (x_field != "coordinate" or b.get("coordinate") is not None)]
    if not sel:
        print(f"Nessun dato per barra {bar_id}")
        return

    q = np.array([b["charge_bar"] for b in sel]) * 1e3

    if x_field == "coordinate":
        x_vals = np.array([b["coordinate"] for b in sel])
        x_label = "position (cm)"
    else:
        dt_LR = np.array([b["t_L"] - b["t_R"] for b in sel])
        mean_dt = np.mean(dt_LR)
        x_vals = dt_LR - mean_dt
        x_label = "dt_LR_corr (ns)"

    x_lo, x_hi = np.percentile(x_vals, 0.), np.percentile(x_vals, 100.)

    def parabola(x, A, B, C):
        return A * x**2 + B * x + C

    def _fit(xf, qf):
        popt, pcov = curve_fit(parabola, xf, qf, p0=[100., 0., 1800], maxfev=10000)
        perr = np.sqrt(pcov.diagonal())
        A, B, C = popt
        if abs(A) > 3 * perr[0]:
            fmv = C - B**2 / (4*A)
        else:
            fmv = C
        return popt, perr, fmv

    # --- fit sui punti grezzi (nessun binning), con rigetto outlier in due passaggi ---
    popt = None
    chi2_dof_str = ""
    outlier_mask_plot = None
    try:
        popt1, perr1, fit_min_val1 = _fit(x_vals, q)
        popt, perr, fit_min_val = popt1, perr1, fit_min_val1

        if outlier_thresh is not None:
            resid_norm_fit = (q / fit_min_val1) - (parabola(x_vals, *popt1) / fit_min_val1)
            keep_out = np.abs(resid_norm_fit) <= outlier_thresh

            popt, perr, fit_min_val = _fit(x_vals[keep_out], q[keep_out])

            outlier_mask_plot = ~keep_out

            x_blue_lo = x_vals[~outlier_mask_plot].min()
            x_blue_hi = x_vals[~outlier_mask_plot].max()
            keep_range = (x_vals >= x_blue_lo) & (x_vals <= x_blue_hi)

            x_vals, q = x_vals[keep_range], q[keep_range]
            outlier_mask_plot = outlier_mask_plot[keep_range]

        A, B, C = popt

        if abs(A) > 3 * perr[0]:
            x_min = -B / (2*A)
        else:
            x_min = float('nan')

        A_n = A / fit_min_val
        B_n = B / fit_min_val
        C_n = C / fit_min_val
        perr_n = perr / fit_min_val

        # --- χ²/ndof e pull sui punti grezzi, usando la risoluzione nota (charge_resolution) ---
        if outlier_mask_plot is not None:
            x_diag = x_vals[~outlier_mask_plot]
            q_diag = q[~outlier_mask_plot]
        else:
            x_diag = x_vals
            q_diag = q

        q_norm_diag  = q_diag / fit_min_val
        pred_norm    = parabola(x_diag, *popt) / fit_min_val
        sigma_norm   = charge_resolution * q_norm_diag

        pull = (q_norm_diag - pred_norm) / sigma_norm

        chi2_val = np.sum(pull**2)
        ndof = len(x_diag) - 3
        chi2_dof_str = f", χ²/ndof={chi2_val:.0f}/{ndof} = {chi2_val/ndof:.2f}" if ndof > 0 else ""

        print(f"  best-fit(norm, raw): A_n*10^3={A_n*10**3}±{perr_n[0]*10**3}, "
              f"B_n*10^3={B_n*10**3}±{perr_n[1]*10**3}, C_n={C_n}±{perr_n[2]}{chi2_dof_str}")
        print(f"  Posizione minimo (vertice): x_min={x_min:.3f} cm")
        if outlier_mask_plot is not None:
            print(f"  Outlier (|q_norm - fit1| > {outlier_thresh}): "
                  f"{outlier_mask_plot.sum()}/{len(outlier_mask_plot)} "
                  f"({100*outlier_mask_plot.sum()/len(outlier_mask_plot):.2f}%)")
    except Exception as e:
        print(f"   Fit fallito: {e}")
        fit_min_val = 1.0

    # --- normalizzazione ---
    norm_factor = fit_min_val if normalize_to_fit_min else 1.0
    q_plot = q / norm_factor
    y_label = "Q_norm" if normalize_to_fit_min else "Q"

    # --- plot ---
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})

    if outlier_mask_plot is not None:
        ax.scatter(x_vals[~outlier_mask_plot], q_plot[~outlier_mask_plot], s=6, alpha=0.6, color="steelblue", label="fit_data")
        ax.scatter(x_vals[outlier_mask_plot],  q_plot[outlier_mask_plot],  s=6, alpha=0.6, color="red", label="outliers")
        y_blue_lo = q_plot[~outlier_mask_plot].min()
        y_blue_hi = q_plot[~outlier_mask_plot].max()
        ax.set_ylim(y_blue_lo, y_blue_hi)
    else:
        ax.scatter(x_vals, q_plot, s=6, alpha=0.6, color="steelblue")
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(f"Charge vs {x_field} — bar {bar_id} — entries {len(q)}")
    ax.grid(True, alpha=0.3)

    if popt is not None:
        x_fit = np.linspace(x_lo, x_hi, 2000)
        ax.plot(x_fit, parabola(x_fit, *popt) / norm_factor, color='black', lw=2)
        ax.legend(fontsize=16)

        # --- pannello residui, sui punti grezzi (niente più bin) ---
        residuals_plot = q_plot - parabola(x_vals, *popt) / norm_factor

        if outlier_mask_plot is not None:
            ax_res.scatter(x_vals[~outlier_mask_plot], residuals_plot[~outlier_mask_plot], s=6, alpha=0.6, color="steelblue")
        else:
            ax_res.scatter(x_vals, residuals_plot, s=6, alpha=0.6, color="black")

        ax_res.axhline(0, color="black", lw=1.5, linestyle="--")
        ax_res.set_ylabel("res. norm.")
        ax_res.set_xlabel(x_label)
        ax_res.grid(True, alpha=0.3)
    else:
        ax_res.set_xlabel(x_label)

    plt.tight_layout()
    plt.show()

def plot_charge_vs_position(bars, bar_id, label="", bins=(200, 200), save=None):
    """Carica di barra vs posizione lungo la barra, punti colorati per densità (log)."""
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b["charge_bar"] is not None and b["charge_bar"] > 0
           and b.get("coordinate") is not None]
    if not sel:
        print(f"Nessun dato per barra {bar_id}")
        return

    x = np.array([b["coordinate"] for b in sel])
    x = x - np.median(x)                                  # centrata in 0
    q = np.array([b["charge_bar"] for b in sel]) * 1e3

    # densità: conteggio del bin 2D in cui cade ogni punto
    H, xe, ye = np.histogram2d(x, q, bins=bins)
    ix = np.clip(np.digitize(x, xe) - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.digitize(q, ye) - 1, 0, H.shape[1] - 1)
    dens = H[ix, iy]
    order = np.argsort(dens)                             # i punti più densi disegnati sopra

    fig, ax = plt.subplots(figsize=(16, 8))
    H, xe, ye = np.histogram2d(x, q, bins=(300, 300),
                               range=[[-30, 30], [0, np.percentile(q, 99.9)]])
    ix = np.clip(np.digitize(x, xe) - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.digitize(q, ye) - 1, 0, H.shape[1] - 1)
    dens = np.maximum(H[ix, iy], 1)
    order = np.argsort(dens)
    sc = ax.scatter(x[order], q[order], c=dens[order], s=3, cmap="turbo",
                    norm=LogNorm(vmin=1, vmax=dens.max()), rasterized=True)
    fig.colorbar(sc, ax=ax, label="Counts (log)")
    ax.set_title(f"Charge vs Position — Bar {bar_id} — entries {len(q)}")
    ax.axvline(0, color="black", lw=2, linestyle="--")
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, np.percentile(q, 99.9))
    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$Q$ (mV$\cdot$ns)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def apply_smile_effect_calibration(bars, path="tw_position_calibration.txt"):
    """
    Carica il file di calibrazione (bar_id, A, dA, B, dB, C, dC) e applica la
    correzione posizionale SOVRASCRIVENDO charge_bar:
        charge_bar = charge_bar / (A*coordinate^2 + B*coordinate + C)
    Barre senza calibrazione disponibile, senza coordinate, o con f(x)<=0
    restano invariate.
    """
    calib = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            bar_id = int(parts[0])
            A, dA, B, dB, C, dC = map(float, parts[1:7])
            calib[bar_id] = (A, B, C)

    n_corrected = 0
    n_skipped = 0

    for b in bars:
        if b.get("charge_bar") is None:
            continue

        if b.get("coordinate") is None or b["bar_id"] not in calib:
            n_skipped += 1
            continue

        A, B, C = calib[b["bar_id"]]
        x = b["coordinate"]
        f_x = A * x**2 + B * x + C

        if f_x <= 0:
            n_skipped += 1
            continue

        b["charge_bar"] = b["charge_bar"] / f_x   # <-- sovrascrive direttamente
        n_corrected += 1

    print(f"Barre corrette: {n_corrected}/{len(bars)}  (saltate: {n_skipped})")
    return bars


# =============================================================================
# 6. CONFRONTO CARICA ORIGINALE vs CARICA FIT + CHI2
# =============================================================================

def plot_charge_fit_vs_orig(bars, bar_id, label=""):
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b.get("charge_bar") is not None
           and b.get("charge_fit_sat") is not None]
    if len(sel) < 10:
        print(f"  Barra {bar_id}: troppo pochi eventi con charge_fit_sat ({len(sel)}), skip.")
        return

    q_orig = np.array([b["charge_bar"]     for b in sel]) * 1e3
    q_fit  = np.array([b["charge_fit_sat"] for b in sel]) * 1e3
    diffs  = (q_fit - q_orig) / q_orig * 100

    # --- debug: salva eventi con discrepanza forte (Q_fit << Q_orig) ---
    bad_thr = 10.0   # %
    bad_idx = np.where(diffs > bad_thr)[0]
    if len(bad_idx) > 0:
        bad_triples = [(sel[i]["run_id"], sel[i]["file_idx"], sel[i]["eventN"],
                        sel[i].get("end", "?"), float(diffs[i]))
                       for i in bad_idx]
        print(f"  {len(bad_triples)} eventi con diff > {bad_thr}%")
        '''
        print(f"  Primi 10:")
        for run_id, file_idx, ev_n, end, diff in bad_triples[:10]:
            print(f"    run={run_id}  file_idx={file_idx}  eventN={ev_n}  diff={diff:.1f}%")
        '''
    
    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for ax in axes:
        ax.set_facecolor('white')
    fig.suptitle(fr"$Q_{{\mathrm{{fit}}}}$ vs $Q_{{\mathrm{{orig}}}}$ — Bar {bar_id} — entries {len(diffs)}", fontsize=18)

    lim = [min(q_orig.min(), q_fit.min()) * 0.95,
           max(q_orig.max(), q_fit.max()) * 1.05]
    if len(sel) > 5000:
        hb = axes[0].hexbin(q_orig, q_fit, gridsize=200, cmap="turbo",
                            norm=LogNorm(),
                            extent=[lim[0], lim[1], lim[0], lim[1]],
                            mincnt=1)
        plt.colorbar(hb, ax=axes[0], label="Counts (log)")
    else:
        axes[0].scatter(q_orig, q_fit, s=3, alpha=0.4, color="steelblue")
    axes[0].plot(lim, lim, color="black", lw=2.5, linestyle="--", label="y=x")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel(r"Q$_{\mathrm{orig}}$ (mV·ns)")
    axes[0].set_ylabel(r"Q$_{\mathrm{fit}}$ (mV·ns)")
    axes[0].set_title("Scatter Plot")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    rng = (np.percentile(diffs, 0.05), np.percentile(diffs, 99.95))
    counts, edges = np.histogram(diffs, bins='fd', range=rng)
    centers = (edges[:-1] + edges[1:]) / 2
    axes[1].hist(diffs, bins=edges, range=rng, histtype="step",
                 color="darkcyan", lw=1.8, zorder=3)
    axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
    try:
        mu0    = centers[np.argmax(counts)]
        sigma0 = (rng[1] - rng[0]) / 6

        mask = (centers >= mu0 - sigma0) & (centers <= mu0 + sigma0)
        #mask = centers <= mu0 + sigma0/2
        x_fit = centers[mask]
        y_fit = counts[mask]
        
        popt, pcov = curve_fit(gauss, x_fit, y_fit,
                                p0=[np.max(counts), mu0, sigma0], maxfev=10000)
        perr  = np.sqrt(np.diag(pcov))
        x_fit = np.linspace(popt[1] - 3*abs(popt[2]),
                             popt[1] + 3*abs(popt[2]), 1000)
        axes[1].plot(x_fit, gauss(x_fit, *popt), color="black", lw=2,
                     linestyle="--", alpha=0.8, zorder=2,
                     label=f"μ={popt[1]:.2f}%, σ={abs(popt[2]):.2f}%")
        axes[1].legend()
        print(f"  charge_diff: N={len(sel)}, μ={popt[1]:.3f}±{perr[1]:.3f} %  "
              f"σ={abs(popt[2]):.3f}±{perr[2]:.3f}%")
    except Exception as e:
        print(f"    Fit gaussiana fallito: {e}")
    axes[1].set_xlabel(r"(Q$_{\mathrm{fit}}$ - Q$_{\mathrm{orig}}$) / Q$_{\mathrm{orig}}$ (%)")
    axes[1].set_ylabel("Counts")
    axes[1].set_title("Relative charge difference")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_chi2(bars, bar_id, label=""):
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b.get("charge_bar") is not None
           and b.get("charge_fit_sat") is not None
           and b.get("chi2_fit_sat") is not None]
    if len(sel) < 10:
        print(f"  Barra {bar_id}: troppo pochi eventi con chi2_fit_sat ({len(sel)}), skip.")
        return

    q_orig = np.array([b["charge_bar"]     for b in sel]) * 1e3
    q_fit  = np.array([b["charge_fit_sat"] for b in sel]) * 1e3
    diffs  = (q_fit - q_orig) / q_orig * 100
    chi2   = np.array([b["chi2_fit_sat"]   for b in sel], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    fig.suptitle(f"chi2 — bar {bar_id} — entries {len(sel)}", fontsize=18)

    # --- sx: distribuzione chi2 ---
    rng_chi2 = (0, np.percentile(chi2, 99.99))
    axes[0].hist(chi2, bins=300, range=rng_chi2, color="red", alpha=0.8)
    axes[0].set_xlabel(r"$\chi^2$", fontsize=18)
    axes[0].set_ylabel("Counts", fontsize=18)
    axes[0].set_yscale("log")
    axes[0].set_title(r"Distribuzione $\chi^2$", fontsize=18)
    axes[0].grid(True, alpha=0.4)

    # --- dx: chi2 vs |diffs| ---
    rng_diffs = (0, np.percentile(diffs, 99.99))
    axes[1].hist2d(abs(diffs), chi2, range = (rng_diffs, rng_chi2),
                   bins=200, cmap="turbo",
                   norm=LogNorm())
    axes[1].set_xlabel(r"|$(Q_{fit} - Q_{orig}) / Q_{orig}$| (%)", fontsize=18)
    axes[1].set_ylabel(r"$\chi^2$", fontsize=18)
    axes[1].set_title(r"$\chi^2$ vs |diffs|", fontsize=18)
    axes[1].grid(True, alpha=0.4)

    plt.show()


# =============================================================================
# 7. CALIBRAZIONE DT -> POSIZIONE (join point + hit grezzi, fit v/offset per barra)
# =============================================================================

def build_hit_index(hits):
    """Indice (run_id, eventN, bar_id) -> hit, per lookup O(1).
    Verifica che la chiave sia univoca (dovrebbe esserlo per costruzione,
    vedi TATWactNtuHit — un solo hit per bar per evento)."""
    idx = {}
    n_collisions = 0
    for h in hits:
        key = (h["run_id"], h["eventN"], h["bar_id"])
        if key in idx:
            n_collisions += 1
            continue
        idx[key] = h

    if n_collisions > 0:
        print(f"[ATTENZIONE] {n_collisions} collisioni trovate nell'indice hit "
              f"(stessa run/evento/bar_id per più hit) — verifica il dataset, "
              f"non dovrebbe succedere.")
    else:
        print(f"Indice hit costruito: {len(idx)} chiavi uniche, nessuna collisione.")

    return idx


def plot_hit_times(bars, bar_id, label="", bins=150):
    """
    Istogrammi di t_L, t_R e dt=t_R-t_L per una barra — utile per capire
    se un taglio netto osservato su dt viene già da un limite sui singoli
    tempi (taglio algoritmico/hardware) o emerge solo dalla loro combinazione
    (effetto geometrico dei bordi barra).
    """
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b["t_L"] is not None
           and b["t_R"] is not None]
    if not sel:
        print(f"Nessun dato per barra {bar_id}")
        return

    t_L = np.array([b["t_L"] for b in sel])
    t_R = np.array([b["t_R"] for b in sel])
    dt = t_R - t_L

    print(f"\n  Bar {bar_id} ({label}): entries={len(sel)}")
    print(f"  t_L range: [{t_L.min():.3f}, {t_L.max():.3f}] ns")
    print(f"  t_R range: [{t_R.min():.3f}, {t_R.max():.3f}] ns")
    print(f"  dt(t_R-t_L) range: [{dt.min():.3f}, {dt.max():.3f}] ns")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.hist(t_L, bins=bins, color="steelblue", alpha=0.8)
    ax1.set_xlabel("t_L (ns)", fontsize=14)
    ax1.set_ylabel("Counts", fontsize=14)
    ax1.set_title(f"t_L — bar {bar_id} ({label})", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.hist(t_R, bins=bins, color="firebrick", alpha=0.8)
    ax2.set_xlabel("t_R (ns)", fontsize=14)
    ax2.set_ylabel("Counts", fontsize=14)
    ax2.set_title(f"t_R — bar {bar_id} ({label})", fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax3.hist(dt, bins=bins, color="seagreen", alpha=0.8)
    ax3.set_xlabel("dt = t_R - t_L (ns)", fontsize=14)
    ax3.set_ylabel("Counts", fontsize=14)
    ax3.set_title(f"dt — bar {bar_id} ({label})", fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def build_deltatime_calibration_data(points, hits):
    """
    Costruisce il dataset per il fit di calibrazione dt->posizione, per ciascuna
    delle 40 barre TW.

    Per ogni point, ricava due coppie (dt, posizione_nota):
      - dt della barra column, con la posizione nota data dal centro fisso
        della barra row accoppiata (verità geometrica, non misurata);
      - dt della barra row, con la posizione nota data dal centro fisso
        della barra column accoppiata.

    Filtri applicati, in ordine (entrambi necessari — SHOE non li applica
    internamente per i dati reali, vedi FindPoints):
      1. Barre riusate nello stesso evento: se una barra (row o column)
         compare in più di un point dello stesso evento, la si scarta —
         segnale di un possibile "ghost match" da pile-up (più hit del
         layer con più statistica che reclamano lo stesso hit sull'altro
         layer). Senza questo filtro, il fit potrebbe assorbire coppie
         (dt, posizione) fisicamente sbagliate.
      2. Z discordante tra l'hit column e l'hit row del point: se le due
         barre non riportano lo stesso Z ricostruito, è probabile che il
         point stia accoppiando due particelle diverse (equivalente
         all'IsPointWithMatchedZ di SHOE, ma verificato qui perché non
         possiamo assumere che fIsZmatch fosse attivo in questa run).

    Ritorna: dict bar_id (0-39) -> lista di tuple (dt, posizione_nota_cm)
    """
    hit_idx = build_hit_index(hits)
    data = {bid: [] for bid in range(40)}   # 0-19 rear/column, 20-39 front/row

    # --- Passo 1: conta quante volte ogni barra (row o column) compare
    #     tra i point dello stesso evento — serve a individuare i "ghost"
    #     (barra reclamata da più di un point) ---
    from collections import defaultdict
    bar_usage_count = defaultdict(int)   # (run_id, eventN, bar_id) -> quante volte usata in quell'evento
    for p in points:
        bar_usage_count[(p["run_id"], p["eventN"], p["column"])]     += 1
        bar_usage_count[(p["run_id"], p["eventN"], p["row"] + 20)]   += 1

    # --- Passo 2: join con gli hit grezzi, applicando i due filtri ---
    n_missing = 0
    n_z_mismatch = 0
    n_reused = 0
    for p in points:
        run_id, evN = p["run_id"], p["eventN"]
        col_bar_id = p["column"]
        row_bar_id = p["row"] + 20   # da indice locale (0-19) a bar_id globale (20-39)

        # filtro 1: scarta se row o column sono condivise con un altro point nello stesso evento
        if bar_usage_count[(run_id, evN, col_bar_id)] > 1 or bar_usage_count[(run_id, evN, row_bar_id)] > 1:
            n_reused += 1
            continue

        h_col = hit_idx.get((run_id, evN, col_bar_id))
        h_row = hit_idx.get((run_id, evN, row_bar_id))

        # può mancare se l'hit era stato scartato a monte in extract_shoe_hit_dt.py
        # (es. fIsValid=False, tempo non finito)
        if h_col is None or h_row is None:
            n_missing += 1
            continue

        # filtro 2: coerenza Z tra i due lati del point
        if h_col["chargeZ"] != h_row["chargeZ"]:
            n_z_mismatch += 1
            continue

        # --- dato per il fit della barra column: dt misurato sulla column,
        #     posizione nota dal centro fisso della barra row accoppiata ---
        dt_col = h_col["t_R"] - h_col["t_L"]
        pos_known_col = LAYER1_Y[p["row"]]
        data[col_bar_id].append((dt_col, pos_known_col))

        # --- simmetrico: dato per il fit della barra row ---
        dt_row = h_row["t_R"] - h_row["t_L"]
        pos_known_row = LAYER0_X[p["column"]]
        data[row_bar_id].append((dt_row, pos_known_row))

    print(f"Point processati: {len(points)}, mancava hit: {n_missing}, "
          f"Z mismatch: {n_z_mismatch}, barre riusate (scartate): {n_reused}")
    return data


def fit_dt_distribution_per_position(bar_data, plot=True, bar_id=None, label=""):
    """
    Raggruppa le coppie (dt, pos_nota) per posizione nominale (le ~20 posizioni
    fisse delle barre sull'altro layer), fitta la distribuzione dei dt per
    ciascuna posizione con una gaussiana. Ritorna una lista di dict
    {pos, mu, dmu, sigma, n} — uno per posizione con statistica sufficiente.
    """
    from collections import defaultdict
    by_pos = defaultdict(list)
    for dt, pos in bar_data:
        by_pos[round(pos, 3)].append(dt)

    results = []
    for pos in sorted(by_pos.keys()):
        dts = np.array(by_pos[pos])
        fit = _fit_tof_sigma(dts)
        if fit is None:
            print(f"    pos={pos:+.1f} cm: troppo pochi punti ({len(dts)}), skip")
            continue

        print(f"    pos={pos:+.1f} cm: mu_dt={fit['mu']:.4f}±{fit['dmu']:.4f} ns, "
              f"sigma_dt={fit['sigma']*1e3:.1f} ps, n={fit['n']}")

        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(dts, bins=fit["bins"], range=fit["rng"], color="seagreen", alpha=0.7)
            x_plot = np.linspace(fit["rng"][0], fit["rng"][1], 500)
            ax.plot(x_plot, gauss(x_plot, *fit["popt"]), color="black", lw=2,
                    label=f"mu={fit['mu']:.3f}±{fit['dmu']:.3f} ns, sigma={fit['sigma']*1e3:.0f} ps")
            ax.set_xlabel("dt = t_R - t_L (ns)", fontsize=14)
            ax.set_ylabel("Counts", fontsize=14)
            ax.set_title(f"Bar {bar_id} ({label}) — pos nota = {pos:+.1f} cm — {fit['n']} entries",
                        fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            #plt.show()

        results.append({"pos": pos, "mu": fit["mu"], "dmu": fit["dmu"],
                        "sigma": fit["sigma"], "n": fit["n"]})

    return results

def plot_deltatime_calibration(dt_arr, pos_arr, bar_id, label="", max_residual_cm=2.5):
    """
    Fit lineare pos = (v/2)*dt + offset sui punti grezzi (dt, posizione_nota).
    Un primo fit preliminare individua gli outlier (residuo > max_residual_cm
    dalla retta), che vengono esclusi; il fit finale gira solo sui punti buoni.
    """
    dt_arr  = np.array(dt_arr)
    pos_arr = np.array(pos_arr)

    def linfunc(dt, v, offset):
        return (v/2)*dt + offset

    try:
        # --- fit preliminare, su tutti i punti ---
        popt0, _ = curve_fit(linfunc, dt_arr, pos_arr, p0=[15.5, 0.0])
        residuals0 = pos_arr - linfunc(dt_arr, *popt0)
        mask = np.abs(residuals0) <= max_residual_cm

        # --- fit finale, solo sui punti entro la soglia ---
        popt, pcov = curve_fit(linfunc, dt_arr[mask], pos_arr[mask], p0=popt0)
        perr = np.sqrt(np.diag(pcov))
        v, offset = popt
        dv, doffset = perr
    except Exception as e:
        print(f"  Bar {bar_id}: fit fallito — {e}")
        return None

    residuals = pos_arr - linfunc(dt_arr, *popt)
    rms = np.std(residuals[mask])
    n_kept = mask.sum()

    print(f"\n  Bar {bar_id} ({label}): v={v:.4f}±{dv:.4f} cm/ns, "
          f"offset={offset:.4f}±{doffset:.4f} cm, "
          f"entries={len(dt_arr)} (tenuti: {n_kept}, {100*n_kept/len(dt_arr):.1f}%, "
          f"esclusi >{max_residual_cm}cm), rms_residui={rms*1e4:.1f} µm")

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})

    ax.scatter(dt_arr[~mask], pos_arr[~mask], s=6, alpha=0.5, color="red", label="outlier")
    ax.scatter(dt_arr[mask], pos_arr[mask], s=6, alpha=0.5, color="steelblue", label="fit_data")
    dt_fit = np.linspace(dt_arr.min(), dt_arr.max(), 500)
    ax.plot(dt_fit, linfunc(dt_fit, *popt), color="black", lw=2,
            label=f"v={v:.3f}±{dv:.3f} cm/ns, offset={offset:.3f}±{doffset:.3f} cm")
    ax.set_ylabel("Posizione nominale (cm)", fontsize=16)
    ax.set_title(f"Calibrazione dt→posizione — Bar {bar_id} ({label}) — {len(dt_arr)} entries",
                fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax_res.scatter(dt_arr[mask], residuals[mask], s=6, alpha=0.5, color="firebrick")
    ax_res.axhline(0, color="black", lw=1.5, linestyle="--")
    ax_res.set_xlabel("dt = t_R - t_L (ns)", fontsize=16)
    ax_res.set_ylabel("residui (cm)", fontsize=16)
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()

    return {"v": v, "dv": dv, "offset": offset, "doffset": doffset,
            "n": len(dt_arr), "n_kept": n_kept, "rms": rms}


def fit_deltatime_calibration_v2(bar_data, bar_id, label="", plot_individual=True, plot_summary=True):
    """
    Calibrazione dt->posizione in due passi:
      1. per ciascuna posizione nota, fitta la distribuzione dei dt con una
         gaussiana -> (mu_dt, dmu_dt) per posizione (fit_dt_distribution_per_position)
      2. fit lineare PESATO mu_dt(pos) = A*pos + B, usando dmu_dt come errore
         sulla y -> converte in v = 2/A, offset = -B/A
    """
    print(f"\n=== Bar {bar_id} ({label}) ===")
    pos_points = fit_dt_distribution_per_position(bar_data, plot=plot_individual,
                                                    bar_id=bar_id, label=label)

    if len(pos_points) < 3:
        print(f"  Bar {bar_id}: troppo poche posizioni valide ({len(pos_points)}), skip fit finale")
        return None

    pos_arr = np.array([p["pos"] for p in pos_points])
    mu_arr  = np.array([p["mu"] for p in pos_points])
    dmu_arr = np.array([p["dmu"] for p in pos_points])

    def linfunc(pos, A, B):
        return A*pos + B

    try:
        popt, pcov = curve_fit(linfunc, pos_arr, mu_arr, sigma=dmu_arr,
                               absolute_sigma=True, p0=[2/15.5, 0.0])
    except Exception as e:
        print(f"  Bar {bar_id}: fit finale fallito — {e}")
        return None

    A, B = popt
    dA, dB = np.sqrt(np.diag(pcov))
    cov_AB = pcov[0, 1]

    v  = 2 / A
    dv = abs(2 / A**2) * dA

    offset  = -B / A
    doffset = np.sqrt((B/A**2)**2 * dA**2 + (1/A)**2 * dB**2
                      - 2*(B/A**2)*(1/A)*cov_AB)

    residuals = mu_arr - linfunc(pos_arr, *popt)
    chi2 = np.sum((residuals/dmu_arr)**2)
    ndof = len(pos_arr) - 2

    print(f"\n  Bar {bar_id} ({label}) — RISULTATO: v={v:.4f}±{dv:.4f} cm/ns, "
          f"offset={offset:.4f}±{doffset:.4f} cm, punti usati={len(pos_arr)}, "
          f"chi2/ndof={chi2:.1f}/{ndof}")

    if plot_summary:
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
        ax.errorbar(pos_arr, mu_arr, yerr=dmu_arr, fmt='o', color="steelblue",
                    ecolor="steelblue", capsize=4, markersize=6)
        pos_fit = np.linspace(pos_arr.min(), pos_arr.max(), 200)
        ax.plot(pos_fit, linfunc(pos_fit, *popt), color="black", lw=2,
                label=f"v={v:.3f}±{dv:.3f} cm/ns, offset={offset:.3f}±{doffset:.3f} cm")
        ax.set_ylabel("mu_dt (ns)", fontsize=14)
        ax.set_title(f"Calibrazione dt<->posizione — Bar {bar_id} ({label})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax_res.errorbar(pos_arr, residuals, yerr=dmu_arr, fmt='o', color="firebrick", capsize=3)
        ax_res.axhline(0, color="black", lw=1.5, linestyle="--")
        ax_res.set_xlabel("Posizione nota (cm)", fontsize=14)
        ax_res.set_ylabel("residui (ns)", fontsize=14)
        ax_res.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {"bar_id": bar_id, "layer": label, "v": v, "dv": dv,
            "offset": offset, "doffset": doffset, "n_points": len(pos_arr),
            "chi2": chi2, "ndof": ndof, "pos_points": pos_points}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=None,
                    help="Cartella con i results_*.pkl (non serve con --load-bars)")
    parser.add_argument("--runs",       nargs="+", type=int, default=None)
    parser.add_argument("--bars",       nargs="+", type=int, default=None)
    parser.add_argument("--tw-method", default="simple",
                        choices=["simple","tangent","bipolar","fit","fit_sat"])
    parser.add_argument("--st-method",  default="simple",
                        choices=["simple","tangent","bipolar","fit","fit_single"])
    parser.add_argument("--q-min",      type=float, default=None)
    parser.add_argument("--q-max",      type=float, default=None)
    parser.add_argument("--z", nargs="+", type=int, default=None,
                        help="seleziona solo le barre con questo/i Z ricostruito/i (es. --z 6, o --z 5 6)")
    parser.add_argument("--save-bars",  default=None,
                        help="Salva lista barre in questo .pkl")
    parser.add_argument("--coincidence", nargs=2, type=int, default=None,
                    metavar=("BAR1", "BAR2"))
    parser.add_argument("--load-hits", default=None,
                    help="Carica barre da hits.pkl (extract_shoe.py) invece di ricostruirle")
    parser.add_argument("--load-bars", default=None,
                    help="Carica barre da bars.pkl (prodotto da --save-bars, pipeline vecchia con tw_method/st_method)")
    parser.add_argument("--single-hit", action="store_true",
                    help="Seleziona solo eventi con una barra per layer")
    parser.add_argument("--chi2-max",   type=float, default=None,
                        help="Taglio massimo sul chi2 del fit saturato")
    parser.add_argument("--load-points", type=str, default=None,
                    help="Percorso al pickle prodotto da extract_shoe_point.py")
    parser.add_argument("--row", type=int, default=None,
                    help="Indice barra row (front) per il test TOF su un singolo incrocio")
    parser.add_argument("--column", type=int, default=None,
                    help="Indice barra column (rear) per il test TOF su un singolo incrocio")
    parser.add_argument("--load-hits-dt", type=str, default=None,
                    help="Percorso al pickle prodotto da extract_shoe_hit_dt.py, per la calibrazione dt->posizione")
    parser.add_argument("--calib", type=int, nargs="+", default=None,
                    help="Lista di bar_id (es. 7 8 9) da calibrare con il fit dt->posizione in due passi")
    parser.add_argument("--calib-scan", type=int, nargs="+", default=None,
                    help="Lista di bar_id (es. 7 8 9) da calibrare con il fit robusto diretto sui punti grezzi (adatto a basse statistiche, es. twscan)")
    parser.add_argument("--plot-times", type=int, nargs="+", default=None,
                    help="Lista di bar_id: plotta gli istogrammi di t_L/t_R separatamente")
    
    args = parser.parse_args()

    # POINTS
    if args.load_points:
        points = load_payload(args.load_points, "points")

        if args.load_hits_dt and args.calib:
            hits = load_payload(args.load_hits_dt, "hits")
            data = build_deltatime_calibration_data(points, hits)
            for bar_id in args.calib:
                label = "rear" if bar_id < 20 else "front"
                fit_deltatime_calibration_v2(data[bar_id], bar_id=bar_id, label=label)

        elif args.load_hits_dt and args.calib_scan:
            hits = load_payload(args.load_hits_dt, "hits")
            data = build_deltatime_calibration_data(points, hits)
            for bar_id in args.calib_scan:
                label = "rear" if bar_id < 20 else "front"
                dts  = [x[0] for x in data[bar_id]]
                poss = [x[1] for x in data[bar_id]]
                plot_deltatime_calibration(dts, poss, bar_id=bar_id, label=label)

        elif args.row is not None and args.column is not None:
            plot_tof_point(points, row=args.row, column=args.column, z=args.z,
                           label=f"TOF front-rear SHOE")
        else:
            print("Specifica --row e --column per il test TOF, --calib per la calibrazione a due stadi, oppure --calib-scan per il fit robusto diretto (basse statistiche).")
        sys.exit(0)
    
    coincidence = tuple(args.coincidence) if args.coincidence else None

    if args.load_hits and args.load_bars:
        parser.error("--load-hits e --load-bars sono alternativi, usane solo uno")

    # --- carico direttamente le barre, da hits.pkl (extract_shoe.py) o da bars.pkl (vecchia pipeline) ---
    if args.load_hits or args.load_bars:
        if args.load_hits:
            with open(args.load_hits, "rb") as f:
                payload = pickle.load(f)
            bars = payload["hits"]
            print(f"\nBarre caricate da hits.pkl: {len(bars)}")
            print(f"  runs={payload.get('runs', '?')}")
            # niente tw_method/st_method qui: hits.pkl viene da extract_shoe.py, con
            # tof/energy già calcolati da SHOE — non c'è un "metodo" da riportare
        else:
            with open(args.load_bars, "rb") as f:
                payload = pickle.load(f)
            bars = payload["bars"]
            print(f"\nBarre caricate da bars.pkl: {len(bars)}")
            print(f"  tw_method={payload.get('tw_method', '?')}, st_method={payload.get('st_method', '?')}, "
                f"runs={payload.get('runs', '?')}")
            # riporta i metodi REALMENTE usati per costruire queste barre (non i default da CLI)
            args.tw_method = payload.get("tw_method", args.tw_method)
            args.st_method = payload.get("st_method", args.st_method)

        events = None   # non servono più

        if args.plot_times:
            for bar_id in args.plot_times:
                label = "rear" if bar_id < 20 else "front"
                plot_hit_times(bars, bar_id=bar_id, label=label)
            sys.exit(0)

        # applica single-hit a posteriori se richiesto
        if args.single_hit:
            from collections import defaultdict
            ev_index = defaultdict(list)
            for b in bars:
                key = (b["run_id"], b["file_idx"], b["eventN"])
                ev_index[key].append(b["bar_id"])
            ok_events = {k for k, bids in ev_index.items()
                         if sum(1 for bid in bids if bid < 20) <= 1
                         and sum(1 for bid in bids if bid >= 20) <= 1}
            bars = [b for b in bars if
                    (b["run_id"], b["file_idx"], b["eventN"]) in ok_events]
            print(f"Dopo single-hit: {len(bars)} barre")

        # applica coincidenza a posteriori se richiesta
        coincidence = tuple(args.coincidence) if args.coincidence else payload.get("coincidence")
        if coincidence is not None:
            bars = apply_coincidence(bars, coincidence[0], coincidence[1])

    else:
        if args.folder is None:
            parser.error("folder è obbligatoria quando non si usa --load-hits o --load-bars")
        # streaming di default — carica un .pkl alla volta per non saturare la RAM
        bars  = build_bars_streaming(args.folder, runs=args.runs,
                                    tw_method=args.tw_method,
                                    st_method=args.st_method, coincidence=coincidence,
                                    single_hit=args.single_hit)
        events = None


    if args.bars is None:
        args.bars = sorted({b["bar_id"] for b in bars})
    
    # --- calibrazione + PID: applicati SOLO se le barre non sono già calibrate ---
    # (se vengono da --load-hits o --load-bars, sono già calibrate — stessa condizione
    # per entrambe le sorgenti, così la pipeline a valle si comporta allo stesso modo)
    if not args.load_hits and not args.load_bars:
        bars = apply_energy_calibration(bars)
        bars = apply_tof_calibration(bars)
        bars = apply_particle_id(bars, n_workers=4)
    
    if args.save_bars:
        payload = {
            "bars":        bars,
            "tw_method":   args.tw_method,
            "st_method":   args.st_method,
            "coincidence": coincidence,
            "runs":        args.runs,
        }
        with open(args.save_bars, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Barre salvate in: {args.save_bars}")

    # --- analisi per barra ---
    label = f"(TW:{args.tw_method}, ST:{args.st_method})"

    # Calcolo posizione
    bars = add_coordinate(bars)

    # Correzione "effetto smile"
    #bars = apply_smile_effect_calibration(bars, "/home/ubuntu_leo/FOOT/calib/tw_position_calibration_new.txt")
    plot_amp_st_sum(bars, amp_st_min=1100)
    
    if coincidence is None:
        # analisi normale per singola barra
        for bid in args.bars:
            print(f"\n{'─'*55}")
            print(f"Barra {bid}")

            # TOF senza filtro carica
            bars_bar = [b for b in bars if b["bar_id"] == bid]
            print(f"  Entrate totali: {len(bars_bar)}")

            # Plot carica
            #plot_charge(bars, bid, label=label, fit_windows=None)
            #plot_charge(bars, bid, label=label, fit_windows=[2000], compare_fit_sat=None)

            # TOF con filtro carica e ST_sum
            n_pre = len(bars_bar)
            bars_q = filter_bars(bars_bar, z_select=args.z, q_min=args.q_min, q_max=args.q_max, chi2_max=args.chi2_max, amp_st_min=1100)
            print(f"  Taglio amp_st_min=1100 mV: escluse {n_pre - len(bars_q)} ({100*(n_pre-len(bars_q))/max(n_pre,1):.2f}%)")
            if args.q_min is None and args.q_max is None:
                print(f"  Nessun filtro carica: {len(bars_q)}")
            else:
                print(f"  Con filtro carica [{args.q_min}, {args.q_max}] mV·ns: "
                    f"{len(bars_q)}")
            # TOF
            #plot_tof(bars_q, bid, label="", window_ns=None, pct_low=0., pct_high=100., bins=100)

            # TOF vs carica
            #plot_tof_vs_charge(bars_q, bid, label=label)

            # "Effetto smile"
            #plot_charge_vs_dt_LR(bars_q, bar_id=bid, label=label, x_field="coordinate")
            #plot_charge_vs_position(bars_q, bar_id=bid, label=label)

            # carica originale vs fit
            #plot_charge_fit_vs_orig(bars_q, bar_id=bid, label=label)

            # chi2 vs diffs
            #plot_chi2(bars_q, bar_id=bid, label=label)


    else:    # filtro in coincidenza
        b1, b2 = coincidence
        print(f"\n{'─'*55}")
        print(f"Coincidenza {b1} ∧ {b2}")

        # TOF e carica barra 1 negli eventi in coincidenza (bars è già filtrato in coincidenza) #

        #plot_charge(bars, bar_id=b1, label=label, compare_fit_sat=False, fit_windows=None)
        bars_b1 = [b for b in bars if b["bar_id"] == b1]   # è una lista, l'ID della barra è b1
        print(f"  Eventi in coincidenza: {len(bars_b1)}")
        bars_q1 = filter_bars(bars_b1, z_select=args.z, q_min=args.q_min, q_max=args.q_max, chi2_max=args.chi2_max, amp_st_min=1100)
        #plot_tof(bars_q1, b1, label=f"barra {b1} coincidenza {b2} {label}")

        #plot_charge(bars, bar_id=b2, label=label, compare_fit_sat=False, fit_windows=None)
        bars_b2 = [b for b in bars if b["bar_id"] == b2]   # è una lista, l'ID della barra è b1
        bars_q2 = filter_bars(bars_b2, z_select=args.z, q_min=args.q_min, q_max=args.q_max, chi2_max=args.chi2_max, amp_st_min=1100)
        #plot_tof(bars_q2, b2, label=f"barra {b1} coincidenza {b2} {label}")

        # eventi in cui ENTRAMBE le barre passano il filtro carica
        eventi_comuni = {(b["run_id"], b["file_idx"], b["eventN"]) for b in bars_q1} & \
                        {(b["run_id"], b["file_idx"], b["eventN"]) for b in bars_q2}
        bars_q1 = [b for b in bars_q1 if (b["run_id"], b["file_idx"], b["eventN"]) in eventi_comuni]
        bars_q2 = [b for b in bars_q2 if (b["run_id"], b["file_idx"], b["eventN"]) in eventi_comuni]
        print(f"  Dopo filtro carica: {len(eventi_comuni)} eventi")

        # ora i plot usano le liste già filtrate per Z e ristrette agli eventi comuni
        plot_charge(bars_q1, bar_id=b1, label=label, compare_fit_sat=False, fit_windows=None)
        plot_charge(bars_q2, bar_id=b2, label=label, compare_fit_sat=False, fit_windows=None)

        # coincidenza: dt = t_bar1 - t_bar2
        bars_dt = [{"bar_id": -1, "tof": b["dt"], "charge_bar": None}
                for b in bars_q1
                if (b["run_id"], b["file_idx"], b["eventN"]) in eventi_comuni
                and b["dt"] is not None]
        plot_tof(bars_dt, bar_id=-1, label=f"t_bar {b1} - t_bar {b2}", window_ns=20., pct_low=1., pct_high=99., bins='fd')
        #plot_tof(bars_q1, b1, label=f"TOF bar {b1}", window_ns=None, pct_low=0.1, pct_high=99.9, bins=100)
        #plot_tof(bars_q2, b2, label=f"TOF bar {b2}", window_ns=None, pct_low=0.1, pct_high=99.9, bins=100)