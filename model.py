import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize    import minimize_scalar
from scipy.optimize    import curve_fit

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
# MODEL_WF.PY — Costruzione waveform modello e model fit
# =============================================================================
#
# Pipeline:
#   1. collect_signal_wfs   — raccoglie wf TW con segnale (+ filtri opzionali)
#   2. resample_wfs         — ricampiona tutte le wf su griglia comune a passo fine
#   3. normalize_wfs        — normalizza all'ampiezza di picco
#   4. align_wfs            — allinea temporalmente al t_0 di riferimento
#   5. build_model          — media bin-a-bin → waveform modello + std per campione
#   6. plot_model           — visualizzazione modello con banda di incertezza
#   7. model_fit_wf         — fit di una singola wf al modello (A, t_arr liberi)
#   8. apply_model_fit      — applica il fit a tutte le wf raccolte
#   9. plot_model_fit       — visualizzazione risultati fit
#
# Parametri globali modificabili:
RESAMPLE_STEP = 0.050   # ns — passo di ricampionamento (50 ps)
T0_REF_TW     = 120.0   # ns — riferimento per TW
T0_REF_ST     = 105.0   # ns — riferimento per ST_sum
CFD_TIME_MIN  = 50.0    # ns — soglia minima su time_cfd
CFD_TIME_MAX  = 200.0   # ns — soglia massima su time_cfd (filtra spike di clock)
T_GRID_START  = 0.0     # ns — inizio griglia fissa
T_GRID_END    = 330.0   # ns — fine griglia fissa

CHI2_SIGMA    = "noise"

# =============================================================================
# 1. RACCOLTA WF CON SEGNALE
# =============================================================================

def collect_st_single(events, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """
    Raccoglie le singole wf ST (non la somma) dagli eventi.
    Aggiunge _eventN a ogni wf per poterla ricollegare all'evento.
    Filtra solo le wf valide con time_cfd nel range.
    
    Ritorna lista di wf-like objects compatibili con resample_wfs.
    """
    wfs = []
    n_bad_cfd   = 0

    for ev in events:
        for wf in ev["ST"]:
            t_cfd = wf.get("time_cfd")
            if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
                n_bad_cfd += 1
                continue
            # aggiungi _eventN per collegare la wf all'evento
            wf["_eventN"] = ev["eventN"]
            wfs.append(wf)

    print(f"\n--- collect_st_single ---")
    print(f"ST singole raccolte:         {len(wfs)}")
    print(f"Scartate (cfd fuori range):  {n_bad_cfd}")

    return wfs


def collect_st_sum(events, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """
    Raccoglie le ST_sum dagli eventi come lista di wf-like objects.
    Struttura compatibile con collect_signal_wfs per usare la stessa pipeline.
    """
    wfs = []
    n_no_sum  = 0
    n_bad_cfd = 0

    for ev in events:
        # controllo che l'evento abbia ST_sum
        st_sum = ev.get("ST_sum")
        if st_sum is None:
            n_no_sum += 1
            continue
        
        # controllo che il segnale di ST_sum sia in un range ragionevole
        t_cfd = st_sum.get("time_cfd")
        if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
            n_bad_cfd += 1
            continue

        st_sum["_eventN"] = ev["eventN"]
        wfs.append(st_sum)

    print(f"\n--- collect_st_sum ---")
    print(f"ST_sum raccolte:         {len(wfs)}")
    print(f"Scartate (no sum):       {n_no_sum}")
    print(f"Scartate (cfd fuori range): {n_bad_cfd}")

    return wfs

def collect_tw(events, bar_ids=None, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """
    Raccoglie tutte le wf TW con signal=True che passano i filtri.

    Parametri
    ----------
    events   — lista eventi dopo la pipeline completa
    bar_ids  — lista di bar_id da includere, None = tutte
    cfd_min/cfd_max — range accettato per time_cfd (ns)

    Ritorna lista di wf (riferimenti agli oggetti originali in events).
    Non copia i dati, punta agli stessi oggetti in events. 
    Questo è importante perché amp_resampled[i] corrisponde sempre a wfs[i] per tutta la pipeline.
    """
    wfs = []
    n_no_signal  = 0
    n_bad_cfd    = 0

    for ev in events:
        for wf in ev["TW"]:
            if not wf.get("signal"):
                n_no_signal += 1 # wf senza segnale
                continue
            
            # se gli do una specifica barra e la barra non corrisponde a quella che gli ho dato: continue
            if bar_ids is not None and wf.get("bar_id") not in bar_ids:
                continue

            t_cfd = wf.get("time_cfd")
            if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
                n_bad_cfd += 1  # wf che non rientrano nel range time_cfd
                continue
            
            wf["_eventN"] = ev["eventN"]   # riferimento all'evento in ogni wf
            wfs.append(wf)

    print(f"\n--- collect_signal_wfs ---")
    if bar_ids:
        print(f"Barre selezionate:  {sorted(bar_ids)}")
    print(f"Wf raccolte:        {len(wfs)}")
    print(f"Scartate (no sig):  {n_no_signal}")
    print(f"Scartate (cfd_time fuori range): {n_bad_cfd}")

    return wfs

# =============================================================================
# 2. RICAMPIONAMENTO
# =============================================================================

def resample_wfs(wfs, step=RESAMPLE_STEP):
    """
    Ricampiona tutte le wf su una griglia comune a passo 'step' ns.
    La griglia va dal max dei t[0] al min dei t[-1] tra tutte le wf.

    Ritorna:
        t_grid        — array 1D: griglia temporale comune (ns)
        amp_resampled — array 2D shape (n_wfs, n_grid) con le ampiezze ricampionate
        wfs           — stessa lista in input (per mantenere corrispondenza indici)
    """
    # griglia comune con range di inizio e fine fissati
    t_grid  = np.arange(T_GRID_START, T_GRID_END, step)

    print(f"\n--- resample_wfs ---")
    print(f"Griglia: {t_grid[0]:.2f} → {t_grid[-1]:.2f} ns  "
          f"({len(t_grid)} campioni, step={step*1e3:.0f} ps)")
    print(f"Memoria stimata: {len(wfs) * len(t_grid) * 8 / 1e9:.2f} GB") 
    # stampa la memoria occupata: #wfs x #campioni di ogni wf x 8 byte per campione (float 64)

    amp_resampled = np.zeros((len(wfs), len(t_grid)), dtype=np.float32)  # float32 per risparmiare memoria (ma adesso uso già float32)

    for i, wf in enumerate(wfs):
        interp = interp1d(wf["time"], wf["amp"], kind="cubic", bounds_error=False, fill_value=0.0)
        # interp1d costruisce una funzione di interpolazione — prende i punti noti (wf["time"], wf["amp"])
        # e crea un oggetto callable che ti permette di valutare la wf in qualsiasi punto temporale
        amp_resampled[i] = interp(t_grid)  # interp(t_grid) valuta questa funzione interpolata su tutti i 6600 punti della griglia comune
        # amp_resampled: array 2D di shape (35000, 6600) — una riga per wf, una colonna per campione

    print(f"Ricampionamento completato.")
    return t_grid, amp_resampled  # array 1D (tempi uguali per tutte le wfs), array 2D (ampiezze di tutte le wfs)


# =============================================================================
# 3. NORMALIZZAZIONE
# =============================================================================

def normalize_wfs(amp_resampled, wfs, method="charge"):
    """
    Normalizza ogni wf ricampionata alla sua ampiezza di picco o alla carica.

    method="amplitude" — normalizza al picco (Usa wf["amplitude"] se disponibile, altrimenti ricalcola dal ricampionato)
    method="charge"    — normalizza alla carica integrata wf["charge"]

    Ritorna:
        amp_norm    — array 2D normalizzato (stesso shape di amp_resampled)
        amplitudes  — array 1D con le ampiezze usate per la normalizzazione
    
    Dopo la normalizzazione ogni wf ha min circa -1.
    """
    amplitudes = np.zeros(len(wfs))  # ampiezze di picco / integrali di carica
    n_fallback = 0

    for i, wf in enumerate(wfs):
        if method == "charge":
            Q = wf.get("charge")  # carica è in V·ns, positiva
            w_norm = Q if (Q is not None and Q > 0) else None
        else:
            w_norm = None

        if w_norm is None:
            if method == "charge":
                n_fallback += 1
            # fallback su ampiezza
            A = wf.get("amplitude")
            w_norm = A if (A is not None and A > 0) else float(np.abs(np.min(amp_resampled[i])))
            w_norm = w_norm if w_norm > 0 else 1.0

        amplitudes[i] = w_norm

    if method == "charge" and n_fallback > 0:
        print(f"  Fallback su amplitude: {n_fallback} wf ({100*n_fallback/len(wfs):.1f}%)")

    # normalizzazione: amp_norm[i] = amp_resampled[i] / norm_factor[i]
    amp_norm = amp_resampled / amplitudes[:, np.newaxis]
    # Per dividere ogni riga di amp_resampled per il suo valore in ampiezza/carica, 
    # numpy deve sapere che amplitudes va applicato lungo le righe e non le colonne.
    # [:, np.newaxis] trasforma amplitudes da shape (35000,) a shape (35000, 1) — 
    # numpy poi fa il broadcast automaticamente e divide ogni elemento [i, j] per amplitudes[i].
    # Senza np.newaxis numpy non saprebbe come allineare le dimensioni e darebbe errore.

    if method == "charge":
        # fattore di scala globale: porta la mediana dei minimi a -1 — i rapporti relativi tra wf rimangono invariati
        minima       = np.abs(np.min(amp_norm, axis=1))
        global_scale = np.median(minima)
        if global_scale > 0:
            amp_norm = amp_norm / global_scale

    print(f"\n--- normalize_wfs (method={method}) ---")
    print(f"Scala media: {np.mean(amplitudes):.4f}  std: {np.std(amplitudes):.4f}")
    print(f"Dopo norm — min medio: {np.mean(np.min(amp_norm, axis=1)):.3f}  "
          f"(atteso -1.0)")

    return amp_norm, amplitudes


# =============================================================================
# 4. ALLINEAMENTO TEMPORALE
# =============================================================================

def align_wfs(amp_norm, wfs, t_grid, t0_ref, max_shift=None):
    """
    Allinea temporalmente le wf normalizzate shiftando ciascuna di
        dt = t0_ref - time_cfd
    in modo che tutti i fronti di salita coincidano a t0_ref.

    Ritorna:
        amp_aligned — array 2D allineato (stesso shape)
        shifts_ns   — array 1D degli shift applicati in ns

    t_grid - dt è la griglia originale traslata di dt ns — ogni punto t_grid[j] - dt è un istante leggermente diverso da t_grid[j]. 
    Valutando la wf su questa griglia shiftata ottiengo l'effetto di aver traslato temporalmente la wf di dt ns,
    ma tutti i valori rimangono sulla griglia originale t_grid.
    È come se spostassi la wf a destra o sinistra di un valore continuo, non intero
    """
    shifts_ns = np.zeros(len(wfs))
    amp_aligned = np.zeros_like(amp_norm)
    n_excluded  = 0

    for i, wf in enumerate(wfs):
        
        t_cfd = wf.get("time_cfd")
        dt = t0_ref - t_cfd        # shift in ns (positivo = sposta a dx)
        shifts_ns[i] = dt          # shift reale, non arrotondato

        # wf con shift anomalo — escludi dal modello mettendo a zero
        if max_shift is not None and abs(dt) > max_shift:
            amp_aligned[i] = 0.0   # peso zero in build_model weighted
            n_excluded += 1
            continue

        interp_wf      = interp1d(t_grid, amp_norm[i], kind="cubic", bounds_error=False, fill_value=0.0) 
        # errore su interpolazione cubica = (step)^4=(0.05)^4 ns = 6e-6 ns = 0.006 ps
        # La wf amp_norm[i] è già sulla griglia t_grid — non ricampiono su una griglia diversa, valuto la stessa wf su una griglia shiftata di dt.
        amp_aligned[i] = interp_wf(t_grid - dt)

    print(f"\n--- align_wfs ---")
    if max_shift is not None:
        mask_ok = np.abs(shifts_ns) <= max_shift
    else:
        mask_ok = np.ones(len(shifts_ns), dtype=bool)
    shifts_ok = shifts_ns[mask_ok]
    print(f"Shift applicati — mean: {np.mean(shifts_ok):.2f} ns  "
        f"std: {np.std(shifts_ok):.2f} ns  "
        f"range: [{np.min(shifts_ok):.2f}, {np.max(shifts_ok):.2f}] ns  "
        f"(tutte: [{np.min(shifts_ns):.2f}, {np.max(shifts_ns):.2f}] ns)")
    if max_shift is not None:
        print(f"Wf con |shift| > {max_shift} ns escluse: {n_excluded} "
              f"({100*n_excluded/len(wfs):.1f}%)")

    return amp_aligned, shifts_ns

# =============================================================================
# 5. COSTRUZIONE DEL MODELLO
# =============================================================================

def build_model(amp_aligned, wfs, method="weighted"):
    """
    Costruisce la waveform modello come media (o mediana) bin-a-bin
    delle wf normalizzate e allineate.

    method:
        "mean"     — media aritmetica (più sensibile agli outlier)
        "median"   — mediana (più robusta)
        "weighted" — pesa le wf a seconda di quanto sono rumorose

    Ritorna:
        model    — array 1D waveform modello
        std      — array 1D std per campione (misura di quanto è "blurred" il modello)
    """
    if method == "weighted":  # usa charge se disponibile, altrimenti amplitude
        weights = np.array([(wf.get("charge") or wf.get("amplitude") or 1.0)**2 /(wf.get("noise") or 1.0)**2 for wf in wfs])  # wfs pesate
        weights /= weights.sum()  # normalizza
        model = np.average(amp_aligned, axis=0, weights=weights)  # media pesata

    elif method == "mean":
        model = np.mean(amp_aligned, axis=0)

    else:
        model = np.median(amp_aligned, axis=0)

    std = np.std(amp_aligned, axis=0)

    # std relativa media nella zona del segnale (campioni con |model| > signal_th)
    signal_th = 0.2  # zona segnale su cui voglio considerare la std relativa
    signal_mask = np.abs(model) > signal_th
    if signal_mask.any():
        std_rel = np.mean(std[signal_mask] / np.abs(model[signal_mask]))
    else:
        std_rel = np.nan

    print(f"\n--- build_model ({method}) ---")
    print(f"Campioni nel modello: {len(model)}")
    print(f"Min template: {np.min(model):.4f}  (atteso ≈ -1.0)")
    print(f"Std media zona segnale (>{signal_th}): {np.mean(std[signal_mask]):.3f}  "
          f"(std relativa: {std_rel*100:.1f} %)")

    return model, std


# =============================================================================
# 4-5. ITERAZIONE DEL MODELLO
# =============================================================================

def build_model_iterative(amp_resampled, amp_norm, wfs, t_grid,
                          n_iter=0, method="weighted", max_wfs=None, t0_ref=T0_REF_TW, max_shift=None,
                          w_rise=20.0, w_tail=1.0, rise_left=5.0, rise_right=5.0,
                          tail=150.0, bracket_half=5.0, optimizer="scalar"):
    """
    Costruisce il modello iterativo.
    Ritorna model, std, amp_aligned pronti per il plot e poi per il fit.
    """
    amp_aligned, shifts = align_wfs(amp_norm, wfs, t_grid, t0_ref=t0_ref, max_shift=max_shift)
    model, std = build_model(amp_aligned, wfs, method=method)

    for it in range(n_iter):
        print(f"\n=== Iterazione {it+1}/{n_iter} ===")
        results_tmp = apply_model_fit(amp_resampled, wfs, t_grid, model, std,
                                      max_wfs=max_wfs,
                                      w_rise=w_rise, w_tail=w_tail,
                                      rise_left=rise_left, rise_right=rise_right,
                                      tail=tail, bracket_half=bracket_half,
                                      t0_ref=t0_ref, optimizer=optimizer)
        # per l'iterazione 0 faccio il fit con time_cfd come riallineamento

        # per le iterazioni >=1 faccio il fit con t_arr come riallineamento
        for i, (res, wf) in enumerate(zip(results_tmp, wfs)):
            wf["_t_align"] = res["t_arr"] if res["success"] else wf.get("time_cfd")

        amp_aligned_new = np.zeros_like(amp_norm)
        for i, wf in enumerate(wfs):
            t_align   = wf.get("_t_align")  # uso come tempo di riallinemento quello calcolato dal fit
            dt        = t0_ref - t_align    # riallineo a partire dal tempo di arrivo ottenuto dal fit
            interp_wf = interp1d(t_grid, amp_norm[i], kind="cubic",
                                 bounds_error=False, fill_value=0.0)
            amp_aligned_new[i] = interp_wf(t_grid - dt)

        model_new, std_new = build_model(amp_aligned_new, wfs, method=method)
        diff = np.mean(np.abs(model_new - model))
        print(f"Variazione modello: {diff:.6f}")
        model, std, amp_aligned = model_new, std_new, amp_aligned_new
        if diff < 1e-5:
            print(f"Convergenza raggiunta all'iterazione {it+1}")
            break

    return model, std, amp_aligned


# =============================================================================
# 6. PLOT DEL MODELLO
# =============================================================================

def plot_model(t_grid, model, std, t0_ref=T0_REF_TW, show_std=False,
               xlim=None, save=None):
    """
    Plotta il template. Con show_std=True aggiunge la banda ±1σ e un secondo
    pannello con la std campione per campione (per ora disattivato).
    """
    if show_std:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    if show_std:
        ax.fill_between(t_grid, model - std, model + std,
                        alpha=0.3, color="crimson", label=r"$\pm1\sigma$")
    ax.plot(t_grid, model, color="crimson", lw=2, label="Template")
    ax.fill_between(t_grid, (model - std), (model + std),
                         alpha=0.3, color="crimson", label="±1σ")
    ax.axvline(t0_ref, color="gray", lw=1.5, linestyle="--",
               label=r"$t_0^{\mathrm{ref}}$")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    if show_std:
        axes[1].plot(t_grid, std, color="steelblue", lw=1.5)
        axes[1].axvline(t0_ref, color="gray", lw=1.5, linestyle="--")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel(r"$\sigma_i$ (a.u.)")
        axes[1].grid(True, alpha=0.4)
    else:
        ax.set_xlabel("Time (ns)")

    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    plt.show()


def plot_model_compare(t_grid, model, std, amp_aligned, wfs,
                       n_wfs=300, end_filter=None,
                       q_min=None, q_max=None):
    """
    Plotta il modello sovrapposto a un campione di wf allineate.

    end_filter : None = tutte, "L" = solo polo L, "R" = solo polo R
    q_min/q_max: filtro sulla carica della singola wf in mV·ns (opzionale)
                 usa wf["charge"] — la carica grezza della singola wf,
                 non la carica di barra.
    """
    idx_pool = list(range(len(amp_aligned)))

    if end_filter is not None:
        idx_pool = [i for i in idx_pool if wfs[i].get("end") == end_filter]

    if q_min is not None:
        idx_pool = [i for i in idx_pool
                    if wfs[i].get("charge") is not None
                    and wfs[i]["charge"] * 1e3 >= q_min]

    if q_max is not None:
        idx_pool = [i for i in idx_pool
                    if wfs[i].get("charge") is not None
                    and wfs[i]["charge"] * 1e3 <= q_max]

    if not idx_pool:
        print(f"Nessuna wf con i filtri richiesti "
              f"(end={end_filter}, q_min={q_min}, q_max={q_max}).")
        return

    idx_sample = np.random.choice(idx_pool,
                                  size=min(n_wfs, len(idx_pool)),
                                  replace=False)

    # costruisci label informativa
    parts = []
    if end_filter:   parts.append("polo Left") if end_filter=="L" else parts.append("polo Right")
    if q_min is not None: parts.append(f"Q>{q_min:.0f} mV·ns")
    if q_max is not None: parts.append(f"Q<{q_max:.0f} mV·ns")
    label = ", ".join(parts) if parts else "tutte le wfs"

    # stampa statistiche del campione selezionato
    charges_sel = [wfs[i]["charge"] * 1e3 for i in idx_sample
                   if wfs[i].get("charge") is not None]
    amps_sel    = [wfs[i]["amplitude"] * 1e3 for i in idx_sample
                   if wfs[i].get("amplitude") is not None]
    print(f"\nplot_model_compare — {len(idx_sample)} wf ({label})")
    if charges_sel:
        print(f"  Charge: mean={np.mean(charges_sel):.0f}, "
              f"min={np.min(charges_sel):.0f}, "
              f"max={np.max(charges_sel):.0f} mV·ns")
    if amps_sel:
        print(f"  Amplitude: mean={np.mean(amps_sel):.1f}, "
              f"min={np.min(amps_sel):.1f}, "
              f"max={np.max(amps_sel):.1f} mV")

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, i in enumerate(idx_sample):
        ax.plot(t_grid, amp_aligned[i], color="steelblue", lw=0.8, alpha=0.5,
                rasterized=True,
                label="Waveforms" if j == 0 else "_nolegend_")
    ax.fill_between(t_grid, model - std, model + std,
                    alpha=0.3, color="crimson")
    ax.plot(t_grid, model, color="crimson", lw=2.5, label="Template")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SATURAZIONE ARTIFICIALE
# =============================================================================

def apply_saturation(amp_resampled, sat_level_mv=250.0):
    """
    Applica saturazione artificiale alle wf ricampionate.
    Tutti i campioni con valore assoluto > sat_level_mv vengono clippati.
    Lavora su amp_resampled in V — converte la soglia da mV a V internamente.

    Parametri
    ----------
    amp_resampled  — array 2D (n_wf x n_campioni) in V
    sat_level_mv   — soglia di saturazione in mV (default 250 mV)

    Ritorna
    -------
    amp_saturated  — array 2D con stessa shape, valori clippati
    n_saturated    — numero di wf con almeno un campione saturato
    frac_clipped   — per ogni wf, frazione di campioni clippati
    """
    sat_level_v   = sat_level_mv / 1e3
    amp_saturated = np.clip(amp_resampled, -sat_level_v, sat_level_v)

    # statistiche
    n_clipped_per_wf = np.sum(np.abs(amp_resampled) > sat_level_v, axis=1)
    n_saturated      = int(np.sum(n_clipped_per_wf > 0))
    frac_clipped     = n_clipped_per_wf / amp_resampled.shape[1]

    print(f"\n--- apply_saturation ---")
    print(f"Soglia: {sat_level_mv:.0f} mV")
    print(f"Wf con almeno 1 campione saturato: {n_saturated}/{len(amp_resampled)} "
          f"({100*n_saturated/len(amp_resampled):.1f}%)")
    if n_saturated > 0:
        frac_ok = frac_clipped[frac_clipped > 0]
        print(f"Campioni saturati per wf (tra quelle saturate): "
              f"mean={np.mean(frac_ok)*100:.1f}%, "
              f"max={np.max(frac_ok)*100:.1f}%")

    return amp_saturated, n_saturated, frac_clipped



def compute_tof_cfd_sat(events, sat_level_mv, q_min=None, q_max=None, bars=None):
    """
    Calcola il TOF applicando saturazione artificiale alle wf TW raw
    e ricalcolando il CFD semplice su quelle saturate.
    ST rimane invariato (CFD semplice su wf non saturate).

      ST = CFD semplice (invariato)
      TW = CFD semplice su wf saturata artificialmente

    Parametri
    ----------
    events       : lista eventi dopo pipeline completa (con bars e time_cfd ST)
    sat_level_mv : float — soglia di saturazione in mV
    q_min, q_max : float o None — filtro carica barra (V·ns)
    bars         : list o None — bar_id da considerare (None = tutte)

    Ritorna
    -------
    tof_list    : list — tutti i TOF
    tof_by_bar  : dict {bar_id: [tof, ...]}
    """
    from cfd_wf import _compute_time_simple   # CFD semplice su singola wf

    sat_level_v = sat_level_mv / 1e3

    tof_list   = []
    tof_by_bar = {}

    for ev in events:
        # tempo ST di riferimento — invariato
        st_sum = ev.get("ST_sum")
        if st_sum is None:
            continue
        t_st = st_sum.get("time_cfd")
        if t_st is None:
            continue

        bar_dict = ev.get("bars", {})
        for bar_id, bar in bar_dict.items():
            if bars is not None and bar_id not in bars:
                continue

            wf_L = bar.get("wf_L")
            wf_R = bar.get("wf_R")
            if wf_L is None or wf_R is None:
                continue
            if not bar.get("signal_L") or not bar.get("signal_R"):
                continue

            # filtro carica
            q_bar = bar.get("charge_bar")
            if q_bar is None or q_bar <= 0:
                continue
            if q_min is not None and q_bar * 1e3 < q_min:
                continue
            if q_max is not None and q_bar * 1e3 > q_max:
                continue

            # applica saturazione e ricalcola CFD semplice
            t_L_sat = None
            t_R_sat = None
            for wf, side in [(wf_L, "L"), (wf_R, "R")]:
                amp_sat = np.clip(wf["amp"], -sat_level_v, sat_level_v)
                wf_sat  = {"amp": amp_sat, "time": wf["time"],
                            "amplitude": float(np.abs(np.min(amp_sat))),
                            "noise": wf.get("noise", 0.0)}
                t_cfd = _compute_time_simple(wf_sat)
                if side == "L":
                    t_L_sat = t_cfd
                else:
                    t_R_sat = t_cfd

            if t_L_sat is None or t_R_sat is None:
                continue

            time_bar = (t_L_sat + t_R_sat) / 2.0
            tof      = time_bar - t_st

            tof_list.append(tof)
            if bar_id not in tof_by_bar:
                tof_by_bar[bar_id] = []
            tof_by_bar[bar_id].append(tof)

    print(f"\n--- compute_tof_cfd_sat (soglia {sat_level_mv:.0f} mV) ---")
    print(f"TOF totali: {len(tof_list)}")
    for bid, tofs in sorted(tof_by_bar.items()):
        print(f"  Barra {bid}: {len(tofs)} TOF")

    return tof_list, tof_by_bar


# =============================================================================
# CALCOLO CARICA DAL FIT
# =============================================================================

def compute_charge_from_fit(results, wfs, t_grid, model, std, t0_ref=T0_REF_TW):
    """
    Calcola la carica integrando la wf ricostruita dal fit (A * model_shifted).
    Confronta con la carica originale wf["charge"].

    Utile per stimare la carica di wf saturate — il fit ricostruisce
    la forma completa anche sopra la soglia di saturazione.
    """
    from charge_wf import _compute_charge_integral, MEAN_OFFSET

    model_interp = interp1d(t_grid, model, kind="cubic", bounds_error=False, fill_value=0.0)

    charges_fit  = []
    charges_orig = []
    amplitudes   = []

    for res, wf in zip(results, wfs):
        if not res["success"]:
            continue

        A     = res["A"]
        t_arr = res["t_arr"]

        # wf ricostruita dal fit
        amp_rec = A * model_interp(t_grid - (t_arr - t0_ref))

        # costruisci wf-like object per _compute_charge_integral
        wf_rec = {
            "amp":      amp_rec,
            "time":     t_grid,
            "pedestal": wf.get("pedestal", 0.0),
        }
        charge_fit = _compute_charge_integral(wf_rec)

        # carica originale già calcolata
        charge_orig = wf.get("charge")

        if charge_orig is not None and charge_fit is not None:
            charges_fit.append(charge_fit * 1e3)   # mV·ns
            charges_orig.append(charge_orig * 1e3)
            amplitudes.append(wf.get("amplitude", 0) * 1e3)

    charges_fit  = np.array(charges_fit)
    charges_orig = np.array(charges_orig)
    amplitudes   = np.array(amplitudes)
    diff         = charges_fit - charges_orig

    print(f"\n--- Confronto carica: fit vs originale ---")
    print(f"N wf: {len(charges_fit)}")
    print(f"Carica originale: mean={np.mean(charges_orig):.0f}, "
          f"std={np.std(charges_orig):.0f} mV·ns")
    print(f"Carica dal fit:   mean={np.mean(charges_fit):.0f}, "
          f"std={np.std(charges_fit):.0f} mV·ns")
    print(f"Differenza:       mean={np.mean(diff):.1f}, "
          f"std={np.std(diff):.1f} mV·ns "
          f"({np.mean(np.abs(diff)/charges_orig)*100:.2f}% relativa)")

    return charges_fit, charges_orig, diff


def plot_charge_bar_from_fit(results, wfs, events, t_grid, model,
                              bar_id=9, t0_ref=T0_REF_TW,
                              fit=True, fit_range=None):
    """
    Per la barra bar_id produce 3 figure separate (polo L, polo R, barra):
      - sx: scatter charge_orig vs charge_fit con retta y=x
      - dx: istogramma differenza relativa (Q_fit-Q_orig)/Q_orig con fit gaussiano
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    from charge_wf import _compute_charge_integral

    model_interp = interp1d(t_grid, model, kind="cubic",
                             bounds_error=False, fill_value=0.0)

    # --- raccolta cariche fit per polo ---
    # ev_charges_fit = {eventN: {"L": q_fit_L, "R": q_fit_R}}
    ev_charges_fit = {}
    for res, wf in zip(results, wfs):
        if not res["success"]:
            continue
        if wf.get("bar_id") != bar_id:
            continue
        ev_n = wf.get("_eventN")
        end  = wf.get("end")
        if ev_n is None or end is None:
            continue
        A       = res["A"]
        t_arr   = res["t_arr"]
        amp_rec = A * model_interp(t_grid - (t_arr - t0_ref))
        wf_rec  = {"amp": amp_rec, "time": t_grid,
                   "pedestal": wf.get("pedestal", 0.0)}
        q_fit = _compute_charge_integral(wf_rec)
        if q_fit is None or q_fit <= 0:
            continue
        ev_charges_fit.setdefault(ev_n, {})[end] = q_fit

    # --- raccolta cariche originali per polo ---
    # ev_charges_orig = {eventN: {"L": q_orig_L, "R": q_orig_R}}
    ev_charges_orig = {}
    for ev in events:
        ev_n = ev["eventN"]
        bar  = ev.get("bars", {}).get(bar_id)
        if bar is None:
            continue
        for end in ("L", "R"):
            wf_end = bar.get(f"wf_{end}")
            if wf_end is None:
                continue
            q_orig = wf_end.get("charge")
            if q_orig is not None and q_orig > 0:
                ev_charges_orig.setdefault(ev_n, {})[end] = q_orig

    print(f"\n--- Carica barra {bar_id}: fit vs originale ---")

    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def _plot_one(title, q_orig_arr, q_fit_arr):
        """Produce una figura con scatter (sx) e diff relativa con gaussiana (dx)."""
        if len(q_orig_arr) == 0:
            print(f"  {title}: nessun dato")
            return

        q_orig_arr = np.array(q_orig_arr) * 1e3   # V·ns → mV·ns
        q_fit_arr  = np.array(q_fit_arr)  * 1e3

        diffs = (q_fit_arr - q_orig_arr) / q_orig_arr * 100

        print(f"\n  {title}: N={len(diffs)}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Charge — bar {bar_id} — {title}", fontsize=14)

        # --- sx: scatter ---
        lim = [min(q_orig_arr.min(), q_fit_arr.min()) * 0.95, 15e3 * 1.05]
        axes[0].scatter(q_orig_arr, q_fit_arr, s=4, alpha=0.6, color="steelblue")
        axes[0].plot(lim, lim, color="red", lw=1.5, linestyle="--", label="y=x")
        axes[0].set_xlim(lim)
        axes[0].set_ylim(lim)
        axes[0].set_xlabel("charge_original (mV·ns)", fontsize=13)
        axes[0].set_ylabel("charge_from_fit (mV·ns)", fontsize=13)
        #axes[0].set_title("Scatter plot", fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.4)

        # --- dx: differenza relativa + gaussiana ---
        rng = (np.percentile(diffs, 0.2), np.percentile(diffs, 99.5))
        counts, edges = np.histogram(diffs, bins=100, range=rng)
        centers = (edges[:-1] + edges[1:]) / 2
        axes[1].hist(diffs, bins=100, range=rng, color="darkcyan", alpha=0.8)
        axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
        if fit:
            try:
                mu0    = centers[np.argmax(counts)]
                sigma0 = (rng[1] - rng[0]) / 6
                popt, pcov = curve_fit(gauss, centers, counts,
                                        p0=[np.max(counts), mu0, sigma0],
                                        maxfev=10000)
                perr  = np.sqrt(np.diag(pcov))
                x_fit = np.linspace(popt[1] - 3*abs(popt[2]),
                                     popt[1] + 3*abs(popt[2]), 1000)
                axes[1].plot(x_fit, gauss(x_fit, *popt), color="black", lw=2,
                             label=f"μ={popt[1]:.2f}%,  "
                                   f"σ={abs(popt[2]):.2f}%")
                axes[1].legend(fontsize=11)
                print(f"    μ_diff={popt[1]:.2f}  "
                      f"σ={abs(popt[2]):.2f}±{perr[2]:.2f}%")
            except Exception as e:
                print(f"    Fit gaussiana fallito: {e}")
        axes[1].set_xlabel("(Q_fit - Q_orig) / Q_orig (%)", fontsize=13)
        axes[1].set_ylabel("Counts", fontsize=13)
        #axes[1].set_title("Differenza relativa", fontsize=13)
        axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()

    # --- polo L ---
    q_orig_L, q_fit_L = [], []
    for ev_n in ev_charges_fit:
        if ev_n not in ev_charges_orig:
            continue
        qf = ev_charges_fit[ev_n].get("L")
        qo = ev_charges_orig[ev_n].get("L")
        if qf and qo:
            q_orig_L.append(qo)
            q_fit_L.append(qf)
    #_plot_one("polo L", q_orig_L, q_fit_L)

    # --- polo R ---
    q_orig_R, q_fit_R = [], []
    for ev_n in ev_charges_fit:
        if ev_n not in ev_charges_orig:
            continue
        qf = ev_charges_fit[ev_n].get("R")
        qo = ev_charges_orig[ev_n].get("R")
        if qf and qo:
            q_orig_R.append(qo)
            q_fit_R.append(qf)
    #_plot_one("polo R", q_orig_R, q_fit_R)

    # --- barra (media geometrica) ---
    q_orig_bar, q_fit_bar = [], []
    for ev_n in ev_charges_fit:
        if ev_n not in ev_charges_orig:
            continue
        qfL = ev_charges_fit[ev_n].get("L")
        qfR = ev_charges_fit[ev_n].get("R")
        qoL = ev_charges_orig[ev_n].get("L")
        qoR = ev_charges_orig[ev_n].get("R")
        if qfL and qfR and qoL and qoR:
            q_orig_bar.append(np.sqrt(qoL * qoR))
            q_fit_bar.append(np.sqrt(qfL * qfR))
    _plot_one("bar", q_orig_bar, q_fit_bar)

    return q_fit_bar, q_orig_bar

# =============================================================================
# 7. MODEL FIT
# =============================================================================

def model_fit_wf(wf_amp, t_grid, model, std, model_interp=None,
                 A_0=None, t_arr_0=None, step=RESAMPLE_STEP, bracket_half=5.0,
                 w_rise=20.0, w_tail=1.0, rise_left=5.0, rise_right=5.0, tail=150.0,
                 optimizer="scalar", t0_ref=T0_REF_TW, sat_level=None, noise=None):
    """
    Fitta una singola wf ricampionata al modello con due parametri liberi:
        A     — ampiezza (scala verticale)
        t_arr — tempo di arrivo (shift orizzontale in ns)

    Minimizza chi2 = sum_i [ (wf_amp[i] - A * model_shifted[i])^2 / sigma_i^2 ]

    Parametri
    ----------
    wf_amp    — array 1D wf ricampionata (non normalizzata, non allineata)
    t_grid    — griglia temporale comune
    modello   — array 1D waveform modello (normalizzata)
    std       — array 1D std del modello per campione
    A_0       — stima iniziale ampiezza (default: abs(min(wf_amp)))
    t_arr_0   — stima iniziale tempo arrivo (default: t_grid[argmin(wf_amp)])

    Ritorna dizionario con:
        A, t_arr         — parametri fittati
        dA, dt_arr       — errori stimati dalla matrice hessiana
        chi2             — chi2 totale
        success          — True se il fit è riuscito
    """
    # parametri iniziali
    if t_arr_0 is None:
        t_arr_0 = float(t_grid[np.argmin(wf_amp)])
    # if A_0 is None:
        # A_0 = float(np.abs(np.min(wf_amp)))

    # np.where applica un floor a 1e-4 per evitare divisioni per zero o valori troppo piccoli che gonfierebbero artificialmente il chi2
    if CHI2_SIGMA == "noise" and noise is not None and noise > 0:
        sigma = np.full(len(wf_amp), noise) # rumore della wf (std della baseline), uguale per tutti i campioni
    else:
        sigma = np.where(std > 1e-4, std, 1e-4) # std del modello per campione

    # tre zone della wf con pesi diversi nel chi2
    t_center  = t_arr_0 if t_arr_0 is not None else t0_ref 
    '''
    È la stima iniziale del tempo di arrivo passata al fit come punto di partenza per la ricerca del minimo del chi2. 
    In pratica è il time_cfd della singola wf — il tempo calcolato dall'algoritmo CFD standard. 
    Serve per due cose: centrare il bracket di ricerca (t_arr_0 - bracket_half, t_arr_0 + bracket_half) 
    e, con la correzione del punto 1, centrare la rise_mask attorno al fronte reale della wf invece che attorno al t0_ref fisso del modello.
    '''
    # per le ST singole il fronte è in posizione variabile ma la maschera è centrata su T0_REF_ST=105 ns fisso. Per la ST_sum è corretto, per le singole no.
    rise_mask = (t_grid >= t_center - rise_left) & (t_grid <= t_center + rise_right)    # fronte di salita
    tail_mask = (t_grid > t_center + rise_right)  & (t_grid <= t_center + tail)  # coda: contribuisce poco al tempo ma aiuta a stabilizzare il fit su A
    weights   = np.where(rise_mask, w_rise, np.where(tail_mask, w_tail, 0.0))  # pesi delle varie parti fittate
    # tutto il resto (baseline prima del segnale, zona dopo 150 ns) pesa 0 — non entra nel chi2

    # maschera saturazione: azzera i campioni oltre la soglia, così il chi2 non è influenzato dai campioni piatti artificialmente
    if sat_level is not None:
        sat_mask = np.abs(wf_amp) >= sat_level   # True dove è saturato
        weights  = np.where(sat_mask, 0.0, weights)

    if model_interp is None:
        model_interp = interp1d(t_grid, model, kind="cubic",
                                bounds_error=False, fill_value=0.0)

    # Quindi _chi2_1d(t_arr) fa tutto in un colpo: dato t_arr, calcola A_opt analiticamente e restituisce il chi2 minimo per quel t_arr. 
    # Il problema a due parametri diventa un problema a un parametro.
    def _chi2_1d(t_arr):
        shift = t_arr - t0_ref
        model_shifted = model_interp(t_grid - shift)  # valuta il modello sulla griglia shiftata di t_arr - T0_REF ns.
        # shift continuo perché model_interp è un interpolatore cubico
        num   = np.sum(wf_amp * model_shifted * weights / sigma**2)  # correlazione pesata tra la wf e il modello shiftato
        den   = np.sum(model_shifted**2  * weights / sigma**2)  # norma pesata del modello
        A_opt = num / den if den > 1e-10 else 1.0
        A_opt = max(A_opt, 1e-6)  # A > 0
        # Per qualsiasi valore fisso di t_arr, il valore ottimale di A che minimizza il chi2 si trova derivando rispetto ad A
        # e uguagliando a zero: dchi2/dA = 0  →  A = sum(wf * model_shifted * w / sigma²) / sum(model_shifted² * w / sigma²)
        res   = (wf_amp - A_opt * model_shifted) * np.sqrt(weights) / sigma
        return float(np.sum(res**2)), A_opt  # A ottimale analitico
        # Il parametro A del fit in questo caso ha le dimensioni di carica / model_min
        # — non è direttamente l'ampiezza in V, ma il chi2 e t_arr non cambiano perché il riscalamento è globale e costante.


    def _chi2_scalar(t_arr):
        val, _ = _chi2_1d(t_arr)
        return val
    # restituisce solo il chi2 scartando A_opt — serve perché minimize_scalar si aspetta una funzione che prende un scalare e restituisce un scalare
    
    # inizializza prima del if/else — necessario per il ramo nelder
    dA     = np.nan
    dt_arr = np.nan
    ndof   = int(np.sum(np.where((t_grid >= t0_ref - rise_left) & (t_grid <= t0_ref + tail), 1, 0))) - 2

    if optimizer == "scalar":
        # bounds: cerco t_arr in un intorno di ±5 ns dal t_arr_0
        bracket = (t_arr_0 - bracket_half, t_arr_0 + bracket_half)

        # funzione di minimizzazione 1d del chi2, con t_arr come unico prametro libero
        result = minimize_scalar(_chi2_scalar,
                             bounds=bracket,
                             method="bounded",
                             options={"xatol": 1e-5}) # controlla la precisione su t_arr —  1e-5 ns = 0.01 ps 

        t_arr_fit       = result.x

        # check: se il fit è converso al bordo del range, probabilmente è sbagliato
        if abs(t_arr_fit - bracket[0]) < 0.01 or abs(t_arr_fit - bracket[1]) < 0.01:
            result.success = False
        chi2_val, A_fit = _chi2_1d(t_arr_fit)  # trova il valore del chi2 e A usando il t_arr uscito dal fit di minimizzazione
    
        # stima errori: errore su t_arr: scan del chi2 ± step attorno al minimo
        try:
            # Stima della derivata seconda del chi2 in t_arr_fit con differenze finite centrate..
            #  Quindi dt_arr = 1/sqrt(d2chi2)
            chi2_p = _chi2_scalar(t_arr_fit + step)
            chi2_m = _chi2_scalar(t_arr_fit - step)
            d2chi2 = (chi2_p - 2*chi2_val + chi2_m) / step**2  # la formula (f(x+h) - 2f(x) + f(x-h)) / h² approssima f''(x)
            dt_arr = 1.0 / np.sqrt(d2chi2) if d2chi2 > 0 else np.nan  # La derivata seconda del chi2 al minimo è l'inverso della varianza sul parametro: sigma²(t_arr) = 1 / (d²chi2/dt_arr²)
            # errore su A: derivata seconda numerica
            eps_A  = A_fit * 0.01
            # pertuba A manualmente
            res_Ap = (wf_amp - (A_fit + eps_A) * model_interp(t_grid - (t_arr_fit - t0_ref)))
            chi2_Ap = float(np.sum((res_Ap * np.sqrt(weights) / sigma)**2))
            d2chi2_A = (chi2_Ap - chi2_val) / eps_A**2
            dA = 1.0 / np.sqrt(d2chi2_A) if d2chi2_A > 0 else np.nan
        except Exception:
            pass

    else:  # nelder
        from scipy.optimize import minimize
        A_0 = float(np.abs(np.min(wf_amp)))

        def _chi2_2d(params):
            A, t_arr = params
            ms  = model_interp(t_grid - (t_arr - t0_ref))
            res = (wf_amp - A * ms) * np.sqrt(weights) / sigma
            return float(np.sum(res**2))

        res2d     = minimize(_chi2_2d, x0=[A_0, t_arr_0],
                            method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-4,
                                    "maxiter": 5000})
        A_fit     = res2d.x[0]
        t_arr_fit = res2d.x[1]
        chi2_val  = res2d.fun
        result    = res2d

    return {
        "A":       A_fit,
        "dA":      dA,
        "t_arr":   t_arr_fit,
        "dt_arr":  dt_arr,
        "chi2":    chi2_val,
        "npts":    int(np.count_nonzero(weights)),
        "success": result.success,
    }


def apply_model_fit(amp_resampled, wfs, t_grid, model, std, t_grid_model=None,
                       max_wfs=None, verbose=True, bracket_half=5.0,
                       w_rise=20.0, w_tail=1.0, rise_left=5.0, rise_right=5.0, tail=150.0,
                       optimizer="scalar", t0_ref=T0_REF_TW, sat_level=None):
    """
    Applica model_fit_wf a tutte le wf raccolte.
    Usa amp_resampled (non normalizzato, non allineato) come input del fit
    I parametri A e t_arr vengono trovati liberamente.
    wfs sono le wf originali, da cui prendere ("amplitude") e ("time_cfd")

    max_wfs — se non None, processa solo le prime max_wfs

    Ritorna lista di dizionari risultato, uno per wf.
    """
    n = len(amp_resampled) if max_wfs is None else min(max_wfs, len(amp_resampled))  # Se wfs=None le prendo tutte, sennò il numero messo
    results = []
    n_failed = 0

    if t_grid_model is None:
        t_grid_model = t_grid
    model_interp = interp1d(t_grid_model, model, kind="cubic", bounds_error=False, fill_value=0.0)

    # se il modello ha griglia diversa dalle wf, interpola std sulla griglia delle wf
    if len(std) != len(t_grid):
        std_interp = interp1d(t_grid_model, std, kind="cubic", bounds_error=False, fill_value=std.mean())
        std = std_interp(t_grid)

    print(f"\n--- apply_model_fit ({n} wf) ---")
    for i in range(n):
        res = model_fit_wf(amp_resampled[i], t_grid, model, std,
                        model_interp=model_interp,
                        A_0=wfs[i].get("amplitude"),
                        t_arr_0=wfs[i].get("time_cfd"),
                        w_rise=w_rise, w_tail=w_tail,
                        rise_left=rise_left, rise_right=rise_right, tail=tail,
                        optimizer=optimizer, t0_ref=t0_ref, bracket_half=bracket_half,
                        sat_level=sat_level, noise=wfs[i].get("noise"))
        results.append(res)

        if not res["success"]:
            n_failed += 1
        if verbose and (i+1) % 1000 == 0:   # verbose=True abilita la stampa del progresso ogni 1000 wf processate
            print(f"  {i+1}/{n} processate...")

    t_arr_list = [r["t_arr"] for r in results if r["success"]]
    A_list     = [r["A"]     for r in results if r["success"]]

    print(f"Fit completati: {n - n_failed}/{n}")
    if t_arr_list:
        print(f"t_arr — mean={np.mean(t_arr_list):.3f} ns,  std={np.std(t_arr_list):.3f} ns")
        print(f"A     — mean={np.mean(A_list)*1e3:.1f} mV,  std={np.std(A_list)*1e3:.1f} mV")

    return results

# =============================================================================
# MODEL FIT — ST SINGOLE
# =============================================================================

def compute_tst_from_single(results_st, wfs_st, min_wfs_per_event=6, chi2_max=None, std_max=None, ch_exclude=None):
    """
    Calcola t_ST per ogni evento mediando i t_arr delle singole wf ST fittate.
    
    Parametri
    ----------
    results_st         — lista risultati da apply_model_fit sulle ST singole
    wfs_st             — lista wf ST singole (stessa corrispondenza indici)
    min_wfs_per_event  — numero minimo di wf ST con fit valido per accettare l'evento
                         (default 6 su 8 — eventi con troppe wf scartate sono rumorosi)
    chi2_max           — scarta fit con chi2 > chi2_max (opzionale)

    Ritorna
    -------
    ev_to_tST_single : dict {eventN: t_ST_mean}
    ev_to_tST_std    : dict {eventN: std_sui_tempi_delle_8_wf}  — utile per diagnostica
    """
    # raccoglie i t_arr e dt_arr per evento
    from collections import defaultdict
    ev_times     = defaultdict(list)  # {eventN: [t_arr_wf0, t_arr_wf1, ...]}
    ev_times_err = defaultdict(list)  # {eventN: [dt_arr_wf0, dt_arr_wf1, ...]}

    for res, wf in zip(results_st, wfs_st):
        if not res["success"]:
            continue
        if ch_exclude is not None and wf.get("channelId") in ch_exclude:
            continue
        if chi2_max is not None and res["chi2"] > chi2_max:
            continue
        ev_n = wf.get("_eventN")
        if ev_n is None:
            continue
        ev_times[ev_n].append(res["t_arr"])
        ev_times_err[ev_n].append(res["dt_arr"])  # errore su t_arr per la media pesata

    ev_to_tST_single = {}
    ev_to_tST_std    = {}
    n_accepted = 0
    n_rejected = 0

    for ev_n, times in ev_times.items():
        if len(times) < min_wfs_per_event:
            n_rejected += 1
            continue

        if std_max is not None and float(np.std(times)) > std_max:
            n_rejected += 1
            continue

        times  = np.array(times)
        errors = np.array(ev_times_err[ev_n])  # dt_arr per ogni wf

        # media pesata per 1/dt_arr² — le wf con fit più preciso pesano di più
        # fallback su media semplice se dt_arr non è disponibile o non valido
        valid = np.isfinite(errors) & (errors > 0)
        if valid.sum() >= 2:
            weights_t = 1.0 / errors[valid]**2
            ev_to_tST_single[ev_n] = float(np.average(times[valid], weights=weights_t))
        else:
            ev_to_tST_single[ev_n] = float(np.mean(times))

        ev_to_tST_std[ev_n] = float(np.std(times))
        n_accepted += 1

    print(f"\n--- compute_tst_from_single ---")
    print(f"Eventi accettati: {n_accepted}  "
          f"(rifiutati per < {min_wfs_per_event} wf valide: {n_rejected})")

    t_means = list(ev_to_tST_single.values())
    t_stds  = list(ev_to_tST_std.values())
    if t_means:
        print(f"t_ST mean — mean={np.mean(t_means):.3f} ns, "
              f"std={np.std(t_means):.3f} ns")
        print(f"Std intra-evento (jitter tra le 8 wf) — "
              f"mean={np.mean(t_stds)*1e3:.1f} ps, "
              f"max={np.max(t_stds)*1e3:.1f} ps")
        # la std intra-evento è il jitter tra le 8 wf ST dello stesso evento
        # se è piccola (<100 ps) le 8 wf sono sincrone e la media è stabile
        # se è grande (>500 ps) c'è jitter fisico o fit instabile

    return ev_to_tST_single, ev_to_tST_std


# =============================================================================
# 8. PLOT RISULTATI FIT
# =============================================================================

def plot_model_fit_results(results, t_grid, amp_resampled, model, n_sample=6, t0_ref=T0_REF_TW, wfs=None, q_min=None, q_max=None):
    """
    Tre plot:
        a) distribuzione t_arr dal model_fit
        b) distribuzione chi2
        c) griglia di n_sample wf con template fittato sovrapposto
    """
    A_list       = np.array([r["A"]        for r in results if r["success"]])
    t_arr_list   = np.array([r["t_arr"]    for r in results if r["success"]])
    chi2_list    = np.array([r["chi2"]     for r in results if r["success"]])

    # --- a) distribuzione t_arr ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Risultati model_fit", fontsize=14)

    counts_t, edges_t = np.histogram(t_arr_list, bins=100)
    mu_peak = edges_t[np.argmax(counts_t)]
    r_tof   = (mu_peak - 2.0, mu_peak + 2.0)
    axes[0].hist(t_arr_list, bins=100, range=(np.min(t_arr_list), np.max(t_arr_list)), color="seagreen", alpha=0.8)
    axes[0].set_xlabel("t_arr (ns)", fontsize=18)
    axes[0].set_ylabel("Conteggi", fontsize=18)
    axes[0].set_title(f"Tempo di arrivo — {len(t_arr_list)} wf", fontsize=18)
    axes[0].grid(True, alpha=0.4)
    axes[0].tick_params(axis='both', labelsize=16)

    # --- b) chi2 ---
    axes[1].hist(chi2_list, bins=100, range=(0, np.percentile(chi2_list, 99)),
                 color="crimson", alpha=0.8)
    axes[1].set_xlabel("chi2", fontsize=18)
    axes[1].set_ylabel("Conteggi", fontsize=18)
    axes[1].set_title(f"Distribuzione chi2", fontsize=18)
    axes[1].grid(True, alpha=0.4)
    axes[1].tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

    # --- c) griglia wf con fit sovrapposto ---
    try:
        trapz_fn = np.trapezoid
    except AttributeError:
        trapz_fn = np.trapz
    charge = trapz_fn(np.abs(amp_resampled), t_grid, axis=1) * 1e3  # mV·ns
    idx_ok = [i for i, r in enumerate(results) if r["success"]
              and (q_min is None or charge[i] >= q_min)
              and (q_max is None or charge[i] <= q_max)]
    if not idx_ok:
        return
    idx_sample = np.random.choice(idx_ok, size=min(n_sample, len(idx_ok)), replace=False)

    cols = min(3, len(idx_sample))
    rows = int(np.ceil(len(idx_sample) / cols))
    #fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4, 3))
    #fig.suptitle("Waveform fitted with template", fontsize=18)
    axes = np.array(axes).flatten()

    model_interp_plot = interp1d(t_grid, model, kind="cubic",
                              bounds_error=False, fill_value=0.0)

    for plot_i, wf_i in enumerate(idx_sample):
        ax  = axes[plot_i]
        res = results[wf_i]
        A, t_arr = res["A"], res["t_arr"]

        # shift del template fittato
        model_shifted = model_interp_plot(t_grid - (t_arr - t0_ref))

        ax.plot(t_grid, amp_resampled[wf_i]*1e3, color="steelblue",
                lw=1.5, label="waveform")
        ax.plot(t_grid, A * model_shifted * 1e3, color="crimson",
                lw=1.5, linestyle="--", label=f"fit")
        ax.axvline(t_arr, color="black", lw=1.8, linestyle=":",
                   label=rf"$t_{{\mathrm{{fit}}}} = {t_arr:.2f}$ ns")
        
        wf_info = wfs[wf_i] if wfs is not None else {}
        bar_id  = wf_info.get("bar_id", "?")
        end     = wf_info.get("end", "?")
        
        q_mv = charge[wf_i]
        #ax.set_title(f"Q={q_mv:.0f} mV·ns, A={A*1e3:.1f}mV")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.1)
        ax.tick_params(axis='both')

    for i in range(len(idx_sample), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_st_event_fit(ev_idx, events, results_st, wfs_st,
                      t_grid_st, amp_resampled_st,
                      model_st=None, models_ch=None,
                      t_grid_model=None, t0_ref=T0_REF_ST, t0_refs_ch=None):
    """
    Collage 2x4 delle 8 wf ST di un singolo evento con il template fittato sovrapposto.
    Analogo al collage ST di read_wf.py ma con fit e info dal model_fit.

    Parametri
    ----------
    ev_idx             — indice dell'evento in events (non eventN)
    events             — lista eventi
    results_st         — lista risultati da apply_model_fit sulle ST singole
    wfs_st             — lista wf ST singole (stessa corrispondenza indici)
    t_grid_st          — griglia temporale comune
    amp_resampled_st   — array ricampionato (n_wf x n_campioni)
    model_st           — template modello
    t0_ref             — riferimento temporale del modello
    """
    ev      = events[ev_idx]
    ev_n    = ev["eventN"]

    # trova gli indici in wfs_st_single che appartengono a questo evento
    idx_ev = [i for i, wf in enumerate(wfs_st)
              if wf.get("_eventN") == ev_n]

    if not idx_ev:
        print(f"Nessuna wf ST trovata per evento {ev_n} (idx={ev_idx}).")
        return

    if t_grid_model is None:
            t_grid_model = t_grid_st

    # se non ci sono modelli per canale, costruisci interp unico fuori dal loop
    if models_ch is None:
        model_interp_default = interp1d(t_grid_model, model_st, kind="cubic",
                                         bounds_error=False, fill_value=0.0)
    else:
        model_interp_default = None


    # t_ST medio dell'evento (media dei t_arr delle wf con fit valido)
    t_arr_valid = [results_st[i]["t_arr"]
                   for i in idx_ev if results_st[i]["success"]]
    t_ST_mean   = float(np.mean(t_arr_valid)) if t_arr_valid else None

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    '''
    fig.suptitle(
        f"ST — Evento {ev_n}  |  "
        f"{len(t_arr_valid)}/{len(idx_ev)} fit validi  |  "
        f"t_ST = {t_ST_mean:.3f} ns" if t_ST_mean else f"ST — Evento {ev_n}",
        fontsize=14)
    '''
    axes = axes.flatten()

    for plot_i, wf_i in enumerate(idx_ev):
        ax  = axes[plot_i]
        wf  = wfs_st[wf_i]
        res = results_st[wf_i]
        ch  = wf.get("channelId")

        # t0_ref specifico del canale se disponibile, altrimenti fallback su t0_ref globale
        t0_ref_ch = t0_refs_ch[ch] if (t0_refs_ch is not None and ch in t0_refs_ch) else t0_ref

        # scegli il modello giusto per questo canale
        if models_ch is not None and ch in models_ch:
            model_interp = interp1d(t_grid_model, models_ch[ch],
                                    kind="cubic", bounds_error=False, fill_value=0.0)
        elif model_interp_default is not None:
            model_interp = model_interp_default
        else:
            # nessun modello disponibile per questo canale
            model_interp = None

        # wf ricampionata
        ax.plot(t_grid_st, amp_resampled_st[wf_i] * 1e3,
                color="steelblue", lw=1.0, label="waveform")

        if res["success"]:
            A, t_arr = res["A"], res["t_arr"]
            model_shifted = model_interp(t_grid_st - (t_arr - t0_ref_ch))
            ax.plot(t_grid_st, A * model_shifted * 1e3,
                    color="crimson", lw=1.8, linestyle="--", label="fit")
            ax.axvline(t_arr, color="black", lw=1.5, linestyle=":",
                   label=rf"$t_{{\mathrm{{fit}}}} = {t_arr:.2f}$ ns")
            ax.set_xlim(50, 250)
            ax.set_title(f"Board {wf['boardId']} Ch {wf['channelId']}")

        if t_ST_mean is not None:
            #ax.axvline(t_ST_mean, color="darkorange", lw=1.2,
             #              linestyle="--", alpha=0.7, label=f"mean={t_ST_mean:.2f}")
            title = (f"Board {wf['boardId']} Ch {wf['channelId']}")
        else:
            title = (f"Board {wf['boardId']} Ch {wf['channelId']}  |  "
                     f"fit FALLITO")
            ax.set_facecolor("#fff0f0")  # sfondo rosato per i fit falliti

        #ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both')

    # nascondi i pannelli vuoti (se < 8 wf per l'evento)
    for i in range(len(idx_ev), nrows * ncols):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 9. CALCOLO TOF
# =============================================================================

def compute_tof_model(results, wfs, events, q_min=None, q_max=None, chi2_max=None, ev_to_tST_model=None):
    """
    Calcola il TOF usando i t_arr dal model_fit invece del CFD standard.
    TOF = time_bar_model - ST_sum["time_cfd"]
    dove time_bar_model = (t_arr_A + t_arr_B) / 2 per la barra.

    Richiede che wfs[i] abbia bar_id, end, e che events abbia ST_sum["time_cfd"].

    Parametri
    ----------
    results  — lista dizionari da apply_model_fit (uno per ogni wf fittata). Ogni dizionario ha i campi A, da, t_arr, dt_arr, chi2, success
    wfs      — lista wf originali, non ricampionate (stessa corrispondenza indici di results → results[i] corrisponde a wfs[i]) 
    (Ogni wf è un dizionario con tutti i campi della pipeline — bar_id, end, _eventN, amplitude, signal ecc.)
    events   — lista eventi (per prendere ST_sum["time_cfd"] e trovare charge_bar per il filtro energia)
    q_min/q_max — filtro energia in mV·ns su charge_bar (opzionale)

    Ritorna:
        tof_list   — lista di tutti TOF calcolati su tutte le barre (ns)
        tof_by_bar — dizionario {bar_id: [tof, tof, ...]} per analisi per barra
    """
    # costruisci mappa evento → t_ST
    ev_to_tST = {} # dizionario semplice {eventN: t_ST}: {0: 105.3, 1: 106.1, 2: 104.8, ...} (eventN → t_ST in ns)
    
    if ev_to_tST_model is not None:
        ev_to_tST = ev_to_tST_model
    else:
        for ev in events:
            st_sum = ev.get("ST_sum")  # Per ogni evento cerca ST_sum["time_cfd"] — il tempo della waveform somma dello ST calcolato dal CFD standard.
            if st_sum and st_sum.get("time_cfd") is not None:
                ev_to_tST[ev["eventN"]] = st_sum["time_cfd"]  # indicizza per eventN: "dato l'evento numero X, qual è il suo t_ST?" con ev_to_tST[X].

    # costruisci mappa (eventN, bar_id) → {A: t_arr_A, B: t_arr_B, charge_bar}
    bar_times = {}  # chiave: (eventN, bar_id)

    for i, (res, wf) in enumerate(zip(results, wfs)):  # itera in parallelo su results e wfs mantenendo la corrispondenza di indice
        if not res["success"]:
            continue
        if chi2_max is not None and res["chi2"] > chi2_max:
            continue
        bar_id = wf.get("bar_id")
        end    = wf.get("end")
        if bar_id is None or end is None:
            continue

        # trova l'evento corrispondente cercando in events
        # wf è un riferimento all'oggetto originale — usiamo boardId e channelId
        # per trovare l'eventN cerchiamo in events quale evento contiene questa wf
        ev_n = wf.get("_eventN")  # campo aggiunto sotto se necessario
        if ev_n is None:
            continue

        key = (ev_n, bar_id)  # Per ogni wf fittata con successo costruisce una chiave (eventN, bar_id)
        if key not in bar_times:
            # Se la chiave non esiste ancora nel dizionario, la inizializza con t_A=None, t_B=None.
            bar_times[key] = {"t_L": None, "t_R": None, "charge_bar": None}
        # scrive il t_arr fittato nel campo giusto — t_L se end="L", t_R se end="R" — usando f"t_{end}" che diventa "t_L" o "t_R".
        bar_times[key][f"t_{end}"] = res["t_arr"]

    '''
    Dopo aver processato tutte le wf, bar_times ha forma tipo:
    {
    (0, 9):  {"t_L": 127.3, "t_R": 127.6},   # evento 0, barra 9
    (0, 29): {"t_L": 128.1, "t_R": 127.9},   # evento 0, barra 29
    (1, 9):  {"t_L": 126.8, "t_R": None},     # evento 1, barra 9 — polo R non fittato
    (1, 29): {"t_L": 128.4, "t_R": 128.2},
    ...
    }
    '''
    # La chiave è una tupla (eventN, bar_id)

    # dizionario {(eventN, bar_id): charge_bar} per poter fare il filtro energia.
    # Recupera charge_bar da ev["bars"][bid] che è già stato calcolato
    ev_charge = {}
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            ev_charge[(ev["eventN"], bid)] = bar.get("charge_bar")

    # calcola TOF
    tof_list   = []
    tof_by_bar = {}

    for (ev_n, bar_id), bt in bar_times.items():  # bar_times.items() restituisce coppie (chiave, valore)
        tL = bt["t_L"]
        tR = bt["t_R"]
        if tL is None or tR is None:
            continue
        t_ST = ev_to_tST.get(ev_n)  # Cerca il t_ST dell'evento corrispondente in ev_to_tST
        if t_ST is None:
            continue

        q = ev_charge.get((ev_n, bar_id))  # Applica il filtro energia usando ev_charge
        if q is not None:
            q_mV = q * 1e3
            if q_min is not None and q_mV < q_min:
                continue
            if q_max is not None and q_mV > q_max:
                continue

        time_bar = (tL + tR) / 2.0  # Calcola time_bar = (tL + tR) / 2 — media dei due poli
        tof      = time_bar - t_ST  # Calcola tof = time_bar - t_ST
        # Aggiunge il TOF a tof_list (lista globale) e a tof_by_bar[bar_id] (lista per barra)
        tof_list.append(tof)
        if bar_id not in tof_by_bar:  # se bar_id non è ancora in tof_by_bar, inizializzalo con una lista vuota, poi appendici il valore
            tof_by_bar[bar_id] = []
        tof_by_bar[bar_id].append(tof)

    print(f"\n--- compute_tof_template ---")
    print(f"TOF calcolati: {len(tof_list)}")
    if tof_list:
        print(f"mean={np.mean(tof_list):.3f} ns,  std={np.std(tof_list):.3f} ns")

    return tof_list, tof_by_bar


def plot_tof_template(tof_list, bar_id=None):
    """Istogramma TOF da model_fit con fit gaussiano."""

    if not tof_list:
        print("Nessun TOF da plottare.")
        return

    data = np.array(tof_list)

    mu_peak    = np.mean(data)
    half_width = 1.0  # ns
    fit_range  = (mu_peak - half_width, mu_peak + half_width)

    counts, bin_edges = np.histogram(data, bins=100, range=fit_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=100, color="steelblue", range=fit_range,
            alpha=0.7, label="TOF data")

    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    mu0    = bin_centers[np.argmax(counts)]
    sigma0 = (fit_range[1] - fit_range[0]) / 6.0
    A0     = np.max(counts)
    p0     = [A0, mu0, sigma0]

    try:
        popt, pcov = curve_fit(gauss, bin_centers, counts, p0=p0, maxfev=10000)
        A, mu, sigma   = popt
        dA, dmu, dsigma = np.sqrt(pcov.diagonal())

        print(f"--- Fit gaussiano TOF (model fit) ---")
        if bar_id is not None:
            print(f"Barra {bar_id}")
        print(f"A = {A:.1f},  mu = ({mu:.1f} ± {dmu:.1f}) ns,  "
              f"sigma = ({abs(sigma)*1e3:.1f} ± {dsigma*1e3:.1f}) ps")

        x_plot = np.linspace(mu - 3*sigma, mu + 3*sigma, 2000)
        ax.plot(x_plot, gauss(x_plot, *popt),
                color="black", lw=2, label=f"sigma_tof = ({abs(sigma)*1e3:.1f} ± {dsigma*1e3:.1f}) ps")
    except Exception as e:
        print(f"\n--- Fit gaussiano TOF fallito: {e} ---")

    title = "TOF da template fit"
    if bar_id is not None:
        title += f" — Barra {bar_id}"
    ax.set_xlabel("TOF (ns)", fontsize=14)
    ax.set_ylabel("Conteggi", fontsize=14)
    ax.set_title(f"{title}  ({len(tof_list)} entries)", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ESECUZIONE STANDALONE
# =============================================================================

if __name__ == "__main__":
    from read_wf      import load_waveforms
    from baseline_wf  import apply_baseline, cleanup_baseline, plot_pedestal
    from amplitude_wf import apply_amplitude, filter_pileup_events, plot_amplitude
    from tw_map       import apply_tw_map
    from charge_wf    import apply_charge
    from cfd_wf       import apply_cfd, build_st_sum
    from bar_analysis import (build_bars, plot_charge_bars, build_bar_times, compute_tof,
                           plot_cfd_times, plot_rise_time_bars, plot_tof)

    file_path = "/home/ubuntu_leo/FOOT/shoe_old/build/Reconstruction/CNAO2025_data/8102/wf_raw018.bin"

    # --- READ ---
    events = load_waveforms(file_path)

    # --- BASELINE + PEDESTAL ---
    events, bl_ST, std_ST, bl_TW, std_TW = apply_baseline(events, method="mean")
    cleanup_baseline(events)
    #plot_pedestal(events)

    # --- AMPLITUDE + PILE-UP ---
    events, amp_ST, amp_TW = apply_amplitude(events, n_sigma=8, noise_std=None, st_amp_max=0.6)
    #plot_amplitude(amp_ST, amp_TW)
    events = filter_pileup_events(events)   # filtro pile-up in V

    # --- CFD --- 
    events = build_st_sum(events)
    events, times_TW, times_ST, times_ST_sum = apply_cfd(events, method_st="simple", method_tw="simple")

    # --- CHARGE ---
    events, charge_TW = apply_charge(events, method="integral")

    # --- bar_analysis ---
    events = apply_tw_map(events)

    events, charge_bars = build_bars(events)
    #plot_charge_bars(charge_bars, fit=True, fit_range=None, n_peaks=2)

    events, bar_times = build_bar_times(events)
    #plot_cfd_times(times_ST_sum, bar_times)
    #plot_rise_time_bars(events)

    # --- TOF (ST:simple_cfd, TW:simple_cfd) ---
    events, tof_list_cfd = compute_tof(events, q_min=10*1e3, q_max=13.5*1e3, bars=None)
    tof_by_bar_cfd = {}
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            if bar.get("tof") is not None:
                tof_by_bar_cfd.setdefault(bid, []).append(bar["tof"])
    #plot_tof(tof_list)

    
    # ================================================================
    # MODEL START COUNTER
    # ================================================================

    ST_BOARD    = 27
    ST_CHANNELS = list(range(8))   # canali 0-7, tutti su board 27

    ST_PARAMS = dict(w_rise=1.0, w_tail=1.0,
                    rise_left=1.0, rise_right=1.5,
                    tail=4.0, bracket_half=5.0,
                    optimizer="scalar")
    
    # ----------------------------------------------------------------
    # METODO 1 — modello e fit su ST_sum
    # ----------------------------------------------------------------

    # 1. raccolta wfs ST_sum
    wfs_st = collect_st_sum(events)

    # 2. ricampionamento wfs ST_sum
    t_grid_st, amp_resampled_st = resample_wfs(wfs_st, step=RESAMPLE_STEP)

    # 3. normalizzazione wfs ST_sum
    amp_norm_st, _ = normalize_wfs(amp_resampled_st, wfs_st, method="amplitude")

    # 4-5. costruzione modello iterativa ST_sum (contiene align_wfs(4) e build_model(5))
    model_st, std_st, amp_aligned_st = build_model_iterative(amp_resampled_st, amp_norm_st, wfs_st, t_grid_st,
                                                             n_iter=0, method="weighted", max_wfs=None, t0_ref=T0_REF_ST, max_shift=10.0, **ST_PARAMS)

    # 6. plot modello wfs ST_sum
    plot_model(t_grid_st, model_st, std_st, t0_ref=T0_REF_ST)
    plot_model_compare(t_grid_st, model_st, std_st, amp_aligned_st, wfs_st, n_wfs=100)

    # 7. fit wfs ST_sum
    results_st_sum = apply_model_fit(amp_resampled_st, wfs_st, t_grid_st, model_st, std_st, t0_ref=T0_REF_ST, **ST_PARAMS)
    t_arr_fit_sum = {wf["_eventN"]: res["t_arr"] for res, wf in zip(results_st_sum, wfs_st) if res["success"]}
    print(f"Metodo 1 (fit ST_sum): {len(t_arr_fit_sum)} eventi con t_ST")
    plot_model_fit_results(results_st_sum, t_grid_st, amp_resampled_st, model_st, n_sample=1, t0_ref=T0_REF_ST)

    # ----------------------------------------------------------------
    # METODO 2 — 8 modelli separati, uno per canale ST
    # ----------------------------------------------------------------

    # 1. raccolta singole wf ST — unica per tutti i canali
    wfs_st_single = collect_st_single(events)

    # 2. ricampionamento singole wf ST unico — griglia comune a tutti i canali
    t_grid_st_single, amp_resampled_st_single = resample_wfs(wfs_st_single, step=RESAMPLE_STEP)

    # indici per canale — i.e. ch_indices[0] = [0, 8, 16, 24, ...]: indici delle wf del canale 0
    ch_indices = {ch: [i for i, wf in enumerate(wfs_st_single) if wf["channelId"] == ch and wf["boardId"] == ST_BOARD] for ch in ST_CHANNELS}
    
    print(f"\n--- Distribuzione wf ST per canale ---")
    for ch in ST_CHANNELS:
        print(f"  Ch {ch}: {len(ch_indices[ch])} wf")

    # loop sui canali — costruzione modello e fit
    models_ch   = {}   # {ch: model}
    stds_ch     = {}   # {ch: std}
    results_ch  = {}   # {ch: results}
    wfs_ch_dict = {}   # {ch: wfs}
    t0_refs_ch = {}    # {ch: t0_ref}

    for ch in ST_CHANNELS:
        idx = ch_indices[ch]
        if not idx:
            print(f"  Ch {ch}: nessuna wf, skip")
            continue

        wfs_ch  = [wfs_st_single[i] for i in idx]
        amp_ch  = amp_resampled_st_single[idx]   # array 2D con solo le ~8000 wf di quel canale.

        print(f"\n=== Canale {ch} — {len(wfs_ch)} wf ===")

        # 3. normalizzazione singole wf ST per ogni canale
        amp_norm_ch, _ = normalize_wfs(amp_ch, wfs_ch, method="amplitude")

        # tempo di riferimento diverso per ogni canale
        T0_REF_ch = float(np.median([t_grid_st_single[np.argmin(amp_ch[i])] for i in range(len(amp_ch))]))
        t0_refs_ch[ch] = T0_REF_ch
        print(f"  T0_REF_ST ch{ch} = {T0_REF_ch:.3f} ns")

        # 4-5. costruzione modello iterativa singole wf ST per ogni canale (contiene align_wfs(4) e build_model(5))
        model_ch, std_ch, amp_aligned_ch = build_model_iterative(amp_ch, amp_norm_ch, wfs_ch, t_grid_st_single, max_shift=10.0,
                                                                n_iter=0, method="weighted", max_wfs=None, t0_ref=T0_REF_ch, **ST_PARAMS)

        # 6. plot modello singole wf ST per canale
        #plot_model(t_grid_st_single, model_ch, std_ch, t0_ref=T0_REF_ch)
        #plot_model_compare(t_grid_st_single, model_ch, std_ch, amp_aligned_ch, wfs_ch, n_wfs=300)

        # 7. fit singole wf ST per canale
        res_ch = apply_model_fit(amp_ch, wfs_ch, t_grid_st_single, model_ch, std_ch, t0_ref=T0_REF_ch, **ST_PARAMS)

        models_ch[ch]   = model_ch
        stds_ch[ch]     = std_ch
        results_ch[ch]  = res_ch   # lista di ~8000 dizionari {A, t_arr, chi2, success, ...}, uno per wf.
        wfs_ch_dict[ch] = wfs_ch

    # ricostruisci results_st_single e wfs_st_single nell'ordine originale
    # necessario per compute_tst_from_single che si aspetta corrispondenza indici
    results_st_single = [None] * len(wfs_st_single)
    for ch in ST_CHANNELS:
        if ch not in results_ch:
            continue
        for res, i in zip(results_ch[ch], ch_indices[ch]):
            results_st_single[i] = res

    # sostituisci None residui con fit fallito così compute_tst_from_single ignora senza crashare.
    dummy_failed = {"A": 0.0, "dA": np.nan, "t_arr": 0.0, "dt_arr": np.nan, "chi2": np.inf, "success": False}
    results_st_single = [r if r is not None else dummy_failed for r in results_st_single]

    # plot evento singolo con i modelli per canale
    plot_st_event_fit(ev_idx=87, events=events, results_st=results_st_single, wfs_st=wfs_st_single, t_grid_st=t_grid_st_single,
                      amp_resampled_st=amp_resampled_st_single, models_ch=models_ch, t_grid_model=t_grid_st_single, t0_refs_ch=t0_refs_ch)

    # calcolo t_ST per evento come media pesata degli 8 canali
    t_arr_fit_single, t_arr_fit_std = compute_tst_from_single(results_st_single, wfs_st_single, min_wfs_per_event=6,
                                                              chi2_max=None, std_max=None, ch_exclude={7})

    # diagnostica jitter intra-evento
    stds = list(t_arr_fit_std.values())
    print(f"\n--- Diagnostica jitter intra-evento (modelli per canale) ---")
    print(f"Percentili jitter (ps): "
          f"10%={np.percentile(stds,10)*1e3:.0f}  "
          f"25%={np.percentile(stds,25)*1e3:.0f}  "
          f"50%={np.percentile(stds,50)*1e3:.0f}  "
          f"75%={np.percentile(stds,75)*1e3:.0f}  "
          f"90%={np.percentile(stds,90)*1e3:.0f}  "
          f"99%={np.percentile(stds,99)*1e3:.0f} ps")
    n_good = sum(1 for s in stds if s < 0.2)
    print(f"Eventi con jitter < 200 ps: {n_good}/{len(stds)} "
          f"({100*n_good/len(stds):.1f}%)")


    # ================================================================
    # CONFRONTO METODI
    # ================================================================

    print(f"\n--- Confronto t_ST evento per evento ---")
    t_sum_dict  = {ev["eventN"]: ev["ST_sum"]["time_cfd"] for ev in events
                   if ev.get("ST_sum") and ev["ST_sum"].get("time_cfd") is not None}

    # diff fit_sum vs CFD_sum
    diffs_sum_cfd = [t_arr_fit_sum[ev_n] - t_sum_dict[ev_n]
                    for ev_n in t_arr_fit_sum if ev_n in t_sum_dict]
    print(f"  fit_ST_sum - CFD_sum:     "
        f"mean={np.mean(diffs_sum_cfd)*1e3:.1f} ps, "
        f"std={np.std(diffs_sum_cfd)*1e3:.1f} ps")

    # diff fit_single vs CFD_sum
    diffs_single_cfd = [t_arr_fit_single[ev_n] - t_sum_dict[ev_n]
                        for ev_n in t_arr_fit_single if ev_n in t_sum_dict]
    print(f"  fit_ST_single - CFD_sum:  "
        f"mean={np.mean(diffs_single_cfd)*1e3:.1f} ps, "
        f"std={np.std(diffs_single_cfd)*1e3:.1f} ps")

    # diff fit_single vs fit_sum
    diffs_single_sum = [t_arr_fit_single[ev_n] - t_arr_fit_sum[ev_n]
                        for ev_n in t_arr_fit_single if ev_n in t_arr_fit_sum]
    print(f"  fit_ST_single - fit_sum:  "
        f"mean={np.mean(diffs_single_sum)*1e3:.1f} ps, "
        f"std={np.std(diffs_single_sum)*1e3:.1f} ps")

    
    # ================================================================
    # MODEL TOF WALL
    # ================================================================

    TW_PARAMS = dict(w_rise=1.0, w_tail=1.0,
                        rise_left=2.0, rise_right=5.0,
                        tail=150.0, bracket_half=5.0,
                        optimizer="scalar")

    # 1. raccolta wfs TW
    bar_ids = [9]
    wfs_tw = collect_tw(events, bar_ids=bar_ids)
    
    # 2. ricampionamento wfs TW
    t_grid_tw, amp_resampled_tw = resample_wfs(wfs_tw, step=RESAMPLE_STEP)
    
    # 3. normalizzazione wfs TW
    amp_norm_tw, amplitudes_tw = normalize_wfs(amp_resampled_tw, wfs_tw, method="charge")

    # 4-5. costruzione modello iterativa TW (contiene align_wfs(4) e build_model(5))
    model_tw, std_tw, amp_aligned_tw = build_model_iterative(amp_resampled_tw, amp_norm_tw, wfs_tw, t_grid_tw, n_iter=0,
                                                            method="weighted", max_wfs=None, t0_ref=T0_REF_TW, max_shift=None,**TW_PARAMS)

    # 6. plot modello
    plot_model(t_grid_tw, model_tw, std_tw, t0_ref=T0_REF_TW)
    plot_model_compare(t_grid_tw, model_tw, std_tw, amp_aligned_tw, wfs_tw, n_wfs=100, end_filter=None, q_min=1000, q_max=None)
    #plot_model_compare(t_grid_tw, model_tw, std_tw, amp_aligned_tw, wfs_tw, n_wfs=300, end_filter="L")  # polo Left
    #plot_model_compare(t_grid_tw, model_tw, std_tw, amp_aligned_tw, wfs_tw, n_wfs=300, end_filter="R")  # polo Right

    # 7. fit wfs TW
    results_tw = apply_model_fit(amp_resampled_tw, wfs_tw, t_grid_tw, model_tw, std_tw, t0_ref=T0_REF_TW, **TW_PARAMS)   

    # 8. plot risultati
    np.random.seed(1)   # stessa waveform a ogni esecuzione
    plot_model_fit_results(results_tw, t_grid_tw, amp_resampled_tw, model_tw, n_sample=1,
                           wfs=wfs_tw, q_min=10000, q_max=13500)

    # 8b. fit con il modello della barra 9 sulle wf della barra 29
    wfs_29 = collect_tw(events, bar_ids=[29])
    t_grid_29, amp_29 = resample_wfs(wfs_29, step=RESAMPLE_STEP)
    results_29 = apply_model_fit(amp_29, wfs_29, t_grid_29, model_tw, std_tw,
                                 t_grid_model=t_grid_tw, t0_ref=T0_REF_TW, **TW_PARAMS)
    np.random.seed(1)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    plot_model_fit_results(results_29, t_grid_29, amp_29, model_tw, n_sample=1,
                           wfs=wfs_29, q_min=10000, q_max=13500)
    # 9. calcolo TOF (ST:simple_cfd, TW:fit)
    tof_list, tof_by_bar = compute_tof_model(results_tw, wfs_tw, events, q_min=10e3, q_max=13.5e3, chi2_max=None, ev_to_tST_model=None)
    
    # ----------------------------------------------------------------
    # TEST SATURAZIONE — costruisci modello su wf originali,
    # applica fit sulle wf saturate artificialmente
    # ----------------------------------------------------------------
    SAT_LEVEL_MV = 200.0  # soglia di saturazione in mV

    # wf saturate artificialmente
    amp_saturated, n_sat, frac_sat = apply_saturation(amp_resampled_tw, sat_level_mv=SAT_LEVEL_MV)

    # fit sulle wf saturate
    results_tw_sat = apply_model_fit(amp_saturated, wfs_tw, t_grid_tw, model_tw, std_tw, **TW_PARAMS, sat_level=SAT_LEVEL_MV/1e3)   # da mV a V

    # plot confronto fit su wf saturate
    plot_model_fit_results(results_tw_sat, t_grid_tw, amp_saturated, model_tw, n_sample=1, t0_ref=T0_REF_TW)

    # TOF con wf saturate (ST:simple_cfd, TW:fit_sat)
    tof_list_sat, tof_by_bar_sat = compute_tof_model(results_tw_sat, wfs_tw, events, q_min=10e3, q_max=13.5e3, chi2_max=None, ev_to_tST_model=None)

    # confronto carica: fit su wf saturate vs carica originale
    charges_fit, charges_orig, diff = compute_charge_from_fit(results_tw_sat, wfs_tw, t_grid_tw, model_tw, std_tw, t0_ref=T0_REF_TW)
    charge_bars_fit, charge_bars_orig = plot_charge_bar_from_fit(results_tw_sat, wfs_tw, events, t_grid_tw, model_tw,
                                                                 bar_id=9, t0_ref=T0_REF_TW, fit=True, fit_range=None)

    # --- TOF (ST:simple_cfd, TW: cfd_wfs_sat) ---
    tof_list_cfd_sat, tof_by_bar_cfd_sat = compute_tof_cfd_sat(events, sat_level_mv=SAT_LEVEL_MV, q_min=10e3, q_max=13.5e3)

    # plot
    for bid in bar_ids:
        plot_tof_template(tof_by_bar_sat[bid],  bar_id=bid)

    
    # ================================================================
    # CALCOLO TOF
    # ================================================================

    # scegli il metodo ST:
    #   t_arr_cfd        — Metodo 1: time_cfd
    #   t_arr_fit_sum    — Metodo 2: fit ST_sum
    #   t_arr_fit_single — Metodo 3: media fit 8 canali separati
    tof_list_sum, tof_by_bar_sum = compute_tof_model(results_tw, wfs_tw, events, q_min=10e3, q_max=13.5e3, chi2_max=None, 
                                                      ev_to_tST_model=t_arr_fit_sum)
    tof_list_single, tof_by_bar_single = compute_tof_model(results_tw, wfs_tw, events, q_min=10e3, q_max=13.5e3, chi2_max=None, 
                                                ev_to_tST_model=t_arr_fit_single)

    def tof_bar_cfd(events, bar_id, ev_to_tST, q_min=10e3, q_max=13.5e3):
        """TOF = time_bar della barra (CFD semplice, media L/R) - t_ST dell'evento.
        Taglio sulla carica di barra in mV·ns."""
        tofs = []
        for ev in events:
            bar = ev.get("bars", {}).get(bar_id)
            if bar is None or bar.get("time_bar") is None or bar.get("charge_bar") is None:
                continue
            if not (q_min <= bar["charge_bar"] * 1e3 <= q_max):
                continue
            t_st = ev_to_tST.get(ev["eventN"])
            if t_st is None:
                continue
            tofs.append(bar["time_bar"] - t_st)
        return tofs

    t_st_cfd = {ev["eventN"]: ev["ST_sum"]["time_cfd"] for ev in events
                if ev.get("ST_sum") and ev["ST_sum"].get("time_cfd") is not None}

    for bid in bar_ids:
        for name, t_st_map in (("CFD su ST_sum", t_st_cfd),
                               ("fit su ST_sum", t_arr_fit_sum),
                               ("fit sui singoli canali", t_arr_fit_single)):
            print(f"\n --- t_bar: CFD | t_ST: {name} ---")
            plot_tof_template(tof_bar_cfd(events, bid, t_st_map), bar_id=bid)

    # --- confronto metodi TW, t_ST = CFD semplice su ST_sum ---
    for bid in bar_ids:
        tof_tw = {"simple": tof_by_bar_cfd[bid], "fit": tof_by_bar[bid],
            "simple_sat": tof_by_bar_cfd_sat[bid], "fit_sat": tof_by_bar_sat[bid]}
        for method in ("tangent", "bipolar"):
            events, *_ = apply_cfd(events, method_st="simple", method_tw=method)
            events, _ = build_bar_times(events)
            tof_tw[method] = tof_bar_cfd(events, bid, t_st_cfd)
        for method, tofs in tof_tw.items():
            print(f"\n --- t_ST: CFD su ST_sum | t_bar: {method} ---")
            plot_tof_template(tofs, bar_id=bid)
'''
    # ================================================================
    # CONFRONTO METODI
    # ================================================================

    print(f"\n--- Confronto t_ST evento per evento ---")
    t_sum_dict  = {ev["eventN"]: ev["ST_sum"]["time_cfd"] for ev in events
                   if ev.get("ST_sum") and ev["ST_sum"].get("time_cfd") is not None}
    # diff fit_sum vs CFD_sum
    diffs_sum_cfd = [t_arr_fit_sum[ev_n] - t_sum_dict[ev_n]
                    for ev_n in t_arr_fit_sum if ev_n in t_sum_dict]
    print(f"  fit_ST_sum - CFD_sum:     "
        f"mean={np.mean(diffs_sum_cfd)*1e3:.1f} ps, "
        f"std={np.std(diffs_sum_cfd)*1e3:.1f} ps")

    # diff fit_single vs CFD_sum
    diffs_single_cfd = [t_arr_fit_single[ev_n] - t_sum_dict[ev_n]
                        for ev_n in t_arr_fit_single if ev_n in t_sum_dict]
    print(f"  fit_ST_single - CFD_sum:  "
        f"mean={np.mean(diffs_single_cfd)*1e3:.1f} ps, "
        f"std={np.std(diffs_single_cfd)*1e3:.1f} ps")

    # diff fit_single vs fit_sum
    diffs_single_sum = [t_arr_fit_single[ev_n] - t_arr_fit_sum[ev_n]
                        for ev_n in t_arr_fit_single if ev_n in t_arr_fit_sum]
    print(f"  fit_ST_single - fit_sum:  "
        f"mean={np.mean(diffs_single_sum)*1e3:.1f} ps, "
        f"std={np.std(diffs_single_sum)*1e3:.1f} ps")
'''   