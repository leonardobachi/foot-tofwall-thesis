"""
results.py — Bar reconstruction from the results_<run>.pkl files and final plots.

For every event the two ends of each TW bar are combined:
    charge_bar = sqrt(Q_L * Q_R),   time_bar = (t_L + t_R)/2,   tof = time_bar - t_ST
with the time method chosen for the TW (--tw-method) and for the ST sum
(--st-method). Then: charge -> energy (Birks), per-bar TOF calibration, charge
identification (Z), position along the bar and, optionally, the charge
equalisation along the bar (--charge-eq).

Charges in the selections are in mV*ns (e.g. carbon: --q-min 10000 --q-max 13500).

Usage:
    python3 results.py <results_dir> [options]

Examples:
    python3 results.py results/ --bars 9 29 --tw-method fit_sat --q-min 10000 --q-max 13500
    python3 results.py results/ --tw-method fit --save-bars bars_fit.pkl
    python3 results.py --load-bars bars_fit.pkl --coincidence 9 29
    python3 results.py --load-bars bars_fit.pkl --bars 9 --plots tof-vs-eloss charge
"""

import os
import sys
import glob
import pickle
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from tw_calibration import apply_energy_calibration, apply_tof_calibration, add_coordinate
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

CFD_MAX_TW = 200.0   # ns — bars with a later end time are rejected
AMP_ST_MIN = 1100    # mV — minimum ST_sum amplitude for the TOF


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


# =============================================================================
# 1. LOADING AND BAR RECONSTRUCTION
# =============================================================================

def _results_files(folder, runs=None):
    if runs:
        files = [os.path.join(folder, f"results_{r}.pkl") for r in runs]
        return [f for f in files if os.path.exists(f)]
    return sorted(glob.glob(os.path.join(folder, "results_*.pkl")))


def load_results(folder, runs=None):
    """All the events of the results_*.pkl files in folder (only the given runs, if any)."""
    files = _results_files(folder, runs)
    if not files:
        print(f"No results_*.pkl file found in {folder}")
        sys.exit(1)

    print(f"\nresults files: {len(files)}")
    all_events = []
    for f in files:
        with open(f, "rb") as fh:
            evs = pickle.load(fh)
        print(f"  {os.path.basename(f)}: {len(evs)} events")
        all_events.extend(evs)

    print(f"Events loaded: {len(all_events)}\n")
    return all_events


def build_bars(events, tw_method="simple", st_method="simple", coincidence=None, single_hit=False):
    """
    One entry per reconstructed bar per event. A bar is reconstructed when
    both ends have a time (method tw_method) below CFD_MAX_TW; charge_bar is
    None if one of the two charges is missing or not positive.

    tw_method: "simple", "tangent", "bipolar", "fit" or "fit_sat"
    st_method: "simple", "tangent" or "bipolar"
    coincidence: (bar1, bar2) keeps only the events with both bars and adds
                 dt = time_bar1 - time_bar2
    single_hit: keeps only the events with at most one bar per layer
    """
    t_tw_key = f"t_{tw_method}"
    t_st_key = f"t_{st_method}"

    bars = []
    n_wf_excluded = 0
    n_bars_incomplete = 0
    n_bars_cfd = 0
    n_bars_both = 0

    for ev in events:
        st_sum = ev.get("ST_sum")
        if st_sum is None:          # no valid ST waveform in the event
            continue

        ev_n     = ev["eventN"]
        run_id   = ev["run_id"]
        file_idx = ev["file_idx"]
        t_st     = st_sum.get(t_st_key)

        bar_wfs = {}
        for wf in ev.get("TW", []):
            bid = wf.get("bar_id")
            if bid is None:
                continue
            bar_wfs.setdefault(bid, {"L": None, "R": None})
            if wf.get("end") in ("L", "R"):
                bar_wfs[bid][wf["end"]] = wf

        ev_bars = {}
        for bid, ends in bar_wfs.items():
            wf_L, wf_R = ends["L"], ends["R"]
            if wf_L is None or wf_R is None:
                continue
            n_bars_both += 1

            t_L = wf_L.get(t_tw_key)
            t_R = wf_R.get(t_tw_key)
            if t_L is None or t_R is None:
                n_bars_incomplete += 1
                continue
            if t_L > CFD_MAX_TW or t_R > CFD_MAX_TW:
                n_bars_cfd += 1
                n_wf_excluded += int(t_L > CFD_MAX_TW) + int(t_R > CFD_MAX_TW)
                continue

            ch_L = wf_L.get("charge")
            ch_R = wf_R.get("charge")
            if ch_L is None or ch_R is None or ch_L <= 0 or ch_R <= 0:
                charge_bar = None
            else:
                charge_bar = float(np.sqrt(ch_L * ch_R))

            time_bar = (t_L + t_R) / 2.0
            tof      = (time_bar - t_st) if t_st is not None else None

            # charge, time and chi2 from the fit of the saturated waveforms
            cfs_L = wf_L.get("charge_fit_sat")
            cfs_R = wf_R.get("charge_fit_sat")
            charge_fit_sat_bar = float(np.sqrt(cfs_L * cfs_R)) \
                if (cfs_L and cfs_R and cfs_L > 0 and cfs_R > 0) else None

            t_fit_sat_L = wf_L.get("t_fit_sat")
            t_fit_sat_R = wf_R.get("t_fit_sat")
            time_bar_fit_sat = (t_fit_sat_L + t_fit_sat_R) / 2.0 \
                if (t_fit_sat_L is not None and t_fit_sat_R is not None) else None

            chi2_L = wf_L.get("chi2_fit_sat")
            chi2_R = wf_R.get("chi2_fit_sat")
            chi2_bar = max(chi2_L, chi2_R) if (chi2_L is not None and chi2_R is not None) else None

            ev_bars[bid] = {
                "run_id":           run_id,
                "file_idx":         file_idx,
                "eventN":           ev_n,
                "bar_id":           bid,
                "layer":            wf_L.get("layer"),
                "amp_st_sum":       st_sum["amplitude"],
                "amp_L":            wf_L.get("amplitude"),
                "amp_R":            wf_R.get("amplitude"),
                "charge_L":         ch_L,
                "charge_R":         ch_R,
                "charge_bar":       charge_bar,
                "t_st":             t_st,
                "t_L":              t_L,
                "t_R":              t_R,
                "time_bar":         time_bar,
                "tof":              tof,
                "dt":               None,
                "charge_fit_sat":   charge_fit_sat_bar,
                "time_bar_fit_sat": time_bar_fit_sat,
                "chi2_fit_sat":     chi2_bar,
            }

        if single_hit:
            layer0 = [bid for bid in ev_bars if ev_bars[bid].get("layer") == 0]
            layer1 = [bid for bid in ev_bars if ev_bars[bid].get("layer") == 1]
            if len(layer0) > 1 or len(layer1) > 1:
                continue

        if coincidence is not None:
            b1, b2 = coincidence
            if b1 not in ev_bars or b2 not in ev_bars:
                continue
            dt = ev_bars[b1]["time_bar"] - ev_bars[b2]["time_bar"]
            ev_bars[b1]["dt"] = dt
            ev_bars[b2]["dt"] = dt

        bars.extend(ev_bars.values())

    n_bars_total = sum(len(set(wf["bar_id"] for wf in ev["TW"] if wf.get("bar_id") is not None)) for ev in events)
    print(f"\nTW waveforms rejected by CFD_MAX_TW: {n_wf_excluded}")
    print(f"Bars rejected by CFD_MAX_TW: {n_bars_cfd} ({100*n_bars_cfd/max(n_bars_total,1):.2f}%)")
    print(f"Bars without time (method {tw_method} failed): {n_bars_incomplete}/{n_bars_both} "
          f"({100*n_bars_incomplete/max(n_bars_both,1):.2f}%)")
    print(f"\nReconstructed bars: {len(bars)} (ST:{st_method}, TW:{tw_method})\n")

    return bars


def build_bars_streaming(folder, runs=None, tw_method="simple",
                         st_method="simple", coincidence=None, single_hit=False):
    """build_bars on one results file at a time, to keep the memory usage low."""
    files = _results_files(folder, runs)
    if not files:
        print(f"No results_*.pkl file found in {folder}")
        sys.exit(1)
    print(f"results files: {len(files)}")

    total_events = 0
    all_bars = []
    for f in files:
        print(f"\n  {os.path.basename(f)}...", end=" ", flush=True)
        with open(f, "rb") as fh:
            events = pickle.load(fh)
        total_events += len(events)
        all_bars.extend(build_bars(events, tw_method=tw_method, st_method=st_method,
                                   coincidence=coincidence, single_hit=single_hit))
        del events

    print(f"\nTotal events: {total_events}")
    print(f"Total bars: {len(all_bars)}")
    return all_bars


def filter_bars(bars, bar_ids=None, q_min=None, q_max=None, chi2_max=None, amp_st_min=None, z_select=None):
    """
    Selection on bar_id, Z (int or list), bar charge (mV*ns), chi2 of the
    saturated fit and ST_sum amplitude (mV). Bars without charge are rejected
    whenever a charge, chi2 or ST amplitude selection is applied.
    """
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
                  and (amp_st_min is None or (b.get("amp_st_sum") is not None and b["amp_st_sum"] * 1e3 >= amp_st_min))]
    return result


def apply_coincidence(bars, b1, b2):
    """Keep only the events with both bars b1 and b2; adds dt = tof_b1 - tof_b2 (on copies of the bars)."""
    ev_index = defaultdict(dict)
    for b in bars:
        ev_index[(b["run_id"], b["file_idx"], b["eventN"])][b["bar_id"]] = b

    result = []
    for key, bids in ev_index.items():
        if b1 not in bids or b2 not in bids:
            continue
        dt = bids[b1]["tof"] - bids[b2]["tof"]
        for bid in (b1, b2):
            entry = dict(bids[bid])
            entry["dt"] = dt
            result.append(entry)

    return result


def apply_charge_equalization(bars, path):
    """
    Charge equalisation along the bar from the file of charge_equalization.py
    (bar_id A dA B dB C dC): charge_bar -> charge_bar / (A*x^2 + B*x + C), with x
    the position along the bar. Bars without calibration, without position or
    with f(x) <= 0 are left unchanged.
    """
    calib = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            A, dA, B, dB, C, dC = map(float, parts[1:7])
            calib[int(parts[0])] = (A, B, C)

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
        b["charge_bar"] = b["charge_bar"] / f_x
        n_corrected += 1

    print(f"Equalised bars: {n_corrected}/{len(bars)}  (skipped: {n_skipped})")
    return bars


# =============================================================================
# 2. PLOTS
# =============================================================================

def plot_amp_st_sum(bars, amp_st_min=AMP_ST_MIN, bins=500, save=None):
    """ST_sum amplitude (one entry per bar) with the amp_st_min cut (mV)."""
    amp = np.array([b["amp_st_sum"] for b in bars if b.get("amp_st_sum") is not None]) * 1e3
    n_cut = int(np.sum(amp < amp_st_min))
    print(f"\nCut amp_st_sum >= {amp_st_min} mV: {n_cut}/{len(amp)} bars rejected "
          f"({100 * n_cut / max(len(amp), 1):.2f}%)")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(amp, bins=bins, range=(0, amp.max()), color="steelblue", alpha=0.8)
    ax.axvspan(0, amp_st_min, color="red", alpha=0.12, label="rejected")
    ax.axvline(amp_st_min, color="red", linestyle="--", lw=2, label=f"cut: {amp_st_min} mV")
    ax.set_yscale("log")
    ax.set_xlabel("ST_sum amplitude (mV)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_charge(bars, bar_id, fit_windows=None, compare_fit_sat=False, q_lines=(10500, 13500)):
    """
    Bar charge histogram (mV*ns), log scale.
        fit_windows      list of peak centres (mV*ns): gaussian fit in ±500 mV*ns
        compare_fit_sat  also draw the charge from the fit of the saturated waveforms
        q_lines          vertical lines (e.g. the carbon selection)
    """
    q = np.array([b["charge_bar"] for b in bars
                  if b["bar_id"] == bar_id and b["charge_bar"] is not None]) * 1e3
    if len(q) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    q_sat = np.array([]) if not compare_fit_sat else np.array(
        [b["charge_fit_sat"] for b in bars
         if b["bar_id"] == bar_id and b.get("charge_fit_sat") is not None]) * 1e3

    vmin = min(q.min(), q_sat.min()) if len(q_sat) else q.min()
    vmax = max(q.max(), q_sat.max()) if len(q_sat) else q.max()

    counts_all, edges, _ = ax.hist(q, bins=500, range=(vmin, 20000), color="darkorange",
                                   histtype="step", lw=2.5, label="measured")
    centers_all = (edges[:-1] + edges[1:]) / 2

    if compare_fit_sat and len(q_sat) > 0:
        ax.hist(q_sat, bins=500, range=(vmin, vmax), color="steelblue",
                histtype="step", lw=2.5, label="fit (saturated)")
        ax.legend()

    for i, qc in enumerate(fit_windows or []):
        qhw = 500
        mask = (centers_all >= qc - qhw) & (centers_all <= qc + qhw)
        if mask.sum() < 5:
            continue
        x_fit, y_fit = centers_all[mask], counts_all[mask]
        try:
            popt, _ = curve_fit(gauss, x_fit, y_fit, p0=[max(y_fit), qc, qhw/3], maxfev=10000)
            mu, sigma = popt[1], np.abs(popt[2])
            x_plot = np.linspace(mu - 2*sigma, mu + 2*sigma, 2000)
            ax.plot(x_plot, gauss(x_plot, *popt), color="black", lw=2)
            ax.axvline(mu - 2*sigma, color="black", lw=1.2, linestyle="--")
            ax.axvline(mu + 2*sigma, color="black", lw=1.2, linestyle="--")
            print(f"  Peak {i+1}: μ={mu:.0f} mV·ns, σ={sigma:.0f} mV·ns, σ/μ={sigma/mu*100:.1f}%, "
                  f"±2σ=[{mu-2*sigma:.0f},{mu+2*sigma:.0f}]")
        except Exception as e:
            print(f"  Fit of peak {i+1} failed: {e}")

    for q_cut in q_lines or []:
        ax.axvline(q_cut, color="black", lw=1.5, linestyle="--")

    ax.set_xlabel("Charge (mV·ns)")
    ax.set_ylabel("Counts")
    ax.set_xlim(right=15000)
    ax.set_title(f"Charge — Bar {bar_id} — {len(q)} entries")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_tof(bars, bar_id, label="", window_ns=None, pct_low=1, pct_high=99, bins=100,
             xlabel="TOF (ns)"):
    """
    TOF histogram of one bar with a gaussian fit within ±1/6 of the histogram
    range around the highest bin. Range: window_ns around the median, or the
    pct_low-pct_high percentiles. Returns {"mu", "sigma_ps", "dsigma_ps", "n"}.
    """
    tofs = np.array([b["tof"] for b in bars if b["bar_id"] == bar_id and b["tof"] is not None])
    if len(tofs) < 50:
        print(f"  Bar {bar_id}: too few entries ({len(tofs)}), skipped.")
        return None

    mu0_init = np.median(tofs)
    if window_ns is not None:
        rng = (mu0_init - window_ns/2, mu0_init + window_ns/2)
    else:
        rng = (np.percentile(tofs, pct_low), np.percentile(tofs, pct_high))

    counts, edges = np.histogram(tofs, bins=bins, range=rng)
    centers = (edges[:-1] + edges[1:]) / 2

    mu0    = centers[np.argmax(counts)]
    sigma0 = (rng[1] - rng[0]) / 6

    mask = (centers >= mu0 - sigma0) & (centers <= mu0 + sigma0)
    x_fit, y_fit = centers[mask], counts[mask]

    try:
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=[max(y_fit), mu0, sigma0], maxfev=10000)
        sigma_ps = abs(popt[2]) * 1e3
        dsigma   = np.sqrt(max(pcov[2,2], 0)) * 1e3
        print(f"\n  TOF: mu={popt[1]:.2f} ns,  σ={sigma_ps:.1f} ± {dsigma:.1f} ps  ({len(tofs)} entries)")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(tofs, bins=bins, range=rng, color="seagreen", alpha=0.7)
        x_plot = np.linspace(popt[1]-3*abs(popt[2]), popt[1]+3*abs(popt[2]), 1000)
        ax.plot(x_plot, gauss(x_plot, *popt), color="black", lw=2, label=f"σ = {sigma_ps:.1f} ps")
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("Counts", fontsize=18)
        ax.set_title(f"{label} — {len(tofs)} entries", fontsize=18)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
        return {"mu": popt[1], "sigma_ps": sigma_ps, "dsigma_ps": dsigma, "n": len(tofs)}
    except Exception as e:
        print(f"  Fit failed for bar {bar_id}: {e}")
        return None


def plot_tof_vs_charge(bars, bar_id=None, tof_range=None, show_bb_curves=True, max_points=100000, save=None):
    """
    E_loss vs TOF, one point per bar, coloured by the reconstructed Z, with the
    Bethe-Bloch curves used for the identification. At most max_points random points.
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

    tofs = np.array([b["tof"]     for b in sel])
    q    = np.array([b["energy"]  for b in sel])
    zs   = np.array([b["chargeZ"] for b in sel])

    if tof_range is None:
        tof_range = (np.percentile(tofs, 0.01), np.percentile(tofs, 99.9))

    Z_COLORS = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "orange", 5: "purple", 6: "gold"}

    fig, ax = plt.subplots(figsize=(12, 9))
    for z_val in sorted(set(zs)):
        mask = zs == z_val
        ax.scatter(tofs[mask], q[mask], color=Z_COLORS.get(z_val, "gray"),
                   s=6, alpha=0.8, label=f"Z={z_val}", rasterized=True)
    ax.legend(fontsize=16, markerscale=4, loc="upper right")

    if show_bb_curves:
        from particle_id import BB_PARAMS, TOF_MIN, TOF_MAX
        x_curve = np.linspace(TOF_MIN, TOF_MAX, 2000)
        for Z, p in BB_PARAMS.get(sel[0]["layer"], {}).items():
            ax.plot(x_curve, p["norm"] * Z**2 * (p["const"] * x_curve**2 - 1), color='black', lw=1.8, alpha=0.9)

    ax.set_xlim(tof_range)
    ax.set_ylim(np.percentile(q, 0.), np.percentile(q, 99.5))
    ax.set_ylabel(r"E$_{\mathrm{loss}}$ (MeV)", fontsize=18)
    ax.set_xlabel("TOF (ns)", fontsize=18)
    ax.set_title(f"TOF vs Energy — bar {bar_id} — {len(sel)} entries", fontsize=18)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=300)
    plt.show()


def plot_charge_vs_dt_LR(bars, bar_id, x_field="dt_LR", normalize_to_fit_min=True,
                         outlier_thresh=None, charge_resolution=0.035):
    """
    Bar charge vs position along the bar (x_field="coordinate", cm) or vs
    t_L - t_R (x_field="dt_LR", centred on its mean), with a parabolic fit on
    the single points. With outlier_thresh, points whose normalised residual
    from a first fit exceeds it are excluded from the final fit. The chi2 uses
    a relative charge resolution charge_resolution.
    """
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b["charge_bar"] is not None
           and b["t_L"] is not None
           and b["t_R"] is not None
           and (x_field != "coordinate" or b.get("coordinate") is not None)]
    if not sel:
        print(f"No data for bar {bar_id}")
        return

    q = np.array([b["charge_bar"] for b in sel]) * 1e3

    if x_field == "coordinate":
        x_vals = np.array([b["coordinate"] for b in sel])
        x_label = "position (cm)"
    else:
        dt_LR = np.array([b["t_L"] - b["t_R"] for b in sel])
        x_vals = dt_LR - np.mean(dt_LR)
        x_label = r"$t_L - t_R$ (ns, centred)"

    x_lo, x_hi = np.percentile(x_vals, 0.), np.percentile(x_vals, 100.)

    def parabola(x, A, B, C):
        return A * x**2 + B * x + C

    def _fit(xf, qf):
        """Parabolic fit; returns popt, perr and the value at the vertex (C if A is compatible with 0)."""
        popt, pcov = curve_fit(parabola, xf, qf, p0=[100., 0., 1800], maxfev=10000)
        perr = np.sqrt(pcov.diagonal())
        A, B, C = popt
        fmv = C - B**2 / (4*A) if abs(A) > 3 * perr[0] else C
        return popt, perr, fmv

    popt = None
    outlier_mask_plot = None
    try:
        popt1, perr1, fit_min_val1 = _fit(x_vals, q)
        popt, perr, fit_min_val = popt1, perr1, fit_min_val1

        if outlier_thresh is not None:
            resid_norm_fit = (q / fit_min_val1) - (parabola(x_vals, *popt1) / fit_min_val1)
            keep_out = np.abs(resid_norm_fit) <= outlier_thresh
            popt, perr, fit_min_val = _fit(x_vals[keep_out], q[keep_out])

            outlier_mask_plot = ~keep_out
            keep_range = (x_vals >= x_vals[keep_out].min()) & (x_vals <= x_vals[keep_out].max())
            x_vals, q = x_vals[keep_range], q[keep_range]
            outlier_mask_plot = outlier_mask_plot[keep_range]

        A, B, C = popt
        x_min = -B / (2*A) if abs(A) > 3 * perr[0] else float('nan')

        A_n, B_n, C_n = A / fit_min_val, B / fit_min_val, C / fit_min_val
        perr_n = perr / fit_min_val

        if outlier_mask_plot is not None:
            x_diag, q_diag = x_vals[~outlier_mask_plot], q[~outlier_mask_plot]
        else:
            x_diag, q_diag = x_vals, q

        q_norm_diag = q_diag / fit_min_val
        pred_norm   = parabola(x_diag, *popt) / fit_min_val
        pull = (q_norm_diag - pred_norm) / (charge_resolution * q_norm_diag)

        chi2_val = np.sum(pull**2)
        ndof = len(x_diag) - 3
        chi2_dof_str = f", χ²/ndof={chi2_val:.0f}/{ndof} = {chi2_val/ndof:.2f}" if ndof > 0 else ""

        print(f"  best fit (normalised): A_n*1e3={A_n*1e3}±{perr_n[0]*1e3}, "
              f"B_n*1e3={B_n*1e3}±{perr_n[1]*1e3}, C_n={C_n}±{perr_n[2]}{chi2_dof_str}")
        print(f"  Vertex: x_min={x_min:.3f}")
        if outlier_mask_plot is not None:
            print(f"  Outliers (|q_norm - fit1| > {outlier_thresh}): "
                  f"{outlier_mask_plot.sum()}/{len(outlier_mask_plot)} "
                  f"({100*outlier_mask_plot.sum()/len(outlier_mask_plot):.2f}%)")
    except Exception as e:
        print(f"   Fit failed: {e}")
        fit_min_val = 1.0

    norm_factor = fit_min_val if normalize_to_fit_min else 1.0
    q_plot = q / norm_factor
    y_label = r"$Q/Q_{\mathrm{min}}$" if normalize_to_fit_min else "Q (mV·ns)"

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    if outlier_mask_plot is not None:
        ax.scatter(x_vals[~outlier_mask_plot], q_plot[~outlier_mask_plot], s=6, alpha=0.6, color="steelblue", label="fitted points")
        ax.scatter(x_vals[outlier_mask_plot],  q_plot[outlier_mask_plot],  s=6, alpha=0.6, color="red", label="outliers")
        ax.set_ylim(q_plot[~outlier_mask_plot].min(), q_plot[~outlier_mask_plot].max())
    else:
        ax.scatter(x_vals, q_plot, s=6, alpha=0.6, color="steelblue")
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(f"Charge vs {x_field} — bar {bar_id} — entries {len(q)}")
    ax.grid(True, alpha=0.3)

    if popt is not None:
        x_fit = np.linspace(x_lo, x_hi, 2000)
        ax.plot(x_fit, parabola(x_fit, *popt) / norm_factor, color='black', lw=2)
        if outlier_mask_plot is not None:
            ax.legend(fontsize=16)

        residuals_plot = q_plot - parabola(x_vals, *popt) / norm_factor
        if outlier_mask_plot is not None:
            ax_res.scatter(x_vals[~outlier_mask_plot], residuals_plot[~outlier_mask_plot], s=6, alpha=0.6, color="steelblue")
        else:
            ax_res.scatter(x_vals, residuals_plot, s=6, alpha=0.6, color="black")
        ax_res.axhline(0, color="black", lw=1.5, linestyle="--")
        ax_res.set_ylabel("residual")
        ax_res.grid(True, alpha=0.3)
    ax_res.set_xlabel(x_label)

    plt.tight_layout()
    plt.show()


def plot_charge_vs_position(bars, bar_id, save=None):
    """Bar charge vs position along the bar (centred on the median), points coloured by local density."""
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b["charge_bar"] is not None and b["charge_bar"] > 0
           and b.get("coordinate") is not None]
    if not sel:
        print(f"No data for bar {bar_id}")
        return

    x = np.array([b["coordinate"] for b in sel])
    x = x - np.median(x)
    q = np.array([b["charge_bar"] for b in sel]) * 1e3

    fig, ax = plt.subplots(figsize=(16, 8))
    H, xe, ye = np.histogram2d(x, q, bins=(300, 300),
                               range=[[-30, 30], [0, np.percentile(q, 99.9)]])
    ix = np.clip(np.digitize(x, xe) - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.digitize(q, ye) - 1, 0, H.shape[1] - 1)
    dens = np.maximum(H[ix, iy], 1)
    order = np.argsort(dens)          # densest points drawn on top
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
    if save:
        fig.savefig(save)
    plt.show()


def plot_charge_fit_vs_orig(bars, bar_id):
    """
    Bar charge from the fit of the saturated waveforms vs the measured charge:
    2D histogram (left) and relative difference with a gaussian fit around the
    peak (right).
    """
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b.get("charge_bar") is not None
           and b.get("charge_fit_sat") is not None]
    if len(sel) < 10:
        print(f"  Bar {bar_id}: too few entries with charge_fit_sat ({len(sel)}), skipped.")
        return

    q_orig = np.array([b["charge_bar"]     for b in sel]) * 1e3
    q_fit  = np.array([b["charge_fit_sat"] for b in sel]) * 1e3
    diffs  = (q_fit - q_orig) / q_orig * 100

    print(f"  {np.sum(diffs > 10.0)} entries with a difference above 10%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(fr"$Q_{{\mathrm{{fit}}}}$ vs $Q_{{\mathrm{{orig}}}}$ — Bar {bar_id} — entries {len(diffs)}", fontsize=18)

    lim = [min(q_orig.min(), q_fit.min()) * 0.95, max(q_orig.max(), q_fit.max()) * 1.05]
    if len(sel) > 5000:
        hb = axes[0].hexbin(q_orig, q_fit, gridsize=200, cmap="turbo", norm=LogNorm(),
                            extent=[lim[0], lim[1], lim[0], lim[1]], mincnt=1, rasterized=True)
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
    axes[1].hist(diffs, bins=edges, range=rng, histtype="step", color="darkcyan", lw=1.8, zorder=3)
    axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
    try:
        mu0    = centers[np.argmax(counts)]
        sigma0 = (rng[1] - rng[0]) / 6
        mask = (centers >= mu0 - sigma0) & (centers <= mu0 + sigma0)
        popt, pcov = curve_fit(gauss, centers[mask], counts[mask],
                               p0=[np.max(counts), mu0, sigma0], maxfev=10000)
        perr  = np.sqrt(np.diag(pcov))
        x_fit = np.linspace(popt[1] - 3*abs(popt[2]), popt[1] + 3*abs(popt[2]), 1000)
        axes[1].plot(x_fit, gauss(x_fit, *popt), color="black", lw=2, linestyle="--", alpha=0.8, zorder=2,
                     label=f"μ={popt[1]:.2f}%, σ={abs(popt[2]):.2f}%")
        axes[1].legend(loc="upper right")
        print(f"  charge difference: N={len(sel)}, μ={popt[1]:.3f}±{perr[1]:.3f} %  "
              f"σ={abs(popt[2]):.3f}±{perr[2]:.3f}%")
    except Exception as e:
        print(f"    Gaussian fit failed: {e}")
    axes[1].set_xlabel(r"(Q$_{\mathrm{fit}}$ - Q$_{\mathrm{orig}}$) / Q$_{\mathrm{orig}}$ (%)")
    axes[1].set_ylabel("Counts")
    axes[1].set_title("Relative charge difference")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_chi2(bars, bar_id):
    """chi2 of the saturated fit: distribution and 2D histogram vs the relative charge difference."""
    sel = [b for b in bars
           if b["bar_id"] == bar_id
           and b.get("charge_bar") is not None
           and b.get("charge_fit_sat") is not None
           and b.get("chi2_fit_sat") is not None]
    if len(sel) < 10:
        print(f"  Bar {bar_id}: too few entries with chi2_fit_sat ({len(sel)}), skipped.")
        return

    q_orig = np.array([b["charge_bar"]     for b in sel]) * 1e3
    q_fit  = np.array([b["charge_fit_sat"] for b in sel]) * 1e3
    diffs  = (q_fit - q_orig) / q_orig * 100
    chi2   = np.array([b["chi2_fit_sat"]   for b in sel], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    fig.suptitle(f"chi2 — bar {bar_id} — entries {len(sel)}", fontsize=18)

    rng_chi2 = (0, np.percentile(chi2, 99.99))
    axes[0].hist(chi2, bins=300, range=rng_chi2, color="red", alpha=0.8)
    axes[0].set_xlabel(r"$\chi^2$", fontsize=18)
    axes[0].set_ylabel("Counts", fontsize=18)
    axes[0].set_yscale("log")
    axes[0].set_title(r"$\chi^2$ distribution", fontsize=18)
    axes[0].grid(True, alpha=0.4)

    rng_diffs = (0, np.percentile(diffs, 99.99))
    axes[1].hist2d(abs(diffs), chi2, range=(rng_diffs, rng_chi2), bins=200, cmap="turbo", norm=LogNorm())
    axes[1].set_xlabel(r"|$(Q_{fit} - Q_{orig}) / Q_{orig}$| (%)", fontsize=18)
    axes[1].set_ylabel(r"$\chi^2$", fontsize=18)
    axes[1].grid(True, alpha=0.4)

    plt.show()


def plot_hit_times(bars, bar_id, label="", bins=150):
    """Histograms of t_L, t_R and t_R - t_L of one bar."""
    sel = [b for b in bars if b["bar_id"] == bar_id and b["t_L"] is not None and b["t_R"] is not None]
    if not sel:
        print(f"No data for bar {bar_id}")
        return

    t_L = np.array([b["t_L"] for b in sel])
    t_R = np.array([b["t_R"] for b in sel])
    dt = t_R - t_L

    print(f"\n  Bar {bar_id} ({label}): entries={len(sel)}")
    print(f"  t_L range: [{t_L.min():.3f}, {t_L.max():.3f}] ns")
    print(f"  t_R range: [{t_R.min():.3f}, {t_R.max():.3f}] ns")
    print(f"  t_R-t_L range: [{dt.min():.3f}, {dt.max():.3f}] ns")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, data, name, color in ((axes[0], t_L, "t_L", "steelblue"),
                                  (axes[1], t_R, "t_R", "firebrick"),
                                  (axes[2], dt, "t_R - t_L", "seagreen")):
        ax.hist(data, bins=bins, color=color, alpha=0.8)
        ax.set_xlabel(f"{name} (ns)", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_title(f"{name} — bar {bar_id} ({label})", fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

PLOT_CHOICES = ["st-amp", "charge", "tof", "tof-vs-eloss", "charge-vs-position", "charge-fit", "chi2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bar reconstruction from results_<run>.pkl and final plots")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Folder with the results_*.pkl files (not needed with --load-bars/--load-hits)")
    parser.add_argument("--runs", nargs="+", type=int, default=None, help="Only these runs")
    parser.add_argument("--bars", nargs="+", type=int, default=None, help="Only these bars (default: all)")
    parser.add_argument("--tw-method", default="simple", choices=["simple", "tangent", "bipolar", "fit", "fit_sat"],
                        help="TW time: CFD method or template fit (fit_sat: fit of the saturated waveforms)")
    parser.add_argument("--st-method", default="simple", choices=["simple", "tangent", "bipolar"],
                        help="CFD method for the ST sum")
    parser.add_argument("--q-min", type=float, default=None, help="Minimum bar charge (mV*ns)")
    parser.add_argument("--q-max", type=float, default=None, help="Maximum bar charge (mV*ns)")
    parser.add_argument("--z", nargs="+", type=int, default=None, help="Only bars with these reconstructed Z")
    parser.add_argument("--chi2-max", type=float, default=None, help="Maximum chi2 of the saturated fit (bar = max of the two ends)")
    parser.add_argument("--single-hit", action="store_true", help="Only events with at most one bar per layer")
    parser.add_argument("--coincidence", nargs=2, type=int, default=None, metavar=("BAR1", "BAR2"),
                        help="Events with both bars: charges and time difference BAR1 - BAR2")
    parser.add_argument("--charge-eq", default=None, help="Charge equalisation file from charge_equalization.py")
    parser.add_argument("--save-bars", default=None, help="Save the reconstructed (calibrated) bars to this .pkl")
    parser.add_argument("--load-bars", default=None, help="Load the bars from a .pkl saved with --save-bars")
    parser.add_argument("--load-hits", default=None, help="Use the SHOE hits of extract_shoe.py --hits instead of the bars")
    parser.add_argument("--plot-times", type=int, nargs="+", default=None,
                        help="Only plot t_L, t_R and t_R-t_L of these bars and exit")
    parser.add_argument("--plots", nargs="+", default=["tof"], choices=PLOT_CHOICES,
                        help="Plots for every bar (default: tof)")
    args = parser.parse_args()

    if args.load_hits and args.load_bars:
        parser.error("--load-hits and --load-bars are alternative")

    coincidence = tuple(args.coincidence) if args.coincidence else None

    if args.load_hits or args.load_bars:
        with open(args.load_hits or args.load_bars, "rb") as f:
            payload = pickle.load(f)
        if args.load_hits:
            bars = payload["hits"]          # tof, energy and Z already from SHOE
            print(f"\nHits loaded: {len(bars)}  (runs={payload.get('runs', '?')})")
        else:
            bars = payload["bars"]
            args.tw_method = payload.get("tw_method", args.tw_method)
            args.st_method = payload.get("st_method", args.st_method)
            print(f"\nBars loaded: {len(bars)}  (TW:{args.tw_method}, ST:{args.st_method}, "
                  f"runs={payload.get('runs', '?')})")

        if args.plot_times:
            for bar_id in args.plot_times:
                plot_hit_times(bars, bar_id=bar_id, label="rear" if bar_id < 20 else "front")
            sys.exit(0)

        if args.single_hit:
            ev_index = defaultdict(list)
            for b in bars:
                ev_index[(b["run_id"], b["file_idx"], b["eventN"])].append(b["bar_id"])
            ok_events = {k for k, bids in ev_index.items()
                         if sum(1 for bid in bids if bid < 20) <= 1
                         and sum(1 for bid in bids if bid >= 20) <= 1}
            bars = [b for b in bars if (b["run_id"], b["file_idx"], b["eventN"]) in ok_events]
            print(f"After single-hit: {len(bars)} bars")

        coincidence = coincidence or payload.get("coincidence")
        if coincidence is not None:
            bars = apply_coincidence(bars, coincidence[0], coincidence[1])

    else:
        if args.folder is None:
            parser.error("the results folder is needed without --load-hits or --load-bars")
        bars = build_bars_streaming(args.folder, runs=args.runs, tw_method=args.tw_method,
                                    st_method=args.st_method, coincidence=coincidence,
                                    single_hit=args.single_hit)
        bars = apply_energy_calibration(bars)
        bars = apply_tof_calibration(bars)
        bars = apply_particle_id(bars, n_workers=4)

        if args.save_bars:
            payload = {"bars": bars, "tw_method": args.tw_method, "st_method": args.st_method,
                       "coincidence": coincidence, "runs": args.runs}
            with open(args.save_bars, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Bars saved to: {args.save_bars}")

    if args.bars is None:
        args.bars = sorted({b["bar_id"] for b in bars})

    bars = add_coordinate(bars)
    if args.charge_eq:
        bars = apply_charge_equalization(bars, args.charge_eq)

    label = f"(TW:{args.tw_method}, ST:{args.st_method})"
    sel_kw = dict(z_select=args.z, q_min=args.q_min, q_max=args.q_max,
                  chi2_max=args.chi2_max, amp_st_min=AMP_ST_MIN)

    if "st-amp" in args.plots:
        plot_amp_st_sum(bars)

    if coincidence is None:
        for bid in args.bars:
            print(f"\n{'─'*55}\nBar {bid}")
            bars_bar = [b for b in bars if b["bar_id"] == bid]
            bars_q = filter_bars(bars_bar, **sel_kw)
            print(f"  Entries: {len(bars_bar)}, after the selection: {len(bars_q)}")

            if "charge" in args.plots:
                plot_charge(bars_bar, bid)
            if "tof" in args.plots:
                plot_tof(bars_q, bid, label=f"Bar {bid} {label}")
            if "tof-vs-eloss" in args.plots:
                plot_tof_vs_charge(bars_q, bid)
            if "charge-vs-position" in args.plots:
                plot_charge_vs_dt_LR(bars_q, bar_id=bid, x_field="coordinate")
                plot_charge_vs_position(bars_q, bar_id=bid)
            if "charge-fit" in args.plots:
                plot_charge_fit_vs_orig(bars_q, bar_id=bid)
            if "chi2" in args.plots:
                plot_chi2(bars_q, bar_id=bid)

    else:
        b1, b2 = coincidence
        print(f"\n{'─'*55}\nCoincidence {b1} & {b2}")

        bars_q1 = filter_bars([b for b in bars if b["bar_id"] == b1], **sel_kw)
        bars_q2 = filter_bars([b for b in bars if b["bar_id"] == b2], **sel_kw)

        # events in which both bars pass the selection
        key = lambda b: (b["run_id"], b["file_idx"], b["eventN"])
        common = {key(b) for b in bars_q1} & {key(b) for b in bars_q2}
        bars_q1 = [b for b in bars_q1 if key(b) in common]
        bars_q2 = [b for b in bars_q2 if key(b) in common]
        print(f"  Events after the selection: {len(common)}")

        plot_charge(bars_q1, bar_id=b1)
        plot_charge(bars_q2, bar_id=b2)

        bars_dt = [{"bar_id": -1, "tof": b["dt"]} for b in bars_q1 if b["dt"] is not None]
        plot_tof(bars_dt, bar_id=-1, label=f"t_bar {b1} - t_bar {b2}", window_ns=20.,
                 pct_low=1., pct_high=99., bins='fd', xlabel=f"t$_{{{b1}}}$ - t$_{{{b2}}}$ (ns)")
