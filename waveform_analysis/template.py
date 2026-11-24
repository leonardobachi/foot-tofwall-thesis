"""
template.py — Template construction and template fit of the waveforms.

Template construction:
    1. collect_tw / collect_st_sum / collect_st_single   select the waveforms
    2. resample_wfs      cubic resampling on a fixed 50 ps grid (0-330 ns)
    3. normalize_wfs     normalisation to the charge (TW) or to the amplitude (ST)
    4. align_wfs         shift so that the CFD time is at t0_ref
    5. build_model       weighted average sample by sample -> template and spread
    (build_model_iterative runs 4-5 and can re-align with the fitted times)

Template fit (model_fit_wf, apply_model_fit): each resampled waveform is fitted
with A * template(t - (t_arr - t0_ref)). For a given t_arr the amplitude A is
computed analytically, so the chi2 is minimised in t_arr only (bounded Brent
search in ±bracket_half around the CFD time). The chi2 uses sigma_noise of the
waveform and is restricted to a window around the leading edge and the tail.
Saturated samples (|amp| >= sat_level) are excluded from the chi2, which allows
the fit of artificially saturated waveforms (apply_saturation).

Usage (template of one bar built on one file, fit and TOF comparison):
    python3 template.py <file.bin> [--bar 9] [--sat 200]
"""

import argparse
from collections import defaultdict

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
# PARAMETERS
# =============================================================================

RESAMPLE_STEP = 0.050   # ns — resampling step (50 ps)
T0_REF_TW     = 120.0   # ns — reference time of the TW template
T0_REF_ST     = 105.0   # ns — reference time of the ST_sum template
CFD_TIME_MIN  = 50.0    # ns — accepted range of the CFD time
CFD_TIME_MAX  = 200.0   # ns   (rejects clock spikes and late signals)
T_GRID_START  = 0.0     # ns — fixed resampling grid
T_GRID_END    = 330.0   # ns

# sigma in the chi2: "noise" = sigma_noise of the waveform (baseline std),
# anything else = spread of the template sample by sample
CHI2_SIGMA    = "noise"


# =============================================================================
# 1. WAVEFORM SELECTION
# =============================================================================

def collect_st_single(events, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """Single ST waveforms with CFD time in (cfd_min, cfd_max); adds wf["_eventN"]."""
    wfs = []
    n_bad_cfd = 0

    for ev in events:
        for wf in ev["ST"]:
            t_cfd = wf.get("time_cfd")
            if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
                n_bad_cfd += 1
                continue
            wf["_eventN"] = ev["eventN"]
            wfs.append(wf)

    print(f"\n--- collect_st_single ---")
    print(f"Single ST waveforms:          {len(wfs)}")
    print(f"Rejected (CFD out of range):  {n_bad_cfd}")

    return wfs


def collect_st_sum(events, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """ST_sum waveforms with CFD time in (cfd_min, cfd_max); adds st_sum["_eventN"]."""
    wfs = []
    n_no_sum  = 0
    n_bad_cfd = 0

    for ev in events:
        st_sum = ev.get("ST_sum")
        if st_sum is None:
            n_no_sum += 1
            continue

        t_cfd = st_sum.get("time_cfd")
        if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
            n_bad_cfd += 1
            continue

        st_sum["_eventN"] = ev["eventN"]
        wfs.append(st_sum)

    print(f"\n--- collect_st_sum ---")
    print(f"ST_sum waveforms:             {len(wfs)}")
    print(f"Rejected (no sum):            {n_no_sum}")
    print(f"Rejected (CFD out of range):  {n_bad_cfd}")

    return wfs


def collect_tw(events, bar_ids=None, cfd_min=CFD_TIME_MIN, cfd_max=CFD_TIME_MAX):
    """
    TW waveforms with signal and CFD time in (cfd_min, cfd_max), optionally
    only for the bars in bar_ids. Returns references to the waveforms in events
    (index i of every later array corresponds to wfs[i]); adds wf["_eventN"].
    """
    wfs = []
    n_no_signal = 0
    n_bad_cfd   = 0

    for ev in events:
        for wf in ev["TW"]:
            if not wf.get("signal"):
                n_no_signal += 1
                continue
            if bar_ids is not None and wf.get("bar_id") not in bar_ids:
                continue

            t_cfd = wf.get("time_cfd")
            if t_cfd is None or not (cfd_min < t_cfd < cfd_max):
                n_bad_cfd += 1
                continue

            wf["_eventN"] = ev["eventN"]
            wfs.append(wf)

    print(f"\n--- collect_tw ---")
    if bar_ids:
        print(f"Selected bars:                {sorted(bar_ids)}")
    print(f"TW waveforms:                 {len(wfs)}")
    print(f"Rejected (no signal):         {n_no_signal}")
    print(f"Rejected (CFD out of range):  {n_bad_cfd}")

    return wfs


# =============================================================================
# 2. RESAMPLING
# =============================================================================

def resample_wfs(wfs, step=RESAMPLE_STEP):
    """
    Cubic resampling of all the waveforms on the fixed grid
    [T_GRID_START, T_GRID_END) with the given step (ns).
    Returns t_grid (1D) and amp_resampled (n_wfs x n_grid, float32).
    """
    t_grid = np.arange(T_GRID_START, T_GRID_END, step)

    print(f"\n--- resample_wfs ---")
    print(f"Grid: {t_grid[0]:.2f} -> {t_grid[-1]:.2f} ns  "
          f"({len(t_grid)} samples, step={step*1e3:.0f} ps)")
    print(f"Estimated memory: {len(wfs) * len(t_grid) * 8 / 1e9:.2f} GB")

    amp_resampled = np.zeros((len(wfs), len(t_grid)), dtype=np.float32)

    for i, wf in enumerate(wfs):
        interp = interp1d(wf["time"], wf["amp"], kind="cubic", bounds_error=False, fill_value=0.0)
        amp_resampled[i] = interp(t_grid)

    return t_grid, amp_resampled


# =============================================================================
# 3. NORMALISATION
# =============================================================================

def normalize_wfs(amp_resampled, wfs, method="charge"):
    """
    Normalise every resampled waveform.
        method="amplitude"  to the peak amplitude
        method="charge"     to the charge wf["charge"] (fallback: amplitude),
                            then a global factor brings the median minimum to -1
    Returns amp_norm (same shape) and the normalisation factors.
    """
    amplitudes = np.zeros(len(wfs))
    n_fallback = 0

    for i, wf in enumerate(wfs):
        if method == "charge":
            Q = wf.get("charge")
            w_norm = Q if (Q is not None and Q > 0) else None
        else:
            w_norm = None

        if w_norm is None:
            if method == "charge":
                n_fallback += 1
            A = wf.get("amplitude")
            w_norm = A if (A is not None and A > 0) else float(np.abs(np.min(amp_resampled[i])))
            w_norm = w_norm if w_norm > 0 else 1.0

        amplitudes[i] = w_norm

    if method == "charge" and n_fallback > 0:
        print(f"  Normalised to the amplitude (no valid charge): {n_fallback} waveforms "
              f"({100*n_fallback/len(wfs):.1f}%)")

    amp_norm = amp_resampled / amplitudes[:, np.newaxis]

    if method == "charge":
        # global scale: median minimum at -1, relative normalisation unchanged
        minima       = np.abs(np.min(amp_norm, axis=1))
        global_scale = np.median(minima)
        if global_scale > 0:
            amp_norm = amp_norm / global_scale

    print(f"\n--- normalize_wfs (method={method}) ---")
    print(f"Mean scale: {np.mean(amplitudes):.4f}  std: {np.std(amplitudes):.4f}")
    print(f"Mean minimum after normalisation: {np.mean(np.min(amp_norm, axis=1)):.3f}  (expected -1.0)")

    return amp_norm, amplitudes


# =============================================================================
# 4. TIME ALIGNMENT
# =============================================================================

def align_wfs(amp_norm, wfs, t_grid, t0_ref, max_shift=None):
    """
    Shift every normalised waveform by dt = t0_ref - time_cfd, so that all the
    CFD times are at t0_ref. The shift is continuous: the waveform is evaluated
    (cubic interpolation) on the grid t_grid - dt. Waveforms with |dt| > max_shift
    are set to zero. Returns amp_aligned and the shifts (ns).
    """
    shifts_ns   = np.zeros(len(wfs))
    amp_aligned = np.zeros_like(amp_norm)
    n_excluded  = 0

    for i, wf in enumerate(wfs):
        t_cfd = wf.get("time_cfd")
        dt = t0_ref - t_cfd
        shifts_ns[i] = dt

        if max_shift is not None and abs(dt) > max_shift:
            amp_aligned[i] = 0.0
            n_excluded += 1
            continue

        interp_wf      = interp1d(t_grid, amp_norm[i], kind="cubic", bounds_error=False, fill_value=0.0)
        amp_aligned[i] = interp_wf(t_grid - dt)

    print(f"\n--- align_wfs ---")
    if max_shift is not None:
        mask_ok = np.abs(shifts_ns) <= max_shift
    else:
        mask_ok = np.ones(len(shifts_ns), dtype=bool)
    shifts_ok = shifts_ns[mask_ok]
    print(f"Shifts — mean: {np.mean(shifts_ok):.2f} ns  "
          f"std: {np.std(shifts_ok):.2f} ns  "
          f"range: [{np.min(shifts_ok):.2f}, {np.max(shifts_ok):.2f}] ns  "
          f"(all: [{np.min(shifts_ns):.2f}, {np.max(shifts_ns):.2f}] ns)")
    if max_shift is not None:
        print(f"Waveforms with |shift| > {max_shift} ns excluded: {n_excluded} "
              f"({100*n_excluded/len(wfs):.1f}%)")

    return amp_aligned, shifts_ns


# =============================================================================
# 5. TEMPLATE
# =============================================================================

def build_model(amp_aligned, wfs, method="weighted"):
    """
    Template as the sample-by-sample average of the aligned waveforms.
        "weighted"  weights (Q / sigma_noise)^2 (Q = charge, or amplitude)
        "mean"      plain mean
        "median"    median
    Returns the template and the spread (std) of the waveforms sample by sample.
    """
    if method == "weighted":
        weights = np.array([(wf.get("charge") or wf.get("amplitude") or 1.0)**2 / (wf.get("noise") or 1.0)**2
                            for wf in wfs])
        weights /= weights.sum()
        model = np.average(amp_aligned, axis=0, weights=weights)

    elif method == "mean":
        model = np.mean(amp_aligned, axis=0)

    else:
        model = np.median(amp_aligned, axis=0)

    std = np.std(amp_aligned, axis=0)

    # mean relative spread in the signal region
    signal_th = 0.2
    signal_mask = np.abs(model) > signal_th
    if signal_mask.any():
        std_rel = np.mean(std[signal_mask] / np.abs(model[signal_mask]))
    else:
        std_rel = np.nan

    print(f"\n--- build_model ({method}) ---")
    print(f"Template samples: {len(model)}")
    print(f"Template minimum: {np.min(model):.4f}  (expected ≈ -1.0)")
    print(f"Mean spread in the signal region (>{signal_th}): {np.mean(std[signal_mask]):.3f}  "
          f"(relative: {std_rel*100:.1f} %)")

    return model, std


def build_model_iterative(amp_resampled, amp_norm, wfs, t_grid,
                          n_iter=0, method="weighted", max_wfs=None, t0_ref=T0_REF_TW, max_shift=None,
                          w_rise=20.0, w_tail=1.0, rise_left=5.0, rise_right=5.0,
                          tail=150.0, bracket_half=5.0, optimizer="scalar"):
    """
    Template built from the waveforms aligned with the CFD time; with n_iter > 0
    the waveforms are fitted, re-aligned with the fitted time t_arr and the
    template is rebuilt, until it changes by less than 1e-5.
    Returns template, spread and aligned waveforms.
    """
    amp_aligned, shifts = align_wfs(amp_norm, wfs, t_grid, t0_ref=t0_ref, max_shift=max_shift)
    model, std = build_model(amp_aligned, wfs, method=method)

    for it in range(n_iter):
        print(f"\n=== Iteration {it+1}/{n_iter} ===")
        results_tmp = apply_model_fit(amp_resampled, wfs, t_grid, model, std,
                                      max_wfs=max_wfs,
                                      w_rise=w_rise, w_tail=w_tail,
                                      rise_left=rise_left, rise_right=rise_right,
                                      tail=tail, bracket_half=bracket_half,
                                      t0_ref=t0_ref, optimizer=optimizer)

        for i, (res, wf) in enumerate(zip(results_tmp, wfs)):
            wf["_t_align"] = res["t_arr"] if res["success"] else wf.get("time_cfd")

        amp_aligned_new = np.zeros_like(amp_norm)
        for i, wf in enumerate(wfs):
            dt        = t0_ref - wf.get("_t_align")
            interp_wf = interp1d(t_grid, amp_norm[i], kind="cubic",
                                 bounds_error=False, fill_value=0.0)
            amp_aligned_new[i] = interp_wf(t_grid - dt)

        model_new, std_new = build_model(amp_aligned_new, wfs, method=method)
        diff = np.mean(np.abs(model_new - model))
        print(f"Template change: {diff:.6f}")
        model, std, amp_aligned = model_new, std_new, amp_aligned_new
        if diff < 1e-5:
            print(f"Converged at iteration {it+1}")
            break

    return model, std, amp_aligned


# =============================================================================
# TEMPLATE PLOTS
# =============================================================================

def plot_model(t_grid, model, std, t0_ref=T0_REF_TW, show_std=False,
               xlim=None, save=None):
    """Template with the ±1 sigma band; show_std=True adds a panel with the spread sample by sample."""
    if show_std:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

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
    if save:
        fig.savefig(save)
    plt.show()


def plot_model_compare(t_grid, model, std, amp_aligned, wfs,
                       n_wfs=300, end_filter=None,
                       q_min=None, q_max=None):
    """
    Template over a random sample of aligned waveforms.
        end_filter    None, "L" or "R"
        q_min, q_max  selection on the single-waveform charge (mV*ns)
    """
    idx_pool = list(range(len(amp_aligned)))

    if end_filter is not None:
        idx_pool = [i for i in idx_pool if wfs[i].get("end") == end_filter]
    if q_min is not None:
        idx_pool = [i for i in idx_pool
                    if wfs[i].get("charge") is not None and wfs[i]["charge"] * 1e3 >= q_min]
    if q_max is not None:
        idx_pool = [i for i in idx_pool
                    if wfs[i].get("charge") is not None and wfs[i]["charge"] * 1e3 <= q_max]

    if not idx_pool:
        print(f"No waveform passes the selection (end={end_filter}, q_min={q_min}, q_max={q_max}).")
        return

    idx_sample = np.random.choice(idx_pool, size=min(n_wfs, len(idx_pool)), replace=False)

    charges_sel = [wfs[i]["charge"] * 1e3 for i in idx_sample if wfs[i].get("charge") is not None]
    amps_sel    = [wfs[i]["amplitude"] * 1e3 for i in idx_sample if wfs[i].get("amplitude") is not None]
    print(f"\nplot_model_compare — {len(idx_sample)} waveforms")
    if charges_sel:
        print(f"  Charge: mean={np.mean(charges_sel):.0f}, "
              f"min={np.min(charges_sel):.0f}, max={np.max(charges_sel):.0f} mV·ns")
    if amps_sel:
        print(f"  Amplitude: mean={np.mean(amps_sel):.1f}, "
              f"min={np.min(amps_sel):.1f}, max={np.max(amps_sel):.1f} mV")

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, i in enumerate(idx_sample):
        ax.plot(t_grid, amp_aligned[i], color="steelblue", lw=0.8, alpha=0.5,
                rasterized=True, label="Waveforms" if j == 0 else "_nolegend_")
    ax.fill_between(t_grid, model - std, model + std, alpha=0.3, color="crimson")
    ax.plot(t_grid, model, color="crimson", lw=2.5, label="Template")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ARTIFICIAL SATURATION
# =============================================================================

def apply_saturation(amp_resampled, sat_level_mv=250.0):
    """
    Clip the resampled waveforms (V) at ±sat_level_mv (mV).
    Returns the clipped array, the number of waveforms with at least one clipped
    sample and the fraction of clipped samples of each waveform.
    """
    sat_level_v   = sat_level_mv / 1e3
    amp_saturated = np.clip(amp_resampled, -sat_level_v, sat_level_v)

    n_clipped_per_wf = np.sum(np.abs(amp_resampled) > sat_level_v, axis=1)
    n_saturated      = int(np.sum(n_clipped_per_wf > 0))
    frac_clipped     = n_clipped_per_wf / amp_resampled.shape[1]

    print(f"\n--- apply_saturation ---")
    print(f"Level: {sat_level_mv:.0f} mV")
    print(f"Waveforms with at least one saturated sample: {n_saturated}/{len(amp_resampled)} "
          f"({100*n_saturated/len(amp_resampled):.1f}%)")
    if n_saturated > 0:
        frac_ok = frac_clipped[frac_clipped > 0]
        print(f"Saturated samples per waveform (saturated waveforms only): "
              f"mean={np.mean(frac_ok)*100:.1f}%, max={np.max(frac_ok)*100:.1f}%")

    return amp_saturated, n_saturated, frac_clipped


def compute_tof_cfd_sat(events, sat_level_mv, q_min=None, q_max=None, bars=None):
    """
    TOF with the simple CFD applied to the TW waveforms clipped at sat_level_mv
    (the ST time is unchanged). q_min, q_max: bar charge selection (mV*ns);
    bars: list of bar_id (None = all).
    Returns tof_list and tof_by_bar {bar_id: [tof, ...]}.
    """
    from cfd_wf import _compute_time_simple

    sat_level_v = sat_level_mv / 1e3

    tof_list   = []
    tof_by_bar = {}

    for ev in events:
        st_sum = ev.get("ST_sum")
        if st_sum is None:
            continue
        t_st = st_sum.get("time_cfd")
        if t_st is None:
            continue

        for bar_id, bar in ev.get("bars", {}).items():
            if bars is not None and bar_id not in bars:
                continue

            wf_L = bar.get("wf_L")
            wf_R = bar.get("wf_R")
            if wf_L is None or wf_R is None:
                continue
            if not bar.get("signal_L") or not bar.get("signal_R"):
                continue

            q_bar = bar.get("charge_bar")
            if q_bar is None or q_bar <= 0:
                continue
            if q_min is not None and q_bar * 1e3 < q_min:
                continue
            if q_max is not None and q_bar * 1e3 > q_max:
                continue

            t_sat = {}
            for wf, side in [(wf_L, "L"), (wf_R, "R")]:
                amp_sat = np.clip(wf["amp"], -sat_level_v, sat_level_v)
                wf_sat  = {"amp": amp_sat, "time": wf["time"],
                           "amplitude": float(np.abs(np.min(amp_sat))),
                           "noise": wf.get("noise", 0.0)}
                t_sat[side] = _compute_time_simple(wf_sat)

            if t_sat["L"] is None or t_sat["R"] is None:
                continue

            tof = (t_sat["L"] + t_sat["R"]) / 2.0 - t_st
            tof_list.append(tof)
            tof_by_bar.setdefault(bar_id, []).append(tof)

    print(f"\n--- compute_tof_cfd_sat (level {sat_level_mv:.0f} mV) ---")
    print(f"TOF entries: {len(tof_list)}")
    for bid, tofs in sorted(tof_by_bar.items()):
        print(f"  Bar {bid}: {len(tofs)}")

    return tof_list, tof_by_bar


# =============================================================================
# CHARGE FROM THE FIT
# =============================================================================

def compute_charge_from_fit(results, wfs, t_grid, model, std, t0_ref=T0_REF_TW):
    """
    Charge of the fitted waveform A * template, integrated as the measured
    charge (charge_wf._compute_charge_integral), compared with wf["charge"].
    For saturated waveforms the fit recovers the part above saturation.
    Returns charges_fit, charges_orig, diff (mV*ns).
    """
    from charge_wf import _compute_charge_integral

    model_interp = interp1d(t_grid, model, kind="cubic", bounds_error=False, fill_value=0.0)

    charges_fit  = []
    charges_orig = []

    for res, wf in zip(results, wfs):
        if not res["success"]:
            continue

        amp_rec = res["A"] * model_interp(t_grid - (res["t_arr"] - t0_ref))
        wf_rec  = {"amp": amp_rec, "time": t_grid, "pedestal": wf.get("pedestal", 0.0)}
        charge_fit  = _compute_charge_integral(wf_rec)
        charge_orig = wf.get("charge")

        if charge_orig is not None and charge_fit is not None:
            charges_fit.append(charge_fit * 1e3)
            charges_orig.append(charge_orig * 1e3)

    charges_fit  = np.array(charges_fit)
    charges_orig = np.array(charges_orig)
    diff         = charges_fit - charges_orig

    print(f"\n--- Charge: fit vs measured ---")
    print(f"Waveforms: {len(charges_fit)}")
    print(f"Measured charge: mean={np.mean(charges_orig):.0f}, std={np.std(charges_orig):.0f} mV·ns")
    print(f"Fitted charge:   mean={np.mean(charges_fit):.0f}, std={np.std(charges_fit):.0f} mV·ns")
    print(f"Difference:      mean={np.mean(diff):.1f}, std={np.std(diff):.1f} mV·ns "
          f"({np.mean(np.abs(diff)/charges_orig)*100:.2f}% relative)")

    return charges_fit, charges_orig, diff


def plot_charge_bar_from_fit(results, wfs, events, t_grid, model,
                             bar_id=9, t0_ref=T0_REF_TW, fit=True, fit_range=None):
    """
    Bar charge sqrt(Q_L * Q_R) from the fitted waveforms vs the measured one,
    for one bar: scatter plot (left) and relative difference with a gaussian
    fit (right). Returns q_fit_bar, q_orig_bar (V*ns).
    """
    from charge_wf import _compute_charge_integral

    model_interp = interp1d(t_grid, model, kind="cubic", bounds_error=False, fill_value=0.0)

    # fitted charge of each end: {eventN: {"L": q, "R": q}}
    ev_charges_fit = {}
    for res, wf in zip(results, wfs):
        if not res["success"] or wf.get("bar_id") != bar_id:
            continue
        ev_n = wf.get("_eventN")
        end  = wf.get("end")
        if ev_n is None or end is None:
            continue
        amp_rec = res["A"] * model_interp(t_grid - (res["t_arr"] - t0_ref))
        q_fit = _compute_charge_integral({"amp": amp_rec, "time": t_grid,
                                          "pedestal": wf.get("pedestal", 0.0)})
        if q_fit is None or q_fit <= 0:
            continue
        ev_charges_fit.setdefault(ev_n, {})[end] = q_fit

    # measured charge of each end
    ev_charges_orig = {}
    for ev in events:
        bar = ev.get("bars", {}).get(bar_id)
        if bar is None:
            continue
        for end in ("L", "R"):
            wf_end = bar.get(f"wf_{end}")
            if wf_end is None:
                continue
            q_orig = wf_end.get("charge")
            if q_orig is not None and q_orig > 0:
                ev_charges_orig.setdefault(ev["eventN"], {})[end] = q_orig

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

    print(f"\n--- Bar {bar_id} charge: fit vs measured ---")
    if not q_orig_bar:
        print("  no data")
        return q_fit_bar, q_orig_bar

    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    q_orig_arr = np.array(q_orig_bar) * 1e3   # mV·ns
    q_fit_arr  = np.array(q_fit_bar)  * 1e3
    diffs = (q_fit_arr - q_orig_arr) / q_orig_arr * 100
    print(f"  N={len(diffs)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Charge — bar {bar_id}", fontsize=14)

    lim = [min(q_orig_arr.min(), q_fit_arr.min()) * 0.95, 15e3 * 1.05]
    axes[0].scatter(q_orig_arr, q_fit_arr, s=4, alpha=0.6, color="steelblue")
    axes[0].plot(lim, lim, color="red", lw=1.5, linestyle="--", label="y=x")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("measured charge (mV·ns)", fontsize=13)
    axes[0].set_ylabel("charge from fit (mV·ns)", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.4)

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
                                   p0=[np.max(counts), mu0, sigma0], maxfev=10000)
            perr  = np.sqrt(np.diag(pcov))
            x_fit = np.linspace(popt[1] - 3*abs(popt[2]), popt[1] + 3*abs(popt[2]), 1000)
            axes[1].plot(x_fit, gauss(x_fit, *popt), color="black", lw=2,
                         label=f"μ={popt[1]:.2f}%,  σ={abs(popt[2]):.2f}%")
            axes[1].legend(fontsize=11)
            print(f"    mu={popt[1]:.2f}%  sigma={abs(popt[2]):.2f}±{perr[2]:.2f}%")
        except Exception as e:
            print(f"    Gaussian fit failed: {e}")
    axes[1].set_xlabel("(Q_fit - Q_orig) / Q_orig (%)", fontsize=13)
    axes[1].set_ylabel("Counts", fontsize=13)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    return q_fit_bar, q_orig_bar


# =============================================================================
# TEMPLATE FIT
# =============================================================================

def model_fit_wf(wf_amp, t_grid, model, std, model_interp=None,
                 A_0=None, t_arr_0=None, step=RESAMPLE_STEP, bracket_half=5.0,
                 w_rise=20.0, w_tail=1.0, rise_left=5.0, rise_right=5.0, tail=150.0,
                 optimizer="scalar", t0_ref=T0_REF_TW, sat_level=None, noise=None):
    """
    Fit of one resampled waveform with A * template(t - (t_arr - t0_ref)).

    chi2 = sum_i w_i (wf_i - A * template_i)^2 / sigma_i^2, with
        sigma_i  sigma_noise of the waveform (CHI2_SIGMA = "noise")
        w_i      w_rise in [t_arr_0 - rise_left, t_arr_0 + rise_right],
                 w_tail up to t_arr_0 + tail, 0 elsewhere and on saturated samples
    For a fixed t_arr, A = sum(wf*T*w/sigma^2) / sum(T^2*w/sigma^2); the chi2 is
    then minimised in t_arr within ±bracket_half of t_arr_0 (the CFD time).
    A fit converging to the edge of the interval is flagged as failed.
    optimizer="nelder" fits A and t_arr together (Nelder-Mead).

    Returns {"A", "dA", "t_arr", "dt_arr", "chi2", "npts", "success"}; the errors
    come from the second derivative of the chi2 and are only indicative
    (resampled points are correlated).
    """
    if t_arr_0 is None:
        t_arr_0 = float(t_grid[np.argmin(wf_amp)])

    if CHI2_SIGMA == "noise" and noise is not None and noise > 0:
        sigma = np.full(len(wf_amp), noise)
    else:
        sigma = np.where(std > 1e-4, std, 1e-4)

    # chi2 window, centred on the initial estimate of the arrival time
    t_center  = t_arr_0 if t_arr_0 is not None else t0_ref
    rise_mask = (t_grid >= t_center - rise_left) & (t_grid <= t_center + rise_right)
    tail_mask = (t_grid > t_center + rise_right)  & (t_grid <= t_center + tail)
    weights   = np.where(rise_mask, w_rise, np.where(tail_mask, w_tail, 0.0))

    if sat_level is not None:
        sat_mask = np.abs(wf_amp) >= sat_level
        weights  = np.where(sat_mask, 0.0, weights)

    if model_interp is None:
        model_interp = interp1d(t_grid, model, kind="cubic", bounds_error=False, fill_value=0.0)

    def _chi2_1d(t_arr):
        """chi2 at fixed t_arr, with the analytic optimal amplitude."""
        model_shifted = model_interp(t_grid - (t_arr - t0_ref))
        num   = np.sum(wf_amp * model_shifted * weights / sigma**2)
        den   = np.sum(model_shifted**2  * weights / sigma**2)
        A_opt = num / den if den > 1e-10 else 1.0
        A_opt = max(A_opt, 1e-6)
        res   = (wf_amp - A_opt * model_shifted) * np.sqrt(weights) / sigma
        return float(np.sum(res**2)), A_opt

    def _chi2_scalar(t_arr):
        val, _ = _chi2_1d(t_arr)
        return val

    dA     = np.nan
    dt_arr = np.nan

    if optimizer == "scalar":
        bracket = (t_arr_0 - bracket_half, t_arr_0 + bracket_half)

        result = minimize_scalar(_chi2_scalar, bounds=bracket, method="bounded",
                                 options={"xatol": 1e-5})

        t_arr_fit = result.x
        if abs(t_arr_fit - bracket[0]) < 0.01 or abs(t_arr_fit - bracket[1]) < 0.01:
            result.success = False
        chi2_val, A_fit = _chi2_1d(t_arr_fit)

        try:
            # errors from the curvature of the chi2 at the minimum (finite differences)
            chi2_p = _chi2_scalar(t_arr_fit + step)
            chi2_m = _chi2_scalar(t_arr_fit - step)
            d2chi2 = (chi2_p - 2*chi2_val + chi2_m) / step**2
            dt_arr = 1.0 / np.sqrt(d2chi2) if d2chi2 > 0 else np.nan

            eps_A   = A_fit * 0.01
            res_Ap  = (wf_amp - (A_fit + eps_A) * model_interp(t_grid - (t_arr_fit - t0_ref)))
            chi2_Ap = float(np.sum((res_Ap * np.sqrt(weights) / sigma)**2))
            d2chi2_A = (chi2_Ap - chi2_val) / eps_A**2
            dA = 1.0 / np.sqrt(d2chi2_A) if d2chi2_A > 0 else np.nan
        except Exception:
            pass

    else:
        from scipy.optimize import minimize
        A_0 = float(np.abs(np.min(wf_amp)))

        def _chi2_2d(params):
            A, t_arr = params
            ms  = model_interp(t_grid - (t_arr - t0_ref))
            res = (wf_amp - A * ms) * np.sqrt(weights) / sigma
            return float(np.sum(res**2))

        res2d     = minimize(_chi2_2d, x0=[A_0, t_arr_0], method="Nelder-Mead",
                             options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000})
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
    model_fit_wf on all the resampled waveforms (not normalised, not aligned),
    using the CFD time and the noise of each waveform. t_grid_model is the grid
    of the template, if different from t_grid. max_wfs limits the number of fits.
    Returns a list of result dicts, one per waveform.
    """
    n = len(amp_resampled) if max_wfs is None else min(max_wfs, len(amp_resampled))
    results = []
    n_failed = 0

    if t_grid_model is None:
        t_grid_model = t_grid
    model_interp = interp1d(t_grid_model, model, kind="cubic", bounds_error=False, fill_value=0.0)

    if len(std) != len(t_grid):
        std_interp = interp1d(t_grid_model, std, kind="cubic", bounds_error=False, fill_value=std.mean())
        std = std_interp(t_grid)

    print(f"\n--- apply_model_fit ({n} waveforms) ---")
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
        if verbose and (i+1) % 1000 == 0:
            print(f"  {i+1}/{n} done...")

    t_arr_list = [r["t_arr"] for r in results if r["success"]]
    A_list     = [r["A"]     for r in results if r["success"]]

    print(f"Successful fits: {n - n_failed}/{n}")
    if t_arr_list:
        print(f"t_arr — mean={np.mean(t_arr_list):.3f} ns,  std={np.std(t_arr_list):.3f} ns")
        print(f"A     — mean={np.mean(A_list)*1e3:.1f},  std={np.std(A_list)*1e3:.1f}  (x1e3)")

    return results


def compute_tst_from_single(results_st, wfs_st, min_wfs_per_event=6, chi2_max=None, std_max=None, ch_exclude=None):
    """
    ST time of each event from the fits of the single ST channels: weighted mean
    (1/dt_arr^2) of the fitted times, if at least min_wfs_per_event channels
    have a valid fit. ch_exclude: channels to ignore; std_max: maximum spread
    of the channel times (ns).
    Returns {eventN: t_ST} and {eventN: spread of the channel times}.
    """
    ev_times     = defaultdict(list)
    ev_times_err = defaultdict(list)

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
        ev_times_err[ev_n].append(res["dt_arr"])

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
        errors = np.array(ev_times_err[ev_n])

        valid = np.isfinite(errors) & (errors > 0)
        if valid.sum() >= 2:
            weights_t = 1.0 / errors[valid]**2
            ev_to_tST_single[ev_n] = float(np.average(times[valid], weights=weights_t))
        else:
            ev_to_tST_single[ev_n] = float(np.mean(times))

        ev_to_tST_std[ev_n] = float(np.std(times))
        n_accepted += 1

    print(f"\n--- compute_tst_from_single ---")
    print(f"Accepted events: {n_accepted}  (rejected: {n_rejected})")

    t_means = list(ev_to_tST_single.values())
    t_stds  = list(ev_to_tST_std.values())
    if t_means:
        print(f"t_ST — mean={np.mean(t_means):.3f} ns, std={np.std(t_means):.3f} ns")
        print(f"Spread of the channel times within an event — "
              f"mean={np.mean(t_stds)*1e3:.1f} ps, max={np.max(t_stds)*1e3:.1f} ps")

    return ev_to_tST_single, ev_to_tST_std


# =============================================================================
# FIT PLOTS
# =============================================================================

def plot_model_fit_results(results, t_grid, amp_resampled, model, n_sample=6, t0_ref=T0_REF_TW,
                           wfs=None, q_min=None, q_max=None):
    """
    Distributions of the fitted time and of the chi2, and n_sample random
    waveforms with the fitted template (q_min, q_max: selection on the
    integral of the resampled waveform, mV*ns).
    """
    t_arr_list = np.array([r["t_arr"] for r in results if r["success"]])
    chi2_list  = np.array([r["chi2"]  for r in results if r["success"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Template fit results", fontsize=14)

    axes[0].hist(t_arr_list, bins=100, range=(np.min(t_arr_list), np.max(t_arr_list)), color="seagreen", alpha=0.8)
    axes[0].set_xlabel("t_arr (ns)", fontsize=18)
    axes[0].set_ylabel("Counts", fontsize=18)
    axes[0].set_title(f"Arrival time — {len(t_arr_list)} waveforms", fontsize=18)
    axes[0].grid(True, alpha=0.4)
    axes[0].tick_params(axis='both', labelsize=16)

    axes[1].hist(chi2_list, bins=100, range=(0, np.percentile(chi2_list, 99)), color="crimson", alpha=0.8)
    axes[1].set_xlabel("chi2", fontsize=18)
    axes[1].set_ylabel("Counts", fontsize=18)
    axes[1].set_title("chi2 distribution", fontsize=18)
    axes[1].grid(True, alpha=0.4)
    axes[1].tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()

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
    fig, axes = plt.subplots(rows, cols, figsize=(4, 3))
    axes = np.array(axes).flatten()

    model_interp_plot = interp1d(t_grid, model, kind="cubic", bounds_error=False, fill_value=0.0)

    for plot_i, wf_i in enumerate(idx_sample):
        ax  = axes[plot_i]
        A, t_arr = results[wf_i]["A"], results[wf_i]["t_arr"]
        model_shifted = model_interp_plot(t_grid - (t_arr - t0_ref))

        ax.plot(t_grid, amp_resampled[wf_i]*1e3, color="steelblue", lw=1.5, label="waveform")
        ax.plot(t_grid, A * model_shifted * 1e3, color="crimson", lw=1.5, linestyle="--", label="fit")
        ax.axvline(t_arr, color="black", lw=1.8, linestyle=":",
                   label=rf"$t_{{\mathrm{{fit}}}} = {t_arr:.2f}$ ns")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.1)

    for i in range(len(idx_sample), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_st_event_fit(ev_idx, events, results_st, wfs_st,
                      t_grid_st, amp_resampled_st,
                      model_st=None, models_ch=None,
                      t_grid_model=None, t0_ref=T0_REF_ST, t0_refs_ch=None):
    """
    The 8 ST waveforms of one event (ev_idx = index in events) with their fitted
    template: one template for all channels (model_st) or one per channel
    (models_ch, with the reference times t0_refs_ch).
    """
    ev_n = events[ev_idx]["eventN"]

    idx_ev = [i for i, wf in enumerate(wfs_st) if wf.get("_eventN") == ev_n]
    if not idx_ev:
        print(f"No ST waveform for event {ev_n} (idx={ev_idx}).")
        return

    if t_grid_model is None:
        t_grid_model = t_grid_st

    if models_ch is None:
        model_interp_default = interp1d(t_grid_model, model_st, kind="cubic",
                                        bounds_error=False, fill_value=0.0)
    else:
        model_interp_default = None

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = axes.flatten()

    for plot_i, wf_i in enumerate(idx_ev):
        ax  = axes[plot_i]
        wf  = wfs_st[wf_i]
        res = results_st[wf_i]
        ch  = wf.get("channelId")

        t0_ref_ch = t0_refs_ch[ch] if (t0_refs_ch is not None and ch in t0_refs_ch) else t0_ref

        if models_ch is not None and ch in models_ch:
            model_interp = interp1d(t_grid_model, models_ch[ch], kind="cubic",
                                    bounds_error=False, fill_value=0.0)
        else:
            model_interp = model_interp_default

        ax.plot(t_grid_st, amp_resampled_st[wf_i] * 1e3, color="steelblue", lw=1.0, label="waveform")

        if res["success"] and model_interp is not None:
            A, t_arr = res["A"], res["t_arr"]
            model_shifted = model_interp(t_grid_st - (t_arr - t0_ref_ch))
            ax.plot(t_grid_st, A * model_shifted * 1e3, color="crimson", lw=1.8, linestyle="--", label="fit")
            ax.axvline(t_arr, color="black", lw=1.5, linestyle=":",
                       label=rf"$t_{{\mathrm{{fit}}}} = {t_arr:.2f}$ ns")
            ax.set_xlim(50, 250)
            ax.set_title(f"Board {wf['boardId']} Ch {wf['channelId']}")
        else:
            ax.set_facecolor("#fff0f0")   # failed fit

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

    for i in range(len(idx_ev), nrows * ncols):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# TOF FROM THE FIT
# =============================================================================

def compute_tof_model(results, wfs, events, q_min=None, q_max=None, chi2_max=None, ev_to_tST_model=None):
    """
    TOF with the fitted TW times: TOF = (t_arr_L + t_arr_R)/2 - t_ST, with t_ST
    the CFD time of ST_sum or, if given, ev_to_tST_model {eventN: t_ST}.
    results[i] must correspond to wfs[i]; q_min, q_max: bar charge selection
    (mV*ns); chi2_max: maximum chi2 of the single fits.
    Returns tof_list and tof_by_bar {bar_id: [tof, ...]}.
    """
    if ev_to_tST_model is not None:
        ev_to_tST = ev_to_tST_model
    else:
        ev_to_tST = {}
        for ev in events:
            st_sum = ev.get("ST_sum")
            if st_sum and st_sum.get("time_cfd") is not None:
                ev_to_tST[ev["eventN"]] = st_sum["time_cfd"]

    # (eventN, bar_id) -> fitted times of the two ends
    bar_times = {}
    for res, wf in zip(results, wfs):
        if not res["success"]:
            continue
        if chi2_max is not None and res["chi2"] > chi2_max:
            continue
        bar_id = wf.get("bar_id")
        end    = wf.get("end")
        ev_n   = wf.get("_eventN")
        if bar_id is None or end is None or ev_n is None:
            continue
        bar_times.setdefault((ev_n, bar_id), {"t_L": None, "t_R": None})[f"t_{end}"] = res["t_arr"]

    ev_charge = {}
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            ev_charge[(ev["eventN"], bid)] = bar.get("charge_bar")

    tof_list   = []
    tof_by_bar = {}

    for (ev_n, bar_id), bt in bar_times.items():
        tL = bt["t_L"]
        tR = bt["t_R"]
        if tL is None or tR is None:
            continue
        t_ST = ev_to_tST.get(ev_n)
        if t_ST is None:
            continue

        q = ev_charge.get((ev_n, bar_id))
        if q is not None:
            q_mV = q * 1e3
            if q_min is not None and q_mV < q_min:
                continue
            if q_max is not None and q_mV > q_max:
                continue

        tof = (tL + tR) / 2.0 - t_ST
        tof_list.append(tof)
        tof_by_bar.setdefault(bar_id, []).append(tof)

    print(f"\n--- compute_tof_model ---")
    print(f"TOF entries: {len(tof_list)}")
    if tof_list:
        print(f"mean={np.mean(tof_list):.3f} ns,  std={np.std(tof_list):.3f} ns")

    return tof_list, tof_by_bar


def plot_tof_template(tof_list, bar_id=None):
    """TOF histogram in ±1 ns around the mean, with a gaussian fit."""
    if not tof_list:
        print("No TOF to plot.")
        return

    data = np.array(tof_list)

    mu_peak    = np.mean(data)
    half_width = 1.0  # ns
    fit_range  = (mu_peak - half_width, mu_peak + half_width)

    counts, bin_edges = np.histogram(data, bins=100, range=fit_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=100, color="steelblue", range=fit_range, alpha=0.7, label="TOF data")

    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    mu0    = bin_centers[np.argmax(counts)]
    sigma0 = (fit_range[1] - fit_range[0]) / 6.0
    A0     = np.max(counts)
    p0     = [A0, mu0, sigma0]

    try:
        popt, pcov = curve_fit(gauss, bin_centers, counts, p0=p0, maxfev=10000)
        A, mu, sigma    = popt
        dA, dmu, dsigma = np.sqrt(pcov.diagonal())

        print(f"--- TOF gaussian fit ---")
        if bar_id is not None:
            print(f"Bar {bar_id}")
        print(f"A = {A:.1f},  mu = ({mu:.1f} ± {dmu:.1f}) ns,  "
              f"sigma = ({abs(sigma)*1e3:.1f} ± {dsigma*1e3:.1f}) ps")

        x_plot = np.linspace(mu - 3*sigma, mu + 3*sigma, 2000)
        ax.plot(x_plot, gauss(x_plot, *popt), color="black", lw=2,
                label=f"sigma_tof = ({abs(sigma)*1e3:.1f} ± {dsigma*1e3:.1f}) ps")
    except Exception as e:
        print(f"\n--- TOF gaussian fit failed: {e} ---")

    title = "TOF"
    if bar_id is not None:
        title += f" — bar {bar_id}"
    ax.set_xlabel("TOF (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(f"{title}  ({len(tof_list)} entries)", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STANDALONE: templates, fits and TOF comparison on one file
# =============================================================================

if __name__ == "__main__":
    from read_wf      import load_waveforms
    from baseline_wf  import apply_baseline, cleanup_baseline
    from amplitude_wf import apply_amplitude, filter_pileup_events
    from tw_map       import apply_tw_map
    from charge_wf    import apply_charge
    from cfd_wf       import apply_cfd, build_st_sum
    from bar_analysis import build_bars, build_bar_times, compute_tof

    parser = argparse.ArgumentParser(description="Template construction and fit on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--bar", type=int, default=9, help="Bar used to build the TW template")
    parser.add_argument("--sat", type=float, default=200.0, help="Artificial saturation level (mV)")
    parser.add_argument("--event", type=int, default=0, help="Event shown with the per-channel ST fits")
    parser.add_argument("--q-min", type=float, default=10e3, help="Minimum bar charge for the TOF (mV*ns)")
    parser.add_argument("--q-max", type=float, default=13.5e3, help="Maximum bar charge for the TOF (mV*ns)")
    args = parser.parse_args()
    Q_SEL = dict(q_min=args.q_min, q_max=args.q_max)

    # --- standard reconstruction ---
    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="mean")
    cleanup_baseline(events)
    events, *_ = apply_amplitude(events)
    events = filter_pileup_events(events)
    events = build_st_sum(events)
    events, *_ = apply_cfd(events, method_st="simple", method_tw="simple")
    events, _ = apply_charge(events, method="integral")
    events = apply_tw_map(events)
    events, _ = build_bars(events)
    events, _ = build_bar_times(events)

    events, _ = compute_tof(events, **Q_SEL)
    tof_by_bar_cfd = {}
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            if bar.get("tof") is not None:
                tof_by_bar_cfd.setdefault(bid, []).append(bar["tof"])

    # ================================================================
    # START COUNTER
    # ================================================================
    ST_BOARD    = 27
    ST_CHANNELS = list(range(8))
    ST_PARAMS = dict(w_rise=1.0, w_tail=1.0, rise_left=1.0, rise_right=1.5,
                     tail=4.0, bracket_half=5.0, optimizer="scalar")

    # method 1: template and fit of ST_sum
    wfs_st = collect_st_sum(events)
    t_grid_st, amp_resampled_st = resample_wfs(wfs_st)
    amp_norm_st, _ = normalize_wfs(amp_resampled_st, wfs_st, method="amplitude")
    model_st, std_st, amp_aligned_st = build_model_iterative(
        amp_resampled_st, amp_norm_st, wfs_st, t_grid_st,
        n_iter=0, method="weighted", t0_ref=T0_REF_ST, max_shift=10.0, **ST_PARAMS)
    plot_model(t_grid_st, model_st, std_st, t0_ref=T0_REF_ST)
    plot_model_compare(t_grid_st, model_st, std_st, amp_aligned_st, wfs_st, n_wfs=100)

    results_st_sum = apply_model_fit(amp_resampled_st, wfs_st, t_grid_st, model_st, std_st,
                                     t0_ref=T0_REF_ST, **ST_PARAMS)
    t_arr_fit_sum = {wf["_eventN"]: res["t_arr"] for res, wf in zip(results_st_sum, wfs_st) if res["success"]}
    print(f"Method 1 (fit of ST_sum): {len(t_arr_fit_sum)} events with t_ST")
    plot_model_fit_results(results_st_sum, t_grid_st, amp_resampled_st, model_st, n_sample=1, t0_ref=T0_REF_ST)

    # method 2: one template per ST channel, t_ST as weighted mean of the channels
    wfs_st_single = collect_st_single(events)
    t_grid_st_single, amp_resampled_st_single = resample_wfs(wfs_st_single)
    ch_indices = {ch: [i for i, wf in enumerate(wfs_st_single)
                       if wf["channelId"] == ch and wf["boardId"] == ST_BOARD] for ch in ST_CHANNELS}

    models_ch, results_ch, t0_refs_ch = {}, {}, {}
    for ch in ST_CHANNELS:
        idx = ch_indices[ch]
        if not idx:
            continue
        wfs_ch = [wfs_st_single[i] for i in idx]
        amp_ch = amp_resampled_st_single[idx]
        print(f"\n=== ST channel {ch} — {len(wfs_ch)} waveforms ===")

        amp_norm_ch, _ = normalize_wfs(amp_ch, wfs_ch, method="amplitude")
        t0_refs_ch[ch] = float(np.median([t_grid_st_single[np.argmin(a)] for a in amp_ch]))
        model_ch, std_ch, _ = build_model_iterative(
            amp_ch, amp_norm_ch, wfs_ch, t_grid_st_single, max_shift=10.0,
            n_iter=0, method="weighted", t0_ref=t0_refs_ch[ch], **ST_PARAMS)
        models_ch[ch]  = model_ch
        results_ch[ch] = apply_model_fit(amp_ch, wfs_ch, t_grid_st_single, model_ch, std_ch,
                                         t0_ref=t0_refs_ch[ch], **ST_PARAMS)

    # results back in the original order (channels without template: failed fit)
    dummy_failed = {"A": 0.0, "dA": np.nan, "t_arr": 0.0, "dt_arr": np.nan, "chi2": np.inf, "success": False}
    results_st_single = [dummy_failed] * len(wfs_st_single)
    for ch, res_list in results_ch.items():
        for res, i in zip(res_list, ch_indices[ch]):
            results_st_single[i] = res

    plot_st_event_fit(ev_idx=args.event, events=events, results_st=results_st_single, wfs_st=wfs_st_single,
                      t_grid_st=t_grid_st_single, amp_resampled_st=amp_resampled_st_single,
                      models_ch=models_ch, t_grid_model=t_grid_st_single, t0_refs_ch=t0_refs_ch)

    t_arr_fit_single, t_arr_fit_std = compute_tst_from_single(results_st_single, wfs_st_single,
                                                              min_wfs_per_event=6, ch_exclude={7})

    stds = np.array(list(t_arr_fit_std.values()))
    print(f"\n--- Spread of the ST channel times within an event ---")
    print("Percentiles (ps): " + "  ".join(f"{p}%={np.percentile(stds, p)*1e3:.0f}" for p in (10, 25, 50, 75, 90, 99)))
    print(f"Events with spread < 200 ps: {np.sum(stds < 0.2)}/{len(stds)} ({100*np.mean(stds < 0.2):.1f}%)")

    t_st_cfd = {ev["eventN"]: ev["ST_sum"]["time_cfd"] for ev in events
                if ev.get("ST_sum") and ev["ST_sum"].get("time_cfd") is not None}

    print(f"\n--- t_ST comparison, event by event ---")
    for name, a, b in (("fit ST_sum - CFD ST_sum", t_arr_fit_sum, t_st_cfd),
                       ("fit channels - CFD ST_sum", t_arr_fit_single, t_st_cfd),
                       ("fit channels - fit ST_sum", t_arr_fit_single, t_arr_fit_sum)):
        d = [a[k] - b[k] for k in a if k in b]
        print(f"  {name:28s} mean={np.mean(d)*1e3:.1f} ps, std={np.std(d)*1e3:.1f} ps")

    # ================================================================
    # TOF-WALL
    # ================================================================
    TW_PARAMS = dict(w_rise=1.0, w_tail=1.0, rise_left=2.0, rise_right=5.0,
                     tail=150.0, bracket_half=5.0, optimizer="scalar")
    bid = args.bar

    wfs_tw = collect_tw(events, bar_ids=[bid])
    t_grid_tw, amp_resampled_tw = resample_wfs(wfs_tw)
    amp_norm_tw, _ = normalize_wfs(amp_resampled_tw, wfs_tw, method="charge")
    model_tw, std_tw, amp_aligned_tw = build_model_iterative(
        amp_resampled_tw, amp_norm_tw, wfs_tw, t_grid_tw, n_iter=0,
        method="weighted", t0_ref=T0_REF_TW, max_shift=None, **TW_PARAMS)

    plot_model(t_grid_tw, model_tw, std_tw, t0_ref=T0_REF_TW)
    plot_model_compare(t_grid_tw, model_tw, std_tw, amp_aligned_tw, wfs_tw, n_wfs=100, q_min=1000)

    results_tw = apply_model_fit(amp_resampled_tw, wfs_tw, t_grid_tw, model_tw, std_tw,
                                 t0_ref=T0_REF_TW, **TW_PARAMS)
    np.random.seed(1)
    plot_model_fit_results(results_tw, t_grid_tw, amp_resampled_tw, model_tw, n_sample=1,
                           wfs=wfs_tw, q_min=10000, q_max=13500)
    tof_list, tof_by_bar = compute_tof_model(results_tw, wfs_tw, events, **Q_SEL)

    # fit of artificially saturated waveforms with the same template
    amp_saturated, _, _ = apply_saturation(amp_resampled_tw, sat_level_mv=args.sat)
    results_tw_sat = apply_model_fit(amp_saturated, wfs_tw, t_grid_tw, model_tw, std_tw,
                                     **TW_PARAMS, sat_level=args.sat/1e3)
    plot_model_fit_results(results_tw_sat, t_grid_tw, amp_saturated, model_tw, n_sample=1, t0_ref=T0_REF_TW)
    tof_list_sat, tof_by_bar_sat = compute_tof_model(results_tw_sat, wfs_tw, events, **Q_SEL)

    compute_charge_from_fit(results_tw_sat, wfs_tw, t_grid_tw, model_tw, std_tw, t0_ref=T0_REF_TW)
    plot_charge_bar_from_fit(results_tw_sat, wfs_tw, events, t_grid_tw, model_tw, bar_id=bid, t0_ref=T0_REF_TW)

    _, tof_by_bar_cfd_sat = compute_tof_cfd_sat(events, sat_level_mv=args.sat, **Q_SEL)

    # ================================================================
    # TOF COMPARISON
    # ================================================================
    def tof_bar_cfd(events, bar_id, ev_to_tST, q_min=10e3, q_max=13.5e3):
        """TOF = CFD bar time - t_ST of the event, with a bar charge selection (mV*ns)."""
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

    # ST methods, TW with the simple CFD
    for name, t_st_map in (("CFD on ST_sum", t_st_cfd),
                           ("fit of ST_sum", t_arr_fit_sum),
                           ("fit of the single channels", t_arr_fit_single)):
        print(f"\n --- t_bar: simple CFD | t_ST: {name} ---")
        plot_tof_template(tof_bar_cfd(events, bid, t_st_map, **Q_SEL), bar_id=bid)

    # TW methods, ST with the simple CFD on ST_sum
    tof_tw = {"simple": tof_by_bar_cfd.get(bid, []), "fit": tof_by_bar.get(bid, []),
              "simple_sat": tof_by_bar_cfd_sat.get(bid, []), "fit_sat": tof_by_bar_sat.get(bid, [])}
    for method in ("tangent", "bipolar"):
        events, *_ = apply_cfd(events, method_st="simple", method_tw=method)
        events, _ = build_bar_times(events)
        tof_tw[method] = tof_bar_cfd(events, bid, t_st_cfd, **Q_SEL)
    for method, tofs in tof_tw.items():
        print(f"\n --- t_ST: simple CFD on ST_sum | t_bar: {method} ---")
        plot_tof_template(tofs, bar_id=bid)
