"""
bar_analysis.py — From waveforms to bar hits: charge, time and TOF of each TW bar.

A bar is hit when both ends have a signal (with the CFD time below CFD_MAX_TW).
    charge   Q = sqrt(Q_L * Q_R)
    time     t = (t_L + t_R) / 2
    TOF      t - t_ST, with t_ST the CFD time of the ST sum waveform

Usage:
    python3 bar_analysis.py <file.bin> [--bars 9 29 ...] [--q-min Q] [--q-max Q]
    (charges in mV*ns)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from read_wf      import load_waveforms
from baseline_wf  import apply_baseline, cleanup_baseline
from amplitude_wf import apply_amplitude, filter_pileup_events
from tw_map       import apply_tw_map
from charge_wf    import apply_charge
from cfd_wf       import apply_cfd, build_st_sum

CFD_MAX_TW  = 220.0   # ns — ends with a later CFD time are treated as without signal


def _gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def _do_fit(ax, data, fit_range):
    """Gaussian fit of the histogram of data in fit_range, drawn on ax. Returns (popt, errors) or None."""
    if len(data) < 5:
        return None
    r = fit_range if fit_range else (np.min(data), np.max(data))
    counts, bin_edges = np.histogram(data, bins=100, range=r)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A0     = np.max(counts)
    mu0    = bin_centers[np.argmax(counts)]
    sigma0 = (r[1] - r[0]) / 6.0

    try:
        popt, pcov = curve_fit(_gauss, bin_centers, counts,
                               p0=[A0, mu0, sigma0], maxfev=10000)
        pcov = np.sqrt(np.diag(pcov))
        A, mu, sigma = popt

        x_plot = np.linspace(mu-3*sigma, mu+3*sigma, 2000)
        ax.plot(x_plot, _gauss(x_plot, *popt), color="black", lw=2)
        return popt, pcov
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def build_bars(events):
    """
    Group the TW waveforms of each event by bar (to be called after apply_charge
    and apply_tw_map). Adds to every event:

    ev["bars"] = {bar_id: {"layer", "wf_L", "wf_R", "signal_L", "signal_R",
                           "charge_bar"}}

    with charge_bar = sqrt(Q_L * Q_R) when both ends have a signal and a positive
    charge, None otherwise. Returns events, charge_bars.
    """
    charge_bars = []

    for ev in events:
        bars = {}
        for wf in ev["TW"]:
            if wf["bar_id"] is None:
                continue

            bid = wf["bar_id"]
            if bid not in bars:
                bars[bid] = {
                    "layer":      wf["layer"],
                    "wf_L":       None,
                    "wf_R":       None,
                    "signal_L":   False,
                    "signal_R":   False,
                    "charge_bar": None
                }
            end = wf["end"]
            bars[bid][f"wf_{end}"] = wf
            # an end with a missing or too late CFD time counts as without signal
            t_cfd = wf.get("time_cfd")
            if wf["signal"] and (t_cfd is None or t_cfd > CFD_MAX_TW):
                bars[bid][f"signal_{end}"] = False
            else:
                bars[bid][f"signal_{end}"] = wf["signal"]

        for bid, bar in bars.items():
            wf_L = bar["wf_L"]
            wf_R = bar["wf_R"]
            if not (bar["signal_L"] and bar["signal_R"]):
                continue

            chL = wf_L["charge"] if wf_L and wf_L.get("charge") is not None and wf_L["charge"] > 0 else None
            chR = wf_R["charge"] if wf_R and wf_R.get("charge") is not None and wf_R["charge"] > 0 else None

            if chL is not None and chR is not None:
                bar["charge_bar"] = np.sqrt(chL * chR)
            else:
                continue
            if bar["charge_bar"] is not None:
                charge_bars.append(bar["charge_bar"])

        ev["bars"] = bars

    print(f"\n--- TW bar charge statistics ---")
    print(f"Bars with charge: {len(charge_bars)}")
    print(f"min={np.min(charge_bars)*1e3:.2f} mV·ns,  max={np.max(charge_bars)*1e3:.2f} mV·ns,  "
          f"mean={np.mean(charge_bars)*1e3:.2f} mV·ns,  std={np.std(charge_bars)*1e3:.2f} mV·ns")

    return events, charge_bars


def build_bar_times(events):
    """
    Bar time as the mean of the CFD times of the two ends (after apply_cfd and
    build_bars). Adds bar["time_L"], bar["time_R"] and bar["time_bar"]
    (None if one end is missing). Returns events, bar_times.
    """
    bar_times = []
    for ev in events:
        if "bars" not in ev:
            continue
        for bid, bar in ev["bars"].items():
            wf_L = bar.get("wf_L")
            wf_R = bar.get("wf_R")
            tL = wf_L.get("time_cfd") if wf_L and bar["signal_L"] else None
            tR = wf_R.get("time_cfd") if wf_R and bar["signal_R"] else None
            bar["time_L"] = tL
            bar["time_R"] = tR

            if tL is not None and tR is not None:
                bar["time_bar"] = (tL + tR) / 2.0
            else:
                bar["time_bar"] = None

            if bar["time_bar"] is not None and bar["time_bar"] > 0:
                bar_times.append(bar["time_bar"])

    print(f"\n--- Bar time statistics ---")
    print(f"Bars with time: {len(bar_times)}")
    if bar_times:
        print(f"mean={np.mean(bar_times):.2f} ns,  std={np.std(bar_times):.2f} ns")

    return events, bar_times


def compute_tof(events, q_min=None, q_max=None, bars=None):
    """
    bars=None: TOF = time_bar - t_ST_sum for every hit bar, stored in bar["tof"].
    bars=[front, rear]: time difference time_bar_front - time_bar_rear between two
    crossing bars hit in the same event (not stored in the events).
    q_min, q_max: charge selection in mV*ns.
    Returns events, tof_list.
    """
    tof_list = []

    for ev in events:
        st_sum = ev.get("ST_sum")
        t_st = st_sum.get("time_cfd") if st_sum else None

        if t_st is None or "bars" not in ev:
            continue

        if bars is None:
            for bid, bar in ev["bars"].items():
                q = bar.get("charge_bar")

                if q_min is not None and (q is None or q * 1e3 < q_min):
                    bar["tof"] = None
                    continue
                if q_max is not None and (q is None or q * 1e3 > q_max):
                    bar["tof"] = None
                    continue

                t_bar = bar.get("time_bar")
                if t_bar is not None:
                    bar["tof"] = t_bar - t_st
                    tof_list.append(bar["tof"])
                else:
                    bar["tof"] = None

        else:
            bar_front, bar_rear = bars

            bar_f = ev["bars"].get(bar_front)
            bar_r = ev["bars"].get(bar_rear)
            if bar_f is None or bar_r is None:
                continue

            t_f = bar_f.get("time_bar")
            t_r = bar_r.get("time_bar")
            if t_f is None or t_r is None:
                continue

            valid = True
            for bar in [bar_f, bar_r]:
                q = bar.get("charge_bar")
                if q_min is not None and (q is None or q * 1e3 < q_min):
                    valid = False
                    break
                if q_max is not None and (q is None or q * 1e3 > q_max):
                    valid = False
                    break
            if not valid:
                continue

            tof_list.append(t_f - t_r)

    if bars is None:
        print(f"\n--- TOF statistics ---")
    else:
        print(f"\n--- Time difference between bars {bars[0]} and {bars[1]} ---")
    print(f"Entries: {len(tof_list)}")
    if tof_list:
        print(f"mean={np.mean(tof_list):.3f} ns,  std={np.std(tof_list):.3f} ns")

    return events, tof_list


# =============================================================================
# PLOTS
# =============================================================================

def plot_charge_bars(charge_bars, fit=True, fit_range=None, n_peaks=1):
    """
    Histogram of the bar charge (mV*ns) with a fit of n_peaks gaussians
    (n_peaks = 1 or 2; the second peak starts at 15000 mV*ns).
    """
    data = np.array(charge_bars) * 1e3  # mV·ns

    if fit_range is None:
        fit_range = (np.min(data), np.max(data))

    counts, bin_edges = np.histogram(data, bins=100, range=fit_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=100, range=fit_range, color="orange", alpha=0.7, label="data")

    if fit:
        def gauss(x, *params):
            """Sum of n_peaks gaussians, params = [A1, mu1, sigma1, A2, mu2, sigma2, ...]."""
            multi_gauss = np.zeros_like(x, dtype=float)
            for i in range(n_peaks):
                A, mu, sigma = params[3*i], params[3*i+1], params[3*i+2]
                multi_gauss += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
            return multi_gauss

        A0   = np.max(counts)
        mu0  = bin_centers[np.argmax(counts)]
        sigma0 = mu0/20
        if n_peaks == 1:
            p0 = [A0, mu0, sigma0]
        else:
            A1 = 246
            mu1 = 15*1e3
            p0 = [A0, mu0, sigma0, A1, mu1, sigma0]
        try:
            popt, pcov = curve_fit(gauss, bin_centers, counts, p0=p0, maxfev=10000)
            pcov = np.sqrt(pcov.diagonal())
            print(f"\n--- Gaussian fit of the bar charge ---")
            for i in range(n_peaks):
                A, mu, sigma = popt[3*i], popt[3*i+1], abs(popt[3*i+2])
                dA, dmu, dsigma = pcov[3*i], pcov[3*i+1], pcov[3*i+2]
                print(f"Peak {i+1}: A = {A:.0f},  mu = ({mu:.0f} ± {dmu:.0f}) mV·ns,  "
                      f"sigma = ({abs(sigma):.0f} ± {dsigma:.0f}) mV·ns,  "
                      f"resolution = {abs(sigma)/mu*100:.1f} %")

                x_plot = np.linspace(mu-2*sigma, mu+2*sigma, 2000)
                ax.plot(x_plot, gauss(x_plot, *popt), color="black", lw=2)
        except Exception as e:
            print(f"Fit failed: {e}")

    ax.set_xlabel("Bar charge (mV·ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(f"TW bar charge, entries = {len(data)}", fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_cfd_times(times_ST_sum, bar_times):
    """Histograms of the ST_sum CFD times and of the bar times."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ST and TW times", fontsize=16)

    axes[0].hist(times_ST_sum, bins=100, color="royalblue")
    axes[0].set_xlabel("ST_sum CFD time (ns)", fontsize=14)
    axes[0].set_ylabel("Counts", fontsize=14)
    axes[0].set_title(f"ST ({len(times_ST_sum)} entries)", fontsize=14)
    axes[0].tick_params(axis='both', labelsize=13)
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(bar_times, bins=100, color="crimson")
    axes[1].set_xlabel("TW bar time (ns)", fontsize=14)
    axes[1].set_ylabel("Counts", fontsize=14)
    axes[1].set_title(f"TW ({len(bar_times)} entries)", fontsize=14)
    axes[1].tick_params(axis='both', labelsize=13)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_rise_time_bars(events):
    """Histogram of the bar rise time (mean of the two ends)."""
    rt_bars = []
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            wf_L = bar.get("wf_L")
            wf_R = bar.get("wf_R")
            rtL = wf_L.get("rise_time") if wf_L else None
            rtR = wf_R.get("rise_time") if wf_R else None
            if rtL is not None and rtR is not None:
                rt_bars.append((rtL + rtR) / 2.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rt_bars, bins=100, range=(2, 8), color="darkcyan", alpha=0.8)
    ax.set_xlabel("Bar rise time (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(f"TW rise time — per bar ({len(rt_bars)} entries)", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_tof(tof_list):
    """TOF histogram in a ±1 ns window around the mean, with a gaussian fit."""
    if not tof_list:
        print("No TOF to plot.")
        return
    data = np.array(tof_list)

    mu_peak = np.mean(data)
    half_width = 1.0  # ns
    fit_range = (mu_peak - half_width, mu_peak + half_width)

    counts, bin_edges = np.histogram(data, bins=100, range=fit_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=100, color="seagreen", range=fit_range, alpha=0.7, label="TOF data")

    mu0    = bin_centers[np.argmax(counts)]
    sigma0 = (fit_range[1] - fit_range[0]) / 6.0
    A0     = np.max(counts)
    p0 = [A0, mu0, sigma0]

    try:
        popt, pcov = curve_fit(_gauss, bin_centers, counts, p0=p0, maxfev=10000)
        A, mu, sigma = popt
        dA, dmu, dsigma = np.sqrt(pcov.diagonal())

        print(f"\n--- Gaussian fit of the TOF ---")
        print(f"A = {A:.1f},  mu = ({mu*1e3:.2f} ± {dmu*1e3:.2f}) ps,  "
              f"sigma = ({abs(sigma*1e3):.2f} ± {dsigma*1e3:.2f}) ps")

        x_plot = np.linspace(mu - 3*sigma, mu + 3*sigma, 2000)
        ax.plot(x_plot, _gauss(x_plot, *popt),
                color="black", lw=2, label=f"sigma = ({abs(sigma*1e3):.2f} ± {dsigma*1e3:.2f}) ps")
    except Exception as e:
        print(f"\n--- TOF gaussian fit failed: {e} ---")

    ax.set_xlabel("TOF (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_title(f"TW time of flight ({len(tof_list)} entries)", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_bar_analysis(events, bar_id, q_min=None, q_max=None,
                      time_range=None, fit=True,
                      fit_range_q=None, fit_range_tof=None):
    """
    Four plots for one bar: times of the two ends, bar time, charge, TOF.
        q_min, q_max   charge selection for the TOF (mV*ns)
        time_range     (min, max) in ns for the time histograms
        fit            gaussian fit of charge and TOF
        fit_range_q    (min, max) in mV*ns for the charge fit
        fit_range_tof  (min, max) in ns for the TOF fit (default: ±1 ns around the peak)
    """
    charges, times_L, times_R, bar_times_list, tofs = [], [], [], [], []

    for ev in events:
        bar = ev.get("bars", {}).get(bar_id)
        if bar is None:
            continue

        q = bar.get("charge_bar")
        if q is not None:
            charges.append(q * 1e3)

        tL = bar.get("time_L")
        tR = bar.get("time_R")
        if tL is not None: times_L.append(tL)
        if tR is not None: times_R.append(tR)

        t_bar = bar.get("time_bar")
        if t_bar is not None:
            bar_times_list.append(t_bar)

        tof = bar.get("tof")
        if tof is not None:
            q_mV = bar.get("charge_bar", 0) * 1e3
            if (q_min is None or q_mV >= q_min) and (q_max is None or q_mV <= q_max):
                tofs.append(tof)

    layer_label = "0 (rear, vertical)" if bar_id < 20 else "1 (front, horizontal)"

    print(f"\n--- Bar {bar_id} — Layer {layer_label} ---")
    print(f"Charges: {len(charges)}  |  Left times: {len(times_L)}  |  "
          f"Right times: {len(times_R)}  |  TOF: {len(tofs)}")

    if not charges and not bar_times_list and not tofs:
        print("No data for this bar.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Bar {bar_id} — Layer {layer_label}", fontsize=15)
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, alpha=0.4)

    # charge
    if charges:
        r_q = fit_range_q if fit_range_q else (np.min(charges), np.max(charges))
        axes[1, 0].hist(charges, bins=100, range=(np.min(charges), np.max(charges)), color="orange", alpha=0.7, label="data")
        if q_min is not None:
            axes[1, 0].axvline(q_min, color="steelblue", lw=1.5, linestyle="--", label=f"q_min={q_min:.0f}")
        if q_max is not None:
            axes[1, 0].axvline(q_max, color="steelblue", lw=1.5, linestyle="--", label=f"q_max={q_max:.0f}")
        if fit:
            result = _do_fit(axes[1, 0], charges, r_q)
            if result:
                popt, pcov = result
                print(f"  Charge fit: μ = {popt[1]:.0f} mV·ns,  "
                      f"σ = {abs(popt[2]):.0f} ± {pcov[2]:.0f} mV·ns,  "
                      f"σ/μ = {abs(popt[2]/popt[1])*100:.1f} %")
        axes[1, 0].set_xlabel("Bar charge (mV·ns)", fontsize=13)
        axes[1, 0].set_ylabel("Counts", fontsize=13)
        axes[1, 0].set_title(f"Charge — {len(charges)} entries", fontsize=13)
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend(fontsize=11)
    else:
        axes[1, 0].set_visible(False)

    # times of the two ends
    if times_L or times_R:
        all_t = times_L + times_R
        t_range = time_range if time_range else (min(all_t), max(all_t))
        if times_L:
            axes[0, 0].hist(times_L, bins=100, range=t_range,
                            color="steelblue", alpha=0.6, label=f"Left end ({len(times_L)})")
        if times_R:
            axes[0, 0].hist(times_R, bins=100, range=t_range,
                            color="crimson", alpha=0.6, label=f"Right end ({len(times_R)})")
        axes[0, 0].set_xlabel("CFD time (ns)", fontsize=13)
        axes[0, 0].set_ylabel("Counts", fontsize=13)
        axes[0, 0].set_title("Times of the L and R ends", fontsize=13)
        axes[0, 0].legend(fontsize=11)
    else:
        axes[0, 0].set_visible(False)

    # bar time
    if bar_times_list:
        axes[0, 1].hist(bar_times_list, bins=100,
                        range=time_range if time_range else None,
                        color="darkcyan", alpha=0.8, label="data")
        axes[0, 1].set_xlabel("Bar time (ns)", fontsize=13)
        axes[0, 1].set_ylabel("Counts", fontsize=13)
        axes[0, 1].set_title(f"Bar time — {len(bar_times_list)} entries", fontsize=13)
        axes[0, 1].legend(fontsize=11)
    else:
        axes[0, 1].set_visible(False)

    # TOF
    if tofs:
        if fit_range_tof:
            r_tof = fit_range_tof
        else:
            counts_tmp, edges_tmp = np.histogram(tofs, bins=200)
            mu_peak = edges_tmp[np.argmax(counts_tmp)]
            r_tof = (mu_peak - 1.0, mu_peak + 1.0)

        axes[1, 1].hist(tofs, bins=100, range=r_tof,
                        color="seagreen", alpha=0.7, label="data")
        if fit:
            result = _do_fit(axes[1, 1], tofs, r_tof)
            if result:
                popt, pcov = result
                print(f"  TOF fit: μ = {popt[1]:.1f} ns,  "
                      f"σ = {abs(popt[2])*1e3:.1f} ± {pcov[2]*1e3:.1f} ps")
        axes[1, 1].set_xlabel("TOF (ns)", fontsize=13)
        axes[1, 1].set_ylabel("Counts", fontsize=13)
        axes[1, 1].set_title(f"TOF — {len(tofs)} entries", fontsize=13)
        axes[1, 1].legend(fontsize=11)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_anomalous_times(events, st_min=None, st_max=90.0,
                         bar_min=140.0, bar_max=None, n_plot=8):
    """
    Check plots: ST_sum waveforms with time_cfd in (st_min, st_max) and bars
    with time_bar in (bar_min, bar_max), with their L and R waveforms.
    """
    st_anomalous = []
    for ev in events:
        st_sum = ev.get("ST_sum")
        if st_sum is None:
            continue
        t = st_sum.get("time_cfd")
        if t is None:
            continue
        if (st_min is None or t > st_min) and (st_max is None or t < st_max):
            st_anomalous.append((ev["eventN"], st_sum, t))

    bar_anomalous = []
    for ev in events:
        for bid, bar in ev.get("bars", {}).items():
            t_bar = bar.get("time_bar")
            if t_bar is None:
                continue
            if (bar_min is None or t_bar > bar_min) and (bar_max is None or t_bar < bar_max):
                bar_anomalous.append((ev["eventN"], bid, bar, bar.get("wf_L"), bar.get("wf_R"), t_bar))

    print(f"ST with t_cfd < {st_max} ns:      {len(st_anomalous)}")
    print(f"Bars with time_bar > {bar_min} ns: {len(bar_anomalous)}")

    if st_anomalous:
        subset = st_anomalous[:n_plot]
        nrows, ncols = (len(subset) + 1) // 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5*nrows))
        fig.suptitle(f"ST — time_cfd < {st_max} ns  [{len(st_anomalous)} total]", fontsize=13)
        axes_flat = np.array(axes).flatten()

        for i, (ev_n, st_sum, t) in enumerate(subset):
            ax = axes_flat[i]
            ax.plot(st_sum["time"], st_sum["amp"]*1e3, color="royalblue", lw=1.0)
            ax.axvline(t, color="red", lw=1.5, linestyle="--", label=f"t={t:.2f} ns")
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
            ax.set_title(f"ev {ev_n} | ST_sum ({st_sum.get('n_summed', '?')} wf)", fontsize=9)
            ax.set_xlabel("t (ns)", fontsize=9)
            ax.set_ylabel("amp (mV)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for i in range(len(subset), nrows*ncols):
            axes_flat[i].set_visible(False)
        plt.tight_layout()
        plt.show()

    if bar_anomalous:
        subset = bar_anomalous[:n_plot]
        nrows, ncols = len(subset), 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5*nrows))
        fig.suptitle(f"Bars — time_bar > {bar_min} ns  [{len(bar_anomalous)} total]", fontsize=13)
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for i, (ev_n, bid, bar, wf_L, wf_R, t_bar) in enumerate(subset):
            for j, (wf, end) in enumerate([(wf_L, "L"), (wf_R, "R")]):
                ax = axes[i, j]
                if wf is None:
                    ax.set_visible(False)
                    continue
                ax.plot(wf["time"], wf["amp"]*1e3,
                        color="steelblue" if end == "L" else "crimson", lw=1.0)
                ax.axhline(0, color="black", lw=0.8, linestyle="--")
                t_cfd = wf.get("time_cfd")
                if t_cfd is not None:
                    ax.axvline(t_cfd, color="red", lw=1.5, linestyle="--", label=f"t={t_cfd:.2f} ns")
                ax.set_title(f"ev {ev_n} | bar {bid} {end} | t_bar={t_bar:.2f} ns | "
                             f"B{wf['boardId']} Ch{wf['channelId']}", fontsize=9)
                ax.set_xlabel("t (ns)", fontsize=9)
                ax.set_ylabel("amp (mV)", fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_amplitude_bars(events, bar_a=9, bar_b=29, amp_range=None):
    """
    Amplitude of the single waveforms (L and R together) of two bars: bar_a,
    bar_b, events with at least one of them hit (OR) and with both hit (AND).
    amp_range: (min, max) in mV.
    """
    amps_a, amps_b, amps_or, amps_and = [], [], [], []

    for ev in events:
        bars = ev.get("bars", {})
        bar_info_a = bars.get(bar_a)
        bar_info_b = bars.get(bar_b)

        active_a = (bar_info_a is not None and bar_info_a.get("signal_L") and bar_info_a.get("signal_R"))
        active_b = (bar_info_b is not None and bar_info_b.get("signal_L") and bar_info_b.get("signal_R"))

        amp_a = [bar_info_a["wf_L"]["amplitude"] * 1e3, bar_info_a["wf_R"]["amplitude"] * 1e3] if active_a else []
        amp_b = [bar_info_b["wf_L"]["amplitude"] * 1e3, bar_info_b["wf_R"]["amplitude"] * 1e3] if active_b else []

        amps_a.extend(amp_a)
        amps_b.extend(amp_b)
        if active_a or active_b:
            amps_or.extend(amp_a + amp_b)
        if active_a and active_b:
            amps_and.extend(amp_a + amp_b)

    print(f"\n--- TW waveform amplitude ---")
    print(f"Bar {bar_a}: {len(amps_a)} waveforms")
    print(f"Bar {bar_b}: {len(amps_b)} waveforms")
    print(f"OR  (bar {bar_a} | bar {bar_b}): {len(amps_or)} waveforms")
    print(f"AND (bar {bar_a} & bar {bar_b}): {len(amps_and)} waveforms")

    all_amps = amps_a + amps_b
    r = amp_range if amp_range else (0.0, max(all_amps) if all_amps else 1000.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("TW waveform amplitude (L and R together)", fontsize=14)

    datasets = [
        (amps_a,   f"Bar {bar_a}",                 "steelblue"),
        (amps_b,   f"Bar {bar_b}",                 "crimson"),
        (amps_or,  f"OR  (bar {bar_a} | {bar_b})", "darkorange"),
        (amps_and, f"AND (bar {bar_a} & {bar_b})", "seagreen"),
    ]

    for ax, (data, title, color) in zip(axes.flat, datasets):
        if data:
            ax.hist(np.array(data), bins=100, range=r, color=color, alpha=0.8)
        ax.set_xlabel("Amplitude (mV)", fontsize=13)
        ax.set_ylabel("Counts", fontsize=13)
        ax.set_title(f"{title} — {len(data)} waveforms", fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bar charge, time and TOF on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--bars", type=int, nargs="+", default=[8, 9, 10, 28, 29, 30],
                        help="Bars for the per-bar plots")
    parser.add_argument("--q-min", type=float, default=10e3, help="Minimum bar charge for the TOF (mV*ns)")
    parser.add_argument("--q-max", type=float, default=13.5e3, help="Maximum bar charge for the TOF (mV*ns)")
    args = parser.parse_args()

    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="mean")
    cleanup_baseline(events)

    events, amp_ST, amp_TW = apply_amplitude(events)
    events = filter_pileup_events(events)
    events = build_st_sum(events)

    events, times_TW, times_ST, times_ST_sum = apply_cfd(events, method_st="simple", method_tw="simple")
    events, charge_TW = apply_charge(events, method="integral")
    events = apply_tw_map(events)

    events, charge_bars = build_bars(events)
    plot_charge_bars(charge_bars, fit=True, n_peaks=2)

    events, bar_times = build_bar_times(events)
    plot_cfd_times(times_ST_sum, bar_times)
    plot_rise_time_bars(events)

    events, tof_list = compute_tof(events, q_min=args.q_min, q_max=args.q_max)
    plot_tof(tof_list)

    for bid in args.bars:
        plot_bar_analysis(events, bar_id=bid, q_min=args.q_min, q_max=args.q_max)

    plot_amplitude_bars(events, bar_a=9, bar_b=29)
