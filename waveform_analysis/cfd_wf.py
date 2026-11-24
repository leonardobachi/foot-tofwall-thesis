"""
cfd_wf.py — Arrival time (CFD), rise time and ST sum waveform.

Three constant-fraction algorithms, the same implemented in SHOE:
    simple   threshold at frac * amplitude on the leading edge, linear
             interpolation between the two samples across the threshold
    tangent  same threshold; least-squares line through the 4 samples around
             it, the time is the zero crossing of the line
    bipolar  S(t) = frac * A(t) - A(t - delay) on a 0.2 ns resampled grid;
             the time is the zero crossing of S(t) before its maximum

The ST time is computed on the sum of the ST channels (ST_sum), resampled on a
common grid before summing to remove the sampling phase between channels.

Usage:
    python3 cfd_wf.py <file.bin> [--event N] [--save FILE]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from read_wf      import load_waveforms
from baseline_wf  import apply_baseline, cleanup_baseline
from amplitude_wf import apply_amplitude, filter_pileup_events

# =============================================================================
# PARAMETERS
# =============================================================================

# bipolar CFD
CFD_FRAC    = 0.2    # fraction of the amplitude
CFD_DEL     = 5.0    # ns — default delay, used if the rise time is not available
CFD_TLEFT   = -20.0  # ns — window around the peak
CFD_TRIGHT  = 20.0   # ns
CFD_SAMPLE  = 0.2    # ns — resampling step (also used for ST_sum)

# simple and tangent CFD
SCFD_FRAC   = 0.2    # fraction of the amplitude

# rise time
RT_LOW     = 0.10
RT_HIGH    = 0.90


# =============================================================================
# SINGLE-WAVEFORM FUNCTIONS
# =============================================================================

def _compute_time_bipolar(wf, frac=CFD_FRAC, delay=None,
                          tleft=CFD_TLEFT, tright=CFD_TRIGHT, step=CFD_SAMPLE):
    """
    Bipolar CFD: S(t) = frac * A(t) - A(t - delay) on a resampled grid around
    the peak; the time is the zero crossing found moving left from the maximum
    of S(t). The delay defaults to the rise time of the waveform (or CFD_DEL).
    Returns the arrival time in ns, or None.
    """
    if delay is None:
        rt = wf.get("rise_time")
        delay = rt if (rt is not None and rt > 0) else CFD_DEL

    amp  = wf["amp"]
    time = wf["time"]

    interp = interp1d(time, amp, kind="cubic", bounds_error=False, fill_value=0.0)

    peak_idx  = np.argmin(amp)
    timepeak  = time[peak_idx]
    tmin      = timepeak + tleft
    tmax      = timepeak + tright

    t_cfd   = np.arange(tmin, tmax, step)
    amp_cfd = frac * interp(t_cfd) - interp(t_cfd - delay)

    if len(amp_cfd) < 4:
        return None

    max_bin = np.argmax(amp_cfd[2:]) + 2

    found = False
    i = max_bin
    while i > 0:
        a1 = amp_cfd[i]
        a2 = amp_cfd[i-1]
        if a1 * a2 < 0:
            found = True
            break
        i -= 1

    if not found:
        return None

    t1, t2 = t_cfd[i-1], t_cfd[i]
    a1, a2 = amp_cfd[i-1], amp_cfd[i]
    m    = (a2 - a1) / (t2 - t1)
    q    = a2 - m * t2
    tarr = -q / m
    return tarr


def _compute_time_simple(wf, frac=SCFD_FRAC):
    """
    Simple CFD: moving left from the peak, find the two samples across the
    threshold -frac * amplitude and interpolate linearly between them.
    Returns the arrival time in ns, or None.
    """
    amp  = wf["amp"]
    time = wf["time"]

    if wf["amplitude"] is None:
        return None

    threshold = - frac * wf["amplitude"]   # signals are negative

    peak_idx = np.argmin(amp)
    i        = peak_idx
    found    = False

    while i > 1:
        a1 = amp[i]
        a2 = amp[i+1]
        t1 = time[i]
        t2 = time[i+1]
        if threshold > a2 and threshold <= a1:
            found = True
            break
        i -= 1

    if not found:
        return None

    m    = (a2 - a1) / (t2 - t1)
    q    = a1 - m * t1
    tarr = (threshold - q) / m
    return tarr


def _compute_time_tangent(wf, frac=SCFD_FRAC):
    """
    Tangent CFD: threshold crossing as in the simple CFD, then least-squares
    line through the 4 samples around it; the time is where the line crosses
    zero (not the threshold). Returns the arrival time in ns, or None.
    """
    amp  = wf["amp"]
    time = wf["time"]

    if wf.get("amplitude") is None:
        return None

    threshold = - frac * wf["amplitude"]
    peak_idx  = np.argmin(amp)
    i_thr     = peak_idx
    found     = False

    while i_thr:
        a1 = amp[i_thr]
        a2 = amp[i_thr+1]
        if threshold > a2 and threshold <= a1:
            found = True
            break
        i_thr -= 1

    if not found:
        return None

    idx_range = [i for i in range(i_thr-1, i_thr+3)]
    if len(idx_range) < 2:
        return None

    t_pts = np.array([time[i] for i in idx_range])
    a_pts = np.array([amp[i]  for i in idx_range])

    # analytic least-squares solution of a = m*t + q
    x   = np.mean(t_pts)
    y   = np.mean(a_pts)
    xy  = np.mean(t_pts * a_pts)
    xsq = np.mean(t_pts ** 2)

    denom = xsq - x*x
    if abs(denom) < 1e-12:
        return None

    m = (xy - x*y) / denom
    q = (xsq*y - x*xy) / denom

    if abs(m) < 1e-12:
        return None
    return -q / m


def _compute_rise_time(wf, rt_low=RT_LOW, rt_high=RT_HIGH):
    """Rise time between 10% and 90% of the amplitude on the leading edge (ns), or None."""
    amp  = wf["amp"]
    time = wf["time"]

    if wf.get("amplitude") is None:
        return None

    thr_low = -rt_low  * wf["amplitude"]
    thr_up  = -rt_high * wf["amplitude"]

    peak_idx = np.argmin(amp)
    i_thr    = peak_idx

    found_low = False
    found_up  = False
    tlow      = None
    tup       = None

    while i_thr > 1:
        a1 = amp[i_thr]
        a2 = amp[i_thr+1]
        t1 = time[i_thr]
        t2 = time[i_thr+1]

        if abs(t2 - t1) < 1e-12:
            i_thr -= 1
            continue

        m = (a2 - a1) / (t2 - t1)
        q = a1 - m * t1

        if not found_up and thr_up > a2 and thr_up <= a1:
            found_up = True
            tup = (thr_up - q) / m

        if not found_low and thr_low > a2 and thr_low <= a1:
            found_low = True
            tlow = (thr_low - q) / m

        if found_low and found_up:
            break
        i_thr -= 1

    if tup is not None and tlow is not None:
        if tup < tlow:
            return None
        return tup - tlow
    return None


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def build_st_sum(events, step=CFD_SAMPLE):
    """
    Sum of the valid (non pile-up) ST waveforms of each event, after resampling
    them with a cubic interpolation on a common grid with the given step (ns).
    Adds ev["ST_sum"] = {"time", "amp", "amplitude", "noise", "n_summed", "time_cfd"}.
    """
    from baseline_wf import BASELINE_START, BASELINE_END

    for ev in events:
        valid_wfs = [wf for wf in ev["ST"] if wf["signal"]]

        if not valid_wfs:
            ev["ST_sum"] = None
            continue

        # common grid: overlap of the time axes of all the valid waveforms
        t_start = max(wf["time"][0]  for wf in valid_wfs)
        t_end   = min(wf["time"][-1] for wf in valid_wfs)
        t_common = np.arange(t_start, t_end, step)

        amp_sum = np.zeros(len(t_common))
        for wf in valid_wfs:
            interp = interp1d(wf["time"], wf["amp"],
                              kind="cubic", bounds_error=False, fill_value=0.0)
            amp_sum += interp(t_common)

        bl_mask = (t_common >= valid_wfs[0]["time"][BASELINE_START]) & \
                  (t_common <= valid_wfs[0]["time"][BASELINE_END])
        noise = np.std(amp_sum[bl_mask]) if bl_mask.any() else 0.0

        ev["ST_sum"] = {
            "time":      t_common,
            "amp":       amp_sum,
            "amplitude": np.abs(np.min(amp_sum)),
            "noise":     noise,
            "n_summed":  len(valid_wfs),
            "time_cfd":  None,                 # filled by apply_cfd
        }

    return events


def apply_cfd(events, method_st="simple", method_tw="bipolar", **kwargs):
    """
    CFD time of every ST waveform, of ST_sum and of every TW waveform with signal.
    method_st, method_tw: "simple", "bipolar" or "tangent".

    Adds wf["time_cfd"] (ns, None if not found) and, for TW, wf["rise_time"] (ns).
    Returns events, times_TW, times_ST, times_ST_sum.
    """
    _cfd_map = {
        "simple":  _compute_time_simple,
        "bipolar": _compute_time_bipolar,
        "tangent": _compute_time_tangent,
    }
    if method_st not in _cfd_map:
        raise ValueError(f"method_st must be 'simple', 'bipolar' or 'tangent'")
    if method_tw not in _cfd_map:
        raise ValueError(f"method_tw must be 'simple', 'bipolar' or 'tangent'")

    _cfd_st = _cfd_map[method_st]
    _cfd_tw = _cfd_map[method_tw]

    for ev in events:
        for wf in ev["ST"]:
            wf["time_cfd"] = _cfd_st(wf)
        st_sum = ev.get("ST_sum")
        if st_sum:
            st_sum["time_cfd"] = _cfd_st(st_sum)
        for wf in ev["TW"]:
            if not wf.get("signal"):
                wf["time_cfd"] = None
                wf["rise_time"] = None
                continue
            wf["time_cfd"]  = _cfd_tw(wf, **kwargs)
            wf["rise_time"] = _compute_rise_time(wf)

    times_ST = [wf["time_cfd"] for ev in events for wf in ev["ST"]
                if wf["time_cfd"] is not None]
    times_ST_sum = [ev["ST_sum"]["time_cfd"] for ev in events
                    if ev.get("ST_sum") and ev["ST_sum"].get("time_cfd") is not None]
    times_TW = [wf["time_cfd"] for ev in events for wf in ev["TW"]
                if wf["time_cfd"] is not None and wf["signal"] and wf["time_cfd"]>0]

    print(f"\n--- CFD statistics (ST={method_st}, TW={method_tw}) ---")
    print(f"ST : {len(times_ST)} waveforms with a time")
    if times_ST:
        print(f"     mean={np.mean(times_ST):.2f} ns,  std={np.std(times_ST):.2f} ns")
    print(f"ST_sum : {len(times_ST_sum)} events with a time")
    if times_ST_sum:
        print(f"     mean={np.mean(times_ST_sum):.2f} ns,  std={np.std(times_ST_sum):.2f} ns")
    print(f"TW : {len(times_TW)} waveforms with a time (signal=True)")
    if times_TW:
        print(f"     mean={np.mean(times_TW):.2f} ns,  std={np.std(times_TW):.2f} ns")

    return events, times_TW, times_ST, times_ST_sum


# =============================================================================
# PLOTS
# =============================================================================

def plot_st_sum_check(events, ev_idx):
    """ST sum waveform of one event, with its CFD time."""
    st_sum = events[ev_idx].get("ST_sum")
    if st_sum is None:
        print(f"ST_sum not available for event {ev_idx}.")
        return

    t   = st_sum["time"]
    amp = st_sum["amp"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, amp*1e3, lw=1.5, color="royalblue")
    ax.axhline(0, color="black", lw=1, linestyle="--")
    if st_sum["time_cfd"] is not None:
        ax.axvline(st_sum["time_cfd"], color="red", lw=2, linestyle="--",
                   label=f"t_CFD = {st_sum['time_cfd']:.2f} ns")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
    ax.tick_params(axis='both')
    plt.tight_layout()
    plt.show()


def plot_cfd_methods(events, ev_idx, wf_idx=None, frac=SCFD_FRAC,
                     tleft=CFD_TLEFT, tright=CFD_TRIGHT, step=CFD_SAMPLE,
                     xlim=None, save=None):
    """
    The three CFD methods on the same TW waveform: simple and tangent (top),
    bipolar signal (bottom). To be called after apply_cfd (uses the rise time).
    """
    ev = events[ev_idx]
    wf = next(w for w in ev["TW"] if w.get("signal")) if wf_idx is None else ev["TW"][wf_idx]
    t, a = wf["time"], wf["amp"]
    thr = -frac * wf["amplitude"]

    t_simple  = _compute_time_simple(wf, frac)
    t_tangent = _compute_time_tangent(wf, frac)
    t_bipolar = _compute_time_bipolar(wf, frac=frac)

    # tangent line through the same 4 samples used by _compute_time_tangent
    i = np.argmin(a)
    while i > 0 and not (thr > a[i+1] and thr <= a[i]):
        i -= 1
    idx = np.arange(i-1, i+3)
    m, q = np.polyfit(t[idx], a[idx], 1)

    # bipolar signal
    rt = wf.get("rise_time")
    delay = rt if (rt is not None and rt > 0) else CFD_DEL
    interp = interp1d(t, a, kind="cubic", bounds_error=False, fill_value=0.0)
    tp = t[np.argmin(a)]
    tb = np.arange(tp + tleft, tp + tright, step)
    sb = frac * interp(tb) - interp(tb - delay)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(t, a*1e3, "o-", ms=3, lw=1.2, color="steelblue", label="waveform")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.axhline(thr*1e3, color="orange", ls="--", lw=1.5, label=f"threshold ({frac:.0%})")
    ax1.axvline(t_simple, color="orange", lw=1.5, label=f"simple: {t_simple:.2f} ns")
    tl = np.linspace(-q/m - 1, t[idx[-1]] + 1, 100)
    ax1.plot(tl, (m*tl + q)*1e3, color="crimson", lw=1.8, label="tangent line")
    ax1.plot(t[idx], a[idx]*1e3, "s", ms=6, mfc="none", color="crimson")
    ax1.axvline(t_tangent, color="crimson", ls=":", lw=1.8, label=f"tangent: {t_tangent:.2f} ns")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(tb, sb*1e3, color="seagreen", lw=1.8,
             label=rf"$S(t) = f\,A(t) - A(t-\delta)$, $\delta$ = {delay:.1f} ns")
    ax2.axhline(0, color="gray", lw=0.8)
    if t_bipolar is not None:
        ax2.axvline(t_bipolar, color="seagreen", ls="--", lw=1.5, label=f"bipolar: {t_bipolar:.2f} ns")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("S(t) (mV)")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax1.set_xlim(xlim if xlim is not None else (tp - 15, tp + 10))
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_cfd_bipolar_check(events, det, ev_idx, wf_idx,
                           frac=CFD_FRAC, delay=CFD_DEL,
                           tleft=CFD_TLEFT, tright=CFD_TRIGHT, step=CFD_SAMPLE):
    """Waveform and bipolar CFD signal of one waveform, with the zero crossing."""
    wf = events[ev_idx][det][wf_idx]
    amp  = wf["amp"]
    time = wf["time"]

    interp_f = interp1d(time, amp, kind="cubic", bounds_error=False, fill_value=0.0)
    peak_idx = np.argmin(amp)
    timepeak = time[peak_idx]
    t_bipolar    = np.arange(timepeak + tleft, timepeak + tright, step)
    amp_bipolar  = frac * interp_f(t_bipolar) - interp_f(t_bipolar - delay)

    t_zero = None
    max_bin = np.argmax(amp_bipolar[2:]) + 2
    i = max_bin
    while i > 0:
        if amp_bipolar[i] * amp_bipolar[i-1] < 0:
            t1, t2 = t_bipolar[i-1], t_bipolar[i]
            a1, a2 = amp_bipolar[i-1], amp_bipolar[i]
            m = (a2-a1)/(t2-t1); q = a2-m*t2
            t_zero = -q/m
            break
        i -= 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    fig.suptitle(f"{det} — Event {events[ev_idx]['eventN']}, "
                 f"Board {wf['boardId']}, Ch {wf['channelId']}", fontsize=13)

    axes[0].plot(time, amp*1e3, lw=1.5, color="steelblue", label="waveform")
    axes[0].axhline(0, color="black", lw=1, linestyle="--")
    if t_zero:
        axes[0].axvline(t_zero, color="red", lw=2, linestyle="--",
                        label=f"t_CFD = {t_zero:.2f} ns")
    axes[0].set_ylabel("Amplitude (mV)", fontsize=13)
    axes[0].set_title("Waveform", fontsize=12)
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.4)
    axes[0].tick_params(axis='both', labelsize=12)

    axes[1].plot(t_bipolar, amp_bipolar*1e3, lw=1.5, color="darkorange",
                 label=f"CFD: {frac}×amp(t) − amp(t−{delay}ns)")
    axes[1].axhline(0, color="black", lw=1.5, linestyle="--", label="zero")
    if t_zero:
        axes[1].axvline(t_zero, color="red", lw=2, linestyle="--",
                        label=f"zero crossing = {t_zero:.2f} ns")
    axes[1].set_xlabel("Time (ns)", fontsize=13)
    axes[1].set_ylabel("CFD amplitude (mV)", fontsize=13)
    axes[1].set_title(f"Bipolar CFD signal (frac={frac}, delay={delay} ns)", fontsize=12)
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.4)
    axes[1].tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_simple_cfd_check(events, det, ev_idx, wf_idx):
    """Waveform with threshold and simple CFD time."""
    wf = events[ev_idx][det][wf_idx]
    if wf.get("time_cfd") is None:
        print("No CFD time for this waveform.")
        return

    t   = wf["time"]
    amp = wf["amp"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, amp*1e3, lw=1.5, label="waveform")
    ax.axvline(wf["time_cfd"], color="red", lw=2, linestyle="--",
               label=f"t_CFD = {wf['time_cfd']:.2f} ns")
    ax.axhline(0, color="black", lw=1, linestyle="--")

    thr = - SCFD_FRAC * wf["amplitude"] * 1e3
    ax.axhline(thr, color="orange", lw=1.5, linestyle="--",
               label=f"threshold = {thr:.1f} mV")

    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Amplitude (mV)", fontsize=14)
    ax.set_title(f"{det} — Event {events[ev_idx]['eventN']}, "
                 f"Board {wf['boardId']}, Ch {wf['channelId']} — Simple CFD", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.show()


def plot_tangent_cfd_check(events, det, ev_idx, wf_idx, frac=SCFD_FRAC):
    """Waveform with threshold, the 4 fitted samples, the tangent line and its zero crossing."""
    wf = events[ev_idx][det][wf_idx]

    amp  = wf["amp"]
    time = wf["time"]

    if wf.get("amplitude") is None:
        print("Invalid amplitude.")
        return

    threshold = -frac * wf["amplitude"]

    i_thr = np.argmin(amp)
    found = False
    while i_thr > 0:
        if threshold > amp[i_thr + 1] and threshold <= amp[i_thr]:
            found = True
            break
        i_thr -= 1

    if not found:
        print("Threshold crossing not found.")
        return

    idx_range = [i for i in range(i_thr - 1, i_thr + 3)]
    t_pts = np.array([time[i] for i in idx_range])
    a_pts = np.array([amp[i]  for i in idx_range])

    x   = np.mean(t_pts)
    y   = np.mean(a_pts)
    xy  = np.mean(t_pts * a_pts)
    xsq = np.mean(t_pts**2)

    denom = xsq - x*x
    if abs(denom) < 1e-12:
        print("Invalid fit.")
        return

    m = (xy - x*y) / denom
    q = (xsq*y - x*xy) / denom

    if abs(m) < 1e-12:
        print("Zero slope.")
        return

    t_cfd = -q / m

    t_fit = np.linspace(t_pts.min() - 1, t_pts.max() + 1, 200)
    a_fit = m * t_fit + q

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(time, amp * 1e3, lw=1.5, color="steelblue", label="waveform")
    ax.axhline(threshold * 1e3, color="orange", lw=1.5, linestyle="--",
               label=f"CFD threshold = {threshold*1e3:.1f} mV")
    ax.scatter(t_pts, a_pts * 1e3, color="darkgreen", s=20, zorder=5)
    ax.plot(t_fit, a_fit * 1e3, color="crimson", lw=2, label="tangent line")
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.axvline(t_cfd, color="red", lw=2, linestyle="--", label=f"t_CFD = {t_cfd:.2f} ns")
    ax.scatter([t_cfd], [0], color="red", s=20, zorder=6)

    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Amplitude (mV)", fontsize=14)
    ax.set_title(f"{det} — Event {events[ev_idx]['eventN']}, "
                 f"Board {wf['boardId']}, Ch {wf['channelId']} — Tangent CFD", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.show()


def plot_cfd_times_wf(times_TW, times_ST):
    """Histograms of the CFD times of the single ST and TW waveforms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CFD time distribution", fontsize=16)

    axes[0].hist(times_ST, bins=100, color="royalblue")
    axes[0].set_xlabel("ST CFD time (ns)", fontsize=14)
    axes[0].set_ylabel("Counts", fontsize=14)
    axes[0].set_title(f"ST — single waveforms ({len(times_ST)} entries)", fontsize=14)
    axes[0].tick_params(axis='both', labelsize=13)
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(times_TW, bins=100, color="crimson")
    axes[1].set_xlabel("TW CFD time (ns)", fontsize=14)
    axes[1].set_ylabel("Counts", fontsize=14)
    axes[1].set_title(f"TW with signal ({len(times_TW)} entries)", fontsize=14)
    axes[1].tick_params(axis='both', labelsize=13)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_rise_time_TW(events):
    """Rise time histogram of the single TW waveforms."""
    rt_TW = [wf["rise_time"] for ev in events for wf in ev["TW"]
             if wf.get("rise_time") is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.array(rt_TW), bins=100, range=(0, 10), color="steelblue", alpha=0.8)
    ax.set_xlabel("Rise time (ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(f"TW rise time — single waveforms ({len(rt_TW)} entries)", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def anomalous_wf(events, mode="cfd", threshold=250.0, n_plot=9, cols=3):
    """
    Grid of anomalous TW waveforms:
        mode="cfd"       time_cfd  > threshold (ns)
        mode="risetime"  rise_time < threshold (ns)
    """
    anomalous = []
    for ev in events:
        for wf in ev["TW"]:
            if not wf.get("signal"):
                continue
            if mode == "cfd":
                val = wf.get("time_cfd")
                if val is not None and val > threshold:
                    anomalous.append((ev, wf, val))
            elif mode == "risetime":
                val = wf.get("rise_time")
                if val is not None and val < threshold:
                    anomalous.append((ev, wf, val))

    print(f"Anomalous waveforms ({mode}, threshold={threshold} ns): {len(anomalous)}")
    if not anomalous:
        return

    subset = anomalous[:n_plot]
    rows   = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows))
    fig.suptitle(f"Anomalous TW waveforms — {mode} {'>' if mode=='cfd' else '<'} {threshold} ns  "
                 f"[{len(anomalous)} total, {len(subset)} shown]", fontsize=13)
    axes = np.array(axes).flatten()

    for idx, (ev, wf, val) in enumerate(subset):
        ax  = axes[idx]
        ax.plot(wf["time"], wf["amp"] * 1e3, color="steelblue", lw=1.2)

        t_cfd = wf.get("time_cfd")
        if t_cfd is not None:
            ax.axvline(t_cfd, color="red", lw=1.5, linestyle="--",
                       label=f"t_cfd={t_cfd:.1f} ns")

        amp_val = wf.get("amplitude")
        if amp_val is not None:
            ax.axhline(-amp_val * 1e3, color="gray", lw=1.0, linestyle=":",
                       label=f"amp={amp_val*1e3:.1f} mV")

        rt = wf.get("rise_time")
        label_val = f"{mode}={val:.2f} ns"
        if rt is not None:
            label_val += f"  rt={rt:.2f} ns"

        ax.set_title(f"ev={ev['eventN']}  B={wf['boardId']} ch={wf['channelId']}\n"
                     f"{label_val}", fontsize=8)
        ax.set_xlabel("t (ns)", fontsize=8)
        ax.set_ylabel("amp (mV)", fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    for idx in range(len(subset), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFD methods: check plots on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--event", type=int, default=0, help="Event index")
    parser.add_argument("--save", default=None, help="Save the CFD-methods figure to this file")
    args = parser.parse_args()

    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="median")
    cleanup_baseline(events)

    events, amp_ST, amp_TW = apply_amplitude(events)
    events = filter_pileup_events(events)
    events = build_st_sum(events)

    events, times_TW, times_ST, times_ST_sum = apply_cfd(events, method_st="simple", method_tw="simple")

    plot_cfd_methods(events, ev_idx=args.event, save=args.save)
    plot_st_sum_check(events, ev_idx=args.event)
    plot_cfd_times_wf(times_TW, times_ST)
    plot_rise_time_TW(events)
