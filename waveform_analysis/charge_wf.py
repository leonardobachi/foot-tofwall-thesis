"""
charge_wf.py — Charge of the TW waveforms.

The charge is the trapezoidal integral of the baseline-subtracted waveform over
the whole acquisition window except the last FINAL_CUT samples (end-of-window
artefacts), minus the pedestal computed in baseline_wf.py. Intervals where the
waveform changes sign are split into two triangles. The result is positive,
in V*ns.

Usage:
    python3 charge_wf.py <file.bin> [--event N] [--save FILE]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from read_wf      import load_waveforms
from baseline_wf  import apply_baseline, cleanup_baseline
from amplitude_wf import apply_amplitude, filter_pileup_events
from cfd_wf       import apply_cfd

# =============================================================================
# PARAMETERS
# =============================================================================

# Only samples below this value (V) are integrated. The value is well above
# any baseline fluctuation, so in practice all samples are used.
MEAN_OFFSET = 0.0008816 * 1e3

FINAL_CUT = 50   # last samples excluded from the integral


# =============================================================================
# SINGLE-WAVEFORM FUNCTIONS
# =============================================================================

def _compute_charge_integral(wf, thr=0.0):
    """Trapezoidal integral of the waveform minus the pedestal, in V*ns (positive)."""
    amp     = wf["amp"]
    time    = wf["time"]
    thr_abs = MEAN_OFFSET + thr
    charge  = 0.0

    for j in range(0, len(amp) - FINAL_CUT):
        a1, a2 = amp[j],  amp[j+1]
        t1, t2 = time[j], time[j+1]

        if a1 > thr_abs:
            continue

        if a1 * a2 < 0:   # sign change within the interval: two triangles
            m     = (a2 - a1) / (t2 - t1)
            q     = a2 - m * t2
            thalf = -q / m
            dq    = 0.5*a1*(thalf-t1) + 0.5*a2*(t2-thalf)
        else:             # same sign: trapezoid
            dq = 0.5 * (a1 + a2) * (t2 - t1)

        if amp[j] < thr_abs:
            charge += dq

    return -(charge + wf["pedestal"])


def _compute_charge_amplitude(wf):
    """Peak amplitude used as charge estimator (V, not V*ns)."""
    if wf.get("amplitude") is None:
        return None
    return wf["amplitude"]


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def apply_charge(events, thr=0.0, method="integral"):
    """
    Charge of every TW waveform with signal.
    method: "integral" (default) or "amplitude".
    Adds wf["charge"] (None if signal=False). Returns events, charge_TW.
    """
    if method not in ("integral", "amplitude"):
        raise ValueError("method must be 'integral' or 'amplitude'")

    for ev in events:
        for wf in ev["TW"]:
            if not wf.get("signal"):
                wf["charge"] = None
                continue
            if method == "integral":
                wf["charge"] = _compute_charge_integral(wf, thr)
            else:
                wf["charge"] = _compute_charge_amplitude(wf)

    charge_TW = [wf["charge"] for ev in events for wf in ev["TW"]
                 if wf["charge"] is not None]
    charge_TW_neg = [wf["charge"] for ev in events for wf in ev["TW"]
                     if wf["charge"] is not None and wf["charge"] < 0]

    print(f"\n--- TW charge statistics (method={method}, waveforms with signal) ---")
    print(f"Waveforms with charge: {len(charge_TW)}")
    if charge_TW:
        print(f"min={np.min(charge_TW)*1e3:.2f},  max={np.max(charge_TW)*1e3:.2f},  "
              f"mean={np.mean(charge_TW)*1e3:.2f},  std={np.std(charge_TW)*1e3:.2f}  (mV·ns)")
        print(f"Waveforms with negative charge: {len(charge_TW_neg)}")

    return events, charge_TW


# =============================================================================
# PLOTS
# =============================================================================

def plot_charge_wf(charge_TW):
    """Charge histogram of the single TW waveforms."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.array(charge_TW)*1e3, bins=100, color="crimson", alpha=0.8)
    ax.set_xlabel("Single-waveform charge (mV·ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(f"TW charge — single waveform, entries = {len(charge_TW)}", fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_charge_check(events, ev_idx, wf_idx=None, xlim=None, save=None):
    """
    Integrated area of one TW waveform (the last FINAL_CUT samples are excluded).
    wf_idx=None: first TW waveform with signal in the event.
    """
    ev = events[ev_idx]
    if wf_idx is None:
        wf = next(w for w in ev["TW"] if w.get("signal"))
    else:
        wf = ev["TW"][wf_idx]

    t = wf["time"]
    a = wf["amp"] * 1e3                 # mV
    n = len(a) - FINAL_CUT              # last integrated sample

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, a, lw=1.2, color="steelblue")
    ax.fill_between(t[:n], a[:n], 0, where=a[:n] < 0, color="orange", alpha=0.5,
                    label=f"Integrated area: $Q$ = {wf['charge']*1e3:.0f} mV$\\cdot$ns")
    ax.axhline(0, color="black", lw=0.8)

    ax.set_title(f"TW — board {wf['boardId']}, channel {wf['channelId']}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (mV)")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def find_double_peak_wfs(events, det="ST", n_wfs=6, min_amp_ratio=0.3, min_separation_ns=10.0, amp_min=0.100):
    """
    Look for waveforms of detector det with two significant peaks (pile-up check):
    a second local minimum deeper than min_amp_ratio times the main peak and at
    least min_separation_ns away from it. Plots the first n_wfs candidates,
    sorted by the ratio between the second and the main peak.
    """
    from scipy.signal import find_peaks

    candidates = []

    for ev in events:
        for wf in ev[det]:
            if not wf.get("signal"):
                continue
            amp      = wf["amp"]
            time     = wf["time"]
            amp_main = wf.get("amplitude", 0)

            if amp_main is None or amp_main < amp_min:
                continue

            # find_peaks looks for maxima: invert the (negative) signal
            amp_inv = -amp[5:len(amp)-FINAL_CUT]
            height_thr = min_amp_ratio * amp_main

            peaks_idx, props = find_peaks(
                amp_inv,
                height=height_thr,
                distance=int(min_separation_ns / (time[1] - time[0])),
                prominence=height_thr * 0.5
            )

            if len(peaks_idx) >= 2:
                peak_amps = sorted(amp_inv[peaks_idx], reverse=True)
                ratio = peak_amps[1] / peak_amps[0]
                candidates.append((ratio, ev["eventN"], wf, len(peaks_idx)))

    candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"\n--- {det} waveforms with a double peak ---")
    print(f"Found: {len(candidates)}  "
          f"(out of {sum(1 for ev in events for wf in ev[det] if wf.get('signal'))} with signal)")
    print(f"Parameters: min_amp_ratio={min_amp_ratio}, min_separation={min_separation_ns} ns, amp_min={amp_min}")
    if candidates:
        print(f"Top 5 by second/main peak ratio:")
        for ratio, ev_n, wf, n_peaks in candidates[:5]:
            print(f"  ev={ev_n} | board={wf['boardId']} ch={wf['channelId']} | "
                  f"amp={wf.get('amplitude',0)*1e3:.1f} mV | "
                  f"n_peaks={n_peaks} | ratio={ratio:.2f}")

    if not candidates:
        print("No waveform with a double peak found.")
        return candidates

    n_plot = min(n_wfs, len(candidates))
    nrows, ncols = (n_plot + 1) // 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5*nrows))
    fig.suptitle(f"{det} — {n_plot} waveforms with a double peak "
                 f"(min_amp_ratio={min_amp_ratio}, sep>{min_separation_ns} ns)",
                 fontsize=13)
    axes_flat = np.array(axes).flatten()

    for i, (ratio, ev_n, wf, n_peaks) in enumerate(candidates[:n_plot]):
        ax = axes_flat[i]
        ax.plot(wf["time"], wf["amp"]*1e3, color="steelblue", lw=1.0)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_title(
            f"ev {ev_n} | board {wf['boardId']} ch {wf['channelId']} | "
            f"amp={wf.get('amplitude',0)*1e3:.1f} mV | "
            f"n_peaks={n_peaks} | ratio={ratio:.2f}",
            fontsize=9)
        ax.set_xlabel("t (ns)", fontsize=9)
        ax.set_ylabel("amp (mV)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_plot, nrows*ncols):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TW charge: check plots on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--event", type=int, default=0, help="Event index")
    parser.add_argument("--save", default=None, help="Save the charge-example figure to this file")
    args = parser.parse_args()

    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="mean")
    cleanup_baseline(events)

    events, amp_ST, amp_TW = apply_amplitude(events)
    events = filter_pileup_events(events)

    events, *_ = apply_cfd(events, method_st="simple", method_tw="simple")
    events, charge_TW = apply_charge(events, method="integral")

    plot_charge_wf(charge_TW)
    plot_charge_check(events, ev_idx=args.event, save=args.save)
