"""
baseline_wf.py — Baseline subtraction, noise and pedestal for ST and TW waveforms.

The baseline is estimated on the samples [BASELINE_START, BASELINE_END), well
before the signal (~64 ns), and subtracted from the whole waveform. The
standard deviation of the same samples is the noise of the channel (sigma_noise),
used later for the signal threshold and in the chi2 of the template fit.

The pedestal is the integral of the baseline region, rescaled to the length of
the charge integration window: it is subtracted from the charge to remove the
residual offset left by the baseline subtraction.

Usage:
    python3 baseline_wf.py <file.bin> [--event N] [--wf-st K] [--wf-tw K]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from read_wf import load_waveforms

# =============================================================================
# PARAMETERS
# =============================================================================

# baseline window: samples [5, 205) -> 200 samples, ~64 ns
BASELINE_START = 5
BASELINE_END   = 205

# pedestal window (same as the baseline)
PEDESTAL_START = 5
PEDESTAL_END   = 205

# Only samples below this value (V) enter the pedestal. The value is well above
# any baseline fluctuation, so in practice all samples are used.
MEAN_OFFSET = 0.0008816 * 1e3

# maximum allowed |amplitude| in the baseline window (used by plot_invalid_baseline)
V_TH_BASELINE = {
    "ST": 0.100,   # 100 mV
    "TW": 0.008,   #   8 mV
}

# last samples excluded from the charge integration (end-of-window artefacts)
FINAL_CUT = 50


# =============================================================================
# SINGLE-WAVEFORM FUNCTIONS
# =============================================================================

def _apply_baseline(wf, det, method):
    """
    Compute and subtract the baseline of one waveform.
    method = "mean" or "median" over [BASELINE_START, BASELINE_END).
    Adds/updates: _amp_orig, _bl, noise, amp, pedestal.
    """
    amp     = wf["amp"]
    samples = amp[BASELINE_START:BASELINE_END]

    bl  = np.mean(samples) if method == "mean" else np.median(samples)
    std = np.std(samples)

    wf["_amp_orig"] = amp.copy()             # original waveform (for plots)
    wf["_bl"]       = bl                     # baseline value (V)
    wf["noise"]     = std                    # sigma_noise (V)
    wf["amp"]       = amp - bl               # baseline-subtracted waveform
    wf["pedestal"]  = _compute_pedestal(wf)  # V*ns

    return bl, std


def _compute_pedestal(wf, thr=0.0):
    """
    Trapezoidal integral of the pedestal window, rescaled to the length of the
    charge integration window (whole waveform minus the last FINAL_CUT samples).
    Returns the pedestal in V*ns, with the same sign convention as the charge.
    """
    amp       = wf["amp"]
    time      = wf["time"]
    threshold = MEAN_OFFSET + thr

    pedestal = 0.0

    for j in range(PEDESTAL_START, PEDESTAL_END):
        a1, a2 = amp[j],  amp[j+1]
        t1, t2 = time[j], time[j+1]

        if a1 * a2 < 0:   # sign change within the interval: two triangles
            m     = (a2 - a1) / (t2 - t1)
            q     = a2 - m * t2
            thalf = -q / m
            dq    = 0.5*a1*(thalf-t1) + 0.5*a2*(t2-thalf)
        else:             # same sign: trapezoid
            dq = 0.5 * (a1 + a2) * (t2 - t1)

        if amp[j] < threshold:
            pedestal += dq

    return -(pedestal * (len(amp)-FINAL_CUT) / (PEDESTAL_END-PEDESTAL_START))


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def apply_baseline(events, method="mean"):
    """
    Compute and subtract the baseline of all ST and TW waveforms.

    Adds to every waveform:
        wf["amp"]       baseline-subtracted amplitude
        wf["noise"]     sigma_noise (std of the baseline samples)
        wf["pedestal"]  pedestal (V*ns), used in the charge computation
        wf["_amp_orig"] original amplitude  (temporary, removed by cleanup_baseline)
        wf["_bl"]       baseline value      (temporary, removed by cleanup_baseline)

    Returns: events, bl_ST, std_ST, bl_TW, std_TW
    """
    bl_ST, std_ST, bl_TW, std_TW = [], [], [], []

    for ev in events:
        for det in ["ST", "TW"]:
            for wf in ev[det]:
                bl, std = _apply_baseline(wf, det, method)

                if det == "ST":
                    bl_ST.append(bl); std_ST.append(std)
                else:
                    bl_TW.append(bl); std_TW.append(std)

    print(f"\n--- Baseline statistics ({method}) ---")
    print(f"ST baseline — {method}: {np.mean(bl_ST)*1e3:.2f} mV,  "
          f"spread: {np.std(bl_ST)*1e3:.2f} mV,  "
          f"noise: {np.mean(std_ST)*1e3:.2f} mV")
    print(f"TW baseline — {method}: {np.mean(bl_TW)*1e3:.2f} mV,  "
          f"spread: {np.std(bl_TW)*1e3:.2f} mV,  "
          f"noise: {np.mean(std_TW)*1e3:.2f} mV")

    ped_ST = [wf["pedestal"] for ev in events for wf in ev["ST"]]
    print(f"ST pedestal — mean: {np.mean(ped_ST)*1e3:.4f} mV·ns,  "
          f"std: {np.std(ped_ST)*1e3:.4f} mV·ns")

    ped_TW = [wf["pedestal"] for ev in events for wf in ev["TW"]]
    print(f"TW pedestal — mean: {np.mean(ped_TW)*1e3:.4f} mV·ns,  "
          f"std: {np.std(ped_TW)*1e3:.4f} mV·ns")

    return events, bl_ST, std_ST, bl_TW, std_TW


def cleanup_baseline(events):
    """Remove the temporary fields _amp_orig and _bl (noise and pedestal are kept)."""
    for ev in events:
        for det in ["ST", "TW"]:
            for wf in ev[det]:
                wf.pop("_amp_orig", None)
                wf.pop("_bl", None)
    return events


# =============================================================================
# PLOTS (to be called before cleanup_baseline)
# =============================================================================

def plot_invalid_baseline(events, n_wfs=8):
    """First n_wfs ST and TW waveforms with a sample above V_TH_BASELINE in the baseline window."""
    invalid_ST = []
    invalid_TW = []

    for ev in events:
        for det, lst in [("ST", invalid_ST), ("TW", invalid_TW)]:
            if len(lst) >= n_wfs:
                continue
            for wf in ev[det]:
                if len(lst) >= n_wfs:
                    break
                _orig   = wf.get("_amp_orig")
                amp     = _orig if _orig is not None else wf.get("amp")
                samples = amp[BASELINE_START:BASELINE_END]
                if np.any(np.abs(samples) > V_TH_BASELINE[det]):
                    lst.append((ev["eventN"], wf, amp))

        if len(invalid_ST) >= n_wfs and len(invalid_TW) >= n_wfs:
            break

    for det, lst in [("ST", invalid_ST), ("TW", invalid_TW)]:
        if not lst:
            print(f"No {det} waveform with invalid baseline found.")
            continue

        fig, axes = plt.subplots(4, 2, figsize=(14, 13))
        fig.suptitle(f"{det} — waveforms with invalid baseline "
                     f"(window [{BASELINE_START}:{BASELINE_END}], "
                     f"V_TH={V_TH_BASELINE[det]*1e3:.0f} mV)", fontsize=13)
        axes_flat = axes.flatten()

        for i in range(8):
            ax = axes_flat[i]
            if i >= len(lst):
                ax.set_visible(False)
                continue

            ev_n, wf, amp = lst[i]
            t       = wf["time"]
            samples = amp[BASELINE_START:BASELINE_END]
            thr     = V_TH_BASELINE[det]

            ax.axvspan(t[BASELINE_START], t[BASELINE_END - 1],
                       alpha=0.15, color="green", label="baseline window")
            ax.axhline( thr * 1e3, color="red", lw=1.2, linestyle="--",
                        label=f"±V_TH = {thr*1e3:.0f} mV")
            ax.axhline(-thr * 1e3, color="red", lw=1.2, linestyle="--")

            ax.plot(t, amp * 1e3, color="steelblue", lw=1.0, alpha=0.9)

            mask    = np.abs(samples) > thr
            idx_bad = np.where(mask)[0] + BASELINE_START
            ax.scatter(t[idx_bad], amp[idx_bad] * 1e3,
                       color="orange", s=25, zorder=5,
                       label=f"{mask.sum()} samples above threshold")

            ax.set_title(f"ev {ev_n} — Board {wf['boardId']} Ch {wf['channelId']}",
                         fontsize=10)
            ax.set_xlabel("Time (ns)", fontsize=9)
            ax.set_ylabel("Amp (mV)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_baseline_check(events, det, ev_idx, wf_idx):
    """One waveform before and after the baseline subtraction."""
    wf = events[ev_idx][det][wf_idx]

    t        = wf["time"]
    amp_orig = wf["_amp_orig"]
    amp_corr = wf["amp"]
    bl       = wf["_bl"]
    std      = wf["noise"]
    ped      = wf["pedestal"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{det} — Event {events[ev_idx]['eventN']}, "
                 f"Board {wf['boardId']}, Ch {wf['channelId']} | "
                 f"pedestal = {ped*1e3:.4f} mV·ns", fontsize=13)

    for ax in axes:
        ax.axvspan(t[BASELINE_START], t[BASELINE_END-1],
                   alpha=0.2, color="green", label="baseline window")
        ax.axvspan(t[PEDESTAL_START], t[PEDESTAL_END-1],
                   alpha=0.3, color="purple", label="pedestal window")
        ax.set_xlabel("Time (ns)", fontsize=14)
        ax.set_ylabel("Amplitude (mV)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=13)

    axes[0].plot(t, amp_orig*1e3, lw=1.5)
    axes[0].axhline(bl*1e3, color="red", lw=1.8, linestyle="--",
                    label=f"baseline = {bl*1e3:.2f} mV")
    axes[0].axhline((bl+std)*1e3, color="orange", lw=1.5, linestyle="--",
                    label=f"±std = {std*1e3:.2f} mV")
    axes[0].axhline((bl-std)*1e3, color="orange", lw=1.5, linestyle="--")
    axes[0].set_title("Before subtraction", fontsize=14)
    axes[0].legend(fontsize=10)

    axes[1].plot(t, amp_corr*1e3, lw=1.5)
    axes[1].axhline(0, color="black", lw=1.8, linestyle="--")
    axes[1].set_title("After subtraction", fontsize=14)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_baseline_check_st_tw(events, ev_idx, wf_idx_st, wf_idx_tw,
                              ylim_st=None, ylim_tw=None, save=None):
    """
    ST and TW waveforms of the same event, side by side, before the baseline
    subtraction, with the baseline window, the baseline value and the ±sigma band.
    ylim_st, ylim_tw: (min, max) in mV to zoom on the baseline.
    save: output file (optional).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, det, wf_idx, ylim in ((axes[0], "ST", wf_idx_st, ylim_st),
                                  (axes[1], "TW", wf_idx_tw, ylim_tw)):
        wf  = events[ev_idx][det][wf_idx]
        t   = wf["time"]
        amp = wf["_amp_orig"] * 1e3     # mV
        bl  = wf["_bl"] * 1e3
        std = wf["noise"] * 1e3

        ax.axvspan(t[BASELINE_START], t[BASELINE_END - 1],
                   alpha=0.2, color="green", label="baseline window")
        ax.plot(t, amp, lw=1.2)
        ax.axhline(bl, color="red", lw=1.8, linestyle="--",
                   label=f"baseline = {bl:.2f} mV")
        ax.axhline(bl + std, color="orange", lw=1.5, linestyle="--",
                   label=f"$\\pm\\sigma$ = {std:.2f} mV")
        ax.axhline(bl - std, color="orange", lw=1.5, linestyle="--")

        ax.set_title(f"{det} — board {wf['boardId']}, channel {wf['channelId']}")
        ax.set_xlabel("Time (ns)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel("Amplitude (mV)")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_pedestal(events):
    """Histogram of the TW pedestals."""
    ped_TW = [wf["pedestal"] for ev in events for wf in ev["TW"] if wf["pedestal"] is not None]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.array(ped_TW)*1e3, bins=50, color="crimson")
    ax.set_xlabel("TW pedestal (mV·ns)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title("TW pedestal distribution", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline subtraction: check plots on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--event", type=int, default=0, help="Event index")
    parser.add_argument("--wf-st", type=int, default=0, help="ST waveform index")
    parser.add_argument("--wf-tw", type=int, default=0, help="TW waveform index")
    parser.add_argument("--save", default=None, help="Save the ST/TW figure to this file")
    args = parser.parse_args()

    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="mean")

    plot_baseline_check_st_tw(events, ev_idx=args.event, wf_idx_st=args.wf_st, wf_idx_tw=args.wf_tw,
                              ylim_st=(-60, 20), ylim_tw=(-30, 10), save=args.save)
    plot_pedestal(events)
