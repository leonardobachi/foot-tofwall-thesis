"""
amplitude_wf.py — Amplitude, signal identification and ST pile-up rejection.

The amplitude is the absolute value of the (negative) peak of the
baseline-subtracted waveform. A TW waveform contains a signal if its amplitude
exceeds N_SIGMA times its own noise; an ST waveform is flagged as pile-up if
its amplitude exceeds ST_AMP_MAX, and events with at least one flagged ST
channel are discarded.

Usage:
    python3 amplitude_wf.py <file.bin> [--event N] [--save-dir DIR]
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from read_wf import load_waveforms
from baseline_wf import apply_baseline, cleanup_baseline

# =============================================================================
# PARAMETERS
# =============================================================================

N_SIGMA            = 8       # TW signal threshold, in units of sigma_noise

FIRST_VALID_SAMPLE = 5       # samples excluded at the beginning and at the end
LAST_VALID_SAMPLE  = 5       # of the waveform when searching for the peak

ST_AMP_MAX         = 0.600   # V — ST pile-up threshold


# =============================================================================
# FUNCTIONS
# =============================================================================

def apply_amplitude(events, n_sigma=N_SIGMA, noise_std=None, st_amp_max=ST_AMP_MAX):
    """
    Compute the amplitude of every waveform and flag the signals.
    Adds to every waveform:
        wf["amplitude"]  absolute value of the negative peak (V)
        wf["signal"]     TW: amplitude > n_sigma * noise
                         ST: amplitude < st_amp_max (False = pile-up)
    If noise_std is given, a fixed noise value is used instead of wf["noise"].
    Returns events, amp_ST, amp_TW (amplitudes of the waveforms with signal).
    """
    for ev in events:
        for det in ["ST", "TW"]:
            for wf in ev[det]:
                threshold = n_sigma * wf["noise"] if noise_std is None else n_sigma * noise_std
                wf["amplitude"] = np.abs(np.min(wf["amp"][FIRST_VALID_SAMPLE:len(wf["amp"])-LAST_VALID_SAMPLE]))
                if det == "TW":
                    wf["signal"] = (wf["amplitude"] > threshold)
                else:
                    wf["signal"] = (wf["amplitude"] < st_amp_max)

    amp_ST     = [wf["amplitude"] for ev in events for wf in ev["ST"] if wf["signal"]]
    amp_TW_all = [wf["amplitude"] for ev in events for wf in ev["TW"]]
    amp_TW     = [wf["amplitude"] for ev in events for wf in ev["TW"] if wf["signal"]]

    n_pileup = sum(1 for ev in events for wf in ev["ST"] if not wf["signal"])

    print(f"\n--- Amplitude statistics ---")
    print(f"ST : min={np.min(amp_ST)*1e3:.1f} mV,  max={np.max(amp_ST)*1e3:.1f} mV,  "
          f"mean={np.mean(amp_ST)*1e3:.1f} mV,  std={np.std(amp_ST)*1e3:.1f} mV")
    print(f"ST pile-up flagged (> {st_amp_max*1e3:.0f} mV): {n_pileup}")
    print(f"TW signal threshold: {n_sigma * np.mean([wf['noise'] for ev in events for wf in ev['TW']])*1e3:.2f} mV ({n_sigma}σ, mean)"
          if noise_std is None else
          f"TW signal threshold: {n_sigma*noise_std:.2f} mV (fixed)")
    print(f"TW with signal: {len(amp_TW)}/{len(amp_TW_all)} waveforms "
          f"({100*len(amp_TW)/len(amp_TW_all):.1f}%)")
    print(f"TW : min={np.min(amp_TW)*1e3:.1f} mV,  max={np.max(amp_TW)*1e3:.1f} mV,  "
          f"mean={np.mean(amp_TW)*1e3:.1f} mV,  std={np.std(amp_TW)*1e3:.1f} mV")

    return events, amp_ST, amp_TW


def filter_pileup_events(events):
    """Remove the events with at least one ST waveform flagged as pile-up (to be called after apply_amplitude)."""
    n_before = len(events)
    events   = [ev for ev in events if all(wf["signal"] for wf in ev["ST"])]
    n_after  = len(events)
    print(f"Events after the ST pile-up cut: {n_after}/{n_before} "
          f"({100*n_after/n_before:.1f}%)")
    return events


# =============================================================================
# PLOTS
# =============================================================================

def plot_amplitude(amp_ST, amp_TW):
    """Amplitude histograms of the ST and TW waveforms with signal."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Amplitude distribution", fontsize=16)

    axes[0].hist(np.array(amp_ST)*1e3, bins=100, color="royalblue")
    axes[0].set_xlabel("ST amplitude (mV)", fontsize=14)
    axes[0].set_ylabel("Counts", fontsize=14)
    axes[0].set_title("ST", fontsize=14)
    axes[0].tick_params(axis='both', labelsize=13)
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(np.array(amp_TW)*1e3, bins=100, color="crimson")
    axes[1].set_xlabel("TW amplitude (mV)", fontsize=14)
    axes[1].set_ylabel("Counts", fontsize=14)
    axes[1].set_title("TW (signal)", fontsize=14)
    axes[1].tick_params(axis='both', labelsize=13)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_signal_identification(events, n_sigma=N_SIGMA, save=None):
    """Histogram of A/sigma_noise for all TW waveforms, with the signal threshold."""
    ratio = np.array([wf["amplitude"] / wf["noise"]
                      for ev in events for wf in ev["TW"] if wf["noise"] > 0])

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.logspace(np.log10(max(ratio.min(), 0.5)), np.log10(ratio.max()), 150)
    ax.hist(ratio, bins=bins, histtype="step", lw=1.5)
    ax.axvline(n_sigma, color="red", linestyle="--", lw=2,
               label=f"threshold: ${n_sigma}\\,\\sigma_{{\\mathrm{{noise}}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$A / \sigma_{\mathrm{noise}}$")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_signal_examples(events, ev_idx, n_sigma=N_SIGMA, save=None):
    """One TW waveform without signal and one with signal from the same event, with the threshold."""
    tw = events[ev_idx]["TW"]
    no_sig = next(wf for wf in tw if not wf["signal"])
    sig    = next(wf for wf in tw if wf["signal"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, wf, title in ((axes[0], no_sig, "no signal"), (axes[1], sig, "signal")):
        thr = n_sigma * wf["noise"] * 1e3
        ax.plot(wf["time"], wf["amp"] * 1e3, lw=1.2)
        ax.axhline(-thr, color="red", linestyle="--", lw=1.8,
                   label=f"$-{n_sigma}\\sigma$ = {-thr:.1f} mV")
        ax.set_title(f"TW {title} — board {wf['boardId']}, channel {wf['channelId']}")
        ax.set_xlabel("Time (ns)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Amplitude (mV)")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_pileup(events, st_amp_max=ST_AMP_MAX, exclude_ch=(7,), save=None):
    """Amplitude of all ST waveforms (before the pile-up cut), with the pile-up threshold."""
    amps = np.array([wf["amplitude"] * 1e3
                     for ev in events for wf in ev["ST"]
                     if wf["channelId"] not in exclude_ch])
    frac = np.mean(amps > st_amp_max * 1e3)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(amps, bins=200, histtype="step", lw=1.5)
    ax.axvline(st_amp_max * 1e3, color="red", linestyle="--", lw=2,
               label=f"pile-up threshold = {st_amp_max*1e3:.0f} mV")
    ax.set_yscale("log")
    ax.set_xlabel("ST amplitude (mV)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
    print(f"ST waveforms above threshold: {100*frac:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal identification and ST pile-up: check plots on one file")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--event", type=int, default=0, help="Event used for plot_signal_examples")
    parser.add_argument("--save-dir", default=None, help="Save the figures in this folder")
    args = parser.parse_args()

    def out(name):
        return os.path.join(args.save_dir, name) if args.save_dir else None

    events = load_waveforms(args.file)
    events, *_ = apply_baseline(events, method="mean")
    cleanup_baseline(events)

    events, amp_ST, amp_TW = apply_amplitude(events, n_sigma=N_SIGMA, st_amp_max=ST_AMP_MAX)

    plot_signal_identification(events, save=out("signal_identification.pdf"))
    plot_signal_examples(events, ev_idx=args.event, save=out("signal_examples.pdf"))
    plot_pileup(events, save=out("st_pileup.pdf"))

    events = filter_pileup_events(events)
    plot_amplitude(amp_ST, amp_TW)
