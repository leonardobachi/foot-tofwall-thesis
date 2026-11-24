"""
read_wf.py — Reader for the binary waveform files (wf_raw*.bin).

The .bin files are written by a modified version of the SHOE WaveDAQ reader
and contain, for every event, all the Start Counter (ST) and TOF-Wall (TW)
waveforms recorded by the DRS4 boards (1024 samples, ~3 GS/s).

File layout, repeated for every event:
    eventN       int32
    nST          int32      number of ST waveforms in the event (always 8)
    nTW          int32      number of TW waveforms in the event (variable)
    then nST + nTW waveforms, each made of:
        detector     int8           0 = ST, 1 = TW
        boardId      int32
        channelId    int32
        time         1024 x float32 (ns)
        amp          1024 x float32 (V)
    -> 12 + (nST + nTW) * 8209 bytes per event

Channels 0-11 of board 166 (the three central bars of both layers) are always
read out; the other TW channels only when the signal exceeds the trigger
threshold.

Usage:
    python3 read_wf.py <file.bin> [--event N] [--det TW|ST] [--wf K]
"""

import math
import struct
import argparse

import numpy as np
import matplotlib.pyplot as plt

N_SAMPLES = 1024


def load_waveforms(filepath):
    """
    Read a .bin file and return a list of events. The whole file is read with
    a single call and parsed in memory, which is much faster than many small
    reads (the file must fit in RAM).

    Each event is a dict {"eventN", "nST", "nTW", "ST": [...], "TW": [...]},
    and each waveform a dict {"detector", "boardId", "channelId", "time", "amp"}.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    total = len(raw)
    offset = 0
    events = []

    while offset + 12 <= total:
        eventN, nST, nTW = struct.unpack_from("=iii", raw, offset)
        offset += 12

        st_waves, tw_waves = [], []

        for _ in range(nST + nTW):
            detector  = struct.unpack_from("=b", raw, offset)[0];  offset += 1
            boardId   = struct.unpack_from("=i", raw, offset)[0];  offset += 4
            channelId = struct.unpack_from("=i", raw, offset)[0];  offset += 4
            time_arr  = np.frombuffer(raw, dtype=np.float32,
                                      count=N_SAMPLES, offset=offset).copy();  offset += 4 * N_SAMPLES
            amp_arr   = np.frombuffer(raw, dtype=np.float32,
                                      count=N_SAMPLES, offset=offset).copy();  offset += 4 * N_SAMPLES

            wf = {"detector":  int(detector),
                  "boardId":   int(boardId),
                  "channelId": int(channelId),
                  "time":      time_arr,
                  "amp":       amp_arr}

            (st_waves if detector == 0 else tw_waves).append(wf)

        events.append({"eventN": eventN, "nST": nST, "nTW": nTW,
                       "ST": st_waves, "TW": tw_waves})

    print(f"\nEvents read: {len(events)}")
    return events


def plot_event_grid(event, det, nrows=4):
    """All the waveforms of one detector ("ST" or "TW") in one event, one panel each."""
    wfs = event[det]
    ncols = math.ceil(len(wfs) / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, wf in zip(axes, wfs):
        ax.plot(wf["time"], wf["amp"] * 1e3)
        ax.set_title(f"Board {wf['boardId']}, Ch {wf['channelId']}", fontsize=16)
        ax.set_xlabel("Time (ns)", fontsize=16)
        ax.set_ylabel("Amp (mV)", fontsize=16)
        ax.grid(True)

    for ax in axes[len(wfs):]:
        ax.set_visible(False)

    fig.suptitle(f"{det} - Event {event['eventN']}", fontsize=16)
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a wf_raw*.bin file and plot one event")
    parser.add_argument("file", help="Path to the .bin file")
    parser.add_argument("--event", type=int, default=0, help="Event index in the file")
    parser.add_argument("--det", choices=["ST", "TW"], default="TW", help="Detector of the single waveform")
    parser.add_argument("--wf", type=int, default=0, help="Waveform index within the detector")
    args = parser.parse_args()

    events = load_waveforms(args.file)
    ev = events[args.event]
    wf = ev[args.det][args.wf]

    print(f"Event {args.event}: nST={ev['nST']}, nTW={ev['nTW']}, "
          f"board={wf['boardId']}, channel={wf['channelId']}")

    plt.figure()
    plt.plot(wf["time"], wf["amp"] * 1e3)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)

    plot_event_grid(ev, "ST")
    plot_event_grid(ev, "TW")
    plt.show()
