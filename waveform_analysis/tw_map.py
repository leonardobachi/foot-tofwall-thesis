"""
tw_map.py — Channel map of the TOF-Wall (CNAO2025 configuration).

The TW has 40 bars (bar_id 0-39) in two layers:
    layer 0 (rear):  bars 0-19,  vertical
    layer 1 (front): bars 20-39, horizontal
Each bar is read at both ends (L and R), each end by a separate WaveDAQ channel.

Source: TATWChannelMapXML.map and WDChannelMap.map (SHOE, CNAO2025).

    board 165: ch 0-15  -> bars 0-7    (layer 0)
    board 166: ch 0-5   -> bars 8-10   (layer 0, central bars)
               ch 6-11  -> bars 28-30  (layer 1, central bars)
    board 170: ch 0-15  -> bars 11-18  (layer 0)
    board 167: ch 0-1   -> bar 19      (layer 0)
               ch 2-15  -> bars 20-26  (layer 1)
    board 168: ch 0-1   -> bar 27      (layer 1)
               ch 2-15  -> bars 31-37  (layer 1)
    board 158: ch 0-3   -> bars 38-39  (layer 1)

Usage as a module:
    from tw_map import get_bar_info
    bar_id, layer, end = get_bar_info(board_id, channel_id)   # end = "L" or "R"

Standalone (prints the map and checks the mapping on one file):
    python3 tw_map.py [<file.bin>]
"""

import argparse

# bar_id -> layer, board and channels of the two ends
BAR_INFO = {
    # layer 0 (rear), board 165
     0: {"layer": 0, "board": 165, "chL":  0, "chR":  1},
     1: {"layer": 0, "board": 165, "chL":  2, "chR":  3},
     2: {"layer": 0, "board": 165, "chL":  4, "chR":  5},
     3: {"layer": 0, "board": 165, "chL":  6, "chR":  7},
     4: {"layer": 0, "board": 165, "chL":  8, "chR":  9},
     5: {"layer": 0, "board": 165, "chL": 10, "chR": 11},
     6: {"layer": 0, "board": 165, "chL": 12, "chR": 13},
     7: {"layer": 0, "board": 165, "chL": 14, "chR": 15},
    # layer 0 (rear), board 166 — central bars, always read out
     8: {"layer": 0, "board": 166, "chL":  0, "chR":  1},
     9: {"layer": 0, "board": 166, "chL":  2, "chR":  3},
    10: {"layer": 0, "board": 166, "chL":  4, "chR":  5},
    # layer 0 (rear), board 170
    11: {"layer": 0, "board": 170, "chL":  0, "chR":  1},
    12: {"layer": 0, "board": 170, "chL":  2, "chR":  3},
    13: {"layer": 0, "board": 170, "chL":  4, "chR":  5},
    14: {"layer": 0, "board": 170, "chL":  6, "chR":  7},
    15: {"layer": 0, "board": 170, "chL":  8, "chR":  9},
    16: {"layer": 0, "board": 170, "chL": 10, "chR": 11},
    17: {"layer": 0, "board": 170, "chL": 12, "chR": 13},
    18: {"layer": 0, "board": 170, "chL": 14, "chR": 15},
    # layer 0 (rear), board 167
    19: {"layer": 0, "board": 167, "chL":  0, "chR":  1},
    # layer 1 (front), board 167
    20: {"layer": 1, "board": 167, "chL":  2, "chR":  3},
    21: {"layer": 1, "board": 167, "chL":  4, "chR":  5},
    22: {"layer": 1, "board": 167, "chL":  6, "chR":  7},
    23: {"layer": 1, "board": 167, "chL":  8, "chR":  9},
    24: {"layer": 1, "board": 167, "chL": 10, "chR": 11},
    25: {"layer": 1, "board": 167, "chL": 12, "chR": 13},
    26: {"layer": 1, "board": 167, "chL": 14, "chR": 15},
    # layer 1 (front), board 168
    27: {"layer": 1, "board": 168, "chL":  0, "chR":  1},
    # layer 1 (front), board 166 — central bars, always read out
    28: {"layer": 1, "board": 166, "chL":  6, "chR":  7},
    29: {"layer": 1, "board": 166, "chL":  8, "chR":  9},
    30: {"layer": 1, "board": 166, "chL": 10, "chR": 11},
    # layer 1 (front), board 168
    31: {"layer": 1, "board": 168, "chL":  2, "chR":  3},
    32: {"layer": 1, "board": 168, "chL":  4, "chR":  5},
    33: {"layer": 1, "board": 168, "chL":  6, "chR":  7},
    34: {"layer": 1, "board": 168, "chL":  8, "chR":  9},
    35: {"layer": 1, "board": 168, "chL": 10, "chR": 11},
    36: {"layer": 1, "board": 168, "chL": 12, "chR": 13},
    37: {"layer": 1, "board": 168, "chL": 14, "chR": 15},
    # layer 1 (front), board 158
    38: {"layer": 1, "board": 158, "chL":  0, "chR":  1},
    39: {"layer": 1, "board": 158, "chL":  2, "chR":  3},
}

# inverse map: (board, channel) -> (bar_id, layer, end)
CH_TO_BAR = {}
for bar_id, info in BAR_INFO.items():
    CH_TO_BAR[(info["board"], info["chL"])] = (bar_id, info["layer"], "L")
    CH_TO_BAR[(info["board"], info["chR"])] = (bar_id, info["layer"], "R")


def get_bar_info(board_id, channel_id):
    """(bar_id, layer, end) of a channel, or None if the channel is not a TW bar (clock, empty, ...)."""
    return CH_TO_BAR.get((board_id, channel_id), None)


def apply_tw_map(events):
    """
    Assign every TW waveform to its bar. Adds wf["bar_id"] (0-39), wf["layer"]
    (0 rear, 1 front) and wf["end"] ("L" or "R"); None for unmapped channels.
    """
    n_mapped   = 0
    n_unmapped = 0
    for ev in events:
        for wf in ev["TW"]:
            result = get_bar_info(wf["boardId"], wf["channelId"])
            if result:
                wf["bar_id"], wf["layer"], wf["end"] = result
                n_mapped += 1
            else:
                wf["bar_id"] = None
                wf["layer"]  = None
                wf["end"]    = None
                n_unmapped  += 1
    print(f"\n--- Bar mapping ---")
    print(f"TW waveforms mapped: {n_mapped},  not mapped: {n_unmapped}")
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the TW channel map and check it on one file")
    parser.add_argument("file", nargs="?", default=None, help="Path to a .bin file (optional)")
    args = parser.parse_args()

    print("=== (board, ch) -> (bar_id, layer, end) ===\n")
    for (board, ch), (bar_id, layer, end) in sorted(CH_TO_BAR.items()):
        print(f"  Board {board:3d}, Ch {ch:2d}  ->  Bar {bar_id:2d}, Layer {layer}, End {end}")

    if args.file:
        from read_wf      import load_waveforms
        from baseline_wf  import apply_baseline, cleanup_baseline
        from amplitude_wf import apply_amplitude

        events = load_waveforms(args.file)
        events, *_ = apply_baseline(events, method="mean")
        cleanup_baseline(events)
        events, *_ = apply_amplitude(events)
        events = apply_tw_map(events)

        print(f"\n--- Event {events[0]['eventN']} ---")
        for wf in events[0]["TW"]:
            print(f"  Board {wf['boardId']:3d}, Ch {wf['channelId']:2d}  ->  "
                  f"bar_id={wf['bar_id']}, layer={wf['layer']}, end={wf['end']}, "
                  f"signal={wf['signal']}")
