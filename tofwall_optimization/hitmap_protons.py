#!/usr/bin/env python3
"""
hitmap_protons.py — Hitmap of the SHOE points with a given Z (default Z=1) on
the 400 row x column intersections of the TW.

Each point with chargeZ == Z is counted in the intersection (row, column)
assigned by SHOE. Output, for every Z:
    - 20x20 table on the terminal (rows = row/front, columns = column/rear)
    - hitmap_Z<z>_<input name>.txt with the same matrix
    - hitmap_z<z>.pdf (log colour scale, counts in the cells if ANNOTATE)

Orientation (bar indices, not cm): y axis = row (front, layer 1), row 0 at
the top; x axis = column (rear, layer 0), column 0 on the left. In local
coordinates the position is 19 - 2*index cm, so the x axis is reversed with
respect to the physical x.

Usage:
    python3 hitmap_protons.py --points points.pkl --output-dir DIR [--z 1 2 ...]
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

N_BARS = 20
LOG_SCALE = True
ANNOTATE = False
ANNOTATE_FONTSIZE = 7
CMAP = "viridis"
FIGSIZE = (7, 6)
PAD = 0.2


def load_points(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    points = payload["points"]
    print(f"Points loaded: {len(points)}  (runs={payload.get('runs', '?')})")
    return points


def build_hitmap(points, z):
    """hitmap[row, column] = number of points with chargeZ == z."""
    hitmap = np.zeros((N_BARS, N_BARS), dtype=np.int64)
    n_sel = 0
    n_bad_idx = 0
    for p in points:
        if p["chargeZ"] != z:
            continue
        r, c = int(p["row"]), int(p["column"])
        if not (0 <= r < N_BARS and 0 <= c < N_BARS):
            n_bad_idx += 1
            continue
        hitmap[r, c] += 1
        n_sel += 1
    print(f"Points with Z={z}: {n_sel}" + (f"  (index out of range: {n_bad_idx})" if n_bad_idx else ""))
    return hitmap


def print_hitmap(hitmap, z):
    width = max(6, len(str(hitmap.max())) + 1)
    print(f"\nHitmap Z={z} — rows: row (front), columns: column (rear)")
    print("row\\col" + "".join(f"{c:>{width}}" for c in range(N_BARS)))
    for r in range(N_BARS):
        print(f"{r:>7}" + "".join(f"{hitmap[r, c]:>{width}}" for c in range(N_BARS)))

    vals = hitmap.ravel()
    r_max, c_max = np.unravel_index(np.argmax(hitmap), hitmap.shape)
    r_min, c_min = np.unravel_index(np.argmin(hitmap), hitmap.shape)
    print(f"\nTotal: {vals.sum()}   empty intersections: {(vals == 0).sum()}/{vals.size}")
    print(f"min: {vals.min()} (row {r_min}, col {c_min})   max: {vals.max()} (row {r_max}, col {c_max})   "
          f"median: {np.median(vals):.0f}")


def save_hitmap_txt(hitmap, z, points_path, out_path):
    header = (f"TW hitmap: number of SHOE points with chargeZ={z} per intersection\n"
              f"input: {points_path}\n"
              f"rows: row (front, layer 1) 0-19; columns: column (rear, layer 0) 0-19\n"
              f"total: {hitmap.sum()}")
    np.savetxt(out_path, hitmap, fmt="%d", delimiter="  ", header=header)
    print(f"\n[INFO] Hitmap saved to: {out_path}")


def plot_hitmap(hitmap, z, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if LOG_SCALE:
        data = np.ma.masked_where(hitmap <= 0, hitmap)
        norm = LogNorm(vmin=max(data.min(), 1), vmax=data.max())
    else:
        data, norm = hitmap, None

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(cmap(0))          # empty cells with the lowest colour of the scale
    im = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    if z == 1:
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar.set_label("Counts")

    if ANNOTATE:
        thr = (np.log10(max(hitmap.max(), 1)) / 2) if LOG_SCALE else hitmap.max() / 2
        for r in range(N_BARS):
            for c in range(N_BARS):
                v = hitmap[r, c]
                level = np.log10(v) if (LOG_SCALE and v > 0) else v
                ax.text(c, r, str(v), ha="center", va="center", fontsize=ANNOTATE_FONTSIZE,
                        color="black" if level > thr else "white")

    ax.set_xticks(range(N_BARS))
    ax.set_yticks(range(N_BARS))
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Bar (rear layer)")
    ax.set_ylabel("Bar (front layer)")
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(f"TW hitmap — Z={z} — {hitmap.sum()} points")
    fig.tight_layout(pad=PAD)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Figure saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Hitmap of the SHOE points with a given Z on the 400 TW intersections")
    parser.add_argument("--points", required=True, help="points.pkl from extract_shoe.py (key 'points')")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--z", type=int, nargs="+", default=[1], help="One or more Z, e.g. --z 1 2 3")
    args = parser.parse_args()

    dataset_label = os.path.splitext(os.path.basename(args.points))[0]

    points = load_points(args.points)
    os.makedirs(args.output_dir, exist_ok=True)
    for z in args.z:
        hitmap = build_hitmap(points, z)
        print_hitmap(hitmap, z)
        save_hitmap_txt(hitmap, z, args.points,
                        os.path.join(args.output_dir, f"hitmap_Z{z}_{dataset_label}.txt"))
        plot_hitmap(hitmap, z, os.path.join(args.output_dir, f"hitmap_z{z}.pdf"))


if __name__ == "__main__":
    main()
