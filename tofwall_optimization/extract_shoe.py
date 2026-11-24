#!/usr/bin/env python3
"""
extract_shoe.py — TW hits and points from the SHOE .root files (uproot).

Outputs (any combination, all read in a single pass over each file):
    --hits     full hits: bar_id (0-39), tof, time_bar, t_L/t_R, coordinate,
               energy, chargeZ, charges and amplitudes of the two ends
               (payload key "hits"; also readable by results.py --load-hits)
    --hits-dt  reduced hits: bar_id, t_L, t_R, chargeZ, eventN, run_id
               (payload key "hits"; for the delta-time calibration)
    --points   points: row, column, tof, tof1, tof2, eloss1, eloss2,
               main_eloss, chargeZ (payload key "points")

Hits with fIsValid false, negative or non-finite energy, or non-finite TOF are
discarded (--keep-all keeps them). Points with non-finite TOF or energy are
discarded.

Run number and event number:
    default   run_id from the number at the beginning of the file name
              (8102_....root), eventN from evtevt.fEventNumber
    --calib   for scan/calibration files: run_id from the last number in the
              file name (..._17.root or ..._17_TWBarCalib.root), eventN = entry
              index in the file

Usage:
    python3 extract_shoe.py <files.root or "pattern*.root"> [--calib] [--workers N]
                            [--hits OUT.pkl] [--hits-dt OUT.pkl] [--points OUT.pkl]
Examples:
    python3 extract_shoe.py "reco/*.root" --hits hits.pkl --points points.pkl --workers 8
    python3 extract_shoe.py "scan/scan_*.root" --calib --hits-dt hits_dt.pkl --points points.pkl
"""

import os
import re
import sys
import glob
import time
import pickle
import argparse

import numpy as np
import uproot
from multiprocessing import Pool, cpu_count

# =============================================================================
# BRANCHES — logical name -> ROOT branch
# =============================================================================

HIT_BR = {
    "bar":    "twrh.fListOfHits.fBar",
    "layer":  "twrh.fListOfHits.fLayer",
    "time":   "twrh.fListOfHits.fTime",
    "tof":    "twrh.fListOfHits.fTimeofflight",
    "timeA":  "twrh.fListOfHits.fTimeA",
    "timeB":  "twrh.fListOfHits.fTimeB",
    "chgA":   "twrh.fListOfHits.fChargeA",
    "chgB":   "twrh.fListOfHits.fChargeB",
    "amplitudeA": "twrh.fListOfHits.fAmplitudeA",
    "amplitudeB": "twrh.fListOfHits.fAmplitudeB",
    "de":     "twrh.fListOfHits.fDe",
    "z":      "twrh.fListOfHits.fChargeZ",
    "coord":  "twrh.fListOfHits.fCoordinate",
    "valid":  "twrh.fListOfHits.fIsValid",
}

POINT_BR = {
    "row":    "twpt.fListOfPoints.fRow",
    "col":    "twpt.fListOfPoints.fColumn",
    "de1":    "twpt.fListOfPoints.fDe1",
    "de2":    "twpt.fListOfPoints.fDe2",
    "mainde": "twpt.fListOfPoints.fMainEloss",
    "tof1":   "twpt.fListOfPoints.fTof1",
    "tof2":   "twpt.fListOfPoints.fTof2",
    "tof":    "twpt.fListOfPoints.fToF",
    "layer":  "twpt.fListOfPoints.fLayer",
    "z":      "twpt.fListOfPoints.fChargeZ",
    "zproba": "twpt.fListOfPoints.fChargeZProba",
}

EVENT_BRANCH = "evtevt.fEventNumber"
STEP_SIZE = "100 MB"

TARGET_HIT_KEYS = {
    "hits":    ["bar", "layer", "time", "tof", "timeA", "timeB", "chgA", "chgB", "de", "z", "coord", "valid", "amplitudeA", "amplitudeB"],
    "hits_dt": ["bar", "layer", "timeA", "timeB", "z", "valid"],
}
TARGET_POINT_KEYS = {
    "points": ["row", "col", "de1", "de2", "mainde", "tof1", "tof2", "tof", "layer", "z", "zproba"],
}


def get_run_id_physical(fpath):
    """Physics runs: number at the beginning of the file name (8102_TWBarCalib.root -> 8102)."""
    basename = os.path.basename(fpath)
    m = re.match(r"^(\d+)", basename)
    return int(m.group(1)) if m else None


def get_run_id_calib(fpath):
    """Scan/calibration files: number before the .root suffix
    (run_twscan_000_17_TWBarCalib.root -> 17)."""
    basename = os.path.basename(fpath)
    m = re.search(r"_(\d+)(?:_TWBarCalib)?\.root$", basename)
    return int(m.group(1)) if m else None


# =============================================================================
# RECORDS — one hit/point -> dict, or None if discarded
# =============================================================================

def _build_hit_record(j, hit_ev, eventN, run_id, bar_id_global, n_dropped, keep_all=False):
    """
    Full hit. charge_fit_sat, amp_st_sum and chi2_fit_sat are placeholders, so
    that the hits have the same fields as the bars of results.py and pass its
    selections.
    """
    if not keep_all and not bool(hit_ev["valid"][j]):
        return None, n_dropped + 1
    de_val = float(hit_ev["de"][j])
    if not keep_all and (de_val < 0 or not np.isfinite(de_val)):
        return None, n_dropped + 1
    tof_val = float(hit_ev["tof"][j])
    if not keep_all and not np.isfinite(tof_val):
        return None, n_dropped + 1

    chg_a = float(hit_ev["chgA"][j])
    chg_b = float(hit_ev["chgB"][j])
    charge_bar_val = (chg_a * chg_b) ** 0.5 if (chg_a > 0 and chg_b > 0) else 0.0

    rec = {
        "bar_id": bar_id_global, "layer": 0 if bar_id_global < 20 else 1,
        "tof": tof_val, "time_bar": float(hit_ev["time"][j]),
        "t_L": float(hit_ev["timeA"][j]), "t_R": float(hit_ev["timeB"][j]),
        "coordinate": float(hit_ev["coord"][j]), "energy": de_val,
        "chargeZ": int(hit_ev["z"][j]), "eventN": eventN, "run_id": run_id, "file_idx": 0,
        "charge_bar": charge_bar_val, "charge_L": chg_a, "charge_R": chg_b,
        "amplitude_L": float(hit_ev["amplitudeA"][j]), "amplitude_R": float(hit_ev["amplitudeB"][j]),
        "charge_fit_sat": None, "amp_st_sum": 9999.0, "chi2_fit_sat": 0.0,
        "valid": bool(hit_ev["valid"][j]),
    }
    return rec, n_dropped


def _build_hit_dt_record(j, hit_ev, eventN, run_id, bar_id_global, n_dropped):
    """Reduced hit for the delta-time calibration (valid hits with finite times)."""
    if not bool(hit_ev["valid"][j]):
        return None, n_dropped + 1
    t_L = float(hit_ev["timeA"][j])
    t_R = float(hit_ev["timeB"][j])
    if not (np.isfinite(t_L) and np.isfinite(t_R)):
        return None, n_dropped + 1
    rec = {"bar_id": bar_id_global, "t_L": t_L, "t_R": t_R,
           "chargeZ": int(hit_ev["z"][j]), "eventN": eventN, "run_id": run_id}
    return rec, n_dropped


def _build_point_record(j, pt_ev, eventN, run_id, n_dropped):
    """Point (finite TOF and main energy loss)."""
    tof_val = float(pt_ev["tof"][j])
    de_val = float(pt_ev["mainde"][j])
    if not np.isfinite(tof_val) or not np.isfinite(de_val):
        return None, n_dropped + 1
    rec = {
        "row": int(pt_ev["row"][j]), "column": int(pt_ev["col"][j]),
        "eloss1": float(pt_ev["de1"][j]), "eloss2": float(pt_ev["de2"][j]),
        "main_eloss": de_val, "tof1": float(pt_ev["tof1"][j]), "tof2": float(pt_ev["tof2"][j]),
        "tof": tof_val, "main_layer": int(pt_ev["layer"][j]), "chargeZ": int(pt_ev["z"][j]),
        "chargeZ_proba": float(pt_ev["zproba"][j]), "eventN": eventN, "run_id": run_id, "file_idx": 0,
    }
    return rec, n_dropped


# =============================================================================
# WORKER — one .root file, all the requested outputs in one pass
# =============================================================================

def process_single_file(args):
    fpath, mode, targets, keep_all = args
    get_run_id = get_run_id_physical if mode == "physical" else get_run_id_calib
    use_event_branch = (mode == "physical")

    run_id = get_run_id(fpath)
    if run_id is None:
        print(f"[WARN] cannot get the run number from {fpath}, file skipped")
        return {t: [] for t in targets}, None, 0, {t: 0 for t in targets}

    t_file_start = time.time()
    try:
        tree = uproot.open(fpath)["tree"]
    except Exception as e:
        print(f"  [ERROR] cannot open {fpath}: {e} — skipped")
        return {t: [] for t in targets}, None, 0, {t: 0 for t in targets}

    need_hit_keys = set()
    need_point_keys = set()
    for t in targets:
        if t in TARGET_HIT_KEYS:
            need_hit_keys |= set(TARGET_HIT_KEYS[t])
        if t in TARGET_POINT_KEYS:
            need_point_keys |= set(TARGET_POINT_KEYS[t])

    hit_branch_map = {k: HIT_BR[k] for k in need_hit_keys}
    point_branch_map = {k: POINT_BR[k] for k in need_point_keys}
    branches_to_read = list(hit_branch_map.values()) + list(point_branch_map.values())
    if use_event_branch:
        branches_to_read.append(EVENT_BRANCH)

    local = {t: [] for t in targets}
    n_dropped = {t: 0 for t in targets}
    n_entries_file = 0
    entry_offset = 0

    for chunk in tree.iterate(branches_to_read, step_size=STEP_SIZE, library="ak"):
        n_local = len(chunk)
        for i in range(n_local):
            evraw = chunk[i]
            eventN = int(evraw[EVENT_BRANCH]) if use_event_branch else entry_offset + i

            if need_hit_keys:
                hit_ev = {k: evraw[hit_branch_map[k]] for k in need_hit_keys}
                n_hits = len(hit_ev["bar"])
                for j in range(n_hits):
                    bar_id_global = int(hit_ev["bar"][j]) + 20 * int(hit_ev["layer"][j])
                    if "hits" in targets:
                        rec, n_dropped["hits"] = _build_hit_record(j, hit_ev, eventN, run_id, bar_id_global, n_dropped["hits"], keep_all)
                        if rec is not None:
                            local["hits"].append(rec)
                    if "hits_dt" in targets:
                        rec, n_dropped["hits_dt"] = _build_hit_dt_record(j, hit_ev, eventN, run_id, bar_id_global, n_dropped["hits_dt"])
                        if rec is not None:
                            local["hits_dt"].append(rec)

            if need_point_keys:
                pt_ev = {k: evraw[point_branch_map[k]] for k in need_point_keys}
                n_pts = len(pt_ev["row"])
                for j in range(n_pts):
                    rec, n_dropped["points"] = _build_point_record(j, pt_ev, eventN, run_id, n_dropped["points"])
                    if rec is not None:
                        local["points"].append(rec)

        n_entries_file += n_local

    dt_file = time.time() - t_file_start
    summary = ", ".join(f"{t}={len(local[t])} (discarded {n_dropped[t]})" for t in targets)
    print(f"[{fpath}]  run_id={run_id}  events={n_entries_file}  {summary}  time={dt_file:.1f}s")

    return local, run_id, n_entries_file, n_dropped


# =============================================================================
# MAIN
# =============================================================================

def extract_files(file_args, outputs, mode="physical", n_workers=None, keep_all=False):
    """
    outputs: {target: output path}, target in "hits", "hits_dt", "points".
    file_args: .root files and/or wildcard patterns (quoted or expanded by the shell).
    """
    if n_workers is None:
        n_workers = 1

    targets = list(outputs.keys())

    file_list = []
    for item in file_args:
        if any(c in item for c in "*?["):
            file_list.extend(glob.glob(item))
        else:
            file_list.append(item)
    file_list = sorted(set(file_list))

    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Outputs: {targets}")
    print(f"[INFO] .root files found: {len(file_list)}")
    print(f"[INFO] Parallel processes: {n_workers}")

    if len(file_list) == 0:
        print("[ERROR] No file found.")
        return None

    all_data = {t: [] for t in targets}
    all_run_ids = set()

    t_start = time.time()
    args_list = [(fpath, mode, targets, keep_all) for fpath in file_list]

    with Pool(n_workers) as pool:
        # results merged as soon as each file is done, to limit the memory peak
        for local, run_id, n_entries_file, n_dropped in pool.imap_unordered(process_single_file, args_list):
            for t in targets:
                all_data[t].extend(local[t])
            if run_id is not None:
                all_run_ids.add(run_id)
            del local

    t_total = time.time() - t_start
    print(f"\n[INFO] Extraction done in {t_total:.1f}s")

    for t in targets:
        data = all_data[t]
        print(f"\n[INFO] === Target: {t} ===")
        print(f"[INFO] Entries: {len(data)}")
        if len(data) == 0:
            print(f"[ERROR] Nothing extracted for '{t}'.")
            continue

        if t == "hits":
            bar_ids = np.array([b["bar_id"] for b in data])
            tofs = np.array([b["tof"] for b in data])
            energies = np.array([b["energy"] for b in data])
            zs = np.array([b["chargeZ"] for b in data])
            print(f"  bar_id range: [{bar_ids.min()}, {bar_ids.max()}]")
            print(f"  tof range: [{tofs.min():.3f}, {tofs.max():.3f}] ns")
            print(f"  energy range: [{energies.min():.3f}, {energies.max():.3f}] MeV")
            print(f"  chargeZ values: {np.unique(zs)}")
            payload = {"hits": data, "tw_method": "SHOE", "st_method": "SHOE",
                       "coincidence": None, "runs": sorted(all_run_ids)}

        elif t == "hits_dt":
            bar_ids = np.array([h["bar_id"] for h in data])
            dts = np.array([h["t_R"] - h["t_L"] for h in data])
            print(f"  bar_id range: [{bar_ids.min()}, {bar_ids.max()}]")
            print(f"  dt(t_R-t_L) range: [{dts.min():.3f}, {dts.max():.3f}] ns")
            payload = {"hits": data, "runs": sorted(all_run_ids)}

        elif t == "points":
            tofs = np.array([p["tof"] for p in data])
            elosses = np.array([p["main_eloss"] for p in data])
            zs = np.array([p["chargeZ"] for p in data])
            print(f"  tof range: [{tofs.min():.3f}, {tofs.max():.3f}] ns")
            print(f"  main_eloss range: [{elosses.min():.3f}, {elosses.max():.3f}] MeV")
            print(f"  chargeZ values: {np.unique(zs)}")
            payload = {"points": data, "runs": sorted(all_run_ids)}

        print(f"  run_id values: {sorted(all_run_ids)}")

        with open(outputs[t], "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(outputs[t]) / 1e6
        print(f"[INFO] {t} saved to: {outputs[t]} ({size_mb:.1f} MB)")

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TW hits and points from the SHOE .root files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+",
                        help=".root files or wildcard patterns")
    parser.add_argument("--hits", metavar="OUT.pkl",
                        help="full hits (key 'hits')")
    parser.add_argument("--hits-dt", metavar="OUT.pkl",
                        help="reduced hits for the delta-time calibration (key 'hits')")
    parser.add_argument("--points", metavar="OUT.pkl",
                        help="points (key 'points')")
    parser.add_argument("--calib", action="store_true",
                        help="scan/calibration files: run_id from the end of the file name, eventN = entry index")
    parser.add_argument("--keep-all", action="store_true", help="no selection on the full hits (--hits)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default 1)")
    args = parser.parse_args()

    outputs = {}
    if args.hits:
        outputs["hits"] = args.hits
    if args.hits_dt:
        outputs["hits_dt"] = args.hits_dt
    if args.points:
        outputs["points"] = args.points

    if not outputs:
        print("Error: give at least one of --hits, --hits-dt, --points")
        sys.exit(1)

    mode = "calib" if args.calib else "physical"
    extract_files(args.files, outputs, mode=mode, n_workers=args.workers, keep_all=args.keep_all)