"""
analyze.py — Waveform analysis of all the .bin files of one run.

For every file: baseline and noise, amplitude and signal identification, ST
pile-up rejection, ST sum, CFD times with the three methods (simple, tangent,
bipolar), charge, bar mapping and, unless --no-fit, the template fit of the TW
waveforms (optionally also on artificially saturated waveforms, --saturation).
The output is results_<run>.pkl: a list of events, each with the ST channels,
the ST sum and the TW waveforms with signal (see process_file for the fields).

The TW template is built on bar 9 of the first file of the run, or loaded from
a file saved with --save-template (the same template for all the runs).

Usage:
    python3 analyze.py <run_dir> <out_dir> --run <run_id> [options]

Examples:
    python3 analyze.py data/8102 results/ --run 8102 --save-template template_tw.pkl
    python3 analyze.py data/8102 results/ --run 8102 --load-template template_tw.pkl --saturation 200 --cores 8
"""

import sys
import matplotlib
matplotlib.use("TkAgg" if "--plot-event" in sys.argv else "Agg")

import os
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

from read_wf      import load_waveforms
from baseline_wf  import apply_baseline, cleanup_baseline
from amplitude_wf import apply_amplitude, filter_pileup_events
from cfd_wf       import apply_cfd, build_st_sum
from charge_wf    import apply_charge
from tw_map       import apply_tw_map
from template import (collect_tw, resample_wfs, normalize_wfs, build_model_iterative,
                   apply_model_fit, apply_saturation, compute_charge_from_fit,
                   plot_model_fit_results, plot_model, RESAMPLE_STEP)

# =============================================================================
# PARAMETERS
# =============================================================================

N_SIGMA    = 8       # TW signal threshold (units of sigma_noise)
ST_AMP_MAX = 0.600   # V — ST pile-up threshold

TEMPLATE_BAR = 9     # bar used to build the TW template

# template-fit parameters of the TW waveforms (see template.model_fit_wf)
TW_PARAMS = dict(w_rise=1.0, w_tail=1.0,
                 rise_left=2.0, rise_right=5.0,
                 tail=150.0, bracket_half=5.0,
                 optimizer="scalar")

CFD_METHODS = ("simple", "tangent", "bipolar")


def _f(x):
    """Native Python float, or None for None/NaN (cleaner pickle output)."""
    if x is None:
        return None
    v = float(x)
    return None if np.isnan(v) else v


def _valid_grid(wf):
    """True if time and amp have the same length and the time axis is increasing."""
    return (len(wf["time"]) == len(wf["amp"])
            and len(wf["time"]) > 1
            and wf["time"][0] < wf["time"][-1])


# =============================================================================
# TW TEMPLATE
# =============================================================================

def _build_model_tw(events):
    """
    TW template from the waveforms of TEMPLATE_BAR, normalised to the charge and
    aligned with the simple CFD time. The reference time is the median peak
    position of the selected waveforms.
    Returns {"model", "std", "t_grid", "t0_ref", "params"}.
    """
    wfs_tw = [wf for wf in collect_tw(events, bar_ids=[TEMPLATE_BAR]) if _valid_grid(wf)]

    t_grid_tw, amp_res_tw = resample_wfs(wfs_tw, step=RESAMPLE_STEP)
    amp_norm_tw, _        = normalize_wfs(amp_res_tw, wfs_tw, method="charge")

    t0_ref_tw = float(np.median([t_grid_tw[np.argmin(amp_res_tw[i])] for i in range(len(amp_res_tw))]))
    print(f"  TW template reference time = {t0_ref_tw:.2f} ns")

    model_tw, std_tw, _ = build_model_iterative(
        amp_res_tw, amp_norm_tw, wfs_tw, t_grid_tw,
        n_iter=0, method="weighted", max_wfs=None,
        t0_ref=t0_ref_tw, max_shift=None, **TW_PARAMS)

    return {
        "model":  model_tw,
        "std":    std_tw,
        "t_grid": t_grid_tw,
        "t0_ref": t0_ref_tw,
        "params": TW_PARAMS,
    }


# =============================================================================
# TW TEMPLATE FIT
# =============================================================================

def _store_fit(wfs, results, out, model_tw):
    """Copy the fit results into out[(eventN, bar_id, end)] = {"t", "chi2", "npts", "charge"}."""
    ok = [(wf, res) for wf, res in zip(wfs, results) if res["success"]]
    for wf, res in ok:
        out[(wf["_eventN"], wf.get("bar_id"), wf.get("end"))] = {
            "t": res["t_arr"], "chi2": res["chi2"], "npts": res["npts"], "charge": None}

    if ok:
        m = model_tw
        charges_fit_mv, _, _ = compute_charge_from_fit(
            [r for _, r in ok], [w for w, _ in ok], m["t_grid"], m["model"], m["std"], t0_ref=m["t0_ref"])
        for (wf, _), charge_mv in zip(ok, charges_fit_mv):
            out[(wf["_eventN"], wf.get("bar_id"), wf.get("end"))]["charge"] = float(charge_mv / 1e3)  # V*ns


def _run_fit_tw(events, model_tw, sat_level_mv=None, plot_event=None):
    """
    Template fit of all the TW waveforms with signal and, if sat_level_mv is
    given, of the same waveforms clipped at sat_level_mv.
    Returns two dicts (normal, saturated) keyed by (eventN, bar_id, end).
    """
    m = model_tw
    fit, fit_sat = {}, {}

    wfs_all = collect_tw(events, bar_ids=None)
    if plot_event is not None:
        wfs_all = [wf for wf in wfs_all if wf["_eventN"] == plot_event]
    wfs_ok = [wf for wf in wfs_all if _valid_grid(wf)]

    n_bad = len(wfs_all) - len(wfs_ok)
    if n_bad > 0:
        print(f"  TW fit: {n_bad} waveforms with an invalid time axis skipped")
    if not wfs_ok:
        return fit, fit_sat

    t_grid, amp_res = resample_wfs(wfs_ok, step=RESAMPLE_STEP)

    results = apply_model_fit(amp_res, wfs_ok, t_grid, m["model"], m["std"],
                              t_grid_model=m["t_grid"], t0_ref=m["t0_ref"], **m["params"])
    _store_fit(wfs_ok, results, fit, m)

    if plot_event is not None:
        plot_model_fit_results(results, t_grid, amp_res, m["model"], n_sample=len(wfs_ok),
                               t0_ref=m["t0_ref"], wfs=wfs_ok)

    if sat_level_mv is not None:
        amp_sat, n_sat, _ = apply_saturation(amp_res, sat_level_mv=sat_level_mv)
        results_sat = apply_model_fit(amp_sat, wfs_ok, t_grid, m["model"], m["std"],
                                      t_grid_model=m["t_grid"], t0_ref=m["t0_ref"],
                                      sat_level=sat_level_mv / 1e3, **m["params"])
        _store_fit(wfs_ok, results_sat, fit_sat, m)

    return fit, fit_sat


# =============================================================================
# ONE FILE
# =============================================================================

def reconstruct(events):
    """Standard waveform reconstruction (everything except the template fit)."""
    events, *_ = apply_baseline(events, method="mean")
    cleanup_baseline(events)
    events, *_ = apply_amplitude(events, n_sigma=N_SIGMA, st_amp_max=ST_AMP_MAX)
    events = filter_pileup_events(events)
    events = build_st_sum(events)

    # the three CFD methods, each stored in t_<method>
    for method in CFD_METHODS:
        events, *_ = apply_cfd(events, method_st=method, method_tw=method)
        for ev in events:
            for wf in ev["ST"] + ev["TW"]:
                wf[f"t_{method}"] = wf.get("time_cfd")
            if ev.get("ST_sum"):
                ev["ST_sum"][f"t_{method}"] = ev["ST_sum"].get("time_cfd")

    # time_cfd back to the simple CFD: used as starting point of the fit
    # and for the CFD time selection in model.collect_*
    for ev in events:
        for wf in ev["ST"] + ev["TW"]:
            wf["time_cfd"] = wf.get("t_simple")
        if ev.get("ST_sum"):
            ev["ST_sum"]["time_cfd"] = ev["ST_sum"].get("t_simple")

    events, _ = apply_charge(events, method="integral")
    events = apply_tw_map(events)

    for ev in events:
        for wf in ev["ST"] + ev["TW"]:
            wf["_eventN"] = ev["eventN"]
        if ev.get("ST_sum"):
            ev["ST_sum"]["_eventN"] = ev["eventN"]

    return events


def process_file(bin_path, file_idx, run_id, do_fit=True, sat_level_mv=None,
                 model_tw_ext=None, plot_event=None):
    """
    Analyse one .bin file and return the list of result events:
        {"run_id", "file_idx" (1-based), "eventN", "nST", "nTW",
         "ST":     [{boardId, channelId, amplitude, noise, pedestal, signal,
                     t_simple, t_tangent, t_bipolar}, ...],
         "ST_sum": {amplitude, noise, n_summed, t_simple, t_tangent, t_bipolar},
         "TW":     [{boardId, channelId, bar_id, layer, end, amplitude, noise,
                     pedestal, signal, charge, rise_time, t_simple, t_tangent,
                     t_bipolar, t_fit, charge_fit, chi2_fit, npts_fit,
                     t_fit_sat, charge_fit_sat, chi2_fit_sat, npts_fit_sat}, ...]}
    Times in ns, amplitudes in V, charges in V*ns; missing values are None.
    """
    events = reconstruct(load_waveforms(bin_path))

    fit, fit_sat = {}, {}
    if do_fit:
        try:
            model_tw = model_tw_ext if model_tw_ext is not None else _build_model_tw(events)
            fit, fit_sat = _run_fit_tw(events, model_tw, sat_level_mv=sat_level_mv, plot_event=plot_event)
        except Exception as e:
            print(f"  Warning: TW fit failed ({e}), t_fit = None")

    def get(d, key, field):
        r = d.get(key)
        return _f(r[field]) if r is not None else None

    result_events = []
    for ev in events:
        ev_n   = ev["eventN"]
        st_sum = ev.get("ST_sum")

        st_list = [{
            "boardId":   wf["boardId"],
            "channelId": wf["channelId"],
            "amplitude": float(wf["amplitude"]),
            "noise":     float(wf["noise"]),
            "pedestal":  float(wf["pedestal"]),
            "signal":    bool(wf["signal"]),
            "t_simple":  _f(wf.get("t_simple")),
            "t_tangent": _f(wf.get("t_tangent")),
            "t_bipolar": _f(wf.get("t_bipolar")),
        } for wf in ev["ST"]]

        st_sum_dict = None
        if st_sum is not None:
            st_sum_dict = {
                "amplitude": float(st_sum["amplitude"]),
                "noise":     float(st_sum["noise"]),
                "n_summed":  int(st_sum["n_summed"]),
                "t_simple":  _f(st_sum.get("t_simple")),
                "t_tangent": _f(st_sum.get("t_tangent")),
                "t_bipolar": _f(st_sum.get("t_bipolar")),
            }

        tw_list = []
        for wf in ev["TW"]:
            if not wf.get("signal"):
                continue
            key = (ev_n, wf.get("bar_id"), wf.get("end"))
            tw_list.append({
                "boardId":        wf["boardId"],
                "channelId":      wf["channelId"],
                "bar_id":         wf.get("bar_id"),
                "layer":          wf.get("layer"),
                "end":            wf.get("end"),
                "amplitude":      float(wf["amplitude"]),
                "noise":          float(wf["noise"]),
                "pedestal":       float(wf["pedestal"]),
                "signal":         bool(wf.get("signal", False)),
                "charge":         _f(wf.get("charge")),
                "rise_time":      _f(wf.get("rise_time")),
                "t_simple":       _f(wf.get("t_simple")),
                "t_tangent":      _f(wf.get("t_tangent")),
                "t_bipolar":      _f(wf.get("t_bipolar")),
                "t_fit":          get(fit, key, "t"),
                "charge_fit":     get(fit, key, "charge"),
                "chi2_fit":       get(fit, key, "chi2"),
                "npts_fit":       get(fit, key, "npts"),
                "t_fit_sat":      get(fit_sat, key, "t"),
                "charge_fit_sat": get(fit_sat, key, "charge"),
                "chi2_fit_sat":   get(fit_sat, key, "chi2"),
                "npts_fit_sat":   get(fit_sat, key, "npts"),
            })

        result_events.append({
            "run_id":   run_id,
            "file_idx": file_idx + 1,
            "eventN":   ev_n,
            "nST":      ev["nST"],
            "nTW":      ev["nTW"],
            "ST":       st_list,
            "ST_sum":   st_sum_dict,
            "TW":       tw_list,
        })

    return result_events


def _process_file_worker(args_tuple):
    """
    Pool worker: runs process_file with the stdout redirected to a log file
    per .bin (logs/<run>_<idx>.txt). Returns (events, file_idx) or the exception.
    """
    bin_path, file_idx, run_id, do_fit, sat_level_mv, log_dir, model_tw_ext = args_tuple
    log_path = os.path.join(log_dir, f"{run_id}_{file_idx+1:04d}.txt")

    real_stdout = sys.stdout
    try:
        with open(log_path, "w") as log_f:
            sys.stdout = log_f
            evs = process_file(bin_path, file_idx=file_idx, run_id=run_id, do_fit=do_fit,
                               sat_level_mv=sat_level_mv, model_tw_ext=model_tw_ext)
        return evs, file_idx
    except Exception as e:
        return e
    finally:
        sys.stdout = real_stdout


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Waveform analysis of all the .bin files of one run")
    parser.add_argument("run_dir", help="Folder with the .bin files of the run (or its parent folder)")
    parser.add_argument("out_dir", help="Output folder")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--no-fit", action="store_true", help="Skip the template fit")
    parser.add_argument("--saturation", type=float, default=None,
                        help="Artificial saturation level in mV: adds the fit of the saturated "
                             "waveforms (t_fit_sat, charge_fit_sat, ...)")
    parser.add_argument("--cores", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--save-template", default=None,
                        help="Build the TW template on the first file and save it to this .pkl")
    parser.add_argument("--load-template", default=None,
                        help="Load the TW template from this .pkl instead of building it for every file")
    parser.add_argument("--file-idx", type=int, default=None,
                        help="Process only the file with this index (1-based)")
    parser.add_argument("--plot-event", type=int, default=None,
                        help="Plot the fitted TW waveforms of this eventN")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) / str(args.run)
    if not run_dir.is_dir():
        run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.run}.pkl"

    t_run_start = time.perf_counter()
    bin_files = sorted(run_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin file found in {run_dir}")
        sys.exit(1)

    if args.file_idx is not None:
        bin_files = [bin_files[args.file_idx - 1]]
        file_idx_offset = args.file_idx - 1
    else:
        file_idx_offset = 0

    do_fit = not args.no_fit

    print(f"\n{'='*60}")
    print(f"Run {args.run}  |  {len(bin_files)} .bin files")
    print(f"Input:  {run_dir}")
    print(f"Output: {out_path}")
    print(f"Fit:    {'yes' if do_fit else 'no'}")
    if args.saturation:
        print(f"Saturation: {args.saturation:.0f} mV")
    print(f"{'='*60}")

    # TW template: loaded, or built on the first file and saved
    model_tw_ext = None
    if args.load_template:
        with open(args.load_template, "rb") as f:
            model_tw_ext = pickle.load(f)
        print(f"TW template loaded from {args.load_template}")
    elif args.save_template:
        events_model = reconstruct(load_waveforms(bin_files[0]))
        model_tw_ext = _build_model_tw(events_model)
        plot_model(model_tw_ext["t_grid"], model_tw_ext["model"],
                   model_tw_ext["std"], t0_ref=model_tw_ext["t0_ref"])
        with open(args.save_template, "wb") as f:
            pickle.dump(model_tw_ext, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nTW template saved to {args.save_template}")

    all_results = []
    n_ok, n_err = 0, 0

    if args.cores == 1:
        for i, bin_path in enumerate(bin_files):
            t0 = time.perf_counter()
            print(f"\n[{i+1:3d}/{len(bin_files)}] {bin_path.name}")
            try:
                evs = process_file(bin_path, file_idx=i + file_idx_offset, run_id=args.run,
                                   do_fit=do_fit, sat_level_mv=args.saturation,
                                   model_tw_ext=model_tw_ext, plot_event=args.plot_event)
                all_results.extend(evs)
                n_ok += 1
                print(f"  -> {len(evs)} events  ({time.perf_counter() - t0:.1f} s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                n_err += 1

    else:
        import multiprocessing as mp
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(str(bp), i + file_idx_offset, args.run, do_fit, args.saturation, str(log_dir), model_tw_ext)
                 for i, bp in enumerate(bin_files)]
        print(f"Starting a pool of {args.cores} processes (per-file logs in {log_dir})")
        t_pool_start = time.perf_counter()
        with mp.Pool(processes=args.cores) as pool:
            for i, result in enumerate(pool.imap(_process_file_worker, tasks, chunksize=1)):
                bin_name = bin_files[i].name
                if isinstance(result, Exception):
                    print(f"  ERROR [{bin_name}]: {result}")
                    n_err += 1
                else:
                    evs, _ = result
                    all_results.extend(evs)
                    n_ok += 1
                    print(f"  [{i+1:3d}/{len(bin_files)}] {bin_name} -> {len(evs)} events  "
                          f"(elapsed: {time.perf_counter()-t_pool_start:.0f} s)")

    with open(out_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1e6
    t_tot = time.perf_counter() - t_run_start
    print(f"\n{'='*60}")
    print(f"Total time: {t_tot:.0f} s  ({t_tot/60:.1f} min)")
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"Events: {len(all_results)}")
    print(f"Files OK: {n_ok}  |  Errors: {n_err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
