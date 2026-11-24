"""
tw_calibration.py — Calibrations applied to the bars reconstructed from the waveforms.

    apply_energy_calibration   charge -> deposited energy (Birks law)
    apply_tof_calibration      per-bar TOF offsets
    add_coordinate             position along the bar from t_R - t_L

Constants from the CNAO2025 calibration of the TW:
    BIRKS_PARAMS     Birks fit Area = s*E / (1 + k*E) for each bar
                     (TW calibration note, CNAO2025)
    TOF_BAR_OFFSET   TATW_Tof_Calibration_perBar.cal (SHOE)
    DELTA_TIME_MAP   TATWdeltaTimeMap.map (SHOE): v (cm/ns) and offset (cm)
    LAYER0_X/LAYER1_Y  bar centres, TATWdetector.geo (SHOE)

Usage as a module:
    from tw_calibration import apply_energy_calibration, apply_tof_calibration, add_coordinate
"""

import numpy as np

# =============================================================================
# CHARGE -> ENERGY (Birks)
# =============================================================================

# bar_id -> k (1/MeV), s (a.u./MeV) and their errors (not available for every bar)
BIRKS_PARAMS = {
     0: {"k": 0.0064985, "s": 0.44507},
     1: {"k": 0.021,     "s": 0.55,    "k_err": 0.007,   "s_err": 0.04},
     2: {"k": 0.009,     "s": 0.32,    "k_err": 0.004,   "s_err": 0.04},
     3: {"k": 0.011,     "s": 0.385,   "k_err": 0.0003,  "s_err": 0.004},
     4: {"k": 0.0104,    "s": 0.355,   "k_err": 0.0004,  "s_err": 0.004},
     5: {"k": 0.0107,    "s": 0.407,   "k_err": 0.0004,  "s_err": 0.004},
     6: {"k": 0.0099,    "s": 0.362,   "k_err": 0.0005,  "s_err": 0.005},
     7: {"k": 0.0103,    "s": 0.329,   "k_err": 0.0005,  "s_err": 0.005},
     8: {"k": 0.0114,    "s": 0.428,   "k_err": 0.0006,  "s_err": 0.008},
     9: {"k": 0.011,     "s": 0.34,    "k_err": 0.002,   "s_err": 0.02},
    10: {"k": 0.0116,    "s": 0.322,   "k_err": 0.0006,  "s_err": 0.005},
    11: {"k": 0.0109,    "s": 0.38,    "k_err": 0.0003,  "s_err": 0.003},
    12: {"k": 0.0103,    "s": 0.327,   "k_err": 0.0003,  "s_err": 0.003},
    13: {"k": 0.0105,    "s": 0.355,   "k_err": 0.0003,  "s_err": 0.003},
    14: {"k": 0.0104,    "s": 0.382,   "k_err": 0.0003,  "s_err": 0.003},
    15: {"k": 0.0106,    "s": 0.489,   "k_err": 0.0003,  "s_err": 0.004},
    16: {"k": 0.01,      "s": 0.36,    "k_err": 0.001,   "s_err": 0.01},
    17: {"k": 0.013,     "s": 0.34,    "k_err": 0.003,   "s_err": 0.02},
    18: {"k": 0.012,     "s": 0.42,    "k_err": 0.003,   "s_err": 0.02},
    19: {"k": 0.0048374, "s": 0.43824},
    20: {"k": 0.01161,   "s": 0.45176},
    21: {"k": 0.01,      "s": 0.52,    "k_err": 0.003,   "s_err": 0.03},
    22: {"k": 0.009,     "s": 0.35,    "k_err": 0.001,   "s_err": 0.01},
    23: {"k": 0.0098,    "s": 0.345,   "k_err": 0.0005,  "s_err": 0.006},
    24: {"k": 0.0103,    "s": 0.5,     "k_err": 0.0003,  "s_err": 0.004},
    25: {"k": 0.0104,    "s": 0.383,   "k_err": 0.0003,  "s_err": 0.003},
    26: {"k": 0.0101,    "s": 0.435,   "k_err": 0.0005,  "s_err": 0.006},
    27: {"k": 0.0092,    "s": 0.342,   "k_err": 0.0005,  "s_err": 0.005},
    28: {"k": 0.0116,    "s": 0.35,    "k_err": 0.0005,  "s_err": 0.005},
    29: {"k": 0.0113,    "s": 0.343,   "k_err": 0.001,   "s_err": 0.01},
    30: {"k": 0.0109,    "s": 0.313,   "k_err": 0.0004,  "s_err": 0.003},
    31: {"k": 0.0096,    "s": 0.317,   "k_err": 0.0003,  "s_err": 0.002},
    32: {"k": 0.0099,    "s": 0.381,   "k_err": 0.0004,  "s_err": 0.004},
    33: {"k": 0.0102,    "s": 0.399,   "k_err": 0.0002,  "s_err": 0.002},
    34: {"k": 0.0102,    "s": 0.372,   "k_err": 0.0002,  "s_err": 0.003},
    35: {"k": 0.0099,    "s": 0.282,   "k_err": 0.0005,  "s_err": 0.004},
    36: {"k": 0.011,     "s": 0.36,    "k_err": 0.001,   "s_err": 0.009},
    37: {"k": 0.011,     "s": 0.33,    "k_err": 0.003,   "s_err": 0.02},
    38: {"k": 0.016618,  "s": 0.41678},
    39: {"k": 0.021454,  "s": 0.39386},
}


def charge_to_energy(bar_id, charge_Vns):
    """
    Deposited energy (MeV) from the bar charge (V*ns) by inverting the Birks law
        Area = s*E / (1 + k*E)  ->  E = Area / (s - k*Area)
    (the Area of the Birks fit is the charge in V*ns). None if Area >= s/k.
    """
    params = BIRKS_PARAMS.get(bar_id)
    if params is None or charge_Vns is None:
        return None

    k, s = params["k"], params["s"]
    denom = s - k * charge_Vns
    if denom <= 0:
        return None

    return charge_Vns / denom


def apply_energy_calibration(bars):
    """
    Charge -> energy for all the bars (vectorised per bar). Adds
    b["energy"] (from charge_bar) and b["energy_fit_sat"] (from charge_fit_sat),
    in MeV, None where not available.
    """
    bar_ids = np.array([b["bar_id"] for b in bars])

    for field_in, field_out in [("charge_bar", "energy"), ("charge_fit_sat", "energy_fit_sat")]:
        charges = np.array([b.get(field_in) if b.get(field_in) is not None else np.nan
                            for b in bars])
        energies = np.full(len(bars), np.nan)

        for bid in np.unique(bar_ids):
            params = BIRKS_PARAMS.get(int(bid))
            if params is None:
                continue
            mask = (bar_ids == bid) & ~np.isnan(charges)
            if not mask.any():
                continue
            k, s = params["k"], params["s"]
            area_au = charges[mask]
            denom = s - k * area_au
            valid = denom > 0
            idx = np.where(mask)[0][valid]
            energies[idx] = area_au[valid] / denom[valid]

        for b, e in zip(bars, energies):
            b[field_out] = None if np.isnan(e) else float(e)

    n_ok     = sum(1 for b in bars if b["energy"] is not None)
    n_ok_sat = sum(1 for b in bars if b["energy_fit_sat"] is not None)
    print(f"\n--- Energy calibration (Birks) ---")
    print(f"Calibrated bars (charge_bar):     {n_ok}/{len(bars)}")
    print(f"Calibrated bars (charge_fit_sat): {n_ok_sat}/{len(bars)}")

    return bars


# =============================================================================
# TOF
# =============================================================================

# per-bar TOF offset (ns), TATW_Tof_Calibration_perBar.cal
TOF_BAR_OFFSET = {
     0: 2.1541,  1: 2.0471,  2: 1.6333,  3: 1.5359,  4: 1.9329,
     5: 1.8846,  6: 1.6797,  7: 1.3351,  8: 1.7053,  9: 1.6088,
    10: 1.5206, 11: 1.7472, 12: 1.7039, 13: 1.7428, 14: 1.7761,
    15: 2.0184, 16: 1.9162, 17: 1.8290, 18: 1.5510, 19: 2.0419,
    20: 2.1009, 21: 1.8634, 22: 1.8685, 23: 2.1468, 24: 1.8427,
    25: 1.7667, 26: 1.4086, 27: 1.6035, 28: 1.4812, 29: 1.7051,
    30: 1.6619, 31: 1.5257, 32: 1.4498, 33: 1.5313, 34: 1.8530,
    35: 1.8635, 36: 1.7608, 37: 1.4064, 38: 2.0085, 39: 2.1090,
}

# additional per-bar shift (ns) of the TOF of this waveform analysis with respect to the
# SHOE reconstruction, for which TOF_BAR_OFFSET was derived; subtracted together with it
TOF_EXTRA_SHIFT = {
     0: 1.19,  1: 1.30,  2: 0.93,  3: 1.27,  4: 1.01,
     5: 0.90,  6: 1.08,  7: 1.08,  8: 1.03,  9: 0.78,
    10: 0.88, 11: 0.90, 12: 0.99, 13: 0.87, 14: 0.87,
    15: 0.81, 16: 0.71, 17: 0.81, 18: 0.76, 19: 1.04,
    20: 0.85, 21: 0.72, 22: 0.70, 23: 0.79, 24: 0.93,
    25: 0.78, 26: 0.97, 27: 1.27, 28: 1.03, 29: 0.96,
    30: 0.94, 31: 1.22, 32: 1.08, 33: 0.89, 34: 0.98,
    35: 0.71, 36: 1.02, 37: 1.09, 38: 1.06, 39: 0.77,
}


def apply_tof_calibration(bars):
    """tof = tof_raw - TOF_BAR_OFFSET[bar] - TOF_EXTRA_SHIFT[bar], in place on b["tof"]."""
    bar_ids = np.array([b.get("bar_id") for b in bars])
    tofs    = np.array([b.get("tof") if b.get("tof") is not None else np.nan for b in bars])

    corrected = tofs.copy()
    n_ok = 0

    for bid in np.unique(bar_ids):
        offset = TOF_BAR_OFFSET.get(bid)
        if offset is None:
            continue
        mask = (bar_ids == bid) & ~np.isnan(tofs)
        extra = TOF_EXTRA_SHIFT.get(bid, 0.0)
        corrected[mask] = tofs[mask] - offset - extra
        n_ok += mask.sum()

    for b, t in zip(bars, corrected):
        b["tof"] = None if np.isnan(t) else float(t)

    print(f"\n--- Per-bar TOF calibration ---")
    print(f"Calibrated bars: {n_ok}/{len(bars)}")

    return bars


# =============================================================================
# POSITION ALONG THE BAR
# =============================================================================

# bar_id -> (v [cm/ns], offset [cm]), TATWdeltaTimeMap.map
DELTA_TIME_MAP = {
     0: (15.280226,  0.414808),   1: (15.391466,  0.134272),
     2: (15.637354, -0.279352),   3: (15.769614,  0.098988),
     4: (15.920442, -0.139538),   5: (15.587784,  0.096428),
     6: (15.855530, -0.090854),   7: (15.678014,  4.609668),
     8: (15.507877,  0.051042),   9: (15.726864,  0.225301),
    10: (15.905313,  0.358378),  11: (15.382804,  0.481756),
    12: (15.812275,  0.481613),  13: (15.611340,  0.151847),
    14: (15.424770,  0.298203),  15: (15.706867,  1.333040),
    16: (15.914541,  1.166399),  17: (15.764314,  0.100573),
    18: (15.731082,  4.762064),  19: (15.592971, -0.100186),
    20: (15.278465,  0.131990),  21: (15.404871,  0.199837),
    22: (15.689667,  0.471156),  23: (15.438470,  0.607979),
    24: (15.396446,  0.068021),  25: (15.355130,  0.515307),
    26: (15.411773,  4.555207),  27: (15.514510, -0.133742),
    28: (15.676826,  0.032309),  29: (15.492343,  0.020220),
    30: (15.478460, -0.052026),  31: (15.350779,  0.089271),
    32: (15.770718,  0.458227),  33: (15.337395,  0.418114),
    34: (15.624340,  0.785334),  35: (15.665695,  0.053866),
    36: (15.508069, -0.018826),  37: (15.676788,  5.022496),
    38: (15.279565, -0.466422),  39: (15.558751, -0.513591),
}


def add_coordinate(bars):
    """b["coordinate"] = (t_R - t_L) * v / 2 + offset (cm), for the bars that do not have it yet."""
    for b in bars:
        if b.get("coordinate") is not None:
            continue
        if b.get("t_L") is None or b.get("t_R") is None:
            b["coordinate"] = None
            continue
        v, offset = DELTA_TIME_MAP.get(b["bar_id"], (None, None))
        if v is None:
            b["coordinate"] = None
            continue
        b["coordinate"] = (b["t_R"] - b["t_L"]) * v / 2 + offset
    return bars


# bar centres (cm), TATWdetector.geo: bar 0 -> +19 cm, bar 19 -> -19 cm
LAYER0_X = {i: 19.0 - 2.0*i for i in range(20)}   # layer 0 (rear, vertical bars): x
LAYER1_Y = {i: 19.0 - 2.0*i for i in range(20)}   # layer 1 (front, horizontal bars): y
