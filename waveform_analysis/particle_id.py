"""
particle_id.py — Charge identification (Z) of the TW hits from (TOF, E_loss).

Same algorithm as TATWparCal::GetChargeZ / ComputeBBDistance in SHOE.
For every candidate Z the energy loss is parametrised as
    E_BB(tof) = Norm * Z^2 * (Const * tof^2 - 1)
and the point of the curve closest to the measured (tof, E_loss) is found by
bisection on the derivative of the euclidean distance. The hit gets the Z of
the closest curve. A Z=1 hit too far below its curve is set to Z=0 (neutron
or unidentified), as in SelectProtonsFromNeutrons.

Parameters: TATW_BBparameters_12C_200.cfg (SHOE, CNAO2025, 12C at 200 MeV/u).
The TOF must have the same calibration as the one used to fit them.

Usage as a module:
    from particle_id import apply_particle_id
    bars = apply_particle_id(bars)    # needs b["energy"], b["tof"], b["layer"]
"""

import numpy as np
from multiprocessing import Pool, cpu_count

# layer (0 = rear, 1 = front, as in tw_map.py) -> Z -> parameters
BB_PARAMS = {
    0: {  # LayerY
        1: {"norm": 0.28632, "const": 0.034845},
        2: {"norm": 0.30184, "const": 0.033549},
        3: {"norm": 0.33114, "const": 0.031323},
        4: {"norm": 0.4285,  "const": 0.025726},
        5: {"norm": 0.49078, "const": 0.023228},
        6: {"norm": 0.65024, "const": 0.01905},
    },
    1: {  # LayerX
        1: {"norm": 0.23786, "const": 0.040868},
        2: {"norm": 0.2485,  "const": 0.039624},
        3: {"norm": 0.27403, "const": 0.036649},
        4: {"norm": 0.2809,  "const": 0.036119},
        5: {"norm": 0.23377, "const": 0.042326},
        6: {"norm": 0.17973, "const": 0.053624},
    },
}

TOF_MIN = 8.0    # ns — search interval on the curve
TOF_MAX = 40.0   # ns

DIST_Z1_MEAN  = -0.055486   # mean and sigma of the signed distance from the Z=1 curve
DIST_Z1_SIGMA = 0.14658
N_SIGMA_PN    = 5.0         # proton/neutron selection threshold

Z_MAX = 6                   # beam charge (12C): Z = 1..6 are tested

DELTA_TOF_MIN = 1e-4        # ns — bisection precision


def _bb_eloss(tof, Z, norm, const):
    return norm * Z**2 * (const * tof**2 - 1)


def _bb_eloss_prime(tof, Z, norm, const):
    return norm * Z**2 * 2 * const * tof


def _dist(tof, edep, x, bb_x):
    return np.sqrt((tof - x)**2 + (edep - bb_x)**2)


def _dist_prime(tof, edep, x, bb_x, bb_prime_x, dist):
    if dist == 0:
        return 0.0
    return (-1.0 / dist) * ((tof - x) + bb_prime_x * (edep - bb_x))


def _min_dist_to_curve(tof, edep, Z, norm, const):
    """
    Distance between (tof, edep) and the closest point of the curve of charge Z.
    The search interval [TOF_MIN, TOF_MAX] is first shrunk until the derivative
    of the distance changes sign, then bisected.
    Returns (distance, E_BB at the closest point) or (None, None).
    """
    xmin, xmax = TOF_MIN, TOF_MAX

    bb_x = _bb_eloss(xmin, Z, norm, const)
    bb_prime_x = _bb_eloss_prime(xmin, Z, norm, const)
    dist = _dist(tof, edep, xmin, bb_x)
    f_prime_min = _dist_prime(tof, edep, xmin, bb_x, bb_prime_x, dist)

    while f_prime_min > 0 and xmin < xmax:
        xmin += DELTA_TOF_MIN * 100
        bb_x = _bb_eloss(xmin, Z, norm, const)
        bb_prime_x = _bb_eloss_prime(xmin, Z, norm, const)
        dist = _dist(tof, edep, xmin, bb_x)
        f_prime_min = _dist_prime(tof, edep, xmin, bb_x, bb_prime_x, dist)

    bb_x = _bb_eloss(xmax, Z, norm, const)
    bb_prime_x = _bb_eloss_prime(xmax, Z, norm, const)
    dist = _dist(tof, edep, xmax, bb_x)
    f_prime_max = _dist_prime(tof, edep, xmax, bb_x, bb_prime_x, dist)

    while f_prime_max < 0 and xmax > xmin:
        xmax -= DELTA_TOF_MIN * 100
        bb_x = _bb_eloss(xmax, Z, norm, const)
        bb_prime_x = _bb_eloss_prime(xmax, Z, norm, const)
        dist = _dist(tof, edep, xmax, bb_x)
        f_prime_max = _dist_prime(tof, edep, xmax, bb_x, bb_prime_x, dist)

    if xmin >= xmax or f_prime_min * f_prime_max >= 0:
        return None, None

    while abs(xmax - xmin) > DELTA_TOF_MIN:
        xhalf = (xmin + xmax) / 2.0
        bb_x = _bb_eloss(xhalf, Z, norm, const)
        bb_prime_x = _bb_eloss_prime(xhalf, Z, norm, const)
        dist = _dist(tof, edep, xhalf, bb_x)
        f_prime_half = _dist_prime(tof, edep, xhalf, bb_x, bb_prime_x, dist)

        if f_prime_min * f_prime_half < 0:
            xmax = xhalf
        elif f_prime_half * f_prime_max < 0:
            xmin = xhalf
        else:
            break

    return dist, bb_x


def get_charge_z(edep, tof, layer):
    """
    Z of a hit with energy loss edep (MeV) and time of flight tof (ns) in the
    given layer: 1-6, 0 for a Z=1 candidate rejected as neutron, None if the
    hit cannot be identified.
    """
    if edep is None or tof is None or edep < 0:
        return None

    params_layer = BB_PARAMS.get(layer)
    if params_layer is None:
        return None

    dist_min = np.inf
    z_sel = None
    dist_z1_signed = None

    for Z in range(1, Z_MAX + 1):
        p = params_layer.get(Z)
        if p is None:
            continue
        dist, bb_x = _min_dist_to_curve(tof, edep, Z, p["norm"], p["const"])
        if dist is None:
            continue
        if dist < dist_min:
            dist_min = dist
            z_sel = Z
        if Z == 1:
            dist_z1_signed = -dist if edep < bb_x else dist

    # only hits too far BELOW the Z=1 curve are rejected
    if z_sel == 1 and dist_z1_signed is not None:
        if (dist_z1_signed - DIST_Z1_MEAN) <= -N_SIGMA_PN * DIST_Z1_SIGMA:
            z_sel = 0

    return z_sel


def _worker_charge_z(args):
    edep, tof, layer = args
    return get_charge_z(edep, tof, layer)


def apply_particle_id(bars, n_workers=None):
    """
    Charge identification of all the bars, in parallel (n_workers processes,
    default: all the cores). Adds b["chargeZ"].
    """
    if n_workers is None:
        n_workers = cpu_count()

    args_list = [(b.get("energy"), b.get("tof"), b.get("layer")) for b in bars]

    print(f"\n--- Particle ID — {n_workers} processes ---")
    with Pool(n_workers) as pool:
        results = pool.map(_worker_charge_z, args_list, chunksize=2000)

    n_ok = {z: 0 for z in range(0, Z_MAX + 1)}
    n_none = 0
    for b, z in zip(bars, results):
        b["chargeZ"] = z
        if z is None:
            n_none += 1
        else:
            n_ok[z] = n_ok.get(z, 0) + 1

    for z in range(1, Z_MAX + 1):
        print(f"  Z={z}: {n_ok.get(z,0)}")
    print(f"  Z=0 (neutron/rejected): {n_ok.get(0,0)}")
    print(f"  Not identified: {n_none}")

    return bars
