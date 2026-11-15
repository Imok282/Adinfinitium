"""Export the real rover brain v3.1 traverse from the end-to-end test on site DM1 for mission control.

Usage (read-only on the input; run with the venv that has numpy):
    ~/venvs/adi/bin/python -I website/tools/export_rover_e2e.py <e2e_arrays_final_v31.npz> website/assets/data/rover_route_dm1_v31.json

Input: mini_work/adinfinitium-final/out/e2e_arrays_final_v31.npz from the Mac mini hand-off (15 Nov 2026), written by
scripts/e2e_test_v31.py --site DM1 --date 2027-06-01 --n 512 --seed 1234 (records/final/e2e_result_final_v31.json).
Rover shown = rover index 0, the lowest index that returned under the strict rule (the rule the test script used; none
chosen by eye). Arrays: traj (step, rover, row/col in 5 m world pixels), battery (step, rover, Wh), targets_rc, target_idx,
world_height, world_hazard (hazard U-Net probability), world_sun_mean (64 x 64 mean visible Sun fraction, physics ray-cast).
Output uses the same schema as rover_route_site23.json (x east, z south, y up; metres from the lander) plus a few extras.
"""
import json
import sys

import numpy as np

PX = 5.0  # metres per world pixel (LOLA 5 m grid)


def main(src, out):
    z = np.load(src, allow_pickle=False)
    traj = z["traj"].astype(float)            # (steps, rovers, 2) row, col
    lr, lc = (float(v) for v in z["lander_rc"])
    H = z["world_height"].astype(float)
    n_steps, n_rov, _ = traj.shape
    i = 0
    assert bool(z["strict_returned"][0]), "rover 0 is expected to be the first strict-rule return"
    t = traj[:, i, :]
    h0 = float(H[int(round(lr)), int(round(lc))])

    def xyz(r, c):
        rr = int(np.clip(round(r), 0, H.shape[0] - 1)); cc = int(np.clip(round(c), 0, H.shape[1] - 1))
        return [round((c - lc) * PX, 1), round(float(H[rr, cc]) - h0, 1), round((r - lr) * PX, 1)]

    points = [xyz(r, c) for r, c in t]
    seg = np.hypot(*np.diff(t, axis=0).T) * PX
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    T = z["targets_rc"][int(z["target_idx"][i])]
    dT = np.hypot(*(t - T).T) * PX
    reach = int(np.argmin(dT)) if bool(z["reached"][i]) else None
    sun = z["world_sun_mean"]; s = H.shape[0] / sun.shape[0]
    sun_frac = [round(float(sun[min(sun.shape[0] - 1, int(r // s)), min(sun.shape[1] - 1, int(c // s))]), 3) for r, c in t]
    hz = z["world_hazard"] >= 0.5
    hb = hz.reshape(hz.shape[0] // 2, 2, hz.shape[1] // 2, 2).any(axis=(1, 3))   # 10 m blocks
    rr, cc = np.nonzero(hb)
    hazard_cells = [[round((c * 2 + 0.5 - lc) * PX, 1), round((r * 2 + 0.5 - lr) * PX, 1)] for r, c in zip(rr, cc)]
    targets = [xyz(r, c) for r, c in z["targets_rc"]]
    d = {
        "meta": {
            "provenance": "modelled",
            "source": "Mac mini hand-off: mini_work/adinfinitium-final/out/e2e_arrays_final_v31.npz (end-to-end test, records/final/e2e_result_final_v31.json), rover 0 of 512",
            "frame": "scene: origin = lander (world centre), x east, z south, y up (height relative to the lander cell); metres",
            "units": "m, s, Wh",
            "note": "MODELLED: rover brain v3.1 (scripted, 0 learned parameters) on NASA LOLA 5 m terrain of site DM1, 2027-06-01; route cost from the hazard U-Net map, light from the physics ray-cast. Targets = PSR edges (dark and cold, NOT confirmed ice). Sun fraction = monthly mean at the rover's cell, not hourly.",
        },
        "site": "DM1", "brain": "v3.1", "run_id": "rover 0", "policy": "rover brain v3.1",
        "outcome": "returned", "date": "2027-06-01", "dt_s": 3600.0,
        "t_s": [k * 3600.0 for k in range(n_steps)],
        "points": points, "dist_m": [round(float(v), 1) for v in dist],
        "total_dist_m": round(float(dist[-1]), 1), "duration_h": n_steps - 1,
        "max_range_m": round(float(np.hypot(*(t - [lr, lc]).T).max() * PX), 1),
        "ice_reach_index": reach, "ice_reach_t_s": reach * 3600.0 if reach is not None else None,
        "target_distance_m": round(float(np.hypot(*(T - [lr, lc])) * PX), 1),
        "min_distance_to_target_m": round(float(dT.min()), 2),
        "end_distance_from_lander_m": round(float(np.hypot(*(t[-1] - [lr, lc])) * PX), 2),
        "ice_targets": targets, "ice_source": "NASA PSR edge (dark and cold; NOT confirmed ice)",
        "hazard_cells_xz": hazard_cells, "hazard_cell_m": 10.0,
        "hazard_rule": "hazard U-Net probability >= 0.5 (any 5 m pixel in a 10 m block)",
        "battery_wh": [round(float(v), 2) for v in z["battery"][:, i]], "battery_cap_wh": 100.0,
        "sun_frac": sun_frac, "sun_label": "monthly mean sunlit fraction at the rover's cell",
        "battery_source": "battery array of the same end-to-end run (simulated)",
        "fleet": {"rovers": int(n_rov), "strict_returned": int(z["strict_returned"].sum()),
                  "strict_lost": int(z["strict_lost"].sum()), "alive_without_completed_trip": int(z["alive_no_trip"].sum()),
                  "rule": "strict: lost = battery empty at the end of any hour, anywhere"},
    }
    with open(out, "w") as f:
        json.dump(d, f, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
