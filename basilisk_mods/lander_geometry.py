"""Lander B geometry for Basilisk, measured from the team's model (lunar_ops/showcase/models/lander_B.glb, KAN-47).

Body frame (Basilisk): +z up (main-engine thrust), +x front, origin = model origin, which sits LANDER_FEET_M above the ground
contact when the lander stands on flat ground. glTF (Y up, front -Z) -> body: x_b = -z_g, y_b = -x_g, z_b = y_g.

Attitude thrusters: the eight `rcs0..7_nozzle` parts. Exhaust direction = from the bracket centre to the nozzle centre (the
model points all eight outward, horizontally, 0.22 m above the origin). Thrust acts opposite to the exhaust. With this
layout yaw authority is small (the nozzles are almost radial); that is the model's geometry, used as is.
Footpads: the `leg0..3_footpad` parts, bottom centre.
"""

from __future__ import annotations

import numpy as np

LANDER_FEET_M = 1.45  # [m] model origin above the footpad contact plane

# [m] nozzle exit positions, body frame (rcs0..rcs7)
RCS_POS_B = np.array([
    [0.411, -0.974, 0.22], [0.974, -0.411, 0.22], [0.974, 0.411, 0.22], [0.411, 0.974, 0.22],
    [-0.411, 0.974, 0.22], [-0.974, 0.411, 0.22], [-0.974, -0.411, 0.22], [-0.411, -0.974, 0.22],
])
# [-] exhaust directions (unit), body frame; thrust direction = -RCS_EXHAUST_B
RCS_EXHAUST_B = np.array([
    [0.47, -0.89, 0.0], [0.89, -0.47, 0.0], [0.89, 0.47, 0.0], [0.47, 0.89, 0.0],
    [-0.47, 0.89, 0.0], [-0.89, 0.47, 0.0], [-0.89, -0.47, 0.0], [-0.47, -0.89, 0.0],
])
RCS_EXHAUST_B = RCS_EXHAUST_B / np.linalg.norm(RCS_EXHAUST_B, axis=1, keepdims=True)

# [m] footpad contact points, body frame (leg0..leg3)
FOOTPADS_B = np.array([[1.55, -1.55, -1.45], [1.55, 1.55, -1.45], [-1.55, 1.55, -1.45], [-1.55, -1.55, -1.45]])


def footpad_points_N(r_N, BN) -> np.ndarray:
    """Inertial positions [m] of the four footpads. ``BN`` maps inertial vectors to body."""
    return np.asarray(r_N)[None, :] + (np.asarray(BN).T @ FOOTPADS_B.T).T


def footpad_report(clearance_m, ground_m) -> dict:
    """First contact and how uneven the ground under the pads is.

    clearance_m  height of each pad above the terrain [m] at the moment of first contact
    ground_m     terrain height under each pad [m]; the plane through these is what the lander settles onto, so its tilt
                 is independent of the lander's own attitude
    """
    c, g = np.asarray(clearance_m, float), np.asarray(ground_m, float)
    A = np.c_[FOOTPADS_B[:, 0], FOOTPADS_B[:, 1], np.ones(4)]
    (a, b, _), *_ = np.linalg.lstsq(A, g, rcond=None)
    return {"first_pad": int(np.argmin(c)), "pad_clearance_m": [round(float(x), 3) for x in c],
            "ground_under_pads_m": [round(float(x), 3) for x in g], "settle_tilt_deg": round(float(np.degrees(np.arctan(np.hypot(a, b)))), 2),
            "ground_spread_m": round(float(g.max() - g.min()), 3)}
