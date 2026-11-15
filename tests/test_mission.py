"""Full-mission Basilisk simulation (KAN-48): geometry helpers (fast) and one complete landing (slow)."""

import numpy as np
import pytest

pytest.importorskip("Basilisk")

from Basilisk.utilities import RigidBodyKinematics as rbk  # noqa: E402

from basilisk_mods.lander_geometry import FOOTPADS_B, footpad_report  # noqa: E402
from basilisk_mods.mission import tilt_only_reference  # noqa: E402


def test_tilt_only_reference_points_z_and_keeps_yaw():
    """The reference puts +z on the wanted direction by the smallest rotation: the body x axis stays as close as possible."""
    sigma = np.array([0.05, -0.1, 0.3])
    d = np.array([0.3, -0.2, 0.93]) / np.linalg.norm([0.3, -0.2, 0.93])
    C_RN = np.array(rbk.MRP2C(tilt_only_reference(sigma, d)))
    assert np.allclose(C_RN.T @ [0, 0, 1], d, atol=1e-9)
    C_BN = np.array(rbk.MRP2C(sigma))
    rot = np.degrees(np.arccos(np.clip((np.trace(C_RN @ C_BN.T) - 1) / 2, -1, 1)))
    tilt = np.degrees(np.arccos(np.clip((C_BN.T @ [0, 0, 1]) @ d, -1, 1)))
    assert abs(rot - tilt) < 1e-6        # the whole rotation is the tilt, no added yaw


def test_footpad_report_tilt_is_the_ground_plane():
    """Settle tilt comes from the ground under the pads, not from the pads' clearances."""
    slope = np.radians(10.0)
    ground = FOOTPADS_B[:, 0] * np.tan(slope)               # a 10 deg plane rising along +x
    rep = footpad_report(clearance_m=[0.3, 0.0, 0.5, 0.4], ground_m=ground)
    assert rep["first_pad"] == 1
    assert abs(rep["settle_tilt_deg"] - 10.0) < 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("attitude", ["rcs", "ideal"])
def test_full_mission_lands(attitude):
    """Orbit -> separation -> deorbit -> braking -> terminal -> touchdown on the real Site01 terrain, with the 8 attitude
    thrusters + differential engine throttle (default) and with the starter's ideal torque."""
    from basilisk_mods.mission import MissionSetup, MissionSim

    ms = MissionSim(MissionSetup(attitude=attitude))
    while not ms.done and ms.t < 6000.0:
        ms.advance(50.0)
    assert ms.report is not None and ms.report.get("outcome") == "ok", ms.report
    assert ms.report["miss_m"] < 20.0
    assert ms.report["vertical_speed_m_s"] < 2.0 and ms.report["horizontal_speed_m_s"] < 0.5
