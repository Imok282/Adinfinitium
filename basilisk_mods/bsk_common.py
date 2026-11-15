"""Shared helpers for every Basilisk scenario in this folder.

Frames (keep them straight — most bugs come from mixing these up):
  N  Moon-centred inertial frame used by Basilisk (metres, m/s). In these starters the Moon
     is a non-rotating sphere at the origin; Arnav's terrain module adds the Moon-fixed frame.
  B  lander body frame. +z_B points "up" out of the top deck; engines push along +z_B.
  L  local frame at the landing target: east, north, up (metres from the target point).

All numbers come from config/mission.yaml via contracts.mission.load_mission().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody

from contracts.mission import MissionConfig, load_mission
from contracts.paths import runs_root

R_MOON_M = 1737.4e3


def out_dir() -> Path:
    """runs/basilisk/ — every Basilisk output (pdi_state.json, telemetry CSV, Vizard files) goes here."""
    d = runs_root() / "basilisk"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_sim(tasks: dict[str, float]):
    """Create a simulation with one process and named tasks. tasks = {"dynTask": 0.05, ...} (seconds)."""
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("mainProcess")
    for name, dt_s in tasks.items():
        proc.addTask(sim.CreateNewTask(name, macros.sec2nano(dt_s)))
    return sim


EARTH_MEAN_DISTANCE_M = 384400.0e3  # [m] mean Earth-Moon distance
_VIZ_KEEP = []  # message payloads must outlive the simulation


def add_moon_gravity(spacecraft_obj, viz_earth: bool = False):
    """Point-mass Moon gravity at the origin. Returns (gravFactory, mu).

    ``viz_earth`` adds the Earth for DRAWING ONLY: Vizard 2.4 cannot draw a scene whose only planet is the Moon (it errors in
    GetAbsolutePlanetPositionUnityCS). Its gravity is switched off (mu = 0) because this starter keeps the Moon fixed, so a real
    Earth pull would be an uncompensated force on the lander. Fixed position at the mean distance along +x of the frame.
    """
    grav = simIncludeGravBody.gravBodyFactory()
    moon = grav.createMoon()
    moon.isCentralBody = True
    if viz_earth:
        from Basilisk.architecture import messaging

        earth = grav.createEarth()
        earth.mu = 0.0  # [m^3/s^2] drawn, not pulling
        st = messaging.SpicePlanetStateMsgPayload()
        st.PlanetName = "earth"
        st.PositionVector = [EARTH_MEAN_DISTANCE_M, 0.0, 0.0]  # [m]
        st.J20002Pfix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        msg = messaging.SpicePlanetStateMsg().write(st)
        earth.planetBodyInMsg.subscribeTo(msg)
        _VIZ_KEEP.extend([st, msg])
    grav.addBodiesTo(spacecraft_obj)
    return grav, moon.mu


def local_frame(target_up_N: np.ndarray) -> np.ndarray:
    """Rows = east, north, up unit vectors (in N) at the target. Not valid exactly at a pole."""
    up = target_up_N / np.linalg.norm(target_up_N)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return np.vstack([east, north, up])


def dcm_with_z_along(z_N: np.ndarray, hint_N: np.ndarray | None = None) -> np.ndarray:
    """DCM [RN] whose third axis (z_R) points along z_N. Used to point the engines."""
    z = z_N / np.linalg.norm(z_N)
    hint = np.array([1.0, 0.0, 0.0]) if hint_N is None else hint_N
    if abs(np.dot(hint, z)) > 0.95:
        hint = np.array([0.0, 1.0, 0.0])
    x = np.cross(hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def mrp_with_z_along(z_N: np.ndarray, hint_N: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(rbk.C2MRP(dcm_with_z_along(z_N, hint_N)))


def mission() -> MissionConfig:
    return load_mission()
