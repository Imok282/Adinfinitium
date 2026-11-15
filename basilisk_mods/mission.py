"""Full lunar landing mission in ONE Basilisk simulation (KAN-48): orbit -> separation -> deorbit -> coast to PDI -> braking ->
terminal descent (learned pilot or classic) -> touchdown on real LOLA terrain.

    sim = MissionSim(MissionSetup(region="lola5m_site01"))
    while not sim.done:
        sim.advance(1.0)            # seconds of simulated time; the caller decides the time warp
        frame = sim.frame()

Physics
  * SPICE (NASA NAIF DE440 kernels in <ADI_DATA>/kernels): Sun, Earth and Moon at the mission epoch. The Moon is at the origin
    (zeroBase), its point-mass gravity is central, Earth and Sun pull as third bodies. The landing site turns with the Moon
    (MOON_PA frame), so the target moves in the inertial frame exactly as the real surface does.
  * Two spacecraft: the lander and the propulsion module (PM). Docked, the PM is carried rigidly 1.66 m below the lander
    (kinematic attachment, re-imposed every control period; no burns happen while docked). Separation is an equal and
    opposite push along the lander's +z axis (momentum conserved).
  * Lander: 5 x 800 N main engines (same layout as lander_descent), 8 x 58 N attitude thrusters at the model's nozzles
    (control torque -> thrForceMapping -> thrFiringSchmitt on/off), one propellant tank feeding both. ``attitude="ideal"``
    keeps the starter's ideal torque instead, for comparison.
  * Guidance: retrograde deorbit burn on the main engines until the periapsis is at 15 km; coast; braking (energy-optimal
    ZEM/ZEV to the terminal gate with the shortest feasible time-to-go, i.e. near full throttle); terminal descent with
    the starter's DescentGuidance (and its pilot hook).
  * Touchdown: the four footpads are checked against the real LOLA heights (first contact, ground tilt under the pads).

Stated simplifications: point-mass Moon (no mascons); the PM's mass does not act on the lander while docked; Moon rotation
is ignored in the velocity seen by the pilot (< 0.1 m/s near the pole); RCS specific impulse is not in mission.yaml, the
main-engine value is used. Provenance: MODELLED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, mrpFeedback, thrFiringSchmitt, thrForceMapping
from Basilisk.simulation import extForceTorque, fuelTank, simpleNav, spacecraft, spiceInterface, thrusterDynamicEffector
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, simIncludeThruster

from basilisk_mods.bsk_common import R_MOON_M, mission, mrp_with_z_along
from basilisk_mods.lander_descent import INERTIA_KG_M2, DescentGuidance, engine_layout, touchdown_report
from basilisk_mods.lander_geometry import RCS_EXHAUST_3AXIS_B, RCS_EXHAUST_MODEL_B, RCS_POS_B, footpad_points_N, footpad_report
from basilisk_mods.lunar_terrain import Terrain, latlon_from_ps, moon_fixed_from_latlon
from basilisk_mods.scenario_orbit_release import PROPULSION_MODULE_MASS_KG
from contracts.paths import data_root, tile_path

KERNELS = ("naif0012.tls", "de440s.bsp", "pck00011.tpc", "moon_pa_de440_200625.bpc", "moon_de440_250416.tf")
ORBIT_ALT_M = 100.0e3  # [m] parking orbit altitude
PDI_ALT_M = 15.0e3  # [m] periapsis = powered-descent initiation
GATE_ALT_M = 2000.0  # [m] braking ends / terminal descent (and the pilot) start here, above the target
GATE_UPRANGE_M = 300.0  # [m] the gate is this far short of the target (inside the pilot's training range, +-600 m)
GATE_V_DOWN_M_S = 45.0  # [m/s] down speed at the gate (training range 30-60)
GATE_V_ALONG_M_S = 10.0  # [m/s] along-track speed at the gate (training range +-25)
BRAKE_RANGE_M = 650.0e3  # [m] ground distance covered while braking from the PDI; calibrated by runs (see tests)
BRAKE_THRUST_FRACTION = 0.88  # [-] braking plans with this share of the available thrust (rest: steering)
BRAKE_FREEZE_TGO_S = 90.0  # [s] below this time-to-go it is counted down instead of re-searched (standard final approach)
DOCK_OFFSET_M = 1.66  # [m] PM centre below the lander origin when docked (from the two models)
SEPARATION_SPEED_M_S = 0.10  # [m/s] relative push-off speed (assumption: soft spring separation)
T_SEPARATION_S = 120.0  # [s] mission time of separation
T_DEORBIT_S = 420.0  # [s] mission time of the deorbit burn start
SLEW_LEAD_S = 120.0  # [s] the lander turns retrograde this long before the burn
DYN_DT_S = 0.01  # [s] dynamics step (thruster on-time pulses are quantised to it)
ATT_DT_S = 0.1  # [s] attitude control period
CTRL_DT_S = 0.2  # [s] guidance period
VIZ_DT_S = 0.1  # [s] Vizard update period


@dataclass
class MissionSetup:
    region: str = "lola5m_site01"
    target_xy: tuple[float, float] | None = None  # [m] tile frame; None = the safe cell nearest the tile centre
    epoch_utc: str = "2027-03-02T00:00:00"  # landing slot of the demo plan (planner plan_000.json)
    attitude: str = "rcs"  # "rcs" (8 attitude thrusters) or "ideal" (ideal torque, the starter's simplification)
    rcs_layout: str = "3axis"  # "3axis" (decided 15 Nov) or "model" (nozzles as drawn in KAN-47: horizontal, weak)
    engine_torque: bool = True  # differential throttle of the outer main engines while they fire
    engine_out_at_s: float | None = None  # [s] mission time; index from DescentGuidance.engine_out_index
    engine_out_index: int = 1
    viz: bool = False  # write a Vizard playback file
    live_viz: bool = False  # stream to a running Vizard
    viz_models: dict = field(default_factory=dict)  # {"lander": obj path, "propulsion_module": obj path}


def kernel_paths() -> list[str]:
    d = data_root() / "kernels"
    missing = [k for k in KERNELS if not (d / k).exists()]
    if missing:
        raise FileNotFoundError(f"missing SPICE kernels in {d}: {missing} (NAIF generic_kernels, see worldmodel/light/ephemeris.py)")
    return [str(d / k) for k in KERNELS]


def epoch_msg(epoch_utc: str):
    t = datetime.fromisoformat(epoch_utc)
    p = messaging.EpochMsgPayload()
    p.year, p.month, p.day, p.hours, p.minutes = t.year, t.month, t.day, t.hour, t.minute
    p.seconds = t.second + t.microsecond * 1e-6  # [s]
    return p, messaging.EpochMsg().write(p)


def pick_target(tile, max_slope_deg: float) -> tuple[float, float]:
    """Safe cell nearest the tile centre: no hazard within 3 cells (backup hazard map) AND every slope within 2 cells under
    half the touchdown slope limit (the four footpads span 3.1 m, so the 5 m slope alone is not enough). Tile metres."""
    from numpy.lib.stride_tricks import sliding_window_view

    from worldmodel.lander.terrain import safe_target_cells   # numpy only

    cells = safe_target_cells(tile)
    flat = sliding_window_view(np.pad(tile.slope, 2, mode="edge"), (5, 5)).max(axis=(-1, -2)) < 0.5 * max_slope_deg
    cells = cells[flat[cells[:, 0], cells[:, 1]]]
    h, w = tile.height.shape
    r, c = cells[int(np.argmin((cells[:, 0] - (h - 1) / 2) ** 2 + (cells[:, 1] - (w - 1) / 2) ** 2))]
    mpp = tile.meters_per_pixel
    return float((c - (w - 1) / 2) * mpp), float(((h - 1) / 2 - r) * mpp)


class Site:
    """Landing site on the tile: Moon-fixed position, the tile axes as Moon-fixed unit vectors, terrain lookups."""

    def __init__(self, terrain: Terrain, target_xy: tuple[float, float]):
        self.terrain, self.xy = terrain, (float(target_xy[0]), float(target_xy[1]))
        self.h = float(terrain.ground_height(*self.xy))  # [m] LOLA height of the target above the 1737.4 km sphere
        self.r_P = self.point_P(*self.xy, self.h)
        e = 5.0  # [m] finite-difference step for the tile axes
        self.ex_P = _unit(self.point_P(self.xy[0] + e, self.xy[1], self.h) - self.point_P(self.xy[0] - e, self.xy[1], self.h))
        self.up_P = _unit(self.r_P)
        self.ex_P = _unit(self.ex_P - self.ex_P @ self.up_P * self.up_P)
        self.ey_P = np.cross(self.up_P, self.ex_P)

    def point_P(self, x, y, h=0.0) -> np.ndarray:
        lat, lon = latlon_from_ps(self.terrain.center_ps_x + x, self.terrain.center_ps_y + y)
        return moon_fixed_from_latlon(lat, lon, h)

    def tile_xy_alt(self, r_P) -> tuple[float, float, float]:
        """Moon-fixed position -> tile x, y [m] and height above the real terrain [m]."""
        x, y, alt = self.terrain.tile_xy_from_moon_fixed(r_P)
        return float(x), float(y), float(alt - self.terrain.ground_height(x, y))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class AttitudeActuation(sysModel.SysModel):
    """Splits the attitude controller's torque request between the main engines and the attitude thrusters (or ideal torque).

    While the guidance fires the main engines, the healthy OUTER engines are throttled unevenly around the guidance's
    collective throttle (least squares, total thrust kept, on-time fraction within [0, 1]) to give pitch/roll torque; the
    residual (and all yaw) goes to the attitude thrusters. Engines off: everything goes to the attitude thrusters.
    Runs every attitude period; engine on-times are written for that period.

    Inputs : torqueInMsg (CmdTorqueBodyMsg) from mrpFeedback; collective throttle and healthy engines from the guidance
    Outputs: engineOnTimeOutMsg (THRArrayOnTimeCmdMsg), rcsTorqueOutMsg (CmdTorqueBodyMsg)
    """

    def __init__(self, guidance: DescentGuidance, use_engines: bool, use_rcs: bool):
        super().__init__()
        self.ModelTag = "attitudeActuation"
        self.g, self.use_engines = guidance, use_engines
        self.torqueInMsg = messaging.CmdTorqueBodyMsgReader()
        self.engineOnTimeOutMsg = messaging.THRArrayOnTimeCmdMsg()
        self.rcsTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        lay = engine_layout()
        # torque [N*m] per unit throttle of each engine about the body origin (the CoM sits within ~0.3 m of it)
        self.T = np.array([np.cross(r, t) * guidance.eng_max for r, t in lay]).T  # [3, 5]
        self.throttles = np.zeros(len(lay))
        self.engine_torque = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self.engineOnTimeOutMsg.write(messaging.THRArrayOnTimeCmdMsgPayload())
        self.rcsTorqueOutMsg.write(messaging.CmdTorqueBodyMsgPayload())

    def UpdateState(self, CurrentSimNanos):
        req = np.array(self.torqueInMsg().torqueRequestBody)
        g = self.g
        base = float(g.last.get("throttle", 0.0))
        ok = g.engines_ok.copy()
        th = np.where(ok, base, 0.0)
        if self.use_engines and base > 0.0:
            outer = [i for i in range(1, len(ok)) if ok[i]]
            if outer:
                A = np.vstack([self.T[:2, outer], np.ones(len(outer)) * 50.0])  # rows: torque x, y; total thrust unchanged
                d, *_ = np.linalg.lstsq(A, np.r_[req[:2], 0.0], rcond=None)
                th[outer] = np.clip(base + d, 0.0, 1.0)  # pulse-modulated: any average on-time fraction is physical
        self.throttles = th
        self.engine_torque = self.T @ th - (self.T @ np.where(ok, base, 0.0))  # what the uneven part adds
        on = th * ATT_DT_S
        cmd = messaging.THRArrayOnTimeCmdMsgPayload()
        cmd.OnTimeRequest = on.tolist() + [0.0] * (messaging.MAX_EFF_CNT - len(on))
        self.engineOnTimeOutMsg.write(cmd, CurrentSimNanos, self.moduleID)
        rest = messaging.CmdTorqueBodyMsgPayload()
        rest.torqueRequestBody = (req - self.engine_torque).tolist()
        self.rcsTorqueOutMsg.write(rest, CurrentSimNanos, self.moduleID)


class MissionGuidance(DescentGuidance):
    """One guidance module for every phase. TERMINAL is the starter's DescentGuidance law unchanged (pilot hook included)."""

    def __init__(self, cfg, mission_sim: MissionSim, mass_fn):
        super().__init__(cfg, mission_sim.target_N(), mass_fn, CTRL_DT_S)
        self.ms = mission_sim
        self.mode = "DOCKED"
        self.braking: dict = {}

    def _point_and_fire(self, CurrentSimNanos, thrust_N, sigma_BN, fire: bool, min_throttle_ok: bool = True):
        """Point +z_B along thrust_N; fire the healthy engines with the part of thrust_N along the current +z_B."""
        mag = float(np.linalg.norm(thrust_N))
        dir_N = thrust_N / mag if mag > 1e-9 else self.ms.up_now()
        ref = messaging.AttRefMsgPayload()
        ref.sigma_RN = tilt_only_reference(sigma_BN, dir_N).tolist()
        self.attRefOutMsg.write(ref, CurrentSimNanos, self.moduleID)
        z_B = np.array(rbk.MRP2C(sigma_BN)).T @ np.array([0.0, 0.0, 1.0])
        n_ok = int(self.engines_ok.sum())
        throttle = 0.0
        if fire and n_ok:
            throttle = float(np.clip(max(float(thrust_N @ z_B), 0.0) / (n_ok * self.eng_max), 0.0, 1.0))
            if min_throttle_ok and throttle < 0.5 * self.min_thr:
                throttle = 0.0
            elif throttle:
                throttle = max(throttle, self.min_thr)
        on = np.where(self.engines_ok, throttle * self.period, 0.0)
        cmd = messaging.THRArrayOnTimeCmdMsgPayload()
        cmd.OnTimeRequest = on.tolist() + [0.0] * (messaging.MAX_EFF_CNT - self.n_eng)
        self.thrOnTimeOutMsg.write(cmd, CurrentSimNanos, self.moduleID)
        err = float(np.degrees(np.arccos(np.clip(z_B @ dir_N, -1, 1))))
        self.last = {"t_s": CurrentSimNanos * macros.NANO2SEC, "phase": self.mode.lower(), "throttle": throttle,
                     "thrust_n": throttle * n_ok * self.eng_max, "engines_ok": n_ok, "source": "classic", "on_time_s": on.tolist(),
                     "pointing_error_deg": err}
        return err

    def UpdateState(self, CurrentSimNanos):
        t_s = CurrentSimNanos * macros.NANO2SEC
        st = self.scStateInMsg()
        r, v, sigma = np.array(st.r_BN_N), np.array(st.v_BN_N), np.array(st.sigma_BN)
        self.apply_failures(t_s)
        mode = self.mode
        if mode in ("DOCKED", "SEPARATED"):
            want = self.ms.up_now(r) * 1.0
            ahead = self.ms.angle_to_site(r, v)  # [rad] still to travel to the site
            burn_at = np.pi + BRAKE_RANGE_M / R_MOON_M  # [rad] the periapsis comes half an orbit after the burn
            if mode == "SEPARATED" and ahead <= burn_at + self.ms.n_c * SLEW_LEAD_S:
                want = -_unit(v)
            self._point_and_fire(CurrentSimNanos, want, sigma, fire=False)
            if mode == "SEPARATED" and ahead <= burn_at:
                self._switch(t_s, "DEORBIT")
        elif mode == "DEORBIT":
            err = self._point_and_fire(CurrentSimNanos, -_unit(v) * 1e6, sigma, fire=True)
            if err > 6.0:  # [deg] do not burn until pointing retrograde
                self._point_and_fire(CurrentSimNanos, -_unit(v) * 1e6, sigma, fire=False)
            if periapsis_radius(r, v, self.mu) <= R_MOON_M + PDI_ALT_M:
                self._point_and_fire(CurrentSimNanos, -_unit(v), sigma, fire=False)
                self._switch(t_s, "COAST")
        elif mode == "COAST":
            self._point_and_fire(CurrentSimNanos, -_unit(v), sigma, fire=False)
            if float(r @ v) >= 0.0 and np.linalg.norm(r) < R_MOON_M + PDI_ALT_M + 20e3:  # [m] at (or just past) periapsis
                self._switch(t_s, "BRAKING")
        elif mode == "BRAKING":
            prev = self.braking.get("tgo_s")
            thrust_max = int(self.engines_ok.sum()) * self.eng_max
            fixed = prev - CTRL_DT_S if prev is not None and prev < BRAKE_FREEZE_TGO_S else None  # final approach: count down
            a_cmd, tgo = self.ms.braking_accel(r, v, self.mass_fn(), thrust_max, tgo_fixed=fixed)
            if fixed is None and prev is not None and tgo > prev - CTRL_DT_S:  # time-to-go only counts down: stay on schedule
                a_cmd, tgo = self.ms.braking_accel(r, v, self.mass_fn(), thrust_max, tgo_fixed=prev - CTRL_DT_S)
            self.braking = {"tgo_s": tgo}
            self._point_and_fire(CurrentSimNanos, self.mass_fn() * a_cmd, sigma, fire=True)
            if tgo <= 2.0 or (self.ms.ground_range(r) < 3.0e3 and self.ms.height_over_site(r) <= GATE_ALT_M):  # [s], [m]
                self._switch(t_s, "TERMINAL")
        elif mode == "TERMINAL":
            self.target_N = self.ms.target_N()
            self.up = _unit(self.target_N)
            super().UpdateState(CurrentSimNanos)
            # same thrust direction, but reached by the smallest tilt from the current attitude (yaw is left free)
            dir_N = np.array(rbk.MRP2C(np.array(self.attRefOutMsg.read().sigma_RN))).T @ np.array([0.0, 0.0, 1.0])
            ref = messaging.AttRefMsgPayload()
            ref.sigma_RN = tilt_only_reference(sigma, dir_N).tolist()
            self.attRefOutMsg.write(ref, CurrentSimNanos, self.moduleID)
        else:  # LANDED / ENDED: engines off, hold attitude
            self._point_and_fire(CurrentSimNanos, self.ms.up_now(r), sigma, fire=False)

    def _switch(self, t_s, mode):
        self.decisions.append((t_s, f"phase -> {mode.lower()}"))
        self.mode = mode


def tilt_only_reference(sigma_BN, dir_N) -> np.ndarray:
    """MRP of the attitude whose +z axis is dir_N, reached from the current attitude by the smallest rotation (no yaw change).

    With attitude thrusters that control only pitch and roll this keeps the commanded error a pure tilt."""
    C_BN = np.array(rbk.MRP2C(np.asarray(sigma_BN)))
    z = C_BN.T @ np.array([0.0, 0.0, 1.0])
    d = _unit(dir_N)
    axis = np.cross(z, d)
    sn, c = np.linalg.norm(axis), float(np.clip(z @ d, -1.0, 1.0))
    if sn < 1e-9:
        return np.asarray(rbk.C2MRP(C_BN)) if c > 0 else mrp_with_z_along(d)
    k = axis / sn
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Q = np.eye(3) + sn * K + (1 - c) * K @ K  # rotation taking z onto d (in the inertial frame)
    return np.asarray(rbk.C2MRP(C_BN @ Q.T))


def periapsis_radius(r, v, mu) -> float:
    rn, v2 = np.linalg.norm(r), float(v @ v)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / rn
    a = 1.0 / (2.0 / rn - v2 / mu)
    return float(a * (1.0 - np.linalg.norm(e_vec)))


class MissionSim:
    def __init__(self, setup: MissionSetup, command_fn=None):
        self.setup, self.cfg = setup, mission()
        self.t = 0.0
        self.done = False
        self.report: dict | None = None
        self.events: list[tuple[float, str]] = []
        self.terrain = Terrain.from_tile(tile_path(setup.region))
        self.site = Site(self.terrain, setup.target_xy or pick_target(self.terrain.tile, self.cfg.lander.touchdown_limits.max_slope_deg))
        self._build(command_fn)

    # ------------------------------------------------------------------ build
    def _build(self, command_fn):
        cfg, s = self.cfg, self.setup
        sim = self.sim = SimulationBaseClass.SimBaseClass()
        proc = sim.CreateNewProcess("mission")
        for name, dt in (("dyn", DYN_DT_S), ("att", ATT_DT_S), ("fsw", CTRL_DT_S), ("viz", VIZ_DT_S)):
            proc.addTask(sim.CreateNewTask(name, macros.sec2nano(dt)))

        # SPICE: Sun, Earth, Moon at the epoch; Moon at the origin
        sp = self.spice = spiceInterface.SpiceInterface(auto_configure_kernels=False)
        sp.ModelTag = "spice"
        sp.addKernelPaths(kernel_paths())
        sp.addPlanetNames(spiceInterface.StringVector(["sun", "earth", "moon"]))
        sp.planetFrames = spiceInterface.StringVector(["IAU_SUN", "IAU_EARTH", "MOON_PA"])
        sp.zeroBase = "moon"
        self._epoch = epoch_msg(s.epoch_utc)
        sp.epochInMsg.subscribeTo(self._epoch[1])
        sim.AddModelToTask("dyn", sp, 100)
        self.epoch = datetime.fromisoformat(s.epoch_utc)

        grav = self.grav = simIncludeGravBody.gravBodyFactory()
        sun, earth, moon = grav.createSun(), grav.createEarth(), grav.createMoon()
        moon.isCentralBody = True
        for i, body in enumerate((sun, earth, moon)):
            body.planetBodyInMsg.subscribeTo(sp.planetStateOutMsgs[i])
        self.mu = moon.mu

        # the Moon's orientation at the epoch, to place the orbit over the site (one SPICE call before the run)
        self.PN0 = moon_attitude_at(s.epoch_utc)

        # lander -----------------------------------------------------------
        lander = self.lander = spacecraft.Spacecraft()
        lander.ModelTag = "lander"
        lander.hub.mHub = cfg.lander.dry_mass_kg
        lander.hub.IHubPntBc_B = np.diag(INERTIA_KG_M2).tolist()
        r0, v0 = self.initial_orbit()
        lander.hub.r_CN_NInit, lander.hub.v_CN_NInit = r0.tolist(), v0.tolist()
        lander.hub.sigma_BNInit = mrp_with_z_along(_unit(r0)).tolist()
        sim.AddModelToTask("dyn", lander, 50)
        grav.addBodiesTo(lander)

        eng = self.engines = thrusterDynamicEffector.ThrusterDynamicEffector()
        eng.ModelTag = "mainEngines"
        fac = simIncludeThruster.thrusterFactory()
        for r_B, t_B in engine_layout():
            fac.create("Blank_Thruster", r_B, t_B, MaxThrust=float(cfg.lander.engines.max_thrust_n), steadyIsp=float(cfg.lander.isp_s),
                       MinOnTime=0.0, cutoffFrequency=1000.0)
        fac.addToSpacecraft("mainEngines", eng, lander)
        sim.AddModelToTask("dyn", eng, 40)

        rcs = self.rcs = thrusterDynamicEffector.ThrusterDynamicEffector()
        rcs.ModelTag = "attitudeThrusters"
        rfac = self.rcs_factory = simIncludeThruster.thrusterFactory()
        rcs_n = float(cfg.lander.attitude_thrusters.thrust_n)
        for p, d in zip(RCS_POS_B, RCS_EXHAUST_3AXIS_B if s.rcs_layout == "3axis" else RCS_EXHAUST_MODEL_B):
            rfac.create("Blank_Thruster", p.tolist(), (-d).tolist(), MaxThrust=rcs_n, steadyIsp=float(cfg.lander.isp_s), MinOnTime=0.0,
                        cutoffFrequency=1000.0)
        rfac.addToSpacecraft("attitudeThrusters", rcs, lander)
        sim.AddModelToTask("dyn", rcs, 40)

        tank = self.tank = fuelTank.FuelTank()
        tank.ModelTag = "propTank"
        tm = fuelTank.FuelTankModelConstantVolume()
        tm.propMassInit = float(cfg.lander.wet_mass_at_release_kg - cfg.lander.dry_mass_kg)
        tm.maxFuelMass = tm.propMassInit
        tm.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        tm.radiusTankInit = 0.6  # [m]
        tank.setTankModel(tm)
        tank.setR_TB_B([0.0, 0.0, -0.3])
        tank.addThrusterSet(eng)
        tank.addThrusterSet(rcs)
        lander.addStateEffector(tank)
        sim.AddModelToTask("dyn", tank, 45)

        nav = self.nav = simpleNav.SimpleNav()
        nav.ModelTag = "nav"
        nav.scStateInMsg.subscribeTo(lander.scStateOutMsg)
        sim.AddModelToTask("dyn", nav, 30)

        guid = self.guid = MissionGuidance(cfg, self, self.mass)
        guid.mu = self.mu
        guid.scStateInMsg.subscribeTo(lander.scStateOutMsg)
        guid.engine_out_at_s = s.engine_out_at_s
        guid.engine_out_index = s.engine_out_index
        guid.command_fn = command_fn
        guid.ground_height_fn = self.ground_above_target
        sim.AddModelToTask("fsw", guid, 20)

        att = self.att_err = attTrackingError.attTrackingError()
        att.ModelTag = "attErr"
        att.attNavInMsg.subscribeTo(nav.attOutMsg)
        att.attRefInMsg.subscribeTo(guid.attRefOutMsg)
        sim.AddModelToTask("att", att, 10)
        veh = messaging.VehicleConfigMsgPayload()
        veh.ISCPntB_B = np.diag(INERTIA_KG_M2).flatten().tolist()
        self._veh = (veh, messaging.VehicleConfigMsg().write(veh))
        ctrl = self.ctrl = mrpFeedback.mrpFeedback()
        ctrl.ModelTag = "mrpFeedback"
        ctrl.guidInMsg.subscribeTo(att.attGuidOutMsg)
        ctrl.vehConfigInMsg.subscribeTo(self._veh[1])
        ctrl.Ki = -1.0
        sim.AddModelToTask("att", ctrl, 9)
        act = self.actuation = AttitudeActuation(guid, use_engines=s.engine_torque, use_rcs=s.attitude == "rcs")
        act.torqueInMsg.subscribeTo(ctrl.cmdTorqueOutMsg)
        sim.AddModelToTask("att", act, 8)
        eng.cmdsInMsg.subscribeTo(act.engineOnTimeOutMsg)
        if s.attitude == "rcs":
            ctrl.K, ctrl.P = 2500.0, 2000.0  # [N*m], [N*m*s] engines give hundreds of N*m in powered flight, thrusters ~40 N*m
            self._thr_cfg = rfac.getConfigMessage()
            fm = self.force_map = thrForceMapping.thrForceMapping()
            fm.ModelTag = "thrForceMapping"
            fm.cmdTorqueInMsg.subscribeTo(act.rcsTorqueOutMsg)
            fm.thrConfigInMsg.subscribeTo(self._thr_cfg)
            fm.vehConfigInMsg.subscribeTo(self._veh[1])
            fm.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1] if s.rcs_layout == "3axis" else [1, 0, 0, 0, 1, 0]  # model layout: no yaw
            fm.thrForceSign = +1
            sim.AddModelToTask("att", fm, 7)
            fs = self.firing = thrFiringSchmitt.thrFiringSchmitt()
            fs.ModelTag = "thrFiringSchmitt"
            fs.thrMinFireTime = 0.02  # [s]
            fs.level_on, fs.level_off = 0.75, 0.25
            fs.baseThrustState = 0
            fs.thrConfInMsg.subscribeTo(self._thr_cfg)
            fs.thrForceInMsg.subscribeTo(fm.thrForceCmdOutMsg)
            sim.AddModelToTask("att", fs, 6)
            rcs.cmdsInMsg.subscribeTo(fs.onTimeOutMsg)
        else:
            ctrl.K, ctrl.P = 3600.0, 1800.0  # [N*m], [N*m*s] the starter's ideal-torque gains
            ext = self.ext = extForceTorque.ExtForceTorque()
            ext.ModelTag = "attitudeTorque"
            lander.addDynamicEffector(ext)
            sim.AddModelToTask("dyn", ext, 35)
            ext.cmdTorqueInMsg.subscribeTo(act.rcsTorqueOutMsg)  # ideal torque takes what the engines do not give

        # propulsion module ------------------------------------------------
        pm = self.pm = spacecraft.Spacecraft()
        pm.ModelTag = "propulsion_module"
        pm.hub.mHub = PROPULSION_MODULE_MASS_KG
        pm.hub.IHubPntBc_B = np.diag(INERTIA_KG_M2).tolist()  # assumption: same order of inertia as the lander
        pm.hub.r_CN_NInit = (r0 - DOCK_OFFSET_M * _unit(r0)).tolist()
        pm.hub.v_CN_NInit = v0.tolist()
        pm.hub.sigma_BNInit = lander.hub.sigma_BNInit
        sim.AddModelToTask("dyn", pm, 49)
        grav.addBodiesTo(pm)

        if s.viz or s.live_viz:
            from Basilisk.utilities import vizSupport

            save = str(Path(tile_path(s.region)).parents[2] / "runs" / "basilisk" / "mission") if s.viz else ""
            kw = {"saveFile": save} if save else {}
            viz = self.viz = vizSupport.enableUnityVisualization(sim, "viz", [lander, pm], thrEffectorList=[[eng, rcs], None],
                                                                 liveStream=bool(s.live_viz), **kw)
            for tag, path in s.viz_models.items():
                if path and Path(path).exists():
                    vizSupport.createCustomModel(viz, modelPath=str(path), simBodiesToModify=[tag])
            vs = viz.settings
            vs.mainCameraTarget = "lander"
            vs.forceStartAtSpacecraftLocalView = 1
            vs.showSpacecraftLabels = 1
            vs.showMissionTime = 1
            vs.ambient = 0.15  # [-] earthshine-like fill; the Sun comes from SPICE
            vs.orbitLinesOn = 1
            vs.spacecraftShadowBrightness = 0.3  # [-]

        sim.InitializeSimulation()
        self._state = {k: lander.dynManager.getStateObject(getattr(lander.hub, n)) for k, n in
                       (("r", "nameOfHubPosition"), ("v", "nameOfHubVelocity"), ("s", "nameOfHubSigma"), ("w", "nameOfHubOmega"))}
        self._pm_state = {k: pm.dynManager.getStateObject(getattr(pm.hub, n)) for k, n in
                          (("r", "nameOfHubPosition"), ("v", "nameOfHubVelocity"), ("s", "nameOfHubSigma"), ("w", "nameOfHubOmega"))}

    # ------------------------------------------------------------------ geometry
    def PN(self) -> np.ndarray:
        """DCM inertial -> Moon-fixed (MOON_PA) now."""
        m = self.spice.planetStateOutMsgs[2].read()
        J = np.array(m.J20002Pfix)
        return J if self.t > 0 and np.abs(J).sum() > 0 else self.PN0

    def target_N(self) -> np.ndarray:
        return self.PN().T @ self.site.r_P

    def up_now(self, r=None) -> np.ndarray:
        return _unit(self.lander_r() if r is None else r)

    def lander_r(self):
        return np.array(self.lander.scStateOutMsg.read().r_BN_N)

    def initial_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Circular polar orbit whose plane contains the site; the start point is placed so that the deorbit at T_DEORBIT_S puts
        the periapsis BRAKE_RANGE_M (ground distance) before the site, and the braking ends over it."""
        s_N = _unit(self.PN0.T @ self.site.r_P)
        pole = self.PN0.T @ np.array([0.0, 0.0, 1.0])
        n = _unit(np.cross(pole, s_N))
        a = R_MOON_M + ORBIT_ALT_M
        n_c = self.n_c = np.sqrt(self.mu / a**3)  # [rad/s]
        theta_pdi = BRAKE_RANGE_M / R_MOON_M
        theta0 = theta_pdi + np.pi + n_c * T_DEORBIT_S  # angle still to travel to the site at t = 0
        r_hat = _rot(s_N, n, -theta0)
        return a * r_hat, np.sqrt(self.mu / a) * np.cross(n, r_hat)

    def ground_above_target(self, r_N) -> float:
        """Terrain height under the lander relative to the target's height [m] (DescentGuidance.ground_height_fn)."""
        x, y, _ = self.terrain.tile_xy_from_moon_fixed(self.PN() @ np.asarray(r_N))
        return float(self.terrain.ground_height(x, y)) - self.site.h

    def angle_to_site(self, r_N, v_N) -> float:
        """In-plane angle [rad, 0..2pi) the lander still has to travel (along its motion) to be over the site."""
        n = _unit(np.cross(r_N, v_N))
        s_hat, r_hat = _unit(self.target_N()), _unit(r_N)
        return float(np.arctan2(np.cross(r_hat, s_hat) @ n, r_hat @ s_hat) % (2 * np.pi))

    def ground_range(self, r_N) -> float:
        """Great-circle distance [m] on the reference sphere from the point under the lander to the site."""
        c = float(np.clip(_unit(r_N) @ _unit(self.target_N()), -1.0, 1.0))
        return float(R_MOON_M * np.arccos(c))

    def height_over_site(self, r_N) -> float:
        return float((np.asarray(r_N) - self.target_N()) @ _unit(self.target_N()))

    def tile_basis_N(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        PT = self.PN().T
        return PT @ self.site.ex_P, PT @ self.site.ey_P, PT @ self.site.up_P

    def braking_accel(self, r, v, mass, max_thrust, tgo_fixed: float | None = None) -> tuple[np.ndarray, float]:
        """Braking guidance in local downrange / crossrange / altitude coordinates (flat-Moon ZEM/ZEV with effective gravity).

        Effective vertical gravity = mu/r^2 - v_h^2/r: zero at orbital speed, full lunar gravity at rest (the curvature of the
        Moon carries a fast lander around, it does not fall). Time-to-go = the shortest for which the energy-optimal
        acceleration profile (linear in time) fits 85 % of the available thrust at both ends. Returns the inertial thrust
        acceleration [m/s^2] and the time-to-go [s].
        """
        tgt = self.target_N()
        up = _unit(r)
        n = _unit(np.cross(r, v))
        along = np.cross(n, up)
        cross = n
        rn = np.linalg.norm(r)
        s_go = R_MOON_M * self.angle_to_site(r, v)
        if s_go > np.pi * R_MOON_M:  # already past the site
            s_go -= 2 * np.pi * R_MOON_M
        c_go = R_MOON_M * float(np.arcsin(np.clip(_unit(tgt) @ cross, -1, 1)))
        h = rn - np.linalg.norm(tgt)  # [m] height above the site's radius
        rel = np.array([-(s_go - GATE_UPRANGE_M), -c_go, h - GATE_ALT_M])  # position relative to the gate
        vel = np.array([v @ along, v @ cross, v @ up])
        v_f = np.array([GATE_V_ALONG_M_S, 0.0, -GATE_V_DOWN_M_S])
        g_eff = self.mu / rn**2 - vel[0] ** 2 / rn
        g = np.array([0.0, 0.0, -g_eff])
        a_max = BRAKE_THRUST_FRACTION * max_thrust / max(mass, 1.0)  # [m/s^2] margin for attitude error
        tg = np.array([[tgo_fixed]]) if tgo_fixed else np.linspace(4.0, 1500.0, 600)[:, None]  # [s] candidate times-to-go
        zem = -(rel + vel * tg + 0.5 * g * tg**2)
        zev = v_f - (vel + g * tg)
        acc = 6.0 * zem / tg**2 - 2.0 * zev / tg  # ZEM/ZEV already contain gravity: this is the thrust acceleration
        acc_end = -6.0 * zem / tg**2 + 4.0 * zev / tg
        peak = np.maximum(np.linalg.norm(acc, axis=1), np.linalg.norm(acc_end, axis=1))
        ok = peak <= a_max
        i = int(np.argmax(ok)) if ok.any() else int(np.argmin(peak))
        a = acc[i]
        return a[0] * along + a[1] * cross + a[2] * up, float(tg[i, 0])

    # ------------------------------------------------------------------ state helpers
    def mass(self) -> float:
        return float(self.lander.hub.mHub + self.fuel())

    def fuel(self) -> float:
        return float(self.tank.fuelTankOutMsg.read().fuelMass)

    def _dock_pm(self):
        """Carry the PM rigidly below the lander (docked)."""
        st = self.lander.scStateOutMsg.read()
        r, v = np.array(st.r_BN_N), np.array(st.v_BN_N)
        BN = np.array(rbk.MRP2C(np.array(st.sigma_BN)))
        z = BN.T @ np.array([0.0, 0.0, 1.0])
        self._pm_state["r"].setState((r - DOCK_OFFSET_M * z).reshape(3, 1).tolist())
        self._pm_state["v"].setState(v.reshape(3, 1).tolist())
        self._pm_state["s"].setState(np.array(st.sigma_BN).reshape(3, 1).tolist())
        self._pm_state["w"].setState(np.array(st.omega_BN_B).reshape(3, 1).tolist())

    def _separate(self):
        st = self.lander.scStateOutMsg.read()
        z = np.array(rbk.MRP2C(np.array(st.sigma_BN))).T @ np.array([0.0, 0.0, 1.0])
        m_l, m_p = self.mass(), PROPULSION_MODULE_MASS_KG
        v = np.array(st.v_BN_N)
        self._state["v"].setState((v + SEPARATION_SPEED_M_S * m_p / (m_l + m_p) * z).reshape(3, 1).tolist())
        self._pm_state["v"].setState((v - SEPARATION_SPEED_M_S * m_l / (m_l + m_p) * z).reshape(3, 1).tolist())
        self.guid.mode = "SEPARATED"
        self.guid.decisions.append((self.t, f"separation: push-off {SEPARATION_SPEED_M_S:.2f} m/s along the lander's +z"))

    # ------------------------------------------------------------------ run
    def advance(self, dt_s: float) -> None:
        """Advance the simulation by dt_s seconds (in guidance periods)."""
        end = self.t + dt_s
        while not self.done and self.t < end - 1e-9:
            if self.guid.mode == "DOCKED":
                self._dock_pm()
            self.t = round(self.t + CTRL_DT_S, 6)
            self.sim.ConfigureStopTime(macros.sec2nano(self.t))
            self.sim.ExecuteSimulation()
            if self.guid.mode == "DOCKED" and self.t >= T_SEPARATION_S:
                self._separate()
            if self.guid.mode == "TERMINAL":
                self._check_touchdown()
            if self.fuel() <= 0.01 and self.guid.mode in ("DEORBIT", "BRAKING", "TERMINAL"):
                self._end({"outcome": "out_of_fuel", "t_s": self.t})

    def _check_touchdown(self):
        st = self.lander.scStateOutMsg.read()
        r_N = np.array(st.r_BN_N)
        BN = np.array(rbk.MRP2C(np.array(st.sigma_BN)))
        PN = self.PN()
        pads_P = [PN @ p for p in footpad_points_N(r_N, BN)]
        clear, ground = [], []
        for p in pads_P:
            x, y, alt = self.terrain.tile_xy_from_moon_fixed(p)
            g = float(self.terrain.ground_height(x, y))
            clear.append(float(alt) - g)
            ground.append(g)
        if min(clear) > 0.0:
            return
        fp = footpad_report(clear, ground)
        rep = touchdown_report(self.cfg, r_N, np.array(st.v_BN_N), st.sigma_BN, slope_deg=fp["settle_tilt_deg"])
        x, y, _ = self.terrain.tile_xy_from_moon_fixed(PN @ r_N)
        rep.update(fp)
        rep.update({"t_s": self.t, "x_m": float(x), "y_m": float(y), "target_xy_m": list(self.site.xy),
                    "miss_m": float(np.hypot(x - self.site.xy[0], y - self.site.xy[1])), "fuel_left_kg": self.fuel()})
        self._end(rep)

    def _end(self, report: dict):
        self.report, self.done = report, True
        self.guid.mode = "LANDED" if report.get("outcome") == "ok" else "ENDED"
        self.guid.decisions.append((self.t, f"END {report.get('outcome')}"))

    # ------------------------------------------------------------------ outputs
    def utc(self) -> str:
        return (self.epoch + timedelta(seconds=self.t)).isoformat(timespec="seconds") + "Z"


def _rot(v, axis, ang):
    """Rotate v about the unit axis by ang [rad] (Rodrigues)."""
    v, k = np.asarray(v, float), _unit(axis)
    return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * (k @ v) * (1 - np.cos(ang))


def moon_attitude_at(epoch_utc: str) -> np.ndarray:
    """J2000 -> MOON_PA DCM at the epoch (a one-step SPICE run)."""
    sim = SimulationBaseClass.SimBaseClass()
    p = sim.CreateNewProcess("p")
    p.addTask(sim.CreateNewTask("t", macros.sec2nano(1.0)))
    sp = spiceInterface.SpiceInterface(auto_configure_kernels=False)
    sp.addKernelPaths(kernel_paths())
    sp.addPlanetNames(spiceInterface.StringVector(["moon"]))
    sp.planetFrames = spiceInterface.StringVector(["MOON_PA"])
    sp.zeroBase = "moon"
    ep = epoch_msg(epoch_utc)
    sp.epochInMsg.subscribeTo(ep[1])
    sim.AddModelToTask("t", sp)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(0)
    sim.ExecuteSimulation()
    return np.array(sp.planetStateOutMsgs[0].read().J20002Pfix)
