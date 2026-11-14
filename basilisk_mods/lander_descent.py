"""STARTER (Tanush): 5-engine powered-descent lander for Basilisk — Python modules only.

What's inside
  build_lander(sim, cfg, ...)   spacecraft + fuel tank + 5 x 800 N thrusters + attitude control chain
  DescentGuidance               Python module: classic ZEM/ZEV guidance -> attitude reference + engine on-times
  touchdown_report(...)         pass/fail against config touchdown limits

Engines: 4 outer + 1 centre, all pushing along +z_B (outer ones canted 5 deg outward).
Throttle is implemented the standard Basilisk way: each control period the guidance commands
an on-time = throttle x period for every healthy engine (pulse-width modulation).

Attitude (SIMPLIFICATION, say so in the demo): the attitude loop is
  guidance -> attTrackingError -> mrpFeedback -> extForceTorque
i.e. an ideal torque instead of modelling the 8 x 58 N RCS thrusters / reaction wheels.
"""

from __future__ import annotations

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, mrpFeedback
from Basilisk.simulation import extForceTorque, fuelTank, simpleNav, spacecraft, thrusterDynamicEffector
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros, simIncludeThruster

from basilisk_mods.bsk_common import R_MOON_M, add_moon_gravity, mrp_with_z_along
from contracts.mission import MissionConfig

ENGINE_CANT_DEG = 5.0
OPPOSITE = {1: 3, 3: 1, 2: 4, 4: 2}     # outer engine pairs (index 0 = centre)
ENGINE_RADIUS_M = 0.8
INERTIA_KG_M2 = [900.0, 900.0, 700.0]   # design assumption for a ~1.5 t lander


def engine_layout():
    """Positions r_B and force directions tHat_B for 4 outer + 1 centre engine (body frame)."""
    c = np.cos(np.radians(ENGINE_CANT_DEG))
    s = np.sin(np.radians(ENGINE_CANT_DEG))
    outer = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    layout = [([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])]                     # centre
    for ex, ey in outer:
        r = [ENGINE_RADIUS_M * ex, ENGINE_RADIUS_M * ey, -1.0]
        t = [-s * ex, -s * ey, c]                                       # force tilted inward = plume canted out
        layout.append((r, t))
    return layout


class DescentGuidance(sysModel.SysModel):
    """Classic ZEM/ZEV guidance to a hover gate, then vertical terminal descent (same law as sim_jax).

    Inputs : scStateInMsg (SCStatesMsg)
    Outputs: attRefOutMsg (AttRefMsg)  — point +z_B along the wanted thrust
             thrOnTimeOutMsg (THRArrayOnTimeCmdMsg) — on-time per engine this control period
    """

    def __init__(self, cfg: MissionConfig, target_N: np.ndarray, mass_fn, control_period_s: float):
        super().__init__()
        self.ModelTag = "descentGuidance"
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.attRefOutMsg = messaging.AttRefMsg()
        self.thrOnTimeOutMsg = messaging.THRArrayOnTimeCmdMsg()

        t = cfg.scenarios.terminal
        self.n_eng = cfg.lander.engines.count
        self.eng_max = cfg.lander.engines.max_thrust_n
        self.min_thr = cfg.lander.engines.min_throttle
        self.max_tilt = np.radians(cfg.lander.max_tilt_deg)
        self.gate = t.hover_gate_altitude_m
        self.v_final = t.final_descent_speed_m_s
        self.switch_h = 8.0
        self.terminal_tilt = np.radians(15.0)   # gentler attitude near the ground
        self.no_chase_alt_m = 10.0
        self.target_N = np.asarray(target_N, float)
        self.up = self.target_N / np.linalg.norm(self.target_N)
        self.mass_fn = mass_fn
        self.period = control_period_s
        self.engine_out_at_s: float | None = None
        self.engine_out_index = 1
        self.engines_ok = np.ones(self.n_eng, bool)
        self.ground_height_fn = lambda r_N: 0.0   # replace with Arnav's terrain lookup (KAN-44)
        # log for telemetry
        self.last = {}
        self.decisions: list[tuple[float, str]] = []

    def Reset(self, CurrentSimNanos):
        self.attRefOutMsg.write(messaging.AttRefMsgPayload())
        self.thrOnTimeOutMsg.write(messaging.THRArrayOnTimeCmdMsgPayload())

    def UpdateState(self, CurrentSimNanos):
        t_s = CurrentSimNanos * macros.NANO2SEC
        st = self.scStateInMsg()
        r = np.array(st.r_BN_N)
        v = np.array(st.v_BN_N)
        sigma_BN = np.array(st.sigma_BN)
        mass = self.mass_fn()

        if self.engine_out_at_s is not None and t_s >= self.engine_out_at_s and self.engines_ok.all():
            i = self.engine_out_index
            self.engines_ok[i] = False
            self.decisions.append((t_s, f"engine {i} FAILED"))
            if i in OPPOSITE:   # outer engines are canted: keep the remaining set balanced
                self.engines_ok[OPPOSITE[i]] = False
                self.decisions.append((t_s, (f"shut down opposite engine {OPPOSITE[i]} for balance -> "
                                            f"{self.engines_ok.sum()} engines")))

        rho = r - self.target_N
        h_above_target = float(np.dot(rho, self.up))
        r_h = rho - h_above_target * self.up
        g_vec = -self.mu * r / np.linalg.norm(r) ** 3
        h = h_above_target - self.gate

        if h > self.switch_h:
            phase = "powered_descent"
            tgo = 2.0 * max(h, 0.0) / max(-np.dot(v, self.up) + self.v_final, 1.0)
            tgo = float(np.clip(tgo, 2.0, 240.0))
            r_f = self.gate * self.up
            v_f = -self.v_final * self.up
            zem = (self.target_N + r_f) - (r + v * tgo + 0.5 * g_vec * tgo**2)
            zev = v_f - (v + g_vec * tgo)
            a_ctrl = 6.0 * zem / tgo**2 - 2.0 * zev / tgo
        else:
            phase = "terminal"
            v_up = float(np.dot(v, self.up))
            v_h = v - v_up * self.up
            h_ground = h_above_target - self.ground_height_fn(r)
            kp = 0.08 if h_ground > self.no_chase_alt_m else 0.0   # last metres: stop chasing position, just null drift
            a_des = -kp * r_h - 0.8 * v_h + 1.2 * (-self.v_final - v_up) * self.up
            a_ctrl = a_des - g_vec

        thrust_N = mass * a_ctrl
        # tilt limit about local up
        mag = np.linalg.norm(thrust_N)
        dir_N = thrust_N / max(mag, 1e-9)
        max_tilt = self.max_tilt if phase == "powered_descent" else self.terminal_tilt
        cos_t = np.dot(dir_N, self.up)
        if cos_t < np.cos(max_tilt):
            horiz = dir_N - cos_t * self.up
            horiz /= max(np.linalg.norm(horiz), 1e-9)
            dir_N = np.cos(max_tilt) * self.up + np.sin(max_tilt) * horiz

        # attitude reference: +z_B along dir_N
        ref = messaging.AttRefMsgPayload()
        ref.sigma_RN = mrp_with_z_along(dir_N).tolist()
        self.attRefOutMsg.write(ref, CurrentSimNanos, self.moduleID)

        # throttle: only the component along the CURRENT body z counts
        BN = np.array(rbk.MRP2C(sigma_BN))
        z_B_in_N = BN.T @ np.array([0.0, 0.0, 1.0])
        along = max(float(np.dot(thrust_N, z_B_in_N)), 0.0)
        n_ok = int(self.engines_ok.sum())
        throttle = along / (n_ok * self.eng_max)
        if throttle < 0.5 * self.min_thr / self.n_eng:
            throttle = 0.0
        throttle = float(np.clip(throttle, 0.0, 1.0))
        on = np.where(self.engines_ok, throttle * self.period, 0.0)
        cmd = messaging.THRArrayOnTimeCmdMsgPayload()
        cmd.OnTimeRequest = on.tolist() + [0.0] * (messaging.MAX_EFF_CNT - self.n_eng)
        self.thrOnTimeOutMsg.write(cmd, CurrentSimNanos, self.moduleID)

        if phase != self.last.get("phase"):
            self.decisions.append((t_s, f"phase -> {phase}"))
        self.last = {"t_s": t_s, "phase": phase, "throttle": throttle, "thrust_n": throttle * n_ok * self.eng_max,
                     "engines_ok": n_ok, "h_m": h_above_target - self.ground_height_fn(r)}


def build_lander(sim, cfg: MissionConfig, r0_N, v0_N, target_N, dyn_task="dynTask", fsw_task="fswTask",
                 control_period_s=0.2, propellant_kg: float | None = None):
    """Create the lander + all modules and wire the messages. Returns a dict of handles."""
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "lander"
    sc.hub.mHub = cfg.lander.dry_mass_kg
    sc.hub.IHubPntBc_B = np.diag(INERTIA_KG_M2).tolist()
    sc.hub.r_CN_NInit = np.asarray(r0_N).tolist()
    sc.hub.v_CN_NInit = np.asarray(v0_N).tolist()
    up = np.asarray(r0_N) / np.linalg.norm(r0_N)
    sc.hub.sigma_BNInit = mrp_with_z_along(up).tolist()
    sim.AddModelToTask(dyn_task, sc)
    _, mu = add_moon_gravity(sc)

    # engines
    thr_eff = thrusterDynamicEffector.ThrusterDynamicEffector()
    thr_eff.ModelTag = "mainEngines"
    factory = simIncludeThruster.thrusterFactory()
    for r_B, t_B in engine_layout():
        factory.create("Blank_Thruster", r_B, t_B, MaxThrust=float(cfg.lander.engines.max_thrust_n),
                       steadyIsp=float(cfg.lander.isp_s), MinOnTime=0.0,
                       cutoffFrequency=1000.0)   # fast engine response so PWM throttling delivers the commanded impulse
    factory.addToSpacecraft("mainEngines", thr_eff, sc)
    sim.AddModelToTask(dyn_task, thr_eff)

    # propellant tank (mass flows out as the engines fire)
    prop = cfg.scenarios.terminal.start_propellant_kg if propellant_kg is None else propellant_kg
    tank = fuelTank.FuelTank()
    tank.ModelTag = "propTank"
    model = fuelTank.FuelTankModelConstantVolume()
    model.propMassInit = float(prop)
    model.maxFuelMass = float(cfg.lander.wet_mass_at_release_kg - cfg.lander.dry_mass_kg)
    model.r_TcT_TInit = [[0.0], [0.0], [0.0]]
    model.radiusTankInit = 0.6
    tank.setTankModel(model)
    tank.setR_TB_B([0.0, 0.0, -0.3])
    tank.addThrusterSet(thr_eff)
    sc.addStateEffector(tank)
    sim.AddModelToTask(dyn_task, tank)

    # attitude chain (ideal torque — see module docstring)
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "nav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask(dyn_task, nav)
    ext = extForceTorque.ExtForceTorque()
    ext.ModelTag = "attitudeTorque"
    sc.addDynamicEffector(ext)
    sim.AddModelToTask(dyn_task, ext)

    tank_rec = tank.fuelTankOutMsg.recorder()
    sim.AddModelToTask(dyn_task, tank_rec)

    def mass_now():
        return float(sc.hub.mHub + tank_mass_now())

    def tank_mass_now():
        return float(tank.fuelTankOutMsg.read().fuelMass)

    guid = DescentGuidance(cfg, np.asarray(target_N), mass_now, control_period_s)
    guid.mu = mu
    guid.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask(fsw_task, guid, 20)
    thr_eff.cmdsInMsg.subscribeTo(guid.thrOnTimeOutMsg)

    att_err = attTrackingError.attTrackingError()
    att_err.ModelTag = "attErr"
    att_err.attNavInMsg.subscribeTo(nav.attOutMsg)
    att_err.attRefInMsg.subscribeTo(guid.attRefOutMsg)
    sim.AddModelToTask(fsw_task, att_err, 10)

    veh = messaging.VehicleConfigMsgPayload()
    veh.ISCPntB_B = np.diag(INERTIA_KG_M2).flatten().tolist()
    veh_msg = messaging.VehicleConfigMsg().write(veh)
    ctrl = mrpFeedback.mrpFeedback()
    ctrl.ModelTag = "mrpFeedback"
    ctrl.K = 3600.0   # ~1 rad/s attitude bandwidth (ideal torque)
    ctrl.P = 1800.0
    ctrl.Ki = -1.0
    ctrl.guidInMsg.subscribeTo(att_err.attGuidOutMsg)
    ctrl.vehConfigInMsg.subscribeTo(veh_msg)
    sim.AddModelToTask(fsw_task, ctrl, 5)
    ext.cmdTorqueInMsg.subscribeTo(ctrl.cmdTorqueOutMsg)

    return {"sc": sc, "engines": thr_eff, "tank": tank, "guidance": guid, "mu": mu,
            "mass_fn": mass_now, "fuel_fn": tank_mass_now, "veh_msg": veh_msg, "nav": nav,
            "att_err": att_err, "ctrl": ctrl, "ext": ext, "tank_rec": tank_rec}


def touchdown_report(cfg: MissionConfig, r_N, v_N, sigma_BN, slope_deg: float = 0.0) -> dict:
    """Check the touchdown against config limits. Returns speeds, tilt and pass/fail reason."""
    up = np.asarray(r_N) / np.linalg.norm(r_N)
    v = np.asarray(v_N)
    v_up = float(np.dot(v, up))
    v_h = float(np.linalg.norm(v - v_up * up))
    BN = np.array(rbk.MRP2C(np.asarray(sigma_BN)))
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(BN.T @ [0, 0, 1], up), -1, 1))))
    lim = cfg.lander.touchdown_limits
    reason = "ok"
    if -v_up > lim.max_vertical_speed_m_s:
        reason = "hard_vertical"
    elif v_h > lim.max_horizontal_speed_m_s:
        reason = "hard_horizontal"
    elif slope_deg > lim.max_slope_deg:
        reason = "steep_slope"
    return {"vertical_speed_m_s": -v_up, "horizontal_speed_m_s": v_h, "tilt_deg": tilt,
            "slope_deg": slope_deg, "outcome": reason, "altitude_check_m": float(np.linalg.norm(r_N) - R_MOON_M)}
