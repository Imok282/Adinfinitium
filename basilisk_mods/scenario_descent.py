"""STARTER (Tanush): fly the 5-engine lander from the terminal-descent start to touchdown.

    python -m basilisk_mods.scenario_descent                       # nominal
    python -m basilisk_mods.scenario_descent --engine-out-at 20    # engine 1 fails at t = 20 s
    python -m basilisk_mods.scenario_descent --alt 2000 --offset-east 400 --no-viz

Writes (contract formats, see docs/FORMATS.md):
  runs/basilisk/telemetry_<id>.csv              -> Arnav's mission-control window
  runs/basilisk/descent_<id>.json               touchdown report
  runs/basilisk/_VizFiles/descent_<id>_UnityViz.bin

Start: terminal-descent conditions (same as the JAX swarm, config/mission.yaml scenarios.terminal).
Next step for KAN-42: start from runs/basilisk/pdi_state.json (15 km, ~1.7 km/s) with a braking phase.
Ground: flat 1737.4 km sphere until Arnav's terrain module (KAN-44) plugs into guidance.ground_height_fn.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from Basilisk.utilities import macros, vizSupport

from basilisk_mods.bsk_common import R_MOON_M, local_frame, mission, new_sim, out_dir
from basilisk_mods.lander_descent import build_lander, touchdown_report
from contracts.provenance import Meta, Provenance
from contracts.runs import write_telemetry

TARGET_LAT_DEG = -89.0   # demo target near the south pole (non-rotating Moon in this starter)
TARGET_LON_DEG = 30.0


def target_point_N(lat_deg=TARGET_LAT_DEG, lon_deg=TARGET_LON_DEG, height_m=0.0) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return (R_MOON_M + height_m) * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def run(run_id="demo", alt=2000.0, offset_east=300.0, offset_north=-200.0, v_down=45.0, v_east=10.0,
        engine_out_at=None, viz=True, max_time=300.0, command_fn=None, ground_height_fn=None, on_step=None,
        rtf=None, live_viz=False) -> dict:
    """Fly one descent. Optional live hooks (all default off, so the plain run is unchanged):

    command_fn        outside pilot, see ``DescentGuidance.command_fn`` (classic backup on any error)
    ground_height_fn  f(r_N) -> ground height [m] above the target point (terrain); default flat
    on_step           f(t_s, handles, guidance, state_msg_payload, target_N, (E, N, U)) called every control period
    rtf               real-time factor: 1.0 = wall clock, None = as fast as possible
    live_viz          stream to a running Vizard (liveStream) instead of only writing the playback file
    """
    import time

    cfg = mission()
    dyn_dt, ctrl_dt = 0.01, 0.2   # dyn step must be << control period: on-time pulses are quantised to dyn_dt
    sim = new_sim({"dynTask": dyn_dt, "fswTask": ctrl_dt})

    tgt = target_point_N()
    E, N, U = local_frame(tgt)
    r0 = tgt + alt * U + offset_east * E + offset_north * N
    v0 = -v_down * U + v_east * E
    h = build_lander(sim, cfg, r0, v0, tgt, control_period_s=ctrl_dt)
    guid = h["guidance"]
    guid.engine_out_at_s = engine_out_at
    guid.command_fn = command_fn
    if ground_height_fn is not None:
        guid.ground_height_fn = ground_height_fn
    if viz or live_viz:
        vizSupport.enableUnityVisualization(sim, "dynTask", h["sc"], saveFile=str(out_dir() / f"descent_{run_id}"),
                                            thrEffectorList=[h["engines"]], liveStream=bool(live_viz))
    sim.InitializeSimulation()

    rows, t, report = [], 0.0, None
    state_msg = h["sc"].scStateOutMsg
    wall0 = time.monotonic()
    while t < max_time:
        t += ctrl_dt
        sim.ConfigureStopTime(macros.sec2nano(t))
        sim.ExecuteSimulation()
        st = state_msg.read()
        if on_step is not None:
            on_step(t, h, guid, st, tgt, (E, N, U))
        if rtf:
            lag = t / rtf - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
        r, v = np.array(st.r_BN_N), np.array(st.v_BN_N)
        ground = guid.ground_height_fn(r)
        alt_now = float(np.dot(r - tgt, U)) - ground
        if abs(t - round(t)) < 1e-6 or alt_now <= 0:          # log once per second
            rel = r - tgt
            fuel = h["fuel_fn"]()
            decision = "; ".join(d for (td, d) in guid.decisions if t - 1.0 < td <= t)
            rows.append({"t_s": round(t, 2), "phase": guid.last.get("phase", ""), "x_m": float(rel @ E),
                         "y_m": float(rel @ N), "alt_m": alt_now, "vx_m_s": float(v @ E), "vy_m_s": float(v @ N),
                         "vz_m_s": float(v @ U), "mass_kg": h["mass_fn"](), "fuel_kg": fuel,
                         "thrust_n": guid.last.get("thrust_n", 0.0), "engines_ok": guid.last.get("engines_ok", 5),
                         "battery_wh": "", "solar_w": "", "earth_visible": "", "decision": decision})
        if alt_now <= 0.0:
            report = touchdown_report(cfg, r, v, st.sigma_BN)
            report.update({"t_s": t, "x_m": float((r - tgt) @ E), "y_m": float((r - tgt) @ N),
                           "fuel_left_kg": h["fuel_fn"](), "engine_out_at_s": engine_out_at})
            rows[-1]["phase"] = "touchdown"
            rows[-1]["decision"] = f"TOUCHDOWN {report['outcome']}"
            break
        if h["fuel_fn"]() <= 0.01:
            report = {"outcome": "out_of_fuel", "t_s": t}
            break
    if report is None:
        report = {"outcome": "timeout", "t_s": t}

    meta = Meta(provenance=Provenance.MODELLED, source="Basilisk 6-DOF rigid body, 5 thrusters, ideal attitude torque",
                frame="local east/north/up at target; flat ground", created_by="basilisk_mods.scenario_descent")
    report["meta"] = meta.model_dump(mode="json")
    write_telemetry(rows, out_dir() / f"telemetry_{run_id}.csv")
    (out_dir() / f"descent_{run_id}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items() if k != "meta"}, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="demo")
    ap.add_argument("--alt", type=float, default=2000.0)
    ap.add_argument("--offset-east", type=float, default=300.0)
    ap.add_argument("--offset-north", type=float, default=-200.0)
    ap.add_argument("--v-down", type=float, default=45.0)
    ap.add_argument("--engine-out-at", type=float, default=None)
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()
    run(a.id, a.alt, a.offset_east, a.offset_north, a.v_down, engine_out_at=a.engine_out_at, viz=not a.no_viz)


if __name__ == "__main__":
    main()
