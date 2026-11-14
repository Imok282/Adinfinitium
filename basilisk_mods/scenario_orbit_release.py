"""STARTER (Tanush): lunar orbit -> release -> deorbit -> coast to powered-descent start.

    python -m basilisk_mods.scenario_orbit_release
    python -m basilisk_mods.scenario_orbit_release --orbit-alt-km 100 --peri-alt-km 15 --no-viz

Writes:
  runs/basilisk/pdi_state.json            state at periapsis (powered-descent initiation)
  runs/basilisk/orbit_altitude.png        altitude vs time
  runs/basilisk/_VizFiles/orbit_release_UnityViz.bin   open in Vizard

Physics: point-mass Moon, impulsive deorbit burn (a real burn takes ~10 s at 4 kN, so the
impulsive model is a good first approximation). Masses from config/mission.yaml.
Release is modelled as dropping the propulsion module's mass; the propulsion module itself
is not simulated after release (add it as a second Spacecraft if you want it in Vizard).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros, orbitalMotion, simHelpers, vizSupport

from basilisk_mods.bsk_common import R_MOON_M, add_moon_gravity, mission, new_sim, out_dir
from contracts.provenance import Meta, Provenance

PROPULSION_MODULE_MASS_KG = 900.0  # design assumption: carrier dry mass at lunar orbit (not in mission.yaml yet)


def run(orbit_alt_km=100.0, peri_alt_km=15.0, inclination_deg=90.0, raan_deg=30.0,
        release_after_s=600.0, viz=True, plot=True) -> dict:
    cfg = mission()
    lander_mass = cfg.lander.wet_mass_at_release_kg

    sim = new_sim({"dynTask": 5.0})
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "lander_stack"
    sc.hub.mHub = lander_mass + PROPULSION_MODULE_MASS_KG
    sim.AddModelToTask("dynTask", sc)
    _, mu = add_moon_gravity(sc)

    oe = orbitalMotion.ClassicElements()
    oe.a = R_MOON_M + orbit_alt_km * 1e3
    oe.e = 0.0
    oe.i = inclination_deg * macros.D2R
    oe.Omega = raan_deg * macros.D2R
    oe.omega = 0.0
    oe.f = 0.0
    r0, v0 = orbitalMotion.elem2rv(mu, oe)
    sc.hub.r_CN_NInit = r0
    sc.hub.v_CN_NInit = v0

    rec = sc.scStateOutMsg.recorder(macros.sec2nano(10.0))
    sim.AddModelToTask("dynTask", rec)
    if viz:
        vizSupport.enableUnityVisualization(sim, "dynTask", sc,
                                            saveFile=str(out_dir() / "orbit_release"))
    sim.InitializeSimulation()

    # 1) circular orbit until release
    sim.ConfigureStopTime(macros.sec2nano(release_after_s))
    sim.ExecuteSimulation()

    # 2) release: the lander continues alone
    sc.hub.mHub = lander_mass

    # 3) impulsive deorbit burn: retrograde, lowering periapsis to peri_alt_km
    pos = sc.dynManager.getStateObject(sc.hub.nameOfHubPosition)
    vel = sc.dynManager.getStateObject(sc.hub.nameOfHubVelocity)
    r = simHelpers.EigenVector3d2np(pos.getState())
    v = simHelpers.EigenVector3d2np(vel.getState())
    ra = np.linalg.norm(r)
    rp = R_MOON_M + peri_alt_km * 1e3
    a_new = 0.5 * (ra + rp)
    v_new = np.sqrt(mu * (2.0 / ra - 1.0 / a_new))
    dv = v_new - np.linalg.norm(v)
    vel.setState(v / np.linalg.norm(v) * v_new)
    prop_used = lander_mass * (1 - np.exp(-abs(dv) / (cfg.lander.isp_s * cfg.moon.g0_m_s2)))

    # 4) coast half an orbit to periapsis = powered-descent initiation (PDI)
    t_coast = np.pi * np.sqrt(a_new**3 / mu)
    sim.ConfigureStopTime(macros.sec2nano(release_after_s + t_coast))
    sim.ExecuteSimulation()

    r_f = simHelpers.EigenVector3d2np(pos.getState())
    v_f = simHelpers.EigenVector3d2np(vel.getState())
    alt = np.linalg.norm(rec.r_BN_N, axis=1) - R_MOON_M
    t = rec.times() * macros.NANO2SEC

    state = {
        "t_s": float(release_after_s + t_coast),
        "r_BN_N_m": r_f.tolist(),
        "v_BN_N_m_s": v_f.tolist(),
        "mass_kg": float(lander_mass - prop_used),
        "altitude_m": float(np.linalg.norm(r_f) - R_MOON_M),
        "speed_m_s": float(np.linalg.norm(v_f)),
        "deorbit_dv_m_s": float(dv),
        "deorbit_propellant_kg": float(prop_used),
        "meta": Meta(provenance=Provenance.MODELLED, source="Basilisk point-mass Moon, impulsive deorbit",
                     frame="Moon-centred inertial (Basilisk N), non-rotating Moon",
                     units={"r_BN_N_m": "m", "v_BN_N_m_s": "m/s", "mass_kg": "kg", "t_s": "s"},
                     created_by="basilisk_mods.scenario_orbit_release").model_dump(mode="json"),
    }
    (out_dir() / "pdi_state.json").write_text(json.dumps(state, indent=2))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t / 60, alt / 1e3)
        ax.axvline(release_after_s / 60, ls="--", c="gray")
        ax.text(release_after_s / 60, alt.max() / 1e3, " release + deorbit", va="top")
        ax.set_xlabel("time [min]")
        ax.set_ylabel("altitude above 1737.4 km sphere [km]")
        ax.set_title("Orbit -> release -> deorbit -> PDI (MODELLED, Basilisk)")
        fig.tight_layout()
        fig.savefig(out_dir() / "orbit_altitude.png", dpi=130)

    print(f"orbit period         : {2 * np.pi * np.sqrt(oe.a**3 / mu) / 60:.1f} min")
    print(f"deorbit delta-v      : {dv:.1f} m/s  ({prop_used:.1f} kg propellant)")
    print(f"PDI altitude         : {state['altitude_m'] / 1e3:.2f} km  after {t_coast / 60:.1f} min coast")
    print(f"PDI speed            : {state['speed_m_s']:.1f} m/s")
    print(f"wrote                : {out_dir() / 'pdi_state.json'}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit-alt-km", type=float, default=100.0)
    ap.add_argument("--peri-alt-km", type=float, default=15.0)
    ap.add_argument("--inclination-deg", type=float, default=90.0)
    ap.add_argument("--release-after-s", type=float, default=600.0)
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()
    run(a.orbit_alt_km, a.peri_alt_km, a.inclination_deg, release_after_s=a.release_after_s, viz=not a.no_viz)


if __name__ == "__main__":
    main()
