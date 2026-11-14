"""Validate any shared file against its contract (KAN-19).

    python -m data_tools.validate_formats data/tiles/demo_craters.npz runs/bench/generation_000.json
    python -m data_tools.validate_formats --selftest      # writes tiny fixtures to a temp dir and checks them

The file type is detected from the name: tiles/*.npz, light/*.npz, generation_*.json,
champion_*.json, telemetry_*.csv, sites*.json, mission*.yaml.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from contracts.light import LightField, PointSeries, load_light, load_points, save_light, save_points
from contracts.mission import load_mission
from contracts.provenance import Meta, Provenance
from contracts.runs import (
    Champion,
    ControlStep,
    GenerationLog,
    MissionGenes,
    read_champion,
    read_generation,
    read_telemetry,
    write_json,
    write_telemetry,
)
from contracts.sites import load_sites
from contracts.tiles import load_tile


def validate(path: Path) -> str:
    name, parent = path.name, path.parent.name
    if name.endswith(".yaml"):
        load_mission(path)
        return "mission config"
    if name.endswith("_points.npz") and parent == "light":
        ps = load_points(path)
        return f"light point series T={len(ps.times_utc)} K={len(ps.cells)} [{ps.meta.provenance.banner}]"
    if name.endswith(".npz") and parent == "light":
        lf = load_light(path)
        v = "v2 (azimuth)" if lf.sun_az_deg is not None else "v1 (no azimuth)"
        return f"light field {lf.sun_fraction.shape} {v} [{lf.meta.provenance.banner}]"
    if name.endswith(".npz"):
        t = load_tile(path)
        return f"tile {t.shape} @ {t.meters_per_pixel} m/px [{t.meta.provenance.banner}]"
    if name.startswith("generation_"):
        g = read_generation(path)
        return f"generation {g.generation}: success {g.success_rate:.1%}"
    if name.startswith("champion_"):
        c = read_champion(path)
        return f"champion {c.champion_id}: {len(c.controls)} control steps, outcome {c.outcome}"
    if name.startswith("telemetry_"):
        return f"telemetry: {len(read_telemetry(path))} rows"
    if name.startswith("sites") and name.endswith(".json"):
        return f"site library: {len(load_sites(path).sites)} sites"
    raise ValueError(f"don't know which contract applies to {path}")


def write_fixtures(d: Path) -> list[Path]:
    from contracts.tiles import save_tile
    from data_tools.make_demo_tile import make_demo_tile

    meta = Meta(provenance=Provenance.GENERATED, source="validate_formats selftest")
    tile = save_tile(make_demo_tile(size=32, mpp=5.0, n_craters=4, region="fixture"),
                     d / "tiles" / "fixture.npz")
    T = 2
    lf = LightField("fixture", "2027-01-01", ["2027-01-01T00:00:00Z", "2027-01-01T01:00:00Z"],
                    np.array([0.0, 3600.0]), np.ones((T, 32, 32)) * 0.5,
                    np.full((T, 32, 32), 2.0), np.full((T, 32, 32), 5.0), meta,
                    sun_az_deg=np.array([10.0, 20.0]), earth_az_deg=np.array([100.0, 101.0]))
    light = save_light(lf, d / "light" / "fixture_2027-01-01.npz")
    cells = np.array([[16, 16], [15, 16], [17, 16]])
    points = save_points(PointSeries.from_field(lf, cells, np.zeros((3, 360))),
                         d / "light" / "fixture_2027-01-01_points.npz")
    gen = write_json(GenerationLog(run="fixture", generation=0, population=1, best_score=0.0,
                                   mean_score=0.0, success_rate=1.0, meta=meta),
                     d / "runs" / "generation_000.json")
    champ = write_json(Champion(run="fixture", generation=0, champion_id="c0", score=1.0,
                                genes=MissionGenes(site="fixture", target_x_m=0.0, target_y_m=0.0),
                                initial_state={"pos_m": [0, 0, 100], "vel_m_s": [0, 0, -5], "mass_kg": 900},
                                controls=[ControlStep(t_s=0.0, thrust_n=[0.0, 0.0, 1500.0])],
                                outcome="ok", meta=meta),
                       d / "runs" / "champion_000.json")
    tel = write_telemetry([{"t_s": 0.0, "phase": "terminal", "alt_m": 100.0, "decision": "start"}],
                          d / "runs" / "telemetry_000.csv")
    sites = d / "sites.json"
    sites.write_text('{"schema_version": 1, "sites": [{"id": "fixture", "name": "Fixture", '
                     '"lat_deg": -89.5, "lon_deg": 0.0, "source": "selftest"}]}', encoding="utf-8")
    return [tile, light, points, gen, champ, tel, sites]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    paths = list(a.paths)
    tmp = None
    if a.selftest:
        tmp = tempfile.TemporaryDirectory()
        paths += write_fixtures(Path(tmp.name))
        from contracts.paths import mission_config_path
        paths.append(mission_config_path())
    if not paths:
        ap.error("give file paths or --selftest")
    bad = 0
    for p in paths:
        try:
            print(f"OK    {p.name:32s} {validate(p)}")
        except Exception as e:  # noqa: BLE001 — report every file, then fail
            bad += 1
            print(f"FAIL  {p.name:32s} {type(e).__name__}: {e}")
    if tmp:
        tmp.cleanup()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
