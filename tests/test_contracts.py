import numpy as np
import pytest
from pydantic import ValidationError

from contracts.light import LightField, load_light, save_light
from contracts.mission import load_mission
from contracts.paths import model_path, repo_root
from contracts.provenance import Meta, Provenance
from contracts.runs import (
    TELEMETRY_COLUMNS,
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
from contracts.tiles import Tile, hazard_from_slope, load_tile, save_tile, slope_from_height
from data_tools.make_demo_tile import make_demo_tile


def meta(p=Provenance.MODELLED):
    return Meta(provenance=p, source="test")


def test_mission_yaml_loads_and_is_consistent():
    cfg = load_mission()
    assert cfg.lander.touchdown_limits.max_slope_deg == 12.0
    assert 1.6 < cfg.thrust_to_weight_at_release() < 1.7
    assert cfg.thrust_to_weight_at_release(engines_out=1) > 1.0
    assert cfg.delta_v_available_m_s() >= cfg.lander.descent_delta_v_budget_m_s


def test_mission_yaml_rejects_unknown_keys(tmp_path):
    import yaml
    raw = yaml.safe_load((repo_root() / "config" / "mission.yaml").read_text())
    raw["lander"]["typo_field"] = 1
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_mission(p)


def test_provenance_banner():
    assert Provenance.GENERATED.banner == "GENERATED — NOT OBSERVED"


def test_tile_roundtrip(tmp_path):
    t = make_demo_tile(size=64, mpp=5.0, seed=1, n_craters=10)
    p = save_tile(t, tmp_path / "t.npz")
    t2 = load_tile(p)
    assert t2.shape == (64, 64)
    assert t2.meta.provenance is Provenance.GENERATED
    assert t2.meta.units["height"] == "m"
    np.testing.assert_allclose(t2.height, t.height, rtol=1e-6)


def test_tile_rejects_bad_shapes():
    z = np.zeros((8, 8), np.float32)
    t = Tile("x", 5.0, z, z, z, z, np.zeros((8, 9), bool), meta())
    with pytest.raises(ValueError):
        t.validate()


def test_slope_and_hazard_backup():
    mpp = 5.0
    x = np.arange(20) * mpp
    ramp = np.tile(np.tan(np.radians(20.0)) * x, (20, 1))   # 20 deg ramp
    s = slope_from_height(ramp, mpp)
    assert abs(float(np.median(s)) - 20.0) < 0.1
    assert hazard_from_slope(s, 12.0).mean() == 1.0


def test_light_roundtrip(tmp_path):
    T, H, W = 3, 4, 5
    lf = LightField("r", "2027-01-01", ["a", "b", "c"], np.array([0.0, 1.0, 2.0]),
                    np.full((T, H, W), 0.5), np.zeros((T, H, W)), np.ones((T, H, W)), meta())
    lf2 = load_light(save_light(lf, tmp_path / "l.npz"))
    assert lf2.earth_visible().all()


def test_run_files_roundtrip(tmp_path):
    g = GenerationLog(run="r", generation=3, population=10, best_score=1, mean_score=0.5,
                      success_rate=0.4, meta=meta())
    assert read_generation(write_json(g, tmp_path / "generation_003.json")).generation == 3
    c = Champion(run="r", generation=3, champion_id="c1", score=9.0,
                 genes=MissionGenes(site="s", target_x_m=0, target_y_m=0),
                 initial_state={"pos_m": [0, 0, 100], "vel_m_s": [0, 0, -10], "mass_kg": 900},
                 controls=[ControlStep(t_s=0.0, thrust_n=[0, 0, 3000])], outcome="ok", meta=meta())
    assert read_champion(write_json(c, tmp_path / "champion_003.json")).controls[0].thrust_n[2] == 3000
    rows = [{"t_s": 0.0, "phase": "terminal", "decision": "start"}]
    got = read_telemetry(write_telemetry(rows, tmp_path / "telemetry_003.csv"))
    assert got[0]["phase"] == "terminal" and tuple(got[0].keys()) == TELEMETRY_COLUMNS


def test_hazard_detector_config_block_and_model_path(monkeypatch, tmp_path):
    h = load_mission().hazard_detector
    assert 0 < h.decision_threshold < 1 and len(h.gap_fill_sigma_px) >= 1
    assert h.gap_fill_sigma_px == sorted(h.gap_fill_sigma_px)  # finest gap-fill width first
    bad = (repo_root() / "config" / "mission.yaml").read_text().replace("decision_threshold: 0.5", "decision_threshold: 1.5")
    f = tmp_path / "m.yaml"
    f.write_text(bad)
    with pytest.raises(ValidationError):
        load_mission(f)
    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    assert model_path("hazard_unet") == tmp_path / "models" / "hazard_unet.pt"


def test_example_sites_load():
    lib = load_sites(repo_root() / "sites" / "sites.example.json")
    assert lib.by_id("chandrayaan3_shiv_shakti").lat_deg < -69


def test_validate_formats_selftest():
    from data_tools.validate_formats import main
    assert main(["--selftest"]) == 0
