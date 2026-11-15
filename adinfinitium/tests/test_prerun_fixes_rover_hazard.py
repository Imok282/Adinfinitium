"""Pre-run fixes: rover always leaves a loadable named checkpoint, the hazard checkpoint shown to the rover is explicit, hazard metrics on laser pixels."""

import numpy as np
import pytest

from contracts.paths import data_root, tile_path
from worldmodel.hazard.evaluate import measured_px_metrics


@pytest.mark.parametrize("arg, env, on_disk, want", [
    (None, None, ("hazard_unet_big", "hazard_unet"), "hazard_unet_big"),
    (None, None, ("hazard_unet",), "hazard_unet"),
    (None, None, (), None),
    (None, "hazard_env", ("hazard_unet_big",), "hazard_env"),
    ("hazard_cli", "hazard_env", ("hazard_unet_big",), "hazard_cli"),
    ("backup", None, ("hazard_unet_big",), None),
])
def test_hazard_name_resolution_order(tmp_path, monkeypatch, arg, env, on_disk, want):
    pytest.importorskip("scipy")
    from worldmodel.rover.world import resolve_hazard_name

    monkeypatch.setenv("ADI_DATA", str(tmp_path))
    monkeypatch.delenv("ADI_HAZARD_NAME", raising=False)
    if env:
        monkeypatch.setenv("ADI_HAZARD_NAME", env)
    (tmp_path / "models").mkdir()
    for n in on_disk:
        (tmp_path / "models" / f"{n}.pt").touch()
    assert resolve_hazard_name(arg) == want


def test_measured_pixel_metrics_use_only_laser_pixels():
    label = np.array([[True, False], [True, False]])
    unet = np.array([[True, False], [False, True]])          # right on the top row, wrong on the bottom row
    mask = np.array([[True, True], [False, False]])
    res = measured_px_metrics(mask, label, ~label, unet)
    assert res["measured_fraction"] == 0.5
    assert res["unet_measured_px"]["f1"] == 1.0 and res["unet_measured_px"]["tp"] + res["unet_measured_px"]["tn"] == 2
    assert res["backup_slope_rule_measured_px"]["f1"] == 0.0
    skipped = measured_px_metrics(None, label, ~label, unet)
    assert "skipped" in skipped["measured_px_note"] and "unet_measured_px" not in skipped


def test_rover_train_always_leaves_a_loadable_named_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    if not tile_path("lola5m_site01").exists() or not (data_root() / "kernels" / "de440s.bsp").exists():
        pytest.skip("real LOLA tile / SPICE kernels not on this machine")
    from worldmodel.rover import train
    from worldmodel.rover.policy import load_checkpoint

    monkeypatch.setenv("ADI_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(train, "model_path", lambda name: tmp_path / f"{name}.pt")
    args = ["--arch", "conv", "--envs", "8", "--horizon", "4", "--updates", "1", "--sites", "Site01", "--dates", "2027-03-01", "--device", "cpu",
            "--hazard-name", "backup", "--name", "t"]
    best = tmp_path / "t.pt"
    for extra in ([], ["--resume"]):                          # the resumed run has no updates left and must still restore the named checkpoint
        best.unlink(missing_ok=True)
        assert train.main(args + extra) == 0
        _, info = load_checkpoint(best)
        assert info["selection"] == train.SELECT_LAST and info["update"] == 1
        assert info["trained_on"] == ["lola5m_site01@2027-03-01"] and info["hazard_source"] == "slope-rule backup"
