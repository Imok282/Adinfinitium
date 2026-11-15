"""Live pilot adapter and the Basilisk pilot hook (KAN-48 live mission)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from contracts.mission import load_mission  # noqa: E402
from worldmodel.lander.live import LivePilot, PilotState, traced_forward  # noqa: E402
from worldmodel.lander.terrain import flat_terrain  # noqa: E402


def _pilot():
    return LivePilot.create(load_mission(), flat_terrain(), (0.0, 0.0), None)


def test_traced_forward_matches_model():
    """The layer-by-layer replay with attention weights gives the same output as PilotFormer.forward."""
    p = _pilot()
    torch.manual_seed(1)
    torch.nn.init.normal_(p.model.mu[-1].weight, std=0.05)
    feats, img = torch.randn(2, 24), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        mu1, v1 = p.model(feats, img)
    mu2, v2, tr = traced_forward(p.model, feats[:1], img[:1])
    assert torch.allclose(mu1[:1], mu2, atol=1e-5) and torch.allclose(v1[:1], v2, atol=1e-5)
    assert tr["tokens"] == 66 and tr["layers"] == 8 and len(tr["rollout"]) == 66
    assert abs(sum(tr["rollout"]) - 1.0) < 1e-3
    assert tr["chain"] and all(0 <= e[0] < 8 and 0 <= e[1] < 66 and 0 <= e[2] < 66 for e in tr["chain"])


def test_untrained_pilot_flies_like_classic():
    """Zero last layer: the untrained network adds nothing to the classic ZEM/ZEV command."""
    p = _pilot()
    assert p.info["trained"] is False
    out = p.act(PilotState(1.0, 300.0, -200.0, 1900.0, (10.0, 0.0, -45.0), 900.0, 5))
    assert np.allclose(out["thrust_enu_n"], out["classic_enu_n"], atol=1e-3)
    assert out["thrust_enu_n"][2] > 0.0                      # pushing up while falling at 45 m/s


def test_basilisk_hook_falls_back_to_classic_on_pilot_error():
    """Physics backup first: a pilot that raises leaves the classic command in charge and is logged."""
    pytest.importorskip("Basilisk")
    from basilisk_mods import scenario_descent

    seen = {}

    def broken(*_):
        raise RuntimeError("test pilot down")

    def on_step(t, h, guid, st, tgt, frame):
        seen["guid"] = guid

    rep = scenario_descent.run("pytest_live", viz=False, max_time=1.0, command_fn=broken, on_step=on_step)
    guid = seen["guid"]
    assert rep["outcome"] == "timeout"
    assert guid.last["source"] == "classic"
    assert any("classic backup used" in d for _, d in guid.decisions)


def test_body_to_enu_quaternion_upright():
    """A lander whose body +z points along local up has the identity attitude about the vertical axis tilt."""
    pytest.importorskip("Basilisk")
    from basilisk_mods.bsk_common import local_frame, mrp_with_z_along
    from scripts.live_mission import body_to_enu_quat

    up = np.array([0.0, np.cos(np.radians(1.0)), -np.sin(np.radians(1.0))])
    E, N, U = local_frame(up)
    w, x, y, z = body_to_enu_quat(mrp_with_z_along(U), E, N, U)
    R_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # body +z in ENU
    assert np.allclose(R_z, [0.0, 0.0, 1.0], atol=1e-6)
