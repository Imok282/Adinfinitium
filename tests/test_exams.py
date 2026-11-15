import numpy as np

from contracts.mission import load_mission
from data_tools.make_demo_tile import make_demo_tile
from worldmodel.exams import EXAM_BY_NAME, EXAMS, classic_factory, pick_targets, run_exam

CFG = load_mission()


def _hazard_within(tile, x, y, r_m):
    h, w = tile.hazard.shape
    mpp = tile.meters_per_pixel
    rows, cols = np.nonzero(tile.hazard >= 0.5)
    hx, hy = (cols - (w - 1) / 2) * mpp, ((h - 1) / 2 - rows) * mpp
    return bool((np.hypot(hx - x, hy - y) <= r_m).any())


def test_near_hazard_targets_are_safe_but_close_to_hazard():
    tile = make_demo_tile(seed=0)
    ts = pick_targets(tile, 4, near_hazard_m=40.0, seed=0)
    assert len(ts) == 4
    for x, y in ts:
        assert not _hazard_within(tile, x, y, 14.0)        # target itself is safe
        assert _hazard_within(tile, x, y, 40.0 * 1.42 + 5.0)   # ...but a hazard is close by (+-40 m square window)
        assert np.hypot(x, y) <= 300.0                     # start offsets stay on the tile
    assert all(np.hypot(a[0] - b[0], a[1] - b[1]) >= 100 for i, a in enumerate(ts) for b in ts[:i])


def test_exam_definitions_are_unique_and_honest():
    assert len({e.name for e in EXAMS}) == len(EXAMS)
    assert EXAM_BY_NAME["tight_fuel"].start_mass_kg < CFG.scenarios.terminal.start_mass_kg
    # unseen exam must not reuse the control tile seed
    assert 0 not in EXAM_BY_NAME["unseen_craters"].tile_seeds


def test_control_exam_classic_passes():
    r = run_exam(EXAM_BY_NAME["control"], classic_factory, CFG, n_per_target=48, targets_per_tile=1)
    assert r["n"] == 48 and r["success_rate"] >= 0.9
