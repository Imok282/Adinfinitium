"""Pre-run fixes in the overnight runner: deadline caps, the no-checkpoint rule, verify reads the run it is given."""

import json

import pytest

from ops.overnight import report, run_all
from ops.overnight.profiles import Stage, get_profile


def _stages():
    return [Stage("light", [["m", "--name", "l"]], 60, train=True), Stage("light_eval", [["m"]], 15),
            Stage("rover", [["m", "--name", "r"]], 150, train=True), Stage("rover_eval", [["m"]], 20),
            Stage("terrain", [["m", "--name", "t"]], 150, train=True, shrink_first=True, skip_below_min=30), Stage("terrain_eval", [["m"]], 25)]


@pytest.mark.parametrize(("left", "idx", "want", "skipped"), [
    (1000, 0, 60, []),            # everything fits: untouched
    (360, 0, 60, []),             # 60 short: terrain alone shrinks (150 -> 90)
    (280, 0, 60, ["terrain"]),    # terrain would get 10 min (< 30): skipped, its eval time is freed, light and rover keep theirs
    (150, 0, 32.9, ["terrain"]),  # still short without terrain: light and rover share the cut, rover is kept
    (45, 4, None, ["terrain"]),   # at the terrain stage itself, 20 min of training time left after its eval: skipped
])
def test_deadline_plan(left, idx, want, skipped):
    st = _stages()
    got, det = run_all.deadline_plan(st[idx], st[idx + 1:], left)
    assert (got is None) == (want is None)
    if want is not None:
        assert got == pytest.approx(want, abs=0.1)
    assert det.get("skipped", []) == skipped
    if left == 150:
        assert det["capped"]["rover"] >= run_all.MIN_TRAIN_MIN


def test_training_stage_without_checkpoint_is_failed_and_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ADI_RUNS", str(tmp_path / "runs"))
    from ops.overnight.profiles import Profile

    prof = Profile("t", [Stage("light", [["platform", "--name", "nockpt"]], 0.2, train=True)])
    monkeypatch.setattr(run_all, "build_command", lambda stage, cmd, python, *a, **k: [python, "-m", "platform"])     # exits 0, writes no checkpoint
    for _ in range(2):                                         # the second (--resume) pass must run it again, not skip it
        run_all.run(prof, "r1", resume=True)
    ev = [json.loads(x) for x in (tmp_path / "runs" / "overnight" / "r1" / "ledger.jsonl").read_text().splitlines()]
    ends = [e for e in ev if e.get("event") == "stage_end"]
    assert len(ends) == 2 and all(e["status"] == "failed" and e["no_checkpoint"] for e in ends)


def test_smoke_profile_is_isolated_and_short():
    p = get_profile("smoke")
    assert p.fresh and p.sandbox and sum(s.budget_min for s in p.stages) <= 25
    names = [run_all.checkpoint_name(s) for s in p.stages if s.train]
    assert all(n.startswith("smoketest_") for n in names)


def test_report_shows_test_sites_first_and_whole(tmp_path):
    d = {"checkpoint_id": "x", "trained_on": [], "regions": {
        "lola5m_site11": {"role": "val", "pad": "v" * 4000},
        "lola5m_site23": {"role": "TEST", "unet": {"f1": 0.123456}, "curve": list(range(400))}}}
    (tmp_path / "hazard").mkdir()
    (tmp_path / "hazard" / "eval.json").write_text(json.dumps(d))
    text = "\n".join(report._headline(tmp_path))
    assert text.index("lola5m_site23") < text.index("lola5m_site11") and "0.123456" in text and "Site23 only" in text
