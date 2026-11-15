"""Overnight kit: plans build, commands get budgets/resume, the ledger keeps failures, the report and verifier cope with missing files."""

import json
import sys

from ops.overnight.profiles import PROFILES, get_profile
from ops.overnight.report import build_report
from ops.overnight.run_all import Ledger, Stage, build_command, run_stage


def test_every_profile_builds():
    for name in PROFILES:
        p = get_profile(name)
        assert p.stages and all(s.cmds for s in p.stages)


def test_train_stage_gets_resume_and_budget():
    s = Stage("x", [["worldmodel.hazard.train"]], budget_min=100, train=True)
    cmd = build_command(s, s.cmds[0], "python", resume=True)
    assert "--resume" in cmd and "--max-minutes" in cmd and cmd[:3] == ["python", "-s", "-m"]


def test_failed_stage_is_recorded_with_error_tail(tmp_path):
    led = Ledger(tmp_path)
    s = Stage("boom", [["ops.nonexistent_module"]], budget_min=1)
    assert run_stage(s, tmp_path, led, sys.executable, False, 1.0, {}) == "failed"
    ev = next(e for e in led.events() if e["event"] == "stage_end")
    assert ev["status"] == "failed" and "error_tail" in ev
    assert "boom" not in led.finished()


def test_report_from_partial_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_RUNS", str(tmp_path / "runs"))
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"event": "run_start", "run_id": "t", "profile": "smoke"}) + "\n")
    out = build_report(tmp_path)
    assert "Run report" in out.read_text()
