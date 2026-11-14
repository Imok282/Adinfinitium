"""One-command overnight runner (4090 PC, Colab, Mac mini). Cross-platform: pure Python, no shell scripting.

    python -m ops.overnight.run_all --profile smoke                 # FIRST on every machine: tiny settings, proves the kit works
    python -m ops.overnight.run_all --profile real4090 --resume     # the 4090 overnight
    python -m ops.overnight.run_all --profile colab --resume
    python -m ops.overnight.run_all --profile real4090 --dry-run    # print the plan and the time budget, run nothing

Every stage is a subprocess (`python -s -m <module> ...`) with its own log, a wall-clock budget and a recorded outcome. A failed stage never stops the
others: it is written to the LEDGER (<ADI_RUNS>/overnight/<run-id>/ledger.jsonl) with the last lines of its log, so failures are kept for the showcase.
`status.json` is refreshed every ~30 s (current stage, last log line, GPU, disk). At the end a RUN_REPORT.md with curves and held-out results is built.
--resume skips stages that already finished (ok or killed at their budget, which keep a resumable checkpoint) and continues training stages from `*_last.pt`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from contracts.paths import repo_root, runs_root
from ops.overnight.profiles import Profile, Stage, get_profile

STATUS_EVERY_S = 30
TAIL_LINES = 30


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def tail(path: Path, n: int = TAIL_LINES) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def gpu_status() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()[0]
        u, mu, mt, tc = (float(x) for x in out.split(","))
        return {"util_pct": u, "mem_used_mb": mu, "mem_total_mb": mt, "temp_c": tc}
    except Exception:  # noqa: BLE001 — no NVIDIA GPU / no nvidia-smi
        return None


def disk_free_gb(path: Path) -> float:
    import shutil

    return round(shutil.disk_usage(path).free / 1e9, 1)


class Ledger:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, event: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": now(), **event}) + "\n")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def finished(self) -> set[str]:
        """Stages that already ended ok or were stopped at their budget (partial but resumable)."""
        return {e["stage"] for e in self.events() if e.get("event") == "stage_end" and e.get("status") in ("ok", "timeout")}

    def attempts(self, stage: str) -> int:
        return sum(1 for e in self.events() if e.get("event") == "stage_start" and e.get("stage") == stage)


def build_command(stage: Stage, cmd: list[str], python: str, resume: bool, budget_scale: float = 1.0) -> list[str]:
    args = list(cmd[1:])
    if stage.train and resume and "--resume" not in args:
        args.append("--resume")
    if stage.train and "--max-minutes" not in args:
        args += ["--max-minutes", f"{max(stage.budget_min * budget_scale * 0.88, 0.2):.2f}"]     # the script sizes its own steps to finish inside the stage budget
    return [python, "-s", "-m", cmd[0], *args]     # -s: no user-site packages; cwd (the repo root) stays importable, so no pip install -e is needed


def run_stage(stage: Stage, run_dir: Path, ledger: Ledger, python: str, resume: bool, budget_scale: float, status: dict) -> str:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    budget_s = max(stage.budget_min * 60.0 * budget_scale, 1.0)
    cmds = [build_command(stage, c, python, resume, budget_scale) for c in stage.cmds]
    ledger.add({"event": "stage_start", "stage": stage.name, "attempt": ledger.attempts(stage.name) + 1, "budget_min": round(budget_s / 60, 2),
                "cmds": [" ".join(c[3:]) for c in cmds]})
    t0 = time.time()
    pending = list(enumerate(cmds))
    running: list[tuple[int, subprocess.Popen, Path, object]] = []
    rcs: dict[int, int] = {}
    timed_out = False
    last_status = 0.0
    while pending or running:
        while pending and len(running) < max(stage.parallel, 1):
            i, c = pending.pop(0)
            log = logs / f"{stage.name}_{i}.log"
            fh = open(log, "w", encoding="utf-8")        # noqa: SIM115 (closed when the process ends)
            running.append((i, subprocess.Popen(c, stdout=fh, stderr=subprocess.STDOUT, cwd=repo_root()), log, fh))
        time.sleep(1.0)
        for item in list(running):
            i, proc, log, fh = item
            if proc.poll() is not None:
                rcs[i] = proc.returncode
                fh.close()
                running.remove(item)
        if time.time() - t0 > budget_s:
            timed_out = True
            for i, proc, _, fh in running:
                proc.terminate()
            time.sleep(5)
            for i, proc, _, fh in running:
                if proc.poll() is None:
                    proc.kill()
                rcs[i] = proc.wait()
                fh.close()
            running.clear()
            pending.clear()
        if time.time() - last_status > STATUS_EVERY_S:
            last = ""
            for _, _, log, _ in running:
                tl = tail(log, 1)
                last = tl[0][:200] if tl else last
            status.update({"stage": stage.name, "stage_elapsed_s": round(time.time() - t0), "last_log_line": last, "gpu": gpu_status(), "disk_free_gb": disk_free_gb(run_dir), "updated": now()})
            (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            last_status = time.time()
    failed = any(rc != 0 for rc in rcs.values()) and not timed_out
    st = "timeout" if timed_out else ("failed" if failed else "ok")
    event = {"event": "stage_end", "stage": stage.name, "status": st, "rc": [rcs.get(i) for i in range(len(cmds))], "seconds": round(time.time() - t0, 1)}
    if st != "ok":
        event["error_tail"] = {f"{stage.name}_{i}": tail(logs / f"{stage.name}_{i}.log") for i in range(len(cmds)) if rcs.get(i, 0) != 0 or st == "timeout"}
    ledger.add(event)
    return st


def plan_text(profile: Profile, budget_scale: float) -> str:
    total = sum(s.budget_min for s in profile.stages) * budget_scale
    lines = [f"profile {profile.name}: {len(profile.stages)} stages, wall-clock budget {total / 60:.1f} h (estimates; the ledger records real durations)"]
    for s in profile.stages:
        lines.append(f"  {s.name:14s} {s.budget_min * budget_scale:6.0f} min  x{len(s.cmds)} cmd{'s' if len(s.cmds) > 1 else ''}"
                     f"{' (parallel ' + str(s.parallel) + ')' if s.parallel > 1 else ''}  {s.note}")
        for c in s.cmds:
            lines.append("      python -s -m " + " ".join(c))
    return "\n".join(lines)


def run(profile: Profile, run_id: str, only: list[str] | None = None, skip: list[str] | None = None, resume: bool = False, dry_run: bool = False,
        budget_scale: float = 1.0, python: str | None = None, run_dir: Path | None = None) -> int:
    python = python or sys.executable
    run_dir = run_dir or runs_root() / "overnight" / run_id
    stages = [s for s in profile.stages if (not only or s.name in only) and not (skip and s.name in skip)]
    sel = Profile(profile.name, stages)
    print(plan_text(sel, budget_scale), flush=True)
    if dry_run:
        return 0
    ledger = Ledger(run_dir)
    ledger.add({"event": "run_start", "run_id": run_id, "profile": profile.name, "stages": sel.names(), "resume": resume, "python": python})
    done = ledger.finished() if resume else set()
    status = {"run_id": run_id, "profile": profile.name, "started": now()}
    results: dict[str, str] = {}
    for s in stages:
        if s.name in done:
            print(f"[{now()}] skip {s.name} (already finished in this run)", flush=True)
            results[s.name] = "skipped"
            continue
        print(f"[{now()}] stage {s.name} ...", flush=True)
        results[s.name] = run_stage(s, run_dir, ledger, python, resume, budget_scale, status)
        print(f"[{now()}] stage {s.name}: {results[s.name]}", flush=True)
    ledger.add({"event": "run_end", "run_id": run_id, "results": results})
    try:
        from ops.overnight.report import build_report

        path = build_report(run_dir)
        print(f"report: {path}", flush=True)
    except Exception as e:  # noqa: BLE001 — a report problem must not hide the run
        ledger.add({"event": "report_failed", "error": repr(e)})
        print(f"report failed: {e!r}", flush=True)
    (run_dir / "status.json").write_text(json.dumps({**status, "stage": "done", "results": results, "updated": now()}, indent=2), encoding="utf-8")
    return 1 if any(v == "failed" for v in results.values()) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke", choices=["smoke", "real4090", "colab", "macmini", "brain4090"])
    ap.add_argument("--run-id", default=None, help="default: <profile>_<timestamp>; reuse the same id with --resume to continue a run")
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--skip", default=None, help="comma-separated stage names")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply every stage budget (e.g. 0.5 for a shorter night)")
    a = ap.parse_args(argv)
    rid = a.run_id or f"{a.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    return run(get_profile(a.profile), rid, split(a.only), split(a.skip), a.resume, a.dry_run, a.budget_scale)


if __name__ == "__main__":
    raise SystemExit(main())
