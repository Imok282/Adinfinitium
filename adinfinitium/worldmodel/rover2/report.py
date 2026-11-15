"""Rover v2 report: v1, v2 and the planners side by side on the held-out worlds, the Site11 selection record, and the pre-registered verdict.

    python -m worldmodel.rover2.report --main-run-dir ~/adinfinitium-run/adinfinitium      # reads <ADI_RUNS>/eval.json, selection.json, train_log.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contracts.paths import model_path, runs_root
from worldmodel.hazard.lola import HOLDOUT_SITES

PLANNERS = ("naive", "cautious", "cautious_wait_light")
MARGIN, DIED_SLACK = 0.01, 0.02


def verdict(summary: dict) -> tuple[str, str]:
    """The rule written in ROVER_V2_PLAN.md before any v2 number existed. `summary` = eval.json's summary (planners + model)."""
    if "model" not in summary or not any(p in summary for p in PLANNERS):
        return "NO RESULT", "the evaluation has no model row or no planner row"
    best = max((p for p in PLANNERS if p in summary), key=lambda p: summary[p]["returned"])
    m, b = summary["model"], summary[best]
    why = (f"model returned {m['returned']:.3f}, died {m['died']:.3f}; best planner by returned = {best}: returned {b['returned']:.3f}, died {b['died']:.3f}")
    if m["returned"] > b["returned"] + MARGIN and m["died"] <= b["died"]:
        return "SUCCESS (beats the best planner)", why
    if m["returned"] >= b["returned"] - MARGIN and m["died"] <= b["died"] + DIED_SLACK:
        return "MATCH (as good as the best planner)", why
    if m["returned"] >= 0.5 * b["returned"]:
        return "PARTIAL (does the mission, worse than the best planner)", why
    return "FAILURE", why


def _load(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _row(name: str, s: dict | None) -> str:
    if not s:
        return f"| {name} | not available | | | |"
    return f"| {name} | {s['reached_ice']:.3f} | {s['returned']:.3f} | {s['died']:.3f} | {s['mean_return']:+.1f} |"


def _sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(v2: dict | None, v1: dict | None, sel: dict | None, log_rows: list[dict], name: str, notes: list[str]) -> str:
    out = ["# Rover v2 report", "", "Rover v1 (PPO from scratch, in the main frozen run) failed and stays reported as failed. Rover v2 is the same network trained AFTER the freeze "
           "by copying the physics planners (DAgger); it is not part of the frozen main run. Method, metric, selection rule and verdict rule were written in ROVER_V2_PLAN.md before any v2 number existed "
           "(one amendment to the training examples and one list of clarifications were added before any v2 held-out number; both are in that file). "
           "The v1 held-out numbers had been seen when v2 was designed; no test site was used to train, select or decide anything in v2.", ""]
    out += [f"- NOTE: {n}" for n in notes] + ([""] if notes else [])
    s2 = (v2 or {}).get("summary", {})
    s1 = (v1 or {}).get("summary", {})
    sites = (v2 or {}).get("test_sites")
    if v2 and sorted(sites or []) != sorted(HOLDOUT_SITES):
        out += [f"## Evaluation on {sites}: NOT the held-out test sites {HOLDOUT_SITES} (a smoke run; the table below is not a result)", ""]
    else:
        out += [f"## Held-out test sites ({' + '.join(HOLDOUT_SITES)}), evaluated by the main run's evaluator", ""]
    if v2:
        out += [f"Worlds finished: {len(v2.get('worlds', {}))} of {v2.get('windows_planned')} (complete: {v2.get('complete')}); {v2.get('rovers_per_world')} rovers per world; "
                f"hazard map: {v2.get('hazard_source')}; v2 checkpoint id {v2.get('checkpoint_id')}, {v2.get('parameters')} parameters.", ""]
    out += ["| driver | reached ice | reached ice and returned (primary) | died | mean return |", "|---|---|---|---|---|",
            _row("rover v1 network (main run, failed)", s1.get("model")), _row("rover v2 network (alone, greedy, no planner fallback)", s2.get("model"))]
    out += [_row(f"planner: {p}" + ("" if p in s2 else " (from the MAIN run's evaluation; the v2 evaluation has no such row)"), s2.get(p) or s1.get(p)) for p in PLANNERS]
    done = bool(v2 and v2.get("complete"))
    if v1 and v2:
        diffs = [abs(s1[p][k] - s2[p][k]) for p in PLANNERS if p in s1 and p in s2 for k in ("reached_ice", "returned", "died")]
        if not done:
            out += ["", "Same worlds and starts as the main run's evaluation: cannot be checked, the v2 evaluation is incomplete."]
        elif diffs and len(diffs) == 9:
            out += ["", f"Same worlds and starts as the main run's evaluation: the planners are deterministic, and their rows differ from the main run's by at most "
                    f"{max(diffs):.4f} ({'the same evaluation' if max(diffs) < 1e-6 else 'a few rovers differ, e.g. GPU rounding' if max(diffs) < 0.005 else 'NOT the same worlds / starts / hazard map: v1 and v2 rows are not comparable'})."]
        else:
            out += ["", "Same worlds and starts as the main run's evaluation: not checked (planner rows missing on one side)."]
    if done:
        v, why = verdict(s2)
    elif v2:
        _, why = verdict(s2)
        v = f"NO RESULT (held-out evaluation incomplete: {len(v2.get('worlds', {}))} of {v2.get('windows_planned')} worlds)"
        why += "; these are provisional numbers over the finished worlds only; the pre-registered metric is the mean over all planned worlds"
    else:
        v, why = "NO RESULT", "no held-out evaluation file"
    out += ["", f"**Pre-registered verdict: {v}.** {why}.", ""]
    if v2:
        out += ["Per world (v2 network | best-returned planner in that world):", "", "| world | v2 ice | v2 returned | v2 died | planner | ice | returned | died |", "|---|---|---|---|---|---|---|---|"]
        for k, w in v2.get("worlds", {}).items():
            if "model" not in w or not any(p in w for p in PLANNERS):
                continue
            bp = max((p for p in PLANNERS if p in w), key=lambda p: w[p]["returned"])
            out.append(f"| {k} | {w['model']['reached_ice']:.3f} | {w['model']['returned']:.3f} | {w['model']['died']:.3f} | {bp} | {w[bp]['reached_ice']:.3f} | {w[bp]['returned']:.3f} | {w[bp]['died']:.3f} |")
        out.append("")
    out += ["## Checkpoint selection on the validation site (Site11 only; not a held-out score)", ""]
    if sel:
        out += [f"Rule: {sel.get('rule')}. Worlds: {', '.join(sel.get('validation_worlds', []))}; {sel.get('rovers_per_world')} rovers each, seed {sel.get('seed')}.", "",
                "| candidate | tag | round | training steps | minute | returned | reached ice | died | selected at that time |", "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {c['candidate']} | {c['tag']} | {c['round']} | {c['steps']} | {c['minutes']:.0f} | {c['returned']:.3f} | {c['reached_ice']:.3f} | {c['died']:.3f} | {'yes' if c['selected'] else 'no'} |"
                for c in sel.get("candidates", [])]
        out += [f"| planner {n} (same starts, reference) | | | | | {r['returned']:.3f} | {r['reached_ice']:.3f} | {r['died']:.3f} | |"
                for n, r in (sel.get("planners_on_the_same_starts") or {}).items()]
        b = sel.get("best")
        out += ["", f"Final choice: candidate {b['candidate']} (round {b['round']}, {b['steps']} training steps)." if b else "No candidate was recorded.", ""]
    else:
        out += ["No selection record found.", ""]
    out += ["## Training (train sites only)", ""]
    if log_rows:
        last = log_rows[-1]
        tail = log_rows[-min(len(log_rows), 32):]
        mean = lambda k: sum(r[k] for r in tail) / len(tail)                                                 # noqa: E731
        dag = [r for r in tail if r.get("net_rovers")]
        out += [f"{last['round']} data-collection episodes over {len({r['world'] for r in log_rows})} different train worlds, {last['steps']} training steps, "
                f"{last['minutes']:.0f} minutes. Over the last {len(tail)} episodes: "
                f"teacher (best planner variant per rover, chosen by simulation) returned {mean('teacher_returned'):.3f}, reached ice {mean('teacher_reached_ice'):.3f}."]
        if dag:
            out.append(f"Network-driven training rovers in those episodes (5% random actions, train sites, not a score): returned "
                       f"{sum(r['net_returned'] for r in dag) / len(dag):.3f}, agreed with the teacher on {sum(r['net_agrees_with_teacher'] for r in dag) / len(dag):.1%} of steps.")
        share = {k: sum(r["variant_share"][k] for r in tail) / len(tail) for k in last["variant_share"]}
        out += ["Teacher variant shares: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()) + ".", ""]
    else:
        out += ["No training log found.", ""]
    ck = model_path(name)
    out += ["## Files", "", f"- driver: `{ck}` sha256 `{_sha256(ck)}` (trained after the freeze; not in FROZEN.json)", "- `eval.json` (held-out), `selection.json` (Site11), "
            "`train_log.jsonl`, `train.log`, `eval.log` in this folder", "",
            "Honest limits: simulation only (kinematic, hourly steps); PSR = dark and cold, not confirmed ice; v2 copies planners, it cannot be expected to do what no planner does; "
            "one validation site.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-run-dir", default=None, help="main run folder; its runs/rover/eval.json gives the v1 row (read only)")
    ap.add_argument("--name", default="rover_driver_v2")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args(argv)
    out = runs_root()
    v1 = _load(Path(a.main_run_dir).expanduser() / "runs" / "rover" / "eval.json") if a.main_run_dir else None
    rows = []
    if (out / "train_log.jsonl").exists():
        rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    text = build(_load(out / "eval.json"), v1, _load(out / "selection.json"), rows, a.name, a.note)
    (out / "ROVER_V2_REPORT.md").write_text(text, encoding="utf-8")
    print(next((line.replace("**", "") for line in text.splitlines() if line.startswith("**Pre-registered verdict")), "no verdict"))
    print(f"wrote {out / 'ROVER_V2_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
