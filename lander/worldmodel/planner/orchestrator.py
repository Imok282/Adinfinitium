"""Orchestrator (L3): a plain state machine that turns a plan into the time-stamped decision log the mission-control window shows
("no commentary, just decisions"): orbit -> descent -> touchdown -> surface (rover out) -> return by deadline -> night -> sunrise.
Times come from the plan and the real light series at the landing cell (SPICE + skyline); every row says why."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from contracts.plan import Plan
from worldmodel.rover.world import RoverWorld

DESCENT_MIN = 12          # assumption: powered descent from the 100 km orbit release to touchdown, minutes (Basilisk scenario order of magnitude)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def decision_log(plan: Plan, w: RoverWorld, stride: int) -> list[dict]:
    t_land, deadline = _t(plan.landing_time_utc), _t(plan.return_deadline_utc)
    t0 = datetime.fromisoformat(w.times_utc[0]) if isinstance(w.times_utc[0], str) else t_land
    r, c = plan.target_cell_rc
    ci, cj = int(min(r // stride, w.sun_frac.shape[1] - 1)), int(min(c // stride, w.sun_frac.shape[2] - 1))
    sun = w.sun_frac[:, ci, cj].astype(np.float32)
    h_land = round((t_land - t0).total_seconds() / 3600)
    rows = [{"t_utc": _fmt(t_land - timedelta(minutes=DESCENT_MIN + 30)), "phase": "orbit", "decision": plan.decision.upper(),
             "reason": f"planner p={plan.p_success:.2f}{' (Navigator backup)' if plan.backup_used else ''}; cell {plan.target_cell_rc}"}]
    if plan.decision == "wait":
        rows.append({"t_utc": rows[0]["t_utc"], "phase": "orbit", "decision": "HOLD", "reason": "no slot cleared the confidence bar; keep orbiting"})
        return rows
    rows += [{"t_utc": _fmt(t_land - timedelta(minutes=DESCENT_MIN)), "phase": "descent", "decision": "DEORBIT_AND_DESCEND", "reason": "landing slot reached"},
             {"t_utc": _fmt(t_land), "phase": "touchdown", "decision": "TOUCHDOWN_CHECK", "reason": f"expected scatter {plan.scatter_radius_m:.0f} m (95%)"},
             {"t_utc": _fmt(t_land + timedelta(hours=1)), "phase": "surface", "decision": "DEPLOY_ROVER", "reason": f"{len(plan.rover_goals_rc)} ice-edge goals (PSR, not confirmed ice)"},
             {"t_utc": _fmt(deadline), "phase": "surface", "decision": "ROVER_RETURN_DEADLINE", "reason": "24 h before the first local night at the lander"}]
    dark = np.nonzero(sun[h_land:] < 0.05)[0]
    if len(dark):
        tn = t_land + timedelta(hours=int(dark[0]))
        rows.append({"t_utc": _fmt(tn), "phase": "night", "decision": "SURVIVAL_MODE", "reason": "Sun below the local skyline (SPICE + skyline)"})
        lit = np.nonzero(sun[h_land + int(dark[0]):] > 0.5)[0]
        if len(lit):
            rows.append({"t_utc": _fmt(tn + timedelta(hours=int(lit[0]))), "phase": "sunrise", "decision": "WAKE_AND_RECHARGE", "reason": "Sun back above the skyline"})
    return rows


def write_decisions(rows: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["t_utc", "phase", "decision", "reason"])
        wr.writeheader()
        wr.writerows(rows)
    return p
