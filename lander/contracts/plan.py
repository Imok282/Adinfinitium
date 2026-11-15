"""Mission plan contract: runs/<run>/plan_XXX.json, written by the mission planner (worldmodel/planner) and read by the orchestrator, the Navigator,
Basilisk and the mission-control window. One file = one decision: where, when, which cell, rover goals, return deadline, and WHY.

Honesty rules: `p_success` is a simulated-outcome estimate (MODELLED), never a guarantee; `backup_used` says whether the plan came from the exact Navigator
ranking instead of the learned model; ice goals are NASA permanently-shadowed-region edges (dark and cold), NOT confirmed ice.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from contracts.provenance import Meta

PLAN_SCHEMA_VERSION = 1
DECISIONS = ("go", "wait")


class Factor(BaseModel):
    name: str
    value: float
    note: str = ""


class RankedCandidate(BaseModel):
    region: str
    date: str = Field(description="start of the 709-hour light axis (YYYY-MM-DD)")
    landing_hour: int = Field(ge=0, description="hours after `date` 00:00 UTC")
    cell_rc: tuple[float, float]
    p_success: float = Field(ge=0.0, le=1.0)
    backup_score: float


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PLAN_SCHEMA_VERSION
    decision: str = Field(description="'go' = land on the first-ranked candidate; 'wait' = no candidate clears the confidence bar before the chosen later time")
    region: str
    landing_time_utc: str
    target_cell_rc: tuple[float, float] = Field(description="landing cell, pixels of the 512 x 512 rover world window")
    scatter_radius_m: float = Field(ge=0.0, description="95th percentile touchdown miss distance of the lander pilot at this cell, simulated")
    rover_goals_rc: list[tuple[float, float]] = Field(default_factory=list)
    return_deadline_utc: str
    p_success: float = Field(ge=0.0, le=1.0)
    p_success_backup_rank1: float | None = None
    backup_used: bool
    rationale: list[Factor] = Field(default_factory=list, description="top factors, largest first")
    ranking: list[RankedCandidate] = Field(default_factory=list)
    meta: Meta


def plan_name(idx: int) -> str:
    return f"plan_{idx:03d}.json"


def write_plan(run_dir: str | Path, idx: int, plan: Plan) -> Path:
    p = Path(run_dir) / plan_name(idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return p


def read_plan(path: str | Path) -> Plan:
    return Plan.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
