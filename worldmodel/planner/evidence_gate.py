"""Fail-closed evidence screen for a candidate landing cell.

This is a research-prototype advisory gate, not a flight controller or a
landing-site certification. It returns REVIEW_ONLY at best; it never returns GO.
It is standalone and is not yet called by the existing plan-generation command.
Source freshness must come from `tile.extra['source_acquired_at_utc']` (or the
explicit argument), never Meta.created_at, which records file processing time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import math
from typing import Any

import numpy as np

from contracts.provenance import Provenance
from contracts.tiles import Tile


class GateStatus(StrEnum):
    NO_GO = "NO_GO"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REVIEW_ONLY = "REVIEW_ONLY"


@dataclass(frozen=True)
class GatePolicy:
    """All thresholds must be supplied by a mission owner; no flight values are assumed here."""

    max_source_age_s: float
    max_meters_per_pixel: float
    min_measured_fraction: float
    max_height_uncertainty_m: float
    min_hazard_confidence: float
    max_hazard_probability: float
    max_slope_deg: float
    footprint_radius_m: float
    approved_calibration_ids: tuple[str, ...]


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    reasons: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reasons": list(self.reasons), "evidence": self.evidence}


def _utc(value: datetime | str | None) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _policy_errors(policy: GatePolicy | None) -> list[str]:
    if policy is None:
        return ["gate_policy_missing"]
    if not isinstance(policy, GatePolicy):
        return ["gate_policy_invalid"]
    numeric = {
        "max_source_age_s": policy.max_source_age_s,
        "max_meters_per_pixel": policy.max_meters_per_pixel,
        "min_measured_fraction": policy.min_measured_fraction,
        "max_height_uncertainty_m": policy.max_height_uncertainty_m,
        "min_hazard_confidence": policy.min_hazard_confidence,
        "max_hazard_probability": policy.max_hazard_probability,
        "max_slope_deg": policy.max_slope_deg,
        "footprint_radius_m": policy.footprint_radius_m,
    }
    errors = []
    values: dict[str, float] = {}
    for name, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            errors.append(f"invalid_policy_{name}")
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            parsed = math.nan
        if not math.isfinite(parsed):
            errors.append(f"invalid_policy_{name}")
        else:
            values[name] = parsed
    if errors:
        return sorted(set(errors))
    if values["max_source_age_s"] < 0:
        errors.append("invalid_policy_max_source_age_s")
    if values["max_meters_per_pixel"] <= 0:
        errors.append("invalid_policy_max_meters_per_pixel")
    if not 0.0 <= values["min_measured_fraction"] <= 1.0:
        errors.append("invalid_policy_min_measured_fraction")
    if values["max_height_uncertainty_m"] < 0:
        errors.append("invalid_policy_max_height_uncertainty_m")
    if not 0.0 <= values["min_hazard_confidence"] <= 1.0:
        errors.append("invalid_policy_min_hazard_confidence")
    if not 0.0 <= values["max_hazard_probability"] <= 1.0:
        errors.append("invalid_policy_max_hazard_probability")
    if not 0.0 < values["max_slope_deg"] < 90.0:
        errors.append("invalid_policy_max_slope_deg")
    if values["footprint_radius_m"] < 0:
        errors.append("invalid_policy_footprint_radius_m")
    calibration_ids = policy.approved_calibration_ids
    if not isinstance(calibration_ids, (tuple, list)) or not calibration_ids or any(
        not isinstance(x, str) or not x.strip() for x in calibration_ids
    ):
        errors.append("approved_calibration_ids_missing")
    return sorted(set(errors))


def screen_candidate(
    tile: Tile,
    candidate_rc: tuple[int, int],
    *,
    target_time_utc: datetime | str,
    policy: GatePolicy | None,
    hazard_probability_grid: np.ndarray | None = None,
    hazard_confidence: float | None = None,
    calibration_id: str | None = None,
    source_acquired_at_utc: datetime | str | None = None,
) -> GateResult:
    """Screen a map-space footprint; any hard hazard is NO_GO, missing evidence is INSUFFICIENT_EVIDENCE.

    REVIEW_ONLY means the configured evidence checks passed and a human/mission
    review may continue. It is deliberately not an automated GO decision.
    """
    if not isinstance(tile, Tile):
        return GateResult(GateStatus.INSUFFICIENT_EVIDENCE, ("tile_invalid",), {})
    try:
        tile.validate()
        provenance = tile.meta.provenance
        mpp = float(tile.meters_per_pixel)
        if not math.isfinite(mpp) or mpp <= 0 or not isinstance(provenance, Provenance):
            raise ValueError("invalid tile metadata")
    except (ValueError, TypeError, AttributeError):
        return GateResult(GateStatus.INSUFFICIENT_EVIDENCE, ("tile_invalid",), {})
    reasons = _policy_errors(policy)
    try:
        rc_evidence = list(candidate_rc) if len(candidate_rc) == 2 else None
    except (TypeError, AttributeError):
        rc_evidence = None
    evidence: dict[str, Any] = {
        "region": tile.region,
        "provenance": provenance.value,
        "meters_per_pixel": float(tile.meters_per_pixel),
        "target_rc": rc_evidence,
        "meta_created_at_is_source_time": False,
    }
    if reasons:
        return GateResult(GateStatus.INSUFFICIENT_EVIDENCE, tuple(reasons), evidence)
    assert policy is not None

    if tile.meta.provenance in (Provenance.UNKNOWN, Provenance.GENERATED):
        reasons.append("terrain_provenance_not_usable_for_safety_screen")
    if tile.meters_per_pixel > policy.max_meters_per_pixel:
        reasons.append("terrain_resolution_too_coarse")

    try:
        if rc_evidence is None:
            raise ValueError("candidate must be exactly (row, col)")
        row_f, col_f = float(candidate_rc[0]), float(candidate_rc[1])
        if not (math.isfinite(row_f) and math.isfinite(col_f)) or not (
            row_f.is_integer() and col_f.is_integer()
        ):
            raise ValueError
        row, col = int(row_f), int(col_f)
    except (TypeError, ValueError, IndexError):
        return GateResult(
            GateStatus.INSUFFICIENT_EVIDENCE,
            tuple(sorted(set(reasons + ["candidate_cell_invalid"]))),
            evidence,
        )

    radius_px = int(math.ceil(policy.footprint_radius_m / tile.meters_per_pixel))
    r0, r1 = row - radius_px, row + radius_px + 1
    c0, c1 = col - radius_px, col + radius_px + 1
    h, w = tile.shape
    evidence["footprint_radius_px"] = radius_px
    if row < 0 or col < 0 or row >= h or col >= w or r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return GateResult(
            GateStatus.INSUFFICIENT_EVIDENCE,
            tuple(sorted(set(reasons + ["candidate_footprint_out_of_bounds"]))),
            evidence,
        )

    patch = np.s_[r0:r1, c0:c1]
    slope_patch = np.asarray(tile.slope[patch], dtype=np.float64)
    hazard_grid = tile.hazard if hazard_probability_grid is None else np.asarray(hazard_probability_grid)
    try:
        grid_ok = (
            np.issubdtype(hazard_grid.dtype, np.number)
            and not np.issubdtype(hazard_grid.dtype, np.bool_)
            and hazard_grid.shape == tile.shape
            and bool(np.isfinite(hazard_grid).all())
            and not bool(np.any((hazard_grid < 0) | (hazard_grid > 1)))
        )
    except (TypeError, ValueError):
        grid_ok = False
    if not grid_ok:
        reasons.append("hazard_probability_grid_invalid")
        hazard_patch = None
    else:
        hazard_patch = np.asarray(hazard_grid[patch], dtype=np.float64)

    if tile.real_mask is None:
        measured_fraction = None
        reasons.append("direct_measurement_mask_missing")
    else:
        measured_fraction = float(np.asarray(tile.real_mask[patch], dtype=bool).mean())
        if measured_fraction < policy.min_measured_fraction:
            reasons.append("local_measured_coverage_below_minimum")
    uncertainty_patch = np.asarray(tile.uncertainty[patch], dtype=np.float64)
    max_uncertainty = float(np.max(uncertainty_patch))
    if np.any(uncertainty_patch < 0):
        reasons.append("height_uncertainty_invalid")
    elif max_uncertainty > policy.max_height_uncertainty_m:
        reasons.append("height_uncertainty_above_maximum")

    source_value = source_acquired_at_utc
    if source_value is None:
        source_value = tile.extra.get("source_acquired_at_utc")
    source_time = _utc(source_value)
    target_time = _utc(target_time_utc)
    if target_time is None:
        reasons.append("target_time_missing_or_invalid")
        source_age_s = None
    elif source_time is None:
        # Deliberately do not fall back to Meta.created_at.
        reasons.append("source_acquisition_time_unknown")
        source_age_s = None
    else:
        source_age_s = (target_time - source_time).total_seconds()
        if source_age_s < 0:
            reasons.append("source_timestamp_after_target_time")
        elif source_age_s > policy.max_source_age_s:
            reasons.append("terrain_source_stale")

    confidence = hazard_confidence if hazard_confidence is not None else tile.extra.get("hazard_confidence")
    try:
        if isinstance(confidence, (bool, np.bool_)):
            raise TypeError("boolean is not a confidence")
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is None or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        reasons.append("hazard_confidence_unavailable_or_invalid")
    elif confidence < policy.min_hazard_confidence:
        reasons.append("hazard_confidence_below_minimum")

    calibration = calibration_id if calibration_id is not None else tile.extra.get("hazard_calibration_id")
    if not calibration or calibration not in policy.approved_calibration_ids:
        reasons.append("hazard_calibration_unapproved")

    max_slope = float(np.max(slope_patch))
    max_probability = None if hazard_patch is None else float(np.max(hazard_patch))
    evidence.update({
        "source_acquired_at_utc": source_time.isoformat() if source_time else None,
        "target_time_utc": target_time.isoformat() if target_time else None,
        "source_age_s": source_age_s,
        "local_measured_fraction": measured_fraction,
        "max_height_uncertainty_m": max_uncertainty,
        "max_slope_deg": max_slope,
        "max_hazard_probability": max_probability,
        "hazard_confidence": confidence,
        "hazard_calibration_id": calibration,
    })

    hard_hazards = []
    if max_slope >= policy.max_slope_deg:
        hard_hazards.append("slope_limit_exceeded")
    if max_probability is not None and max_probability >= policy.max_hazard_probability:
        hard_hazards.append("hazard_probability_limit_reached")
    if hard_hazards:
        return GateResult(GateStatus.NO_GO, tuple(sorted(set(reasons + hard_hazards))), evidence)
    if reasons:
        return GateResult(GateStatus.INSUFFICIENT_EVIDENCE, tuple(sorted(set(reasons))), evidence)
    return GateResult(GateStatus.REVIEW_ONLY, ("evidence_thresholds_passed_human_review_required",), evidence)


__all__ = ["GatePolicy", "GateResult", "GateStatus", "screen_candidate"]
