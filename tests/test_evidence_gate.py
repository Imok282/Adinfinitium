from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile
from worldmodel.planner.evidence_gate import GatePolicy, GateStatus, screen_candidate

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
POLICY = GatePolicy(
    max_source_age_s=24 * 60 * 60,
    max_meters_per_pixel=5.0,
    min_measured_fraction=1.0,
    max_height_uncertainty_m=0.5,
    min_hazard_confidence=0.9,
    max_hazard_probability=0.5,
    max_slope_deg=12.0,
    footprint_radius_m=1.0,
    approved_calibration_ids=("test-calibration-v1",),
)


def make_tile(*, provenance=Provenance.OBSERVED, mpp=1.0, source_time="2026-11-17T00:00:00Z"):
    shape = (9, 9)
    extra = {}
    if source_time is not None:
        extra["source_acquired_at_utc"] = source_time
    return Tile(
        region="synthetic-gate-fixture",
        meters_per_pixel=mpp,
        height=np.zeros(shape, np.float32),
        slope=np.zeros(shape, np.float32),
        hazard=np.zeros(shape, np.float32),
        uncertainty=np.full(shape, 0.1, np.float32),
        psr_mask=np.zeros(shape, bool),
        real_mask=np.ones(shape, bool),
        meta=Meta(provenance=provenance, source="synthetic unit-test fixture", created_at=NOW.isoformat()),
        extra=extra,
    )


def screen(tile, **kwargs):
    return screen_candidate(
        tile,
        (4, 4),
        target_time_utc=NOW,
        policy=POLICY,
        hazard_confidence=0.99,
        calibration_id="test-calibration-v1",
        **kwargs,
    )


def test_fresh_supported_candidate_is_review_only_never_go():
    result = screen(make_tile())
    assert result.status is GateStatus.REVIEW_ONLY
    assert result.status.value != "GO"
    assert "evidence_thresholds_passed_human_review_required" in result.reasons


def test_missing_source_time_does_not_fall_back_to_recent_meta_created_at():
    tile = make_tile(source_time=None)
    assert tile.meta.created_at == NOW.isoformat()
    result = screen(tile)
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "source_acquisition_time_unknown" in result.reasons
    assert result.evidence["meta_created_at_is_source_time"] is False


def test_stale_source_fails_closed():
    result = screen(make_tile(source_time="2026-11-15T00:00:00Z"))
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "terrain_source_stale" in result.reasons


def test_coarse_source_fails_closed():
    result = screen(make_tile(mpp=10.0))
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "terrain_resolution_too_coarse" in result.reasons


def test_text_policy_threshold_fails_closed():
    policy = replace(POLICY, max_meters_per_pixel="5")
    result = screen_candidate(
        make_tile(),
        (4, 4),
        target_time_utc=NOW,
        policy=policy,
        hazard_confidence=0.99,
        calibration_id="test-calibration-v1",
    )
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "invalid_policy_max_meters_per_pixel" in result.reasons


def test_low_confidence_fails_but_exact_floor_is_eligible_for_review():
    low = screen_candidate(make_tile(), (4, 4), target_time_utc=NOW, policy=POLICY,
                           hazard_confidence=0.899, calibration_id="test-calibration-v1")
    at_floor = screen_candidate(make_tile(), (4, 4), target_time_utc=NOW, policy=POLICY,
                                hazard_confidence=0.9, calibration_id="test-calibration-v1")
    assert low.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "hazard_confidence_below_minimum" in low.reasons
    assert at_floor.status is GateStatus.REVIEW_ONLY


def test_missing_or_unapproved_calibration_fails_closed():
    missing = screen_candidate(
        make_tile(), (4, 4), target_time_utc=NOW, policy=POLICY, hazard_confidence=0.99
    )
    unknown = screen_candidate(make_tile(), (4, 4), target_time_utc=NOW, policy=POLICY,
                               hazard_confidence=0.99, calibration_id="not-approved")
    assert missing.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert unknown.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "hazard_calibration_unapproved" in missing.reasons


def test_sparse_direct_measurements_and_high_uncertainty_fail_closed():
    sparse = make_tile()
    sparse.real_mask[3:6, 3:6] = False
    sparse_result = screen(sparse)
    assert sparse_result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "local_measured_coverage_below_minimum" in sparse_result.reasons

    uncertain = make_tile()
    uncertain.uncertainty[3, 3] = 1.0
    uncertain_result = screen(uncertain)
    assert uncertain_result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "height_uncertainty_above_maximum" in uncertain_result.reasons


def test_generated_only_input_never_passes_evidence_gate():
    result = screen(make_tile(provenance=Provenance.GENERATED))
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "terrain_provenance_not_usable_for_safety_screen" in result.reasons


@pytest.mark.parametrize("unsafe_rc", [(4, 4), (3, 3)])
def test_false_safe_model_probability_cannot_override_unsafe_slope(unsafe_rc):
    tile = make_tile()
    tile.slope[unsafe_rc] = 18.0
    model_says_safe = np.zeros(tile.shape, np.float32)
    result = screen_candidate(tile, (4, 4), target_time_utc=NOW, policy=POLICY,
                              hazard_probability_grid=model_says_safe, hazard_confidence=0.99,
                              calibration_id="test-calibration-v1")
    assert float(model_says_safe[unsafe_rc]) == 0.0
    assert result.status is GateStatus.NO_GO
    assert "slope_limit_exceeded" in result.reasons
    assert result.evidence["max_hazard_probability"] == 0.0


def test_hazard_threshold_is_inclusive_and_out_of_bounds_is_not_clipped_safe():
    tile = make_tile()
    p = np.zeros(tile.shape, np.float32)
    p[4, 4] = 0.5
    unsafe = screen_candidate(tile, (4, 4), target_time_utc=NOW, policy=POLICY,
                              hazard_probability_grid=p, hazard_confidence=0.99,
                              calibration_id="test-calibration-v1")
    edge = screen_candidate(tile, (0, 0), target_time_utc=NOW, policy=POLICY,
                            hazard_confidence=0.99, calibration_id="test-calibration-v1")
    assert unsafe.status is GateStatus.NO_GO
    assert "hazard_probability_limit_reached" in unsafe.reasons
    assert edge.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "candidate_footprint_out_of_bounds" in edge.reasons


def test_missing_policy_and_invalid_probability_grid_fail_closed():
    tile = make_tile()
    missing_policy = screen_candidate(tile, (4, 4), target_time_utc=NOW, policy=None)
    invalid_grid = np.zeros(tile.shape, np.float32)
    invalid_grid[4, 4] = np.nan
    invalid = screen_candidate(tile, (4, 4), target_time_utc=NOW, policy=POLICY,
                               hazard_probability_grid=invalid_grid, hazard_confidence=0.99,
                               calibration_id="test-calibration-v1")
    assert missing_policy.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "gate_policy_missing" in missing_policy.reasons
    assert invalid.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "hazard_probability_grid_invalid" in invalid.reasons

    bad_policy = screen_candidate(
        tile,
        (4, 4),
        target_time_utc=NOW,
        policy=replace(POLICY, max_source_age_s=float("nan")),
    )
    assert bad_policy.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "invalid_policy_max_source_age_s" in bad_policy.reasons


@pytest.mark.parametrize("candidate_rc", [(4,), (4, 4, 0)])
def test_candidate_coordinate_must_have_exactly_two_values(candidate_rc):
    result = screen_candidate(
        make_tile(),
        candidate_rc,
        target_time_utc=NOW,
        policy=POLICY,
        hazard_confidence=0.99,
        calibration_id="test-calibration-v1",
    )
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "candidate_cell_invalid" in result.reasons
    assert result.evidence["target_rc"] is None


def test_nonnumeric_hazard_grid_fails_closed():
    tile = make_tile()
    invalid_grid = np.full(tile.shape, "not-a-probability", dtype=object)
    result = screen_candidate(
        tile,
        (4, 4),
        target_time_utc=NOW,
        policy=POLICY,
        hazard_probability_grid=invalid_grid,
        hazard_confidence=0.99,
        calibration_id="test-calibration-v1",
    )
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "hazard_probability_grid_invalid" in result.reasons


def test_boolean_hazard_confidence_fails_closed():
    result = screen_candidate(
        make_tile(),
        (4, 4),
        target_time_utc=NOW,
        policy=POLICY,
        hazard_confidence=True,
        calibration_id="test-calibration-v1",
    )
    assert result.status is GateStatus.INSUFFICIENT_EVIDENCE
    assert "hazard_confidence_unavailable_or_invalid" in result.reasons


def test_gate_is_deterministic_and_does_not_mutate_tile():
    tile = make_tile()
    before = [a.copy() for a in (tile.height, tile.slope, tile.hazard, tile.uncertainty, tile.real_mask)]
    first = screen(tile).to_dict()
    second = screen(tile).to_dict()
    assert first == second
    arrays = (tile.height, tile.slope, tile.hazard, tile.uncertainty, tile.real_mask)
    for actual, expected in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
