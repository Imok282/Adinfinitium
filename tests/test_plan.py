import pytest
from pydantic import ValidationError

from contracts.plan import Plan, read_plan, write_plan
from contracts.provenance import Meta, Provenance


def _plan(**kw) -> Plan:
    base = {"decision": "go", "region": "lola5m_site23", "landing_time_utc": "2027-03-04T10:00:00Z", "target_cell_rc": (256.0, 250.0), "scatter_radius_m": 22.0,
                "rover_goals_rc": [(200.0, 300.0)], "return_deadline_utc": "2027-03-14T10:00:00Z", "p_success": 0.61, "backup_used": False,
                "meta": Meta(provenance=Provenance.MODELLED, source="simulated outcomes", created_by="test")}
    base.update(kw)
    return Plan(**base)


def test_roundtrip(tmp_path):
    p = write_plan(tmp_path, 3, _plan())
    assert p.name == "plan_003.json"
    assert read_plan(p) == _plan(meta=read_plan(p).meta)


def test_rejects_probability_out_of_range_and_unknown_fields():
    with pytest.raises(ValidationError):
        _plan(p_success=1.4)
    with pytest.raises(ValidationError):
        _plan(surprise="x")
