import json

import pytest

from scripts.build_native_delivery_comparison import digest, endpoint


@pytest.fixture
def closed_endpoint(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    write("protocol.json", {"methods": ["H"], "recovery": {"H": {"lineage_controls": 30720}}})
    write("summary.json", {"status": "complete"})
    write("guard_final.json", {"exit_code": 0, "reason": None})
    write("infrastructure/server_stop.json", {"status": "owned_empty_server_stopped"})
    write("H/summary.json", {"physics_steps": 0})
    write("H/config.json", {})
    write("H/model.zip", {"fake": "test-only bytes"})
    write("H/normalization.pkl", {"fake": "test-only bytes"})
    episodes = [dict(case_id=i, scenario="native_curve", success=i != 23,
        termination_reason="success" if i != 23 else "out_of_bounds", physics_steps=10,
        actions=1, interventions=0, plan_failures=0) for i in range(24)]
    write("H/evaluation.json", dict(episodes=episodes, model_sha256=digest(tmp_path/"H/model.zip"),
        normalization_sha256=digest(tmp_path/"H/normalization.pkl")))
    write("H/endpoint_raw_audit.json", dict(
        evaluation_sha256=digest(tmp_path/"H/evaluation.json"), summary_sha256=digest(tmp_path/"H/summary.json"),
        config_sha256=digest(tmp_path/"H/config.json"), protocol_sha256=digest(tmp_path/"protocol.json"),
        evaluation_control_frames_verified=240))
    write("strict_delivery.json", {"strict_delivery_gate_passed": True, "counts": {"controls": 240}})
    return tmp_path


def test_all_cases_and_failure_are_retained(closed_endpoint):
    row = endpoint(closed_endpoint, 11, "H")
    assert row["successes"] == 23 and row["unsafe"] == 1
    assert len(row["cases"]) == 24 and row["controls"] == 240


def test_changed_model_cannot_reuse_endpoint_score(closed_endpoint):
    (closed_endpoint/"H/model.zip").write_bytes(b"different policy")
    with pytest.raises(ValueError, match="model changed"):
        endpoint(closed_endpoint, 11, "H")


def test_uncleaned_server_is_not_closed_evidence(closed_endpoint):
    (closed_endpoint/"infrastructure/server_stop.json").write_text('{"status":"still_running"}')
    with pytest.raises(ValueError, match="cleanup"):
        endpoint(closed_endpoint, 11, "H")


def test_engine_count_must_cover_every_control(closed_endpoint):
    (closed_endpoint/"strict_delivery.json").write_text('{"strict_delivery_gate_passed":true,"counts":{"controls":239}}')
    with pytest.raises(ValueError, match="accounting"):
        endpoint(closed_endpoint, 11, "H")
