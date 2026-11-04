import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from publication_report import metrics, comparison
from batch_status import alive
import os
import time
import pytest


def test_no_rpc_observation_is_not_a_policy_failure_or_zero_collision():
    result = metrics([{"status":"infrastructure_unresolved", "success":False}])
    assert result["completed_cases"] == 0
    assert result["infrastructure_unresolved"] == 1
    assert result["safe_completion"] is None
    assert result["collision_rate"] is None
    assert result["physical_step_intervention_rate"] is None
    assert result["scheduled_completion_lower_bound"] == 0
    assert comparison([{"name":"H","seed":44,"metrics":result}], "H", "B1") == {}


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation time check")
def test_live_pid_cannot_match_a_heartbeat_older_than_its_creation():
    assert alive(os.getpid(), time.time()) is True
    assert alive(os.getpid(), 0) is False
