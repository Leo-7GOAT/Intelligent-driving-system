"""Fault injection for recorded evidence, not a substitute for live CARLA."""
from copy import deepcopy
import pytest
from scripts.analyze_native_road_learning import verify_trace


def records():
    trace, actual = [], {}
    for frame in (101, 102):
        state = {"x": frame/100, "y": 2., "time": (frame-100)*.05}
        trace.append({"state": state, "backend_evidence": {"episode_id": "e1", "frame": frame, "imu_frame": frame}})
        actual[("e1", frame)] = {"state": deepcopy(state), "imu_frame": frame}
    return trace, actual


def test_evidence_rejects_altered_states_duplicate_and_missing_frames():
    trace, actual = records()
    used, previous = set(), {}
    verify_trace(trace, actual, used, previous)
    assert len(used) == 2
    with pytest.raises(AssertionError):
        verify_trace(trace, actual, used, previous)
    altered = deepcopy(trace)
    altered[1]["state"]["x"] += .01
    with pytest.raises(AssertionError):
        verify_trace(altered, actual, set(), {})
    gap = deepcopy(trace)
    gap[1]["backend_evidence"].update(frame=103, imu_frame=103)
    raw = deepcopy(actual)
    raw[("e1", 103)] = {"state": gap[1]["state"], "imu_frame": 103}
    with pytest.raises(AssertionError):
        verify_trace(gap, raw, set(), {})


def test_evidence_rejects_stale_imu_frame():
    trace, actual = records()
    trace[1]["backend_evidence"]["imu_frame"] = 101
    with pytest.raises(AssertionError):
        verify_trace(trace, actual, set(), {})
