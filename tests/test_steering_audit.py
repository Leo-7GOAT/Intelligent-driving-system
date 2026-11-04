"""Analysis math only; synthetic rows do not validate the live CARLA sweep."""
import pytest
from scripts.carla_steering_audit import summarize_phase


def test_signed_mapping_and_registered_measurement_window():
    rows = [{"step": step, "phase": {"normalized_steer": .25},
             "measurement": {"state": {"steering": -.27 if 100 <= step < 140 else 10., "v": 1.},
                             "frame": 1000+step, "collisions": []}} for step in range(160)]
    result = summarize_phase(rows, 1.08)
    assert result["actual_steer_mean_rad"] == pytest.approx(-.27)
    assert result["old_linear_mapping_error_max_abs_rad"] == pytest.approx(0.)
    assert result["measurement_frames"] == list(range(1100, 1140))
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_phase(rows[:120], 1.08)
