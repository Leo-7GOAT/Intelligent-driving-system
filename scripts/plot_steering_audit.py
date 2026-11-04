"""Rebuild actuator/native-map diagnostic figures from saved measurements only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from carla_steering_audit import summarize_phase


def plot(batch):
    batch = Path(batch)
    result = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((batch / "calibration/calibration.json").read_text(encoding="utf-8"))
    survey = json.loads((batch / "native_routes.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(raw) != result["control_ticks"] or len({r["measurement"]["frame"] for r in raw}) != len(raw):
        raise ValueError("Control-frame count or uniqueness mismatch")
    for phase in result["phases"]:
        rows = [r for r in raw if r["phase_id"] == phase["phase_id"]]
        rebuilt = summarize_phase(rows, calibration["recommended_adapter"]["steer_gain"])
        for key in ("actual_speed_mean_m_s", "actual_steer_mean_rad", "old_linear_mapping_error_mean_rad", "old_linear_mapping_error_max_abs_rad"):
            if not np.isclose(phase[key], rebuilt[key], atol=1e-12, rtol=0):
                raise ValueError("Summary differs from raw physical measurements")
        frames = [r["measurement"]["frame"] for r in rows]
        if frames != list(range(phase["initial_frame"]+1, phase["initial_frame"]+1+len(rows))):
            raise ValueError("Nonconsecutive actual control frames")
        if any(r["measurement"]["imu_frame"] != r["measurement"]["frame"] for r in rows):
            raise ValueError("Sensor/world mismatch")
        if not np.allclose(np.diff([r["world_elapsed_seconds"] for r in rows]), .05, atol=1e-5, rtol=0):
            raise ValueError("Physical timestamp mismatch")
    (batch / "measurement_verification.json").write_text(json.dumps({
        "scope": "Raw physical-frame and summary reconstruction, not independent performance validation",
        "control_frames": len(raw), "phases": len(result["phases"]), "frame_sensor_time_checks_passed": True,
        "inputs_sha256": {name: hashlib.sha256((batch / name).read_bytes()).hexdigest()
                          for name in ("responses.jsonl", "summary.json", "native_routes.json")}}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for speed in (.5, 1.5):
        rows = sorted([p for p in result["phases"] if p["phase"]["target_speed_m_s"] == speed],
                      key=lambda p: p["phase"]["normalized_steer"])
        x = [p["phase"]["normalized_steer"] for p in rows]
        label = f"Speed setpoint {speed:g} m/s"
        axes[0, 0].plot(x, [p["actual_steer_mean_rad"] for p in rows], "o-", label=label)
        axes[0, 1].plot(x, [p["old_linear_mapping_error_mean_rad"] for p in rows], "o-", label=label)
        axes[1, 0].plot(x, [p["actual_speed_mean_m_s"] for p in rows], "o-", label=label)
        axes[1, 0].axhline(speed, linestyle=":", color="gray", linewidth=.8)
    gain = calibration["recommended_adapter"]["steer_gain"]
    axes[0, 0].plot([-1, 1], [gain, -gain], "--", color="black", label="Existing linear mapping")
    axes[0, 0].set(title="Actual CARLA wheel response", ylabel="Equivalent front steering (rad)")
    axes[0, 1].axvspan(-.5/gain, .5/gain, color="green", alpha=.08, label="Policy command range")
    axes[0, 1].set(title="Measured minus existing mapping", ylabel="Signed error (rad)")
    axes[1, 0].set(title="Measured speed, not requested speed", ylabel="Actual longitudinal speed (m/s)")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Normalized raw CARLA steer")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    selected = []
    for sign in (-1, 1):
        for target in (30, 60):
            eligible = [r for r in survey["candidates"] if r["heading_change_degrees"]*sign > 0]
            if eligible:
                selected.append(min(eligible, key=lambda r: (abs(abs(r["heading_change_degrees"])-target), r["geometry_id"])))
    for index, route in enumerate(selected):
        points = np.array([[p["x"], p["y"]] for p in route["points"]])
        # Translate each example for readability; keep each actual shape and scale.
        points -= points[0]
        axes[1, 1].plot(points[:, 0], points[:, 1], label=f'{route["heading_change_degrees"]:.1f} degrees')
    axes[1, 1].set(title="Native centerline examples: geometry only", xlabel="CARLA x from start (m)", ylabel="CARLA y from start (m)")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(alpha=.2)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Development actuator audit: 22 phases / 3,520 real control frames; no PPO evaluation")
    output = batch / "steering_native_audit.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    (batch / "plotted_native_geometry_ids.json").write_text(json.dumps([r["geometry_id"] for r in selected], indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    plot(parser.parse_args().batch)
