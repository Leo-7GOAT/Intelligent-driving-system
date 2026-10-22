"""Verify and plot all saved native-road control frames, including comfort spikes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from navstack.trajectory import Trajectory, tracking_error
from navstack.vehicle import VehicleState


def analyze(batch):
    batch = Path(batch)
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    results = []
    for episode in summary["episodes"]:
        case = batch / ("case_%02d" % episode["case_id"])
        trace = case / "transitions.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(trace, "rt", encoding="utf-8")]
        if len(rows) != episode["control_ticks"]:
            raise ValueError("Native control count mismatch")
        frames = [r["measurement"]["frame"] for r in rows]
        if frames != list(range(frames[0], frames[0]+len(frames))) or any(r["measurement"]["imu_frame"] != f for r, f in zip(rows, frames)):
            raise ValueError("Native sensor/world frame mismatch")
        times = np.array([r["measurement"]["state"]["time"] for r in rows])
        if not np.allclose(np.diff(times), .05, atol=1e-6, rtol=0):
            raise ValueError("Native physical timestamp mismatch")
        ref_data = json.loads((case / "reference.json").read_text(encoding="utf-8"))
        reference = Trajectory(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in ref_data.items()})
        states = [VehicleState(**r["measurement"]["state"]) for r in rows]
        errors = [tracking_error(s, reference) for s in states]
        for key in ("lateral", "heading", "speed"):
            rms = float(np.sqrt(np.mean([e[key]**2 for e in errors])))
            if not np.isclose(rms, episode["tracking_rms"][key], atol=1e-12, rtol=0):
                raise ValueError("Tracking summary differs from raw frames")
        acceleration = np.array([s.acceleration for s in states])
        velocity = np.array([s.v for s in states])
        delta_t = np.diff(times)
        derived_acceleration = np.diff(velocity)/delta_t
        jerk = np.diff(acceleration)/delta_t
        yaw_delta = (np.diff([s.yaw for s in states])+np.pi) % (2*np.pi)-np.pi
        lateral = (velocity[1:]+velocity[:-1])/2*yaw_delta/delta_t
        road_misses = sum(bool(r["measurement"]["road_surface_probe"]["offroad_points_world"]) for r in rows)
        result = {**episode, "raw_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                  "verified_frame_count": len(frames), "offroad_sample_frames": road_misses,
                  "minimum_actual_acceleration_m_s2": float(acceleration.min()),
                  "maximum_actual_acceleration_m_s2": float(acceleration.max()),
                  "minimum_velocity_difference_acceleration_m_s2": float(derived_acceleration.min()),
                  "acceleration_measurement_vs_velocity_difference_rms": float(np.sqrt(np.mean((acceleration[1:]-derived_acceleration)**2))),
                  "jerk_rms_m_s3": float(np.sqrt(np.mean(jerk**2))), "max_abs_jerk_m_s3": float(np.max(abs(jerk))),
                  "derived_lateral_acceleration_rms_m_s2": float(np.sqrt(np.mean(lateral**2))),
                  "comfort_difference_samples": len(jerk)}
        results.append(result)
        ax = axes[episode["case_id"]]
        ax.plot(reference.x, reference.y, "--", color="gray", label="Fixed native reference")
        ax.plot([s.x for s in states], [s.y for s in states], label="Actual CARLA rear axle")
        ax.plot(reference.x[-1], reference.y[-1], "x", color="red", label="Goal")
        ax.set(title=f'{episode["heading_change_degrees"]:.1f} deg: {episode["termination_reason"]}', xlabel="Local x (m)", ylabel="Local y (m)")
        ax.axis("equal")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        label = f'Case {episode["case_id"]}'
        axes[4].plot(times, velocity, label=label)
        axes[5].plot(times, acceleration, label=label)
    axes[4].set(title="Actual speed\n(no post-termination padding)", xlabel="Simulation time (s)", ylabel="Longitudinal speed (m/s)")
    axes[5].set(title="Actual longitudinal acceleration\n(spikes retained)", xlabel="Simulation time (s)", ylabel="Acceleration (m/s²)")
    for ax in axes[4:]:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Native-road fixed-reference MPC diagnosis: four development routes, no PPO")
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "native_control.png", dpi=160)
    plt.close(fig)
    record = {"scope": "All actual frames retained; velocity/yaw derivatives are measured-state diagnostics, not IMU acceleration channels. First derivative sample omitted, not zero-filled.",
              "source_summary_sha256": hashlib.sha256((batch / "summary.json").read_bytes()).hexdigest(),
              "total_control_frames": sum(r["verified_frame_count"] for r in results), "episodes": results}
    (output / "verified_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"frames": record["total_control_frames"], "outcomes": [r["termination_reason"] for r in results],
          "min_actual_acceleration": [r["minimum_actual_acceleration_m_s2"] for r in results],
          "min_velocity_difference_acceleration": [r["minimum_velocity_difference_acceleration_m_s2"] for r in results],
          "jerk_rms": [r["jerk_rms_m_s3"] for r in results]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    analyze(parser.parse_args().batch)
