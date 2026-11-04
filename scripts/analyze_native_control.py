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


def bind_controller_frames(batch,protocol,episodes):
    """Bind every saved controller measurement/request to its raw bridge RPC."""
    from scripts.audit_native_method_endpoint import read_prefix
    requests={};responses={};hashes={}
    for kind,destination,field in [('requests',requests,'request'),('responses',responses,'response')]:
        files=list((batch/'infrastructure').glob(f'bridge_*_{kind}.jsonl.gz'))
        if len(files)!=1:raise ValueError('Controller probe requires one complete owned bridge stream')
        rows,error=read_prefix(files[0])
        if error:raise ValueError('Controller bridge stream is truncated')
        hashes[files[0].name]=hashlib.sha256(files[0].read_bytes()).hexdigest()
        for row in rows:
            value=row[field];key=(value.get('episode_id'),value.get('request_id'))
            if key in destination:raise ValueError('Duplicate controller RPC identity')
            destination[key]=value
    expected={key for key,value in requests.items() if value['op']=='step'};used=set()
    for episode in episodes:
        case_id=episode['case_id'];folder=batch/f'case_{case_id:02d}'
        scene=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
        traces=[json.loads(line) for line in gzip.open(folder/'transitions.jsonl.gz','rt',encoding='utf-8')]
        if len(traces)!=episode['control_ticks']:raise ValueError('Controller trace count differs')
        for row in traces:
            measurement=row['measurement'];key=(measurement['episode_id'],measurement['request_id'])
            request=requests.get(key)
            if key in used or key not in expected or responses.get(key)!=measurement:raise ValueError('Saved controller measurement differs from actual bridge acknowledgement')
            if request['expected_frame']!=measurement['frame']-1 or any(request[k]!=row['requested_command'][k] for k in ('acceleration','steering')):raise ValueError('Controller requested command/frame differs from owned RPC')
            used.add(key)
        episode_id=traces[0]['measurement']['episode_id']
        begins=[responses[key] for key,r in requests.items() if r['op']=='begin' and r['episode_id']==episode_id and key in responses]
        if len(begins)!=1 or begins[0]['scene']!=scene or scene['native_layout']['geometry_id']!=protocol['routes'][case_id]['geometry_id']:
            raise ValueError('Controller reset scene/route differs from its raw owned episode')
    if used!=expected:raise ValueError('Unaccounted actual controller commands')
    return {'actual_control_frames_bound':len(used),'bridge_stream_sha256':hashes,
            'scope':'All saved measurements and requested commands exactly match owned raw bridge RPC streams;not a Python plant reconstruction.'}


def analyze(batch):
    batch = Path(batch)
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8')) if (batch/'protocol.json').is_file() else {}
    binding=bind_controller_frames(batch,protocol,summary['episodes']) if protocol.get('comparison_controller') else None
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
    controller=protocol.get('comparison_controller')
    fig.suptitle(f"Native-road fixed-reference {controller}: four development routes, no PPO" if controller else "Native-road fixed-reference MPC diagnosis: four development routes, no PPO")
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "native_control.png", dpi=160)
    plt.close(fig)
    record = {"scope": "All actual frames retained; velocity/yaw derivatives are measured-state diagnostics, not IMU acceleration channels. First derivative sample omitted, not zero-filled.",
              "source_summary_sha256": hashlib.sha256((batch / "summary.json").read_bytes()).hexdigest(),
              "total_control_frames": sum(r["verified_frame_count"] for r in results), "episodes": results}
    if binding:record['bridge_frame_binding']=binding
    (output / "verified_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"frames": record["total_control_frames"], "outcomes": [r["termination_reason"] for r in results],
          "min_actual_acceleration": [r["minimum_actual_acceleration_m_s2"] for r in results],
          "min_velocity_difference_acceleration": [r["minimum_velocity_difference_acceleration_m_s2"] for r in results],
          "jerk_rms": [r["jerk_rms_m_s3"] for r in results]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    analyze(parser.parse_args().batch)
