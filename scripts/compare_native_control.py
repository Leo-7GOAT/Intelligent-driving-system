"""Compare declared actuator experiments on the same four native references."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    protocols = [json.loads((p / "protocol.json").read_text(encoding="utf-8")) for p in (baseline, candidate)]
    controller_pair=any(p.get('comparison_controller') for p in protocols)
    if controller_pair and ({p.get('comparison_controller') for p in protocols}!={'mpc20','pursuit_pi'} or protocols[0]['source_hashes']!=protocols[1]['source_hashes']):
        raise ValueError('Paired controllers require both declared variants and identical complete frozen source')
    for key in ("routes", "target_speed_m_s", "episode_seconds", "dt", "mpc_horizon",
                "goal_tolerance_m", "goal_speed_m_s", "goal_yaw_tolerance_rad"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError("Unmatched protocol field: " + key)
    calibrations = [json.loads((p / "calibration/calibration.json").read_text(encoding="utf-8")) for p in (baseline, candidate)]
    actuator_configurations = [dict(c["recommended_adapter"]) for c in calibrations]
    if controller_pair and calibrations[0]!=calibrations[1]:raise ValueError('Controller comparison requires exactly the same actuator calibration')
    for calibration in calibrations:
        calibration.pop("status", None)
        calibration.pop("throttle_fit_sha256", None)
        for key in ("longitudinal_mode", "drivetrain_mode", "acceleration_integral_gain", "acceleration_filter_seconds", "throttle_model"):
            calibration["recommended_adapter"].pop(key, None)
    if calibrations[0] != calibrations[1]:
        raise ValueError("Physical calibration or legacy actuator gains differ")
    for name in ("navstack/mpc.py", "navstack/trajectory.py", "navstack/vehicle.py"):
        if protocols[0]["source_hashes"][name] != protocols[1]["source_hashes"][name]:
            raise ValueError("Controller/reference source differs: " + name)
    summaries = [json.loads((p / "analysis/verified_summary.json").read_text(encoding="utf-8")) for p in (baseline, candidate)]
    if len(summaries[0]["episodes"]) != 4 or len(summaries[1]["episodes"]) != 4:
        raise ValueError("Keep every registered case, including failures")
    rows = []
    for old, new in zip(summaries[0]["episodes"], summaries[1]["episodes"]):
        if old["geometry_id"] != new["geometry_id"]:
            raise ValueError("Different case geometry")
        case_name = "case_%02d" % old["case_id"]
        refs = [json.loads((p / case_name / "reference.json").read_text(encoding="utf-8")) for p in (baseline, candidate)]
        for key in ("x", "y", "yaw", "s", "kappa", "v", "a", "t"):
            if not (np.array_equal(refs[0][key],refs[1][key]) if controller_pair else np.allclose(refs[0][key], refs[1][key], atol=1e-6, rtol=0)):
                raise ValueError("Different reference " + key)
        if controller_pair and refs[0]['start_time']!=refs[1]['start_time']:raise ValueError('Different reference clock')
        pair = {"case_id": old["case_id"], "geometry_id": old["geometry_id"]}
        for label, batch, episode in (("baseline", baseline, old), ("candidate", candidate, new)):
            trace = batch / case_name / "transitions.jsonl.gz"
            records = [json.loads(line) for line in gzip.open(trace, "rt", encoding="utf-8")]
            pair[label] = {key: episode[key] for key in ("success", "termination_reason", "actual_seconds",
                "actual_path_length_m", "final_goal_distance_m", "tracking_rms", "mpc_failures",
                "collisions", "offroad_sample_frames", "jerk_rms_m_s3", "minimum_actual_acceleration_m_s2", "full_control_cycle")}
            pair[label].update(raw_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                strong_deceleration_frames=sum(r["measurement"]["state"]["acceleration"] < -5 for r in records),
                coast_frames=sum(r["measurement"].get("actuation_mode") == "mild_deceleration_coast" for r in records))
        rows.append(pair)
    result = {"scope": "Four paired development routes, no independent repetitions or PPO claim; no samples removed or padded.",
              "reference_tolerance": "All eight arrays equal within 1e-6 absolute; measured reset velocity numerical noise only",
              "baseline": str(baseline), "candidate": str(candidate), "actuator_configurations": actuator_configurations,
              "episodes": rows}
    if controller_pair:
        result.update(comparison_kind='Fixed-reference controller,MPC20versus timed pursuit/PI',
                      reference_tolerance='All eight arrays exactly equal;fixed rest reference and shared start clock',
                      shared_calibration=True,controllers=[p['comparison_controller'] for p in protocols])
    (candidate / "analysis/paired_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{ "case": r["case_id"], "success": [r[k]["success"] for k in ("baseline", "candidate")],
        "jerk": [r[k]["jerk_rms_m_s3"] for k in ("baseline", "candidate")],
        "speed_rms": [r[k]["tracking_rms"]["speed"] for k in ("baseline", "candidate")],
        "spikes": [r[k]["strong_deceleration_frames"] for k in ("baseline", "candidate")]} for r in rows]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate)
