"""Fit throttle-only dynamics from measured frames; braking is outside this model."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def fit(batch):
    batch = Path(batch)
    groups, hashes = [], {}
    for case_id in range(4):
        trace = batch / ("case_%02d" % case_id) / "transitions.jsonl.gz"
        hashes[trace.relative_to(batch).as_posix()] = hashlib.sha256(trace.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in gzip.open(trace, "rt", encoding="utf-8")]
        x, y = [], []
        for i in range(12, len(rows)):
            # Fit only throttle dynamics. Exclude 0.6s after any brake command;
            # all these excluded frames remain in the full performance report.
            if any(r["measurement"]["vehicle_control"]["brake"] > 0 for r in rows[i-12:i+1]):
                continue
            speed = rows[i]["measurement"]["state"]["v"]
            if not .3 < speed < 2.:
                continue
            before = rows[i-4]["measurement"]["state"]
            interval = rows[i]["measurement"]["state"]["time"]-before["time"]
            if abs(interval-.2) > 1e-6:
                raise ValueError("Irregular measurement timestamps")
            acceleration = (speed-before["v"])/interval
            throttle = np.mean([r["measurement"]["vehicle_control"]["throttle"] for r in rows[i-4:i]])
            x.append([throttle, speed, 1.])
            y.append(acceleration)
        groups.append((np.array(x), np.array(y)))
    train_x = np.concatenate([g[0] for g in groups[:3]])
    train_y = np.concatenate([g[1] for g in groups[:3]])
    coefficients, _, rank, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
    if rank != 3 or coefficients[0] <= 0 or not np.isfinite(coefficients).all():
        raise ValueError("Unidentified throttle model")
    result = {"scope": "Throttle-only development fit, not braking model or independent confirmation",
              "equation": "mean_acceleration = throttle_gain*mean_throttle + speed_coefficient*speed + intercept",
              "training_cases": [0, 1, 2], "held_out_case": 3, "averaging_seconds": .2,
              "speed_interval_m_s": [.3, 2.], "brake_exclusion_seconds": .6,
              "coefficients": dict(zip(("throttle_gain", "speed_coefficient", "intercept"), coefficients.tolist())),
              "raw_sha256": hashes,
              "case_metrics": [{"case_id": i, "samples": len(y), "rms_m_s2": float(np.mean((x@coefficients-y)**2)**.5)}
                               for i, (x, y) in enumerate(groups)]}
    output = batch / "analysis/throttle_fit.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError("Preserve registered fit")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    fit(parser.parse_args().batch)
