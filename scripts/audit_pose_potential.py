"""Offline PBRS identity audit on existing physical traces; no new policy scores."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.config import EnvConfig
from navstack.reward import RewardCalculator
from navstack.vehicle import VehicleState


def audit(batch, job_id, target):
    batch = Path(batch)
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
    config = EnvConfig.from_dict(job["env"])
    original = RewardCalculator(config.reward, config.control_dt)
    changed = EnvConfig.from_dict(config.to_dict())
    changed.reward.heading_potential_weight = .1
    changed.reward.heading_potential_radius = 6.
    shaped = RewardCalculator(changed.reward, changed.control_dt)
    run = batch / protocol["run_group"] / job_id
    trace = run / ("validation_%d_trace.jsonl.gz" % target)
    cases = {}
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["case_id"]
            case = cases.setdefault(identity, {"before": VehicleState(**row["info"]["initial_scenario"]["initial"]),
                    "goal": row["info"]["initial_scenario"]["goal"], "steps": 0, "discounted_added_reward": 0., "max_abs_local_change": 0.})
            for tick in row["info"]["trace"]:
                after = VehicleState(**tick["state"])
                terminal = bool(tick["reward_components"]["event"])
                old = original.calculate(case["before"], after, case["goal"], 0, terminated=terminal)[1]["potential"]
                new = shaped.calculate(case["before"], after, case["goal"], 0, terminated=terminal)[1]["potential"]
                if abs(old-tick["reward_components"]["potential"]) > 1e-8:
                    raise ValueError("Original potential reconstruction mismatch")
                case["discounted_added_reward"] += original.gamma**case["steps"]*(new-old)
                case["max_abs_local_change"] = max(case["max_abs_local_change"], abs(new-old))
                case["steps"] += 1
                case["before"] = after
    # Every registered scenario starts aligned to its final goal; extra initial
    # potential is zero, and true task termination sets extra final potential zero.
    if any(abs(c["discounted_added_reward"]) > 1e-6 for c in cases.values()):
        raise ValueError("Terminal heading shaping did not telescope on recorded cases")
    result = {"scope": "Reward identity on unchanged trajectories, not changed policy performance", "input_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
              "cases": [{"case_id": k, **{key:v for key,v in c.items() if key not in ("before", "goal")}} for k,c in cases.items()]}
    (run / ("pose_potential_audit_%d.json" % target)).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    audit(args.batch, args.job, args.target)
