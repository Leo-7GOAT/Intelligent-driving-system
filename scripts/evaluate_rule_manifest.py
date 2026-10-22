"""B3 rule-only control on the same fixed final cases; zero PPO training budget."""

import argparse, gzip, hashlib, json, sys, time
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--source-root", required=True)
    args = p.parse_args()
    batch = Path(args.batch).resolve()
    source = Path(args.source_root).resolve()
    root = Path(__file__).resolve().parents[1]
    deadline = json.loads((batch / "batch.json").read_text(encoding="utf-8"))[
        "deadline_epoch"
    ]
    required = [
        batch / "formal_v5" / f"{m}_seed{s}" / "validation_1000000.json"
        for s in (11,)
        for m in ("H", "B1")
    ]
    run = batch / "baselines/formal_B3_final_v5"
    run.mkdir(parents=True, exist_ok=True)
    protocol = {
        "epoch": time.time(),
        "policy": "rule_only_B3",
        "training_physics_steps": 0,
        "source_root": str(source),
        "cases_sha256": hashlib.sha256(
            (root / "configs/lightweight_final_cases.json").read_bytes()
        ).hexdigest(),
        "wait_for_primary_1M": True,
        "wait_for_paired_seed": 11,
        "independent_training_seed": None,
    }
    if not (run / "protocol.json").exists():
        (run / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
    while not all(f.exists() for f in required):
        if time.time() > deadline - 180 or (batch / "STOP").exists():
            sys.exit(0)
        time.sleep(5)
    if (run / "summary.json").exists():
        raise RuntimeError("Completed B3 final already exists; refuse repeat")
    sys.path.insert(0, str(source))
    from navstack.env import NavigationEnv, rule_action
    from navstack.config import EnvConfig
    from navstack.experiments import append_json, write_json, jsonable

    checkpoint = Path(
        json.loads(
            (batch / "formal_v5/H_seed11/latest.json").read_text(encoding="utf-8")
        )["checkpoint"]
    )
    cfg = EnvConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["env"]
    )
    cfg.method = "B3"
    cfg.split = "test"
    write_json(
        run / "config.json",
        {
            "env": cfg.to_dict(),
            "policy": "rule_only_B3",
            "source_checkpoint_config_only": str(checkpoint),
            "training_physics_steps": 0,
        },
    )
    cases = json.loads(
        (root / "configs/lightweight_final_cases.json").read_text(encoding="utf-8")
    )
    records = []
    started = time.time()
    with gzip.open(run / "transitions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            if time.time() > deadline - 60 or (batch / "STOP").exists():
                break
            cfg.scenario = case["scenario"]
            env = NavigationEnv(cfg)
            env.reset(seed=case["seed"])
            append_json(
                run / "scenes.jsonl",
                {
                    "case_id": index,
                    "case": case,
                    "initial_scenario": env.initial_scenario,
                },
            )
            while not env.done:
                action = rule_action(env)
                _, _, _, _, info = env.step(action)
                stream.write(
                    json.dumps(
                        {
                            "case_id": index,
                            "policy": "rule_only_B3",
                            "action": action,
                            "mode": info["mode"],
                            "plan": info["plan"],
                            "trace": info["trace"],
                        },
                        default=jsonable,
                    )
                    + "\n"
                )
                if time.time() > deadline - 60 or (batch / "STOP").exists():
                    break
            if not env.done:
                append_json(
                    run / "censored.jsonl",
                    {"case_id": index, "reason": "budget_or_stop"},
                )
                env.close()
                break
            row = {
                "case_id": index,
                "case": case,
                "policy": "rule_only_B3",
                **info["episode_metrics"],
            }
            records.append(row)
            append_json(run / "episodes.jsonl", row)
            print(
                json.dumps({"case_id": index, "reason": row["termination_reason"]}),
                flush=True,
            )
            env.close()
    write_json(
        run / "summary.json",
        {
            "policy": "rule_only_B3",
            "training_physics_steps": 0,
            "independent_training_seed": None,
            "completed": len(records),
            "scheduled": len(cases),
            "complete": len(records) == len(cases),
            "success_rate": (
                sum(r["success"] for r in records) / len(cases)
                if len(records) == len(cases)
                else None
            ),
            "evaluation_physics_steps": sum(r["physics_steps"] for r in records),
            "wall_seconds": time.time() - started,
        },
    )
