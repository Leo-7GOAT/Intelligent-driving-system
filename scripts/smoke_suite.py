"""Bounded engineering suite. No formal training and no final test-set tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navstack.config import EnvConfig
from navstack.experiments import train, evaluate, new_run, write_json
from navstack.benchmark import benchmark
from navstack.carla_adapter import probe


def main():
    suite = new_run("runs", "smoke_suite")
    records = []
    for method, hold in (("H", 1), ("B1", 1), ("B2", 1), ("B1", 10)):
        config = EnvConfig(method=method, flat_hold=hold, deadline=30.0)
        run, summary = train(config, seed=7, root=suite, wall_seconds=120)
        config.split = "validation"
        evaluation, metrics = evaluate(config, run, episodes=4, root=suite, seed=7)
        records.append(
            {
                "method": method,
                "flat_hold": hold,
                "training_run": str(run),
                "evaluation_run": str(evaluation),
                "training": summary,
                "evaluation": metrics,
            }
        )
    rule, rule_metrics = evaluate(
        EnvConfig(method="B3", split="validation"),
        episodes=4,
        root=suite,
        seed=7,
        policy="rule",
    )
    controller = benchmark(suite, 240)
    write_json(
        suite / "suite_summary.json",
        {
            "records": records,
            "rule_run": str(rule),
            "rule": rule_metrics,
            "benchmark": str(controller),
            "carla": probe(),
            "scope": "single-seed engineering smoke, not formal comparison",
        },
    )
    print(f"SUITE={suite}")


if __name__ == "__main__":
    main()
