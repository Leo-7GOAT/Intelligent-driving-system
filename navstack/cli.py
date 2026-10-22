import argparse
import json
from pathlib import Path
from .config import EnvConfig


def main():
    parser = argparse.ArgumentParser(
        description="Physical PPO / Hybrid A* / MPC research runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "baseline-smoke"):
        p = commands.add_parser(name)
        p.add_argument("--method", choices=["H", "B1", "B2", "B3"], default="H")
        p.add_argument("--config", type=Path)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument(
            "--scenario",
            choices=["mixed", "empty", "static", "narrow", "dynamic", "unseen"],
            default=None,
        )
        p.add_argument("--root", default="runs")
        p.add_argument("--deadline", type=float, default=None)
        p.add_argument("--flat-hold", type=int, default=1)
        if name == "train":
            p.add_argument(
                "--profile", choices=["smoke", "debug", "formal"], default="smoke"
            )
            p.add_argument("--physics-budget", type=int)
            p.add_argument("--n-envs", type=int, default=1)
            p.add_argument("--wall-seconds", type=float)
        else:
            p.add_argument("--run", type=Path)
            p.add_argument("--episodes", type=int, default=4)
            p.add_argument(
                "--split", choices=["validation", "test"], default="validation"
            )
    p = commands.add_parser("report")
    p.add_argument("--root", default="runs")
    p.add_argument("--output", default="reports")
    p = commands.add_parser("benchmark")
    p.add_argument("--root", default="runs")
    p.add_argument("--steps", type=int, default=120)
    p = commands.add_parser("carla-probe")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--root", default="runs")
    p = commands.add_parser("select")
    p.add_argument("candidates", nargs="+", type=Path)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--root", default="runs")
    args = parser.parse_args()
    from .experiments import PROFILES, train, evaluate, report

    if args.command == "report":
        return report(args.root, args.output)
    if args.command == "select":
        from .selection import select_checkpoints

        return select_checkpoints(args.candidates, args.episodes, args.root)
    if args.command == "benchmark":
        from .benchmark import benchmark

        return benchmark(args.root, args.steps)
    if args.command == "carla-probe":
        from .carla_adapter import probe
        from .experiments import new_run, write_json

        result = probe(args.host, args.port)
        run = new_run(args.root, "carla_probe")
        write_json(run / "probe.json", result)
        print(result)
        return
    if args.config:
        config = EnvConfig.from_dict(
            json.loads(args.config.read_text(encoding="utf-8"))
        )
        if config.method != args.method:
            raise ValueError("--method differs from --config")
    elif args.command == "evaluate" and args.run:
        config = EnvConfig.from_dict(
            json.loads((args.run / "config.json").read_text(encoding="utf-8"))["env"]
        )
        if config.method != args.method:
            raise ValueError(
                "--method differs from checkpoint; incompatible transfer requires explicit experiment"
            )
    else:
        config = EnvConfig(method=args.method, flat_hold=args.flat_hold)
    if args.scenario:
        config.scenario = args.scenario
    if args.deadline is not None:
        config.deadline = args.deadline
    elif args.command == "train" and not args.config:
        config.deadline = PROFILES[args.profile]["deadline"]
    config.__post_init__()
    if args.command == "train":
        config.split = "train"
        return train(
            config,
            args.profile,
            args.seed,
            args.root,
            args.physics_budget,
            args.n_envs,
            args.wall_seconds,
        )
    config.split = args.split
    if args.command == "evaluate":
        return evaluate(config, args.run, args.episodes, args.root, args.seed)
    if config.method not in ("B2", "B3"):
        raise ValueError("baseline-smoke supports B2/B3; H/B1 must use actual training")
    return evaluate(
        config,
        None,
        args.episodes,
        args.root,
        args.seed,
        "rule" if config.method == "B3" else "cycle",
    )


if __name__ == "__main__":
    main()
