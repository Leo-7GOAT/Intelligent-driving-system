"""Development audit before expanding budgets; no H-success filtering."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import math
import numpy as np
from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv, ACTIONS
from navstack.vehicle import VehicleState, VehiclePlant
from navstack.experiments import write_json


def audit(output, terminal_goal_prefix=False):
    rows = []
    for scenario in ("empty", "static", "narrow", "dynamic"):
        for seed in (11, 22, 33):
            c = EnvConfig(
                scenario=scenario,
                deadline=45,
                planner=PlannerConfig(max_seconds=None),
                terminal_goal_prefix=terminal_goal_prefix,
            )
            env = NavigationEnv(c)
            env.reset(seed=seed)
            states = [
                env.state,
                VehicleState(20, env.goal[1]),
                VehicleState(25, env.goal[1], v=0.75),
            ]
            for label, state in zip(("start", "approach", "parking"), states):
                env.plant = VehiclePlant(c.vehicle, state)
                modes = []
                for action in range(len(ACTIONS)):
                    mode, detail = env._prepare_reference(action)
                    modes.append(
                        {
                            "action": action,
                            "mode": mode,
                            "detail": detail,
                            "stop_reference": bool(
                                env.reference is not None and env.reference.v[-1] == 0
                            ),
                        }
                    )
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "probe": label,
                        "state": asdict(state),
                        "in_collision": env.world.collision(
                            (state.x, state.y, state.yaw), c.vehicle
                        ),
                        "actions": modes,
                    }
                )
            # Independent constructive witness: smooth lateral shift around the entire central
            # obstruction zone. Geometry only, no Hybrid A*, no model selection or rejection.
            witnesses = []
            for shift in (0, -4, 4, -6, 6):
                x = np.linspace(4, c.width - 7, 500)
                u = (x - 4) / (c.width - 11)
                y = env.goal[1] + shift * np.sin(np.pi * u) ** 2
                dy = np.gradient(y, x)
                yaw = np.arctan(dy)
                k = np.gradient(yaw, x) / np.sqrt(1 + dy**2)
                free = all(env.world.free(p, c.vehicle, 0.05) for p in zip(x, y, yaw))
                length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
                witnesses.append(
                    {
                        "shift": shift,
                        "geometry_clear": free,
                        "max_curvature": float(max(abs(k))),
                        "curvature_feasible": bool(
                            max(abs(k)) < 1 / c.vehicle.min_turn_radius
                        ),
                        "length": length,
                        "time_at_1mps_plus_stop": length + 3,
                    }
                )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "independent_witnesses": witnesses,
                    "interpretation": "absence of a witness is inconclusive, never excludes a case",
                }
            )
    # Verify the existing straight action can actually park through MPC and the plant.
    env = NavigationEnv(
        EnvConfig(
            scenario="empty",
            deadline=45,
            planner=PlannerConfig(max_seconds=None),
            terminal_goal_prefix=terminal_goal_prefix,
        )
    )
    env.reset(seed=11)
    done = False
    while not done:
        _, _, done, _, info = env.step(6)
    result = {
        "rows": rows,
        "parking_fixed_action_diagnostic": info["episode_metrics"],
        "discounts": {
            m: {
                "gamma": (c := EnvConfig(method=m, deadline=45)).gamma,
                "gae": 0.95 ** (c.substeps * c.control_dt),
                "physics_per_action": c.substeps,
            }
            for m in ("H", "B1", "B2")
        },
        "deadline": 45,
        "terminal_goal_prefix": terminal_goal_prefix,
        "protocol": "all seed/scenario cases retained; diagnostics are not PPO results",
    }
    write_json(output, result)
    print(
        {
            "output": str(output),
            "parking_success": info["success"],
            "probes": len(rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--terminal-goal-prefix", action="store_true")
    args = p.parse_args()
    audit(Path(args.output), args.terminal_goal_prefix)
