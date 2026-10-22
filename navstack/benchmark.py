"""Fixed-reference controller comparisons; raw data and failures are retained."""

from dataclasses import asdict, replace
import math
import time
import numpy as np
from .config import VehicleConfig
from .vehicle import VehiclePlant, VehicleState
from .planning import primitive
from .trajectory import parameterize, tracking_error
from .mpc import MPC
from .world import World
from .metrics import timing_summary
from .experiments import new_run, write_json, append_json


def benchmark(root="runs", steps=120):
    run = new_run(root, "controller_benchmark")
    nominal = VehicleConfig()
    start = (4.0, 10.0, 0.0)
    dt = 0.05
    spatial = np.vstack([start, primitive(start, 0.15, 16, nominal.wheelbase, 0.1)])
    reference = parameterize(spatial, 1.5, 0.0, nominal, stop=True)
    np.savez(
        run / "fixed_reference.npz",
        **{
            key: getattr(reference, key)
            for key in ("x", "y", "yaw", "s", "kappa", "v", "a", "t")
        }
    )
    variants = [
        ("horizon10", nominal, 10),
        ("horizon20", nominal, 20),
        (
            "horizon20_mismatch",
            replace(
                nominal,
                wheelbase_scale=1.2,
                acceleration_scale=0.8,
                steer_tau=0.35,
                input_delay_steps=2,
            ),
            20,
        ),
    ]
    summaries = []
    for name, plant_config, horizon in variants:
        plant = VehiclePlant(plant_config, VehicleState(*start))
        controller = MPC(nominal, dt, horizon)
        world = World(40, 30)
        control_times = []
        qp_times = []
        errors = []
        jerks = []
        lateral = []
        failures = 0
        distance = 0.0
        event = None
        started = time.perf_counter()
        for index in range(steps):
            before = plant.state
            result = controller.step(before, reference, plant.previous_command)
            failures += result.status != "solved"
            after = plant.step(result.command, dt)
            error = tracking_error(after, reference)
            errors.append(error["lateral"])
            jerks.append((after.acceleration - before.acceleration) / dt)
            lateral.append(
                after.v**2
                * math.tan(after.steering)
                / (plant_config.wheelbase * plant_config.wheelbase_scale)
            )
            distance += math.hypot(after.x - before.x, after.y - before.y)
            control_times.append(result.diagnostics["control_seconds"])
            qp_times.append(result.diagnostics["qp_seconds"])
            append_json(
                run / (name + ".jsonl"),
                {
                    "index": index,
                    "state": asdict(after),
                    "command": asdict(result.command),
                    "status": result.status,
                    "tracking": error,
                    "diagnostics": result.diagnostics,
                },
            )
            if world.collision((after.x, after.y, after.yaw), plant_config):
                event = "collision"
                break
            if world.out_of_bounds((after.x, after.y, after.yaw), plant_config):
                event = "out_of_bounds"
                break
        summaries.append(
            {
                "variant": name,
                "plant": asdict(plant_config),
                "horizon": horizon,
                "physics_steps": index + 1,
                "reference_target_speed": 1.5,
                "lateral_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "jerk_rms": float(np.sqrt(np.mean(np.square(jerks)))),
                "lateral_acceleration_rms": float(np.sqrt(np.mean(np.square(lateral)))),
                "distance_m": distance,
                "progress_m": float(
                    reference.s[-1]
                    - math.hypot(
                        plant.state.x - reference.x[-1], plant.state.y - reference.y[-1]
                    )
                ),
                "qp_failures": failures,
                "qp_failure_rate": failures / (index + 1),
                "safety_event": event,
                "final_goal_distance": math.hypot(
                    plant.state.x - reference.x[-1], plant.state.y - reference.y[-1]
                ),
                "wall_seconds": time.perf_counter() - started,
                "control_timing": timing_summary(control_times, dt),
                "qp_timing": timing_summary(qp_times, dt),
            }
        )
    write_json(
        run / "benchmark.json",
        {
            "reference_shared": True,
            "speed_shared": True,
            "steps_requested": steps,
            "variants": summaries,
            "limitation": "controlled engineering probe, not tuned controller superiority or hardware real-time certification",
        },
    )
    print(run)
    return run
