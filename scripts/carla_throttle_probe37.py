"""Bounded low-throttle step response and coast, with measured CARLA states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def run(batch, manual_first_gear=False):
    batch = Path(batch).resolve()
    sys.path.insert(0, str(batch / "source"))
    import carla
    from scripts.carla_bridge37 import Bridge
    from scripts.carla_steering_audit import guard
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen diagnostic source changed")
    output = batch / ("throttle_probe_manual1" if manual_first_gear else "throttle_probe")
    output.mkdir()  # Never overwrite or retry a partly completed probe.
    shutil.copy2(__file__, output / "recipe.py")
    phases = [{"throttle": value} for value in (.02, .04, .06, .08, .10, .12)]
    write = lambda path, data: path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write(output / "protocol.json", {"scope": "Raw actuator diagnosis, no PPO or controller comparison", "phases": phases,
          "manual_first_gear": manual_first_gear, "step_ticks": 80, "coast_ticks": 40,
          "deadline_epoch": protocol["deadline_epoch"], "automatic_retries": 0,
          "initial_speed_scope": "Start from brake-held measured reset; 4s constant throttle then 2s coast, no forced velocity",
          "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    bridge, results, ticks = None, [], 0
    try:
        guard(batch, protocol)
        bridge = Bridge(batch / "calibration/calibration.json", batch / "carla_tick_owner.lock")
        with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for index, phase in enumerate(phases):
                guard(batch, protocol)
                bridge.begin({"episode_id": "brake_probe_%02d" % index, "scenario": "empty", "condition": 7301})
                initial = bridge.read()
                rows = []
                for step in range(120):
                    throttle = phase["throttle"] if step < 80 else 0.
                    bridge.actor.apply_control(carla.VehicleControl(throttle=throttle, hand_brake=False, manual_gear_shift=manual_first_gear, gear=1 if manual_first_gear else 0))
                    bridge.tick()
                    ticks += 1
                    control = bridge.actor.get_control()
                    row = {"phase_id": index, "stage": "step" if step < 80 else "coast", "step": step, "phase": phase,
                           "measurement": bridge.read(), "actual_control": {"throttle": control.throttle, "brake": control.brake,
                           "hand_brake": control.hand_brake, "gear": control.gear, "reverse": control.reverse}}
                    rows.append(row)
                    stream.write(json.dumps(row)+"\n")
                time.sleep(.3)
                final = bridge.read()
                result = {"phase_id": index, "phase": phase, "initial_measurement": initial, "terminal_measurement": final,
                          "first_tick_velocity_difference_acceleration": (rows[0]["measurement"]["state"]["v"]-initial["state"]["v"])/.05,
                          "first_tick_reported_acceleration": rows[0]["measurement"]["state"]["acceleration"],
                          "step_final_speed": rows[79]["measurement"]["state"]["v"],
                          "maximum_speed": max(r["measurement"]["state"]["v"] for r in rows),
                          "maximum_reported_acceleration": max(r["measurement"]["state"]["acceleration"] for r in rows),
                          "minimum_reported_acceleration": min(r["measurement"]["state"]["acceleration"] for r in rows),
                          "any_hand_brake": any(r["actual_control"]["hand_brake"] for r in rows),
                          "collision_events": final["collisions"]}
                results.append(result)
                stream.flush()
                write(output / "partial_summary.json", {"phases": results, "control_ticks": ticks})
                print(json.dumps({k: result[k] for k in ("phase", "step_final_speed", "maximum_speed", "maximum_reported_acceleration", "minimum_reported_acceleration", "any_hand_brake")}), flush=True)
        write(output / "summary.json", {"scope": "diagnostic_not_PPO", "phases": results, "control_ticks": ticks, "complete": True})
    except BaseException as exc:
        write(output / "failure.json", {"error": repr(exc), "completed_phases": len(results), "control_ticks": ticks})
        raise
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--manual-first-gear", action="store_true")
    args = parser.parse_args()
    run(args.batch, args.manual_first_gear)
