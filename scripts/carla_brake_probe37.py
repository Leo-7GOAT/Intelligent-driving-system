"""Bounded raw coast/brake response following measured low-speed threshold crossing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def run(batch):
    batch = Path(batch).resolve()
    sys.path.insert(0, str(batch / "source"))
    import carla
    from scripts.carla_bridge37 import Bridge
    from scripts.carla_steering_audit import guard
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["source_hashes"].items():
        if hashlib.sha256((batch / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen diagnostic source changed")
    output = batch / "brake_probe"
    output.mkdir()  # Never overwrite or retry a partly completed probe.
    shutil.copy2(__file__, output / "recipe.py")
    phases = [{"target_initial_speed": speed, "brake": brake} for speed in (.5, 1., 1.5, 2.5) for brake in (0., .01, .04, .085, .2)]
    write = lambda path, data: path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write(output / "protocol.json", {"scope": "Raw actuator diagnosis, no PPO or controller comparison", "phases": phases,
          "acceleration_throttle": .3, "max_acceleration_ticks": 160, "probe_ticks": 20,
          "deadline_epoch": protocol["deadline_epoch"], "automatic_retries": 0,
          "initial_speed_scope": "Begin braking immediately on actual speed crossing, not forced speed or equilibrium",
          "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    bridge, results, ticks = None, [], 0
    try:
        guard(batch, protocol)
        bridge = Bridge(batch / "calibration/calibration.json", batch / "carla_tick_owner.lock")
        with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for index, phase in enumerate(phases):
                guard(batch, protocol)
                bridge.begin({"episode_id": "brake_probe_%02d" % index, "scenario": "empty", "condition": 7301})
                initial = None
                for step in range(160):
                    bridge.actor.apply_control(carla.VehicleControl(throttle=.3, hand_brake=False))
                    bridge.tick()
                    ticks += 1
                    measured = bridge.read()
                    stream.write(json.dumps({"phase_id": index, "stage": "acceleration", "step": step, "measurement": measured})+"\n")
                    if measured["state"]["v"] >= phase["target_initial_speed"]:
                        initial = measured
                        break
                if initial is None:
                    raise RuntimeError("Target initial speed never reached")
                rows = []
                for step in range(20):
                    bridge.actor.apply_control(carla.VehicleControl(throttle=0., brake=phase["brake"], hand_brake=False))
                    bridge.tick()
                    ticks += 1
                    control = bridge.actor.get_control()
                    row = {"phase_id": index, "stage": "probe", "step": step, "phase": phase, "measurement": bridge.read(),
                           "actual_control": {"throttle": control.throttle, "brake": control.brake,
                                              "hand_brake": control.hand_brake, "gear": control.gear,
                                              "reverse": control.reverse, "manual_gear_shift": control.manual_gear_shift}}
                    rows.append(row)
                    stream.write(json.dumps(row)+"\n")
                time.sleep(.3)
                final = bridge.read()
                result = {"phase_id": index, "phase": phase, "initial_measurement": initial, "terminal_measurement": final,
                          "first_tick_velocity_difference_acceleration": (rows[0]["measurement"]["state"]["v"]-initial["state"]["v"])/.05,
                          "first_tick_reported_acceleration": rows[0]["measurement"]["state"]["acceleration"],
                          "minimum_reported_acceleration": min(r["measurement"]["state"]["acceleration"] for r in rows),
                          "any_hand_brake": any(r["actual_control"]["hand_brake"] for r in rows),
                          "collision_events": final["collisions"]}
                results.append(result)
                stream.flush()
                write(output / "partial_summary.json", {"phases": results, "control_ticks": ticks})
                print(json.dumps({k: result[k] for k in ("phase", "first_tick_velocity_difference_acceleration", "minimum_reported_acceleration", "any_hand_brake")}), flush=True)
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
    run(parser.parse_args().batch)
