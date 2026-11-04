"""Run with isolated CPython 3.7. Actual CARLA control responses, diagnostic only."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import carla


def vector(v):
    return [v.x, v.y, v.z]


def geometry(actor):
    tr = actor.get_transform()
    box = actor.bounding_box
    return {"location": vector(tr.location), "yaw_degrees": tr.rotation.yaw,
            "bbox_location": vector(box.location), "bbox_extent": vector(box.extent),
            "wheels": [{"position": vector(w.position), "max_steer_angle": w.max_steer_angle,
                        "radius": w.radius} for w in actor.get_physics_control().wheels]}


def select_spawn(world):
    candidates = []
    for i, tr in enumerate(world.get_map().get_spawn_points()):
        wp = world.get_map().get_waypoint(tr.location)
        nexts = wp.next(45)
        for end in nexts:
            delta = (end.transform.rotation.yaw - tr.rotation.yaw + 180) % 360 - 180
            if abs(delta) < 2 and abs(end.transform.location.z-tr.location.z) < 1:
                # Require intermediate waypoints to remain straight, not just endpoints.
                good = all(any(abs((q.transform.rotation.yaw-tr.rotation.yaw+180)%360-180)<2
                               for q in wp.next(d)) for d in (10,20,30,40))
                if good:
                    candidates.append((i, tr, wp.lane_width))
    if not candidates:
        raise RuntimeError("No straight calibration spawn found")
    return candidates[0], [{"index": i, "location": vector(t.location), "yaw": t.rotation.yaw,
                            "lane_width": w} for i,t,w in candidates]


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output.parent / "carla_tick_owner.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    client = carla.Client("127.0.0.1",2000); client.set_timeout(20)
    world = client.get_world()
    actors = [a for a in world.get_actors() if a.type_id.startswith(("vehicle.","walker.","sensor."))]
    if actors:
        lock.unlink(); raise RuntimeError("Other actors present: refusing world control")
    original = world.get_settings()
    owned = []
    records = []
    try:
        settings = world.get_settings(); settings.synchronous_mode=True
        settings.fixed_delta_seconds=.05; settings.substepping=True
        settings.max_substep_delta_time=.01; settings.max_substeps=10
        settings.no_rendering_mode=True; world.apply_settings(settings)
        (index, spawn, _), candidates = select_spawn(world)
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name","codex_calibration_diagnostic")
        raw_geometries=[]
        phases=[("throttle02",.2,0), ("throttle04",.4,0), ("throttle06",.6,0),
                ("steer_positive",.25,.25), ("steer_negative",.25,-.25)]
        with (output/"responses.jsonl").open("w") as stream:
            for phase, throttle, steer in phases:
                actor=world.spawn_actor(blueprint,spawn); owned.append(actor)
                actor.set_autopilot(False)
                events=[]
                sensor=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=actor)
                owned.append(sensor); sensor.listen(lambda e: events.append({"frame":e.frame,"other":e.other_actor.id}))
                actor.apply_control(carla.VehicleControl(brake=1))
                for _ in range(30): world.tick()
                raw_geometries.append({"phase":phase,"geometry":geometry(actor)})
                for step in range(160):
                    control=carla.VehicleControl(throttle=throttle if step<80 else 0,
                                                steer=steer if 20<=step<80 else 0,
                                                brake=0 if step<80 else .4)
                    actor.apply_control(control); frame=world.tick()
                    tr=actor.get_transform(); velocity=actor.get_velocity(); accel=actor.get_acceleration()
                    f=tr.get_forward_vector()
                    row={"phase":phase,"step":step,"frame":frame,"t":step*.05,
                         "location":vector(tr.location),"yaw_degrees":tr.rotation.yaw,
                         "v":velocity.x*f.x+velocity.y*f.y,"a":accel.x*f.x+accel.y*f.y,
                         "throttle":control.throttle,"brake":control.brake,"steer":control.steer,
                         "wheel_steer_degrees":[actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),
                                                actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)]}
                    records.append(row); stream.write(json.dumps(row)+"\n")
                time.sleep(.2)
                raw_geometries[-1]["collisions"]=events
                raw_geometries[-1]["end_geometry"]=geometry(actor)
                sensor.stop(); sensor.destroy(); owned.remove(sensor)
                actor.destroy(); owned.remove(actor); world.tick()
                print(json.dumps({"phase":phase,"ticks":160,"collisions":len(events)}),flush=True)
        result={"status":"measured","label":"diagnostic_not_PPO","map":world.get_map().name,
                "spawn_index":index,"spawn_location":vector(spawn.location),"spawn_yaw":spawn.rotation.yaw,
                "candidates":candidates,"geometry_samples":raw_geometries,
                "client_version":client.get_client_version(),"server_version":client.get_server_version(),
                "physics_ticks_measured":len(records),"dt":.05}
        (output/"raw_calibration.json").write_text(json.dumps(result,indent=2))
    finally:
        for actor in reversed(owned):
            if hasattr(actor,"stop"): actor.stop()
            actor.destroy()
        world.apply_settings(original)
        lock.unlink()


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    run(Path(p.parse_args().output))
