"""Main-environment client. Every measured state comes from the CPython 3.7 CARLA owner."""
from collections import deque
import json
import gzip
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from .vehicle import VehicleState, Command
from .world import World, MovingObstacle
from .env import NavigationEnv


class InfrastructureFailure(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, batch, timeout=30, record_dir=None):
        self.batch=Path(batch).resolve();root=Path(__file__).resolve().parents[1]
        # Frozen source snapshots reuse the isolated installed client runtime;
        # the bridge script itself still comes from the snapshot, not live code.
        interpreter=Path(os.environ.get("NAVSTACK_CARLA_PYTHON",str(root/".carla-python37/runtime/python.exe")))
        if not interpreter.is_file():raise FileNotFoundError("CARLA client interpreter not found: "+str(interpreter))
        audit_id=uuid.uuid4().hex[:8]
        self.stderr=(self.batch/"infrastructure"/f"bridge_{audit_id}.log").open("w")
        self.request_audit_path=self.batch/"infrastructure"/f"bridge_{audit_id}_requests.jsonl.gz"
        self.response_audit_path=self.batch/"infrastructure"/f"bridge_{audit_id}_responses.jsonl.gz"
        self.request_audit=gzip.open(self.request_audit_path,"wt",encoding="utf-8",compresslevel=1)
        recording_args=["--record-dir",str(Path(record_dir).resolve())] if record_dir is not None else []
        self.process=subprocess.Popen([str(interpreter),str(root/"scripts/carla_bridge37.py"),
            "--calibration",str(self.batch/"calibration/calibration.json"),"--lock",str(self.batch/"carla_tick_owner.lock")]+recording_args,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        self.messages=queue.Queue();self.timeout=timeout;self.request_id=0;self.episode_id=None;self.total_ticks=0
        def read():
            # Record before Gym/SB3 auto-reset: a later reset failure must not erase
            # the preceding terminal control frames from the physical evidence.
            with gzip.open(self.response_audit_path,"wt",encoding="utf-8",compresslevel=1) as audit:
                for line in self.process.stdout:
                    if line.startswith("CARLA_JSON:"):
                        result=json.loads(line[len("CARLA_JSON:"):])
                        audit.write(json.dumps({"received_epoch":time.time(),"response":result})+"\n")
                        self.messages.put(result)
                    else:self.stderr.write(line);self.stderr.flush()
            self.messages.put({"ok":False,"error":"bridge exited","infrastructure_failure":True})
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()

    def exchange(self, request):
        self.request_audit.write(json.dumps({"attempted_epoch":time.time(),"request":request})+"\n")
        try:
            self.process.stdin.write(json.dumps(request)+"\n");self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise InfrastructureFailure("Bridge pipe closed: "+str(exc)) from exc
        until=time.monotonic()+self.timeout
        while True:
            try:result=self.messages.get(timeout=max(.01,until-time.monotonic()))
            except queue.Empty:raise InfrastructureFailure("Bridge response timed out; episode aborted, no unacknowledged re-tick")
            if result.get("request_id") not in (None,request["request_id"]):
                continue
            if not result.get("ok"):
                self.last_failure_response=result
                raise InfrastructureFailure(result.get("error","unknown bridge failure"))
            if result.get("episode_id")!=request["episode_id"]:raise InfrastructureFailure("Mismatched response episode")
            return result

    def request(self, op, **payload):
        if op=="begin":self.episode_id=uuid.uuid4().hex
        self.request_id+=1
        request={"op":op,"episode_id":self.episode_id,"request_id":self.request_id,**payload}
        self.last_request=request
        result=self.exchange(request)
        self.total_ticks+=result.get("reset_ticks",0)+(1 if op=="step" else 0)
        return result

    def close(self):
        if getattr(self,"closed",False):return
        self.closed=True
        if self.process.poll() is None:
            try:self.process.stdin.close()
            except OSError:pass
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Own bridge process only; never terminate an unrelated CARLA server.
                self.process.terminate();self.process.wait(timeout=5)
        self.reader.join(timeout=2);self.request_audit.close();self.stderr.close()


class CarlaWorld(World):
    sensor_collision_truth = True
    measured_offroad = False
    def out_of_bounds(self, pose, vehicle, margin=0.0):
        return self.measured_offroad or super().out_of_bounds(pose, vehicle, margin)
    def advance(self, dt):
        pass  # measured actor positions have already arrived with the CARLA frame


class BridgePlant:
    def __init__(self, client, result, world):
        self.client=client;self.world=world;self.delay_queue=deque();self.collision_history=False
        self.update(result)

    def update(self,result):
        self._state=VehicleState(**result["state"]);self.frame=result["frame"]
        self.previous_command=Command(**result["command"])
        self.world.dynamic=[MovingObstacle(**o) for o in result["dynamic"]]
        self.collision_history |= bool(result["collisions"])
        probe=result.get("road_surface_probe")
        self.world.measured_offroad |= bool(probe and probe["offroad_points_world"])
        self.evidence={k:result[k] for k in ("frame","request_id","episode_id","imu_frame","collisions","sensor_health")}
        for key in ("vehicle_control","speed_setpoint","terminal_sensor_drain_seconds","road_surface_probe"):
            if key in result:self.evidence[key]=result[key]

    @property
    def state(self):return self._state

    def step(self,command,dt):
        if abs(dt-.05)>1e-10:raise ValueError("CARLA requires .05 seconds")
        self.update(self.client.request("step",expected_frame=self.frame,
            acceleration=command.acceleration,steering=command.steering))
        return self.state

    def confirm_terminal(self):self.update(self.client.request("finalize"))


class CarlaNavigationEnv(NavigationEnv):
    def __init__(self, config, client, cases=None):
        self.client=client;self.cases=cases;self.case_index=0
        super().__init__(config)

    def _create_episode(self,options=None):
        case=(options or {}).get("case")
        if case is None:
            case=self.cases[self.case_index%len(self.cases)] if self.cases else {
                "scenario":str(self.np_random.choice(["static","curve","dynamic"])),
                "condition":int(self.np_random.integers(0,10))}
        self.case_index+=1
        if case.get("scenario")=="native_curve" and not self.config.road_observation:
            raise ValueError("Native curves require shared measured road observations; old models support fixed-reference diagnostics only")
        if self.config.road_observation and case.get("scenario")!="native_curve":
            raise ValueError("Measured road observation currently requires a native route")
        result=self.client.request("begin",**case,road_observation=self.config.road_observation);scene=result["scene"]
        world=CarlaWorld(scene["width"],scene["height"],scene["static"])
        if self.config.road_observation:
            from .road_grid import RoadGrid
            world.road=RoadGrid(scene["road_grid"])
            if (world.road.width,world.road.height)!=(world.width,world.height):
                raise InfrastructureFailure("Road grid/scene extent mismatch")
        plant=BridgePlant(self.client,result,world)
        if world.out_of_bounds((plant.state.x,plant.state.y,plant.state.yaw),self.config.vehicle):
            raise InfrastructureFailure("Initial measured vehicle footprint is outside registered road geometry")
        self.scene_evidence=scene
        return world,plant,scene["goal"],scene["name"]
