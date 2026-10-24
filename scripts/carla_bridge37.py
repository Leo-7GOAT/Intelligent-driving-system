"""Only CARLA RPC and actuation; PPO/planning/MPC stay in the main Python environment."""
import argparse,json,math,os,sys,time,threading,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import carla
from navstack.bridge_protocol import EpisodeProtocol
from navstack.process_lease import ProcessLease


class Bridge:
    def __init__(self, calibration, lock, record_dir=None, host="127.0.0.1", port=2000):
        self.cal=json.loads(Path(calibration).read_text(encoding="utf-8")); self.lock=Path(lock)
        if not isinstance(host,str) or not host or any(c.isspace() for c in host):raise ValueError("invalid CARLA RPC host")
        if isinstance(port,bool) or not isinstance(port,int) or not 1024<=port<=65533:raise ValueError("invalid CARLA RPC base port")
        self.rpc_host=host;self.rpc_port=port
        self.record_dir=record_dir;self.recorder=None
        self.lease=ProcessLease(self.lock).acquire()
        try:
            self._connect()
        except BaseException:
            self.lease.close()
            raise

    def _connect(self):
        self.client=carla.Client(self.rpc_host,self.rpc_port);self.client.set_timeout(15)
        if self.client.get_client_version()!="0.9.15" or self.client.get_server_version()!="0.9.15":
            raise RuntimeError("client/server must both be 0.9.15")
        self.world=self.client.get_world();self.map=self.world.get_map();self.original=self.world.get_settings()
        if any(a.type_id.startswith(("vehicle.","walker.","sensor.")) for a in self.world.get_actors()):
            raise RuntimeError("World contains other actors; refusing control")
        self.owned=[];self.retired_ids=set();self.actor=None;self.events=[];self.frame=None
        settings=self.world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
        settings.no_rendering_mode=self.record_dir is None;self.world.apply_settings(settings)
        if self.record_dir is not None:
            # Camera frame metadata was inconsistent when enabling rendering in
            # an already-running asynchronous world. Start the recorded world
            # under synchronous settings, as verified by the isolated live probe.
            self.world=self.client.reload_world(False)
            self.map=self.world.get_map()
        self.protocol=EpisodeProtocol()

    def clear(self):
        had_actors=bool(self.owned)
        failures=[]
        survivors=[]
        for a in reversed(self.owned):
            try:
                if hasattr(a,"stop"):a.stop()
            except Exception as exc:
                failures.append("stop actor %s: %s" % (a.id,exc))
            try:
                if a.is_alive and not a.destroy():
                    raise RuntimeError("destroy returned false")
                self.retired_ids.add(a.id)
            except Exception as exc:
                survivors.append(a)
                failures.append("destroy actor %s: %s" % (a.id,exc))
        self.owned=list(reversed(survivors));self.actor=None
        if failures:
            raise RuntimeError("Owned actor cleanup incomplete: " + "; ".join(failures))
        if had_actors:
            # A cleanup-only tick lets Unreal retire destroyed physics objects before
            # the next episode spawns replacements. This is never a policy interaction.
            self.world.tick(15.)
            time.sleep(.05)

    def loc(self,x,y,z=None):
        z=getattr(self,"spawn_z",.6) if z is None else z
        return carla.Location(x=self.origin[0]+x*self.co+y*self.si,
                              y=self.origin[1]+x*self.si-y*self.co,z=z)

    def local(self,x,y):
        dx=x-self.origin[0];dy=y-self.origin[1]
        return dx*self.co+dy*self.si,dx*self.si-dy*self.co

    def transform(self,x,y,yaw=0):
        return carla.Transform(self.loc(x,y),carla.Rotation(yaw=self.angle-math.degrees(yaw)))

    def spawn_vehicle(self,x,y,role):
        bp=self.world.get_blueprint_library().find("vehicle.tesla.model3");bp.set_attribute("role_name",role)
        if self.record_dir is not None and bp.has_attribute("color"):
            bp.set_attribute("color","255,80,0" if role=="codex_ppo_ego" else "30,100,220")
        a=self.world.spawn_actor(bp,self.transform(x,y));self.owned.append(a);a.set_autopilot(False)
        a.apply_control(carla.VehicleControl(brake=1,hand_brake=True));return a

    def begin(self,r):
        self.road_geometry_version=r.get("road_geometry_version","carla_driving_64_v1")
        if self.road_geometry_version not in ("carla_driving_64_v1","carla_driving_refined_union_64_v2","carla_driving_indexed_union_64_v3"):
            raise ValueError("Unknown road geometry version")
        if self.road_geometry_version != "carla_driving_64_v1" and (not r.get("road_observation") or r.get("scenario")!="native_curve"):
            raise ValueError("Refined geometry requires shared native road observation")
        if self.road_geometry_version == "carla_driving_indexed_union_64_v3" and not hasattr(self,"road_surface_index"):
            from navstack.lane_projection import DrivingSurfaceIndex
            self.road_surface_index=DrivingSurfaceIndex(self.map,carla.LaneType.Driving)
        if r.get("road_observation",False) and self.cal["recommended_adapter"].get("longitudinal_mode") in ("acceleration_pi_v1","acceleration_ff_v2"):
            raise ValueError("Measured-road PPO does not yet observe experimental actuator internal state")
        if (self.cal["recommended_adapter"].get("longitudinal_mode") in ("acceleration_pi_v1","acceleration_ff_v2")
                and r.get("scenario")!="native_curve" and not r.get("diagnostic_actuation",False)):
            raise ValueError("Experimental acceleration PI requires explicit diagnostic scope; training state interface not integrated")
        self.clear();self.events=[];self.imu_frame=-1;self.sensor_event=threading.Event();self.active_episode=r["episode_id"]
        if any(a.id not in self.retired_ids and a.type_id.startswith(("vehicle.","walker.","sensor.")) for a in self.world.get_actors()):
            raise RuntimeError("Foreign actors appeared; refusing to tick")
        condition=int(r.get("condition",0));scenario=r.get("scenario","static")
        self.native_layout=None;self.spawn_z=.6
        if scenario=="native_curve":
            from navstack.native_route import local_layout
            self.native_layout=local_layout(r["native_route"],self.map.name,hashlib.sha256(self.map.to_opendrive().encode()).hexdigest())
            self.angle=self.native_layout["angle"];self.origin=self.native_layout["origin"]
            self.spawn_z=self.native_layout["spawn_z"]
            self.width=self.native_layout["width"];self.height=self.native_layout["height"]
        else:
            tr=self.map.get_spawn_points()[18];self.angle=tr.rotation.yaw
            angle=math.radians(self.angle)
            # Existing straight-road scenes retain their measured14mroad edges.
            self.origin=(tr.location.x-5.25*math.sin(angle),tr.location.y+5.25*math.cos(angle))
            self.width=34.;self.height=14.
        angle=math.radians(self.angle);self.co=math.cos(angle);self.si=math.sin(angle)
        self.road_grid=None
        if self.native_layout and r.get("road_observation",False):
            from navstack.native_route import sample_road_grid
            def is_driving(x,y):
                if self.road_geometry_version == "carla_driving_indexed_union_64_v3":
                    return self.road_surface_index.contains(self.loc(x,y,z=self.spawn_z-.6))
                if self.road_geometry_version == "carla_driving_refined_union_64_v2":
                    from navstack.lane_projection import on_driving_surface
                    point=self.loc(x,y,z=self.spawn_z-.6)
                    return on_driving_surface(self.map,point,carla.LaneType.Driving)
                waypoint=self.map.get_waypoint(self.loc(x,y),project_to_road=False,lane_type=carla.LaneType.Driving)
                return waypoint is not None and abs(waypoint.transform.location.z-(self.spawn_z-.6))<1.5
            self.road_grid=sample_road_grid(self.native_layout,is_driving,self.road_geometry_version)
        self.dynamic=[];self.static_actors=[]
        case_rng=random.Random(210000+condition)
        start_y=case_rng.uniform(6.5,7.5);goal_y=start_y
        self.walker_speed=case_rng.uniform(.5,.75)
        obstacle_x=case_rng.uniform(15.6,16.8)
        self.rear=self.cal["rear_axle_actor_x"]
        start_x=4.
        if self.native_layout:
            start_x,start_y,_=self.native_layout["path"][0]
        self.actor=self.spawn_vehicle(start_x-self.rear,start_y,"codex_ppo_ego")
        if scenario=="static":
            self.static_actors.append(self.spawn_vehicle(obstacle_x,case_rng.uniform(6.7,7.3),"codex_static_obstacle"))
        elif scenario=="curve":
            goal_y=start_y+2.5
            self.static_actors.append(self.spawn_vehicle(15,4.,"codex_curve_inner_boundary"))
            self.static_actors.append(self.spawn_vehicle(23,11.5,"codex_curve_outer_boundary"))
        elif scenario=="dynamic":
            bp=self.world.get_blueprint_library().find("walker.pedestrian.0001")
            if bp.has_attribute("is_invincible"):bp.set_attribute("is_invincible","false")
            walker=self.world.spawn_actor(bp,self.transform(obstacle_x,2.))
            self.owned.append(walker);self.dynamic=[walker]
        elif scenario not in ("empty","native_curve"):raise ValueError("unknown CARLA scenario")
        sensor=self.world.spawn_actor(self.world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=self.actor)
        self.owned.append(sensor)
        self.collision_sensor=sensor
        episode=r["episode_id"]
        def collision(e):
            self.events.append({"episode_id":episode,"frame":e.frame,"timestamp":e.timestamp,
                                "other_actor":e.other_actor.id,"impulse":[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z]})
        sensor.listen(collision)
        imu=self.world.spawn_actor(self.world.get_blueprint_library().find("sensor.other.imu"),carla.Transform(),attach_to=self.actor)
        self.owned.append(imu)
        def heartbeat(e):
            self.imu_frame=max(self.imu_frame,e.frame);self.sensor_event.set()
        imu.listen(heartbeat)
        self.recorder=None
        if self.record_dir is not None:
            from navstack.camera_recording import FrameRecorder
            self.recorder=FrameRecorder(self.record_dir,episode)
            camera_bp=self.world.get_blueprint_library().find("sensor.camera.rgb")
            for key,value in (("image_size_x","640"),("image_size_y","360"),("fov","80"),("sensor_tick","0.0")):
                camera_bp.set_attribute(key,value)
            # Fixed overhead view of the real road; this actor never controls ego.
            camera=self.world.spawn_actor(camera_bp,carla.Transform(self.loc(17,7,26),carla.Rotation(pitch=-90,yaw=self.angle+90)))
            self.owned.append(camera)
            camera.listen(self.recorder.save)
        # A newly subscribed IMU may not emit its birth frame. The first five
        # existing brake-held reset ticks warm up the stream; all remaining
        # reset ticks and every policy control still require the matching frame.
        for settling_tick in range(30):self.tick(require_sensor=settling_tick>=5)
        self.events=[];self.t0=self.world.get_snapshot().timestamp.elapsed_seconds
        self.command={"acceleration":0.,"steering":0.};self.speed_target=0.;self.condition=condition
        self.acceleration_controller=None
        if self.cal["recommended_adapter"].get("longitudinal_mode") in ("acceleration_pi_v1","acceleration_ff_v2"):
            from navstack.carla_actuation import AccelerationPI
            self.acceleration_controller=AccelerationPI(self.cal["recommended_adapter"])
        self.goal=self.native_layout["goal"] if self.native_layout else [27.,goal_y,0.]
        static=[]
        for a in self.static_actors:
            points=[self.local(v.x,v.y) for v in a.bounding_box.get_world_vertices(a.get_transform())]
            static.append([min(v[0] for v in points),min(v[1] for v in points),max(v[0] for v in points),max(v[1] for v in points)])
        # Verify the actual installed road, retaining the result for every case.
        misses=[]
        for x in ([] if self.native_layout else range(0,35,2)):
            for y in (.1,3.5,7.,10.5,13.9):
                if self.map.get_waypoint(self.loc(x,y),project_to_road=False,lane_type=carla.LaneType.Driving) is None:
                    misses.append([x,y])
        if misses:raise RuntimeError("Road rectangle does not match driving surface: "+str(misses))
        return {**self.read(),"reset_sensor_warmup_ticks":5,"reset_ticks":30+(1 if len(self.retired_ids)>0 else 0),"scene":{"name":scenario,"width":self.width,"height":self.height,"goal":self.goal,
                "static":static,"map":self.map.name,"origin":self.origin,"angle":self.angle,
                "road_surface_probe_misses":misses,"longitudinal_edges":"task boundaries", "condition":condition,
                "generator_version":"native_curve_v1" if self.native_layout else "seeded_geometry_v1",
                "native_layout":self.native_layout,"road_grid":self.road_grid,
                "boundary_scope":"shared_sampled_driving_grid_v1" if self.road_grid else ("sampled_measured_footprint_on_native_driving_surface" if self.native_layout else "measured_straight_road_rectangle"),
                "walker_control_speed":self.walker_speed,"reset_ticks":30+(1 if len(self.retired_ids)>0 else 0)}}

    def tick(self,require_sensor=True):
        before=self.world.get_snapshot().frame
        self.frame=self.world.tick(15.)
        if self.frame!=before+1:raise RuntimeError("Unexpected frame jump: another tick owner suspected")
        if require_sensor:
            end=time.monotonic()+3
            while self.imu_frame<self.frame and time.monotonic()<end:
                self.sensor_event.wait(.02);self.sensor_event.clear()
            if self.imu_frame<self.frame:raise RuntimeError("IMU frame heartbeat timeout")
        # Camera validation runs fence the actual render callback before the
        # next physics tick, preventing a fast synchronous client from outrunning
        # the renderer. Normal training has no camera and no additional wait.
        if self.recorder is not None:
            self.recorder.finish(self.frame,timeout=5.,timestamp=self.world.get_snapshot().timestamp.elapsed_seconds)

    def read(self):
        if self.recorder is not None and self.recorder.error:
            raise RuntimeError("Camera recording failed: " + self.recorder.error)
        if not self.collision_sensor.is_alive or not self.collision_sensor.is_listening:
            raise RuntimeError("Collision sensor unhealthy")
        tr=self.actor.get_transform();f=tr.get_forward_vector();v=self.actor.get_velocity();a=self.actor.get_acceleration()
        yaw=math.radians(tr.rotation.yaw);x,y=self.local(tr.location.x+self.rear*math.cos(yaw),tr.location.y+self.rear*math.sin(yaw))
        wheels=[self.actor.get_wheel_steer_angle(k) for k in (carla.VehicleWheelLocation.FL_Wheel,carla.VehicleWheelLocation.FR_Wheel)]
        delta=-math.atan(sum(math.tan(math.radians(w)) for w in wheels)/2)
        dynamic=[]
        for actor in self.dynamic:
            p=actor.get_location();vel=actor.get_velocity();xx,yy=self.local(p.x,p.y)
            dynamic.append({"x":xx,"y":yy,"vx":vel.x*self.co+vel.y*self.si,"vy":vel.x*self.si-vel.y*self.co,"radius":.5})
        state={"x":x,"y":y,"yaw":(math.radians(self.angle)-yaw+math.pi)%(2*math.pi)-math.pi,
               "v":v.x*f.x+v.y*f.y,"steering":delta,"acceleration":a.x*f.x+a.y*f.y,
               "time":self.world.get_snapshot().timestamp.elapsed_seconds-self.t0}
        road_probe=None
        if self.native_layout:
            misses=[];raw_misses=[]
            rear_x=tr.location.x+self.rear*math.cos(yaw);rear_y=tr.location.y+self.rear*math.sin(yaw)
            vehicle=self.cal["vehicle"]
            for forward in (-vehicle["rear"],0.,vehicle["front"]*.5,vehicle["front"]):
                for lateral in (-vehicle["width"]/2,0.,vehicle["width"]/2):
                    point=carla.Location(x=rear_x+forward*math.cos(yaw)-lateral*math.sin(yaw),
                                         y=rear_y+forward*math.sin(yaw)+lateral*math.cos(yaw),z=tr.location.z)
                    raw_inside=self.map.get_waypoint(point,project_to_road=False,lane_type=carla.LaneType.Driving) is not None
                    if not raw_inside:
                        raw_misses.append([point.x,point.y,point.z])
                    if self.road_geometry_version == "carla_driving_indexed_union_64_v3":
                        inside=self.road_surface_index.contains(point)
                    elif self.road_geometry_version == "carla_driving_refined_union_64_v2":
                        from navstack.lane_projection import on_driving_surface
                        inside=on_driving_surface(self.map,point,carla.LaneType.Driving)
                    else:
                        inside=raw_inside
                    if not inside:misses.append([point.x,point.y,point.z])
            road_probe={"sample_count":12,"offroad_points_world":misses,"raw_offroad_points_world":raw_misses,
                        "geometry_version":self.road_geometry_version,"scope":"sampled actual rear-axle footprint, not continuous swept-body proof"}
        return {"state":state,"frame":self.frame,"imu_frame":self.imu_frame,"road_surface_probe":road_probe,
                "world_elapsed_seconds":self.world.get_snapshot().timestamp.elapsed_seconds,"collisions":[e for e in self.events if e["episode_id"]==self.active_episode],
                "dynamic":dynamic,"command":self.command,"sensor_health":"frame_heartbeat_ok"}

    def step(self,r):
        if r.get("expected_frame")!=self.frame:raise RuntimeError("Client expected_frame mismatch")
        a=float(r["acceleration"]);delta=float(r["steering"])
        if not math.isfinite(a) or not math.isfinite(delta):raise ValueError("nonfinite command")
        a=max(-2.5,min(1.5,a));delta=max(-.5,min(.5,max(self.command["steering"]-.03,min(self.command["steering"]+.03,delta))))
        state=self.read()["state"];v=state["v"];g=self.cal["recommended_adapter"]
        # Desired-speed integrator is an actuator setpoint, never returned as vehicle state.
        self.speed_target=max(0.,min(3.,v+.5,max(v-.5,self.speed_target+a*.05)))
        from navstack.carla_actuation import longitudinal_control
        if self.acceleration_controller is not None:
            throttle,braking,actuation_mode=self.acceleration_controller.step(v,self.speed_target,a,state["acceleration"],g)
        else:
            throttle,braking,actuation_mode=longitudinal_control(v,self.speed_target,a,g)
        steer=max(-1.,min(1.,-delta/g["steer_gain"]))
        drivetrain=g.get("drivetrain_mode","automatic")
        if drivetrain not in ("automatic","manual_first_v1"):raise ValueError("Unknown drivetrain version")
        manual=drivetrain=="manual_first_v1"
        self.actor.apply_control(carla.VehicleControl(throttle=throttle,brake=braking,steer=steer,
                                 manual_gear_shift=manual,gear=1 if manual else 0))
        for actor in self.dynamic:
            # Same condition-dependent open-loop control; policy only sees current measured state.
            actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(self.si,-self.co,0),speed=self.walker_speed))
        self.command={"acceleration":a,"steering":delta};self.tick()
        applied=self.actor.get_control()
        return {**self.read(),"applied_vehicle_control":{"throttle":applied.throttle,"brake":applied.brake,"steer":applied.steer,
                "gear":applied.gear,"manual_gear_shift":applied.manual_gear_shift,"hand_brake":applied.hand_brake,"reverse":applied.reverse},"vehicle_control":{"throttle":throttle,"brake":braking,"steer":steer},"speed_setpoint":self.speed_target,"actuation_mode":actuation_mode,"actuation_state":self.acceleration_controller.state() if self.acceleration_controller else None}

    def handle(self,r):
        if r["op"]=="begin":return self.begin(r)
        if r["op"]=="step":return self.step(r)
        if r["op"]=="finalize":
            # No extra tick and no waiting for absent collision messages each frame.
            # Terminal fence drains delayed callbacks before any success is awarded.
            time.sleep(.3)
            if self.imu_frame<self.frame:raise RuntimeError("Missing terminal sensor heartbeat")
            if self.recorder is not None:self.recorder.finish(self.frame)
            return {**self.read(),"terminal_sensor_drain_seconds":.3}
        if r["op"]=="read":return self.read()
        raise ValueError("unknown operation")

    def close(self):
        try:
            self.clear()
            # This batch owns the server. Leave an empty, unticked, non-rendering world
            # between jobs to avoid repeated renderer/physics mode changes and idle GPU use.
            settings=self.world.get_settings();settings.synchronous_mode=True;settings.no_rendering_mode=True
            self.world.apply_settings(settings)
        finally:
            self.lease.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--calibration",required=True);p.add_argument("--lock",required=True);p.add_argument("--record-dir")
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=2000);args=p.parse_args()
    b=None
    try:
        b=Bridge(args.calibration,args.lock,args.record_dir,args.host,args.port)
        for line in sys.stdin:
            try:
                request=json.loads(line)
                response=b.protocol.execute(request,b.handle)
            except Exception as exc:response={"ok":False,"error":str(exc),"infrastructure_failure":True}
            if not response.get("ok"):
                response["bridge_diagnostic"]={"last_tick_frame":b.frame,"imu_frame":getattr(b,"imu_frame",None),
                    "active_episode":getattr(b,"active_episode",None),"owned_actor_ids":[a.id for a in b.owned]}
            print("CARLA_JSON:"+json.dumps(response),flush=True)
    except Exception as exc:
        print("CARLA_JSON:"+json.dumps({"ok":False,"infrastructure_failure":True,"error":repr(exc)}),flush=True)
    finally:
        if b:b.close()
