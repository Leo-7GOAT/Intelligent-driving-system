import argparse,json,sys,time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navstack.carla_bridge import BridgeClient,CarlaNavigationEnv
from navstack.config import EnvConfig,VehicleConfig,PlannerConfig
from navstack.experiments import append_json,write_json,new_run
from navstack.env import rule_action


def calibrated_config(batch,method="H",source_config=None):
    cal=json.loads((Path(batch)/"calibration/calibration.json").read_text(encoding="utf-8"))
    if source_config is None:
        return EnvConfig(method=method,deadline=45,vehicle=VehicleConfig(**cal["vehicle"]),planner=PlannerConfig(max_seconds=None))
    # Transfer preserves policy observation/action semantics and the frozen low
    # level. Only the explicitly calibrated vehicle geometry/actuation changes.
    config=EnvConfig.from_dict(source_config.to_dict())
    if config.method != method:raise ValueError("CARLA transfer method mismatch")
    config.vehicle=VehicleConfig(**cal["vehicle"])
    return config


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",required=True);p.add_argument("--scenario",default="empty")
    p.add_argument("--full-reference",action="store_true")
    p.add_argument("--collision-probe",action="store_true")
    args=p.parse_args();run=new_run(Path(args.batch)/"carla","diagnostic_"+args.scenario)
    client=BridgeClient(args.batch)
    try:
        env=CarlaNavigationEnv(calibrated_config(args.batch,"B1" if args.collision_probe else "H"),client)
        obs,_=env.reset(options={"case":{"scenario":args.scenario,"condition":0}})
        first=env.plant.frame;duplicate=client.exchange(client.last_request)
        assert first==duplicate["frame"]
        write_json(run/"scene.json",env.scene_evidence)
        if args.full_reference:
            from navstack.planning import HybridAStar
            from navstack.trajectory import parameterize,dynamic_safe
            planner=HybridAStar(env.config.vehicle,PlannerConfig(max_seconds=None,max_nodes=10000))
            plan=planner.plan((env.state.x,env.state.y,env.state.yaw),env.goal,env.world)
            write_json(run/"full_goal_plan.json",{"status":plan.status,"nodes":plan.expanded_nodes,"path":plan.path})
            if plan.status!="success":raise RuntimeError("Diagnostic full-goal planning failed: "+plan.status)
            reference=parameterize(plan.path,1.,env.state.v,env.config.vehicle,env.state.time,stop=True)
            def prepare(action):
                env.reference=reference
                return "reference",{"status":"diagnostic_fixed_full_goal","planning_time":plan.planning_time}
            env._prepare_reference=prepare
        duplicate_tick_verified=False
        while not env.done:
            action=[1.,0.] if args.collision_probe else (6 if args.scenario=="empty" or args.full_reference else rule_action(env))
            _,_,_,_,info=env.step(action)
            if not duplicate_tick_verified:
                previous=env.plant.frame;duplicate=client.exchange(client.last_request)
                assert duplicate["frame"]==previous
                current=client.request("read");assert current["frame"]==previous
                duplicate_tick_verified=True
            append_json(run/"transitions.jsonl",{"policy":"diagnostic_fixed" if args.scenario=="empty" else "diagnostic_rule",
                                                  "action":action,**info})
        write_json(run/"summary.json",{"policy":"diagnostic_not_PPO","duplicate_begin_did_not_tick":True,
                                        "duplicate_step_did_not_tick":duplicate_tick_verified,"full_reference":args.full_reference,
                                        **info["episode_metrics"],"final_state":asdict(env.state)})
        print(json.dumps({"run":str(run),"state":asdict(env.state),"reason":info["termination_reason"]}),flush=True)
    finally:client.close()
