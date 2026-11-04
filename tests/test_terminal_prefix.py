from navstack.config import EnvConfig,PlannerConfig
from navstack.env import NavigationEnv
from navstack.vehicle import VehiclePlant,VehicleState


def test_selected_goal_crossing_action_can_park_before_unused_endpoint_leaves_map():
    old=NavigationEnv(EnvConfig(scenario="empty",planner=PlannerConfig(max_seconds=None)))
    new=NavigationEnv(EnvConfig(scenario="empty",planner=PlannerConfig(max_seconds=None),terminal_goal_prefix=True))
    for env in (old,new):
        env.reset(seed=11);env.plant=VehiclePlant(env.config.vehicle,VehicleState(26.16,env.goal[1]))
    assert old._prepare_reference(12)[0]=="plan_failed"
    before=new.state
    mode,detail=new._prepare_reference(12)
    assert mode=="reference" and detail["terminal_goal_prefix"]
    assert new.state is before  # planning did not move the vehicle
    for _ in range(30):
        _,_,done,_,info=new.step(12)
        if done:break
    assert info["success"] and new.episode["distance"]>.1 and new.state.v<=.25


def test_non_goal_crossing_action_is_not_replaced_by_goal_directed_rule():
    env=NavigationEnv(EnvConfig(scenario="empty",terminal_goal_prefix=True,planner=PlannerConfig(max_seconds=None)))
    env.reset(seed=11)
    _,detail=env._prepare_reference(12)
    assert not detail["terminal_goal_prefix"]
