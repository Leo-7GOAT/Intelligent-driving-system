from navstack.config import EnvConfig,PlannerConfig
from navstack.env import NavigationEnv,rule_action
from navstack.imitation import replay_demonstration


def test_demonstration_commands_replay_in_actual_direct_plant():
    cfg=EnvConfig(method='B3',scenario='empty',obstacle_num=0,subgoal_mode='goal_frame_terminal',deadline=45,
        planner=PlannerConfig(max_seconds=None,analytic_expansion=True,vectorized_primitives=True))
    options={'diagnostic_route_length':6.}
    env=NavigationEnv(cfg);env.reset(seed=8101,options=options);ticks=[]
    while not env.done:
        _,_,_,_,info=env.step(rule_action(env));ticks.extend(info['trace'])
    assert info['success'];env.close()
    replay=replay_demonstration(cfg,8101,ticks,options)
    assert replay['metrics']['success'] and replay['max_state_difference']<1e-4
    assert len(replay['observations'])==len(ticks)==replay['metrics']['physics_steps']
    assert replay['observations'].dtype.name=='float32'
    assert replay['actions'].shape==(len(ticks),2)
