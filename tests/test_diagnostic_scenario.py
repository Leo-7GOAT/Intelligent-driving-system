from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from scripts.audit_policy_inference import explicit_diagnostic_scenario


def test_requested_static_probe_does_not_become_empty_curriculum():
    cfg=EnvConfig(method='B1',scenario='mixed',split='train',obstacle_num=0,
        route_stages=[{'until_physics_steps':1000000,'route_m':6.}],route_rehearsal_probability=.25)
    cfg=explicit_diagnostic_scenario(cfg,'static')
    env=NavigationEnv(cfg)
    env.reset(seed=7301,options={'diagnostic_route_length':23.})
    assert env.initial_scenario['name']=='static'
    assert len(env.world.static)==1 and env.goal[0]==27.
    env.close()
