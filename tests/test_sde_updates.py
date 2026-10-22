import time
from navstack.config import EnvConfig,PlannerConfig
from navstack.research import cumulative_train


def test_continuous_sde_performs_real_optimizer_updates(tmp_path):
    config=EnvConfig(method="B1",curriculum=True,terminal_goal_prefix=True,deadline=45,planner=PlannerConfig(max_seconds=None))
    result=cumulative_train(tmp_path/"sde",config,11,128,time.time()+60,rollout_physics=64,epochs=2,
                            ppo_options={"use_sde":True,"sde_sample_freq":10,"log_std_init":-2,"target_kl":.03})
    assert result["optimizer_steps"]>0 and result["actor_l2_change"]>0 and result["reload_exact"]
