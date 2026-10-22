import json,time,pickle
from navstack.config import EnvConfig,PlannerConfig
from navstack.research import cumulative_train


def test_optimizer_rng_and_physics_resume_cumulatively(tmp_path):
    tmp_path = tmp_path / "中文路径"
    tmp_path.mkdir()
    c=EnvConfig(method="B1",scenario="empty",deadline=45,planner=PlannerConfig(max_seconds=None))
    a=cumulative_train(tmp_path/"run",c,11,128,time.time()+60,rollout_physics=64,epochs=2)
    b=cumulative_train(tmp_path/"run",c,11,256,time.time()+60,rollout_physics=64,epochs=2)
    assert a["physics_steps"]==128 and b["physics_steps"]==256
    assert a["optimization_phases"]==2 and b["optimization_phases"]==4
    assert b["ppo_epoch_updates"]==8 and b["reload_exact"]
    assert b["actor_l2_change"]>a["actor_l2_change"]>0
    with open(b["checkpoint"]+"/resume.pkl","rb") as stream:r=pickle.load(stream)
    assert r["env"]["total_physics_steps"]==256
