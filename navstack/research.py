"""Cumulative, resumable PPO runs with physical-step budgets and compressed raw evidence."""
from dataclasses import asdict
from contextlib import contextmanager
import copy,gzip,json,os,pickle,random,shutil,time,uuid
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from .config import EnvConfig
from .env import NavigationEnv
from .experiments import actor_vector,append_json,write_json,manifest,signature,jsonable
from .mpc import MPC


@contextmanager
def preserve_global_rng():
    """Reload verification must not reseed an in-progress CPU training process."""
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
    try:yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)


def atomic_json(path, data):
    path=Path(path);tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    write_json(tmp,data)
    for attempt in range(20):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.01)


def resources_ok(root):
    free=shutil.disk_usage(root).free
    if free<8*1024**3:return False,"disk_free_below_8GiB"
    if os.name=="nt":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
                ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
        memory=Memory();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        if memory.avail_phys<4*1024**3:return False,"memory_free_below_4GiB"
    return True,None


class ResearchRecorder(BaseCallback):
    def __init__(self,run,deadline,counts=None):
        super().__init__();self.run=Path(run);self.deadline=deadline
        self.counts=counts or {"physics_steps":0,"actions":0,"episodes":0,"optimization_phases":0,
                               "optimizer_steps":0,"reward_components":{},"modes":{},"action_histogram":{}}
        self.stopped_reason=None;self.last_health=0
        self.stream=gzip.open(self.run/f"transitions_{self.counts['physics_steps']:09d}_{time.time_ns()}.jsonl.gz","at",encoding="utf-8")

    def _on_step(self):
        info=self.locals["infos"][0];c=self.counts
        c["physics_steps"]+=info["executed_substeps"];c["actions"]+=1
        c["modes"][info["mode"]]=c["modes"].get(info["mode"],0)+1
        action=self.locals["actions"][0]
        if np.ndim(action)==0:
            key=str(int(action));c["action_histogram"][key]=c["action_histogram"].get(key,0)+1
        for tick in info["trace"]:
            for key,value in tick["reward_components"].items():
                c["reward_components"][key]=c["reward_components"].get(key,0)+value
        row={"physics_steps":c["physics_steps"],"episode_id":c["episodes"],"policy_action":action,
             "mode":info["mode"],"plan":info["plan"],"trace":info["trace"],"termination_reason":info["termination_reason"]}
        row.update({key:info.get(key) for key in ("execution_feedback_seconds","environment_decision_seconds")})
        self.stream.write(json.dumps(row,default=jsonable,allow_nan=False)+"\n")
        if "episode_metrics" in info:
            append_json(self.run/"episodes.jsonl",{"episode_id":c["episodes"],"at_physics_steps":c["physics_steps"],
                        "initial_scenario":info["initial_scenario"],**info["episode_metrics"]})
            c["episodes"]+=1
        now=time.time()
        if now-self.last_health>5:
            self.stream.flush();self.last_health=now
            ok,reason=resources_ok(self.run)
            if not ok:self.stopped_reason=reason
            atomic_json(self.run/"heartbeat.json",{"epoch":now,"pid":os.getpid(),"physics_steps":c["physics_steps"],
                        "actions":c["actions"],"optimization_phases":c["optimization_phases"]})
        if now>=self.deadline:self.stopped_reason="wall_budget_reached"
        if (self.run.parent.parent/"STOP").exists():self.stopped_reason="watchdog_or_user_stop"
        # Finish the bounded rollout so the observation and optimizer checkpoint agree.
        # The queue reserves 60 seconds before the hard batch deadline for this drain.
        return True

    def close(self):self.stream.close()


def save_checkpoint(run,model,vec,recorder,initial_actor,config,seed,reason):
    counts=recorder.counts
    folder=run/f"checkpoint_{counts['physics_steps']:09d}_{counts['optimization_phases']:05d}"
    # Save immutable checkpoint directories, publish latest only after all files exist.
    if folder.exists():folder=run/(folder.name+"_"+str(time.time_ns()))
    folder.mkdir()
    model.save(folder/"model.zip");vec.save(folder/"normalization.pkl")
    env=vec.venv.envs[0]
    state={k:v for k,v in env.__dict__.items() if k not in ("controller","planner")}
    snapshot={"env":state,"last_obs":model._last_obs,"last_episode_starts":model._last_episode_starts,
              "last_original_obs":getattr(model,"_last_original_obs",None),"random":random.getstate(),
              "numpy_random":np.random.get_state(),"torch_random":torch.get_rng_state(),"initial_actor":initial_actor,
              "counts":counts,"vec_returns":vec.returns.copy()}
    with (folder/"resume.pkl").open("wb") as stream:pickle.dump(snapshot,stream)
    write_json(folder/"config.json",{"env":config.to_dict(),"signature":signature(config),"seed":seed,
                                     "resume_note":"Physical state/RNG/optimizer/normalization retained; OSQP workspace warm start recreated"})
    with preserve_global_rng():
        reloaded=PPO.load(folder/"model.zip",device="cpu")
    exact=all(torch.equal(v,reloaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
    summary={**counts,"seed":seed,"method":config.method,"backend":"lightweight","flat_hold":config.flat_hold,
             "shared_shield":config.shared_shield,"ppo_epoch_updates":model._n_updates,
             "simulation_seconds":counts["physics_steps"]*config.control_dt,"saved_epoch":time.time(),
             "reason":reason,"reload_exact":exact,"actor_l2_change":float(torch.linalg.vector_norm(actor_vector(model)-initial_actor)),
             "checkpoint":str(folder.resolve()),"task_deadline_seconds":config.deadline}
    write_json(folder/"training_summary.json",summary)
    atomic_json(run/"latest.json",summary)
    return summary


def cumulative_train(run,config,seed,target,deadline,rollout_physics=2560,epochs=10,ppo_options=None,resume_learning_rate=None):
    torch.set_num_threads(1);run=Path(run);run.mkdir(parents=True,exist_ok=True)
    ppo_options=ppo_options or {}
    steps=max(2,rollout_physics//config.substeps)
    raw=DummyVecEnv([lambda:NavigationEnv(config)])
    if (run/"latest.json").exists():
        if (run/"ppo_options.json").exists() and json.loads((run/"ppo_options.json").read_text(encoding="utf-8")) != ppo_options:
            raise ValueError("Resume PPO options differ; use a new versioned run")
        previous=json.loads((run/"latest.json").read_text(encoding="utf-8"));checkpoint=Path(previous["checkpoint"])
        saved=json.loads((checkpoint/"config.json").read_text(encoding="utf-8"))
        if saved["env"]!=config.to_dict() or saved["seed"]!=seed:raise ValueError("Resume configuration/seed mismatch")
        vec=VecNormalize.load(checkpoint/"normalization.pkl",raw);vec.training=True
        model=PPO.load(checkpoint/"model.zip",env=vec,device="cpu",force_reset=False)
        if resume_learning_rate is not None:
            if not np.isfinite(resume_learning_rate) or not 0 < resume_learning_rate <= .01:
                raise ValueError("Invalid explicit resume learning rate")
            from stable_baselines3.common.utils import get_schedule_fn
            old_rate=float(model.lr_schedule(1.0))
            model.learning_rate=float(resume_learning_rate)
            model.lr_schedule=get_schedule_fn(float(resume_learning_rate))
            if old_rate != resume_learning_rate:
                append_json(run/"learning_rate_interventions.jsonl",{
                    "checkpoint":checkpoint.name,"previous":old_rate,"new":resume_learning_rate,
                    "scope":"Explicit development branch; weights/optimizer/RNG/env restored before further on-policy training"})
        with (checkpoint/"resume.pkl").open("rb") as stream:snapshot=pickle.load(stream)
        raw.envs[0].__dict__.update(snapshot["env"])
        raw.envs[0].controller=MPC(config.vehicle,config.control_dt,config.mpc_horizon)
        model._last_obs=snapshot["last_obs"];model._last_episode_starts=snapshot["last_episode_starts"]
        model._last_original_obs=snapshot["last_original_obs"];vec.returns=snapshot["vec_returns"]
        random.setstate(snapshot["random"]);np.random.set_state(snapshot["numpy_random"]);torch.set_rng_state(snapshot["torch_random"])
        initial=snapshot["initial_actor"];counts=snapshot["counts"]
    else:
        from .candidate_policy import CandidateVecNormalize, CandidateActorCriticPolicy, FlatCandidateActorCriticPolicy
        policy_name = ppo_options.get("policy", "mlp")
        if policy_name not in ("mlp", "candidate_shared", "flat_candidate_pool", "bounded_mean"):
            raise ValueError("Unknown PPO policy architecture")
        if policy_name == "candidate_shared" and (not config.candidate_observation or config.method == "B1"):
            raise ValueError("Shared candidate policy requires candidate observations and discrete actions")
        if policy_name == "flat_candidate_pool" and (not config.candidate_observation or config.method != "B1"):
            raise ValueError("Pooled flat policy requires candidate observations and direct B1 controls")
        if policy_name == "bounded_mean" and config.method != "B1":
            raise ValueError("Bounded-mean policy requires direct B1 controls")
        normalizer = CandidateVecNormalize if config.candidate_observation else VecNormalize
        raw.seed(seed);vec=normalizer(raw,norm_obs=True,norm_reward=False,gamma=config.gamma,clip_obs=10)
        from .bounded_policy import BoundedMeanActorCriticPolicy
        policy_class = {"mlp": "MlpPolicy", "candidate_shared": CandidateActorCriticPolicy, "flat_candidate_pool": FlatCandidateActorCriticPolicy,
                        "bounded_mean": BoundedMeanActorCriticPolicy}[policy_name]
        model=PPO(policy_class,vec,n_steps=steps,batch_size=min(64,steps),n_epochs=epochs,learning_rate=ppo_options.get("learning_rate",3e-4),
                  gamma=config.gamma,gae_lambda=.95**(config.substeps*config.control_dt),ent_coef=ppo_options.get("entropy",.01),
                  use_sde=ppo_options.get("use_sde",False),sde_sample_freq=ppo_options.get("sde_sample_freq",-1),
                  target_kl=ppo_options.get("target_kl"),
                  policy_kwargs={"net_arch":{"pi":[64,64],"vf":[64,64]},"log_std_init":ppo_options.get("log_std_init",0)},seed=seed,device="cpu")
        if config.method=="B1" and ppo_options.get("forward_mean_bias"):
            with torch.no_grad():model.policy.action_net.bias[0]=ppo_options["forward_mean_bias"]
        initial=actor_vector(model).clone();counts=None
        write_json(run/"version.json",manifest())
        write_json(run/"resolved_config.json",{"env":config.to_dict(),"signature":signature(config),"seed":seed,
                   "rollout_physics_nominal":rollout_physics,"rollout_actions":steps,"epochs":epochs,
                   "gamma":config.gamma,"gae_lambda":model.gae_lambda,"backend":"lightweight",
                   "deadline_epoch":deadline,"network_parameters":sum(p.numel() for p in model.policy.parameters())})
        write_json(run/"ppo_options.json",ppo_options)
    model.set_logger(configure(str(run/"sb3"/str(time.time_ns())),["csv"]))
    recorder=ResearchRecorder(run,deadline,counts);started=time.time();last_save=recorder.counts["physics_steps"]
    recorder.counts.setdefault("optimizer_steps",0)
    def count_optimizer_step(*unused):recorder.counts["optimizer_steps"]+=1
    optimizer_hook=model.policy.optimizer.register_step_post_hook(count_optimizer_step)
    try:
        while recorder.counts["physics_steps"]<target and not recorder.stopped_reason:
            old=model._n_updates
            actor_before=actor_vector(model).clone()
            model.learn(steps,callback=recorder,reset_num_timesteps=False)
            if model._n_updates>old:
                recorder.counts["optimization_phases"]+=1
                row={"physics_steps":recorder.counts["physics_steps"],"actions":recorder.counts["actions"],
                     "simulation_seconds":recorder.counts["physics_steps"]*.05,"phase":recorder.counts["optimization_phases"],
                     "ppo_epoch_updates":model._n_updates,"epoch":time.time(),"session_wall_seconds":time.time()-started}
                row.update({"optimizer_steps":recorder.counts["optimizer_steps"],
                            "actor_update_l2":float(torch.linalg.vector_norm(actor_vector(model)-actor_before))})
                row.update({k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith("train/") and np.isscalar(v) and np.isfinite(v)})
                append_json(run/"optimization.jsonl",row)
                print(json.dumps({"run":str(run),**row}),flush=True)
            if recorder.counts["physics_steps"]-last_save>=10000 or recorder.counts["optimization_phases"]==1:
                save_checkpoint(run,model,vec,recorder,initial,config,seed,"periodic")
                last_save=recorder.counts["physics_steps"]
        return save_checkpoint(run,model,vec,recorder,initial,config,seed,recorder.stopped_reason or "milestone_reached")
    except Exception as exc:
        append_json(run/"failures.jsonl",{"epoch":time.time(),"error":repr(exc),"counts":recorder.counts})
        # An exception may interrupt collect_rollouts between env.step and _last_obs.
        # Keep the last complete checkpoint; raw partial interactions remain in the logs.
        raise
    finally:optimizer_hook.remove();recorder.close();vec.close()
