"""Bounded real CARLA training for the new shared-road interface (development)."""
import argparse
import gzip
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from navstack.carla_bridge import BridgeClient, CarlaNavigationEnv
from navstack.config import EnvConfig, VehicleConfig, RewardConfig
from navstack.experiments import actor_vector, write_json, signature, append_json, jsonable
from navstack.native_route import select_routes
from navstack.research import ResearchRecorder, resources_ok
from navstack.research import atomic_json
from navstack.independent_clip_ppo import IndependentClipPPO, load_ppo, GLOBAL_CLIP, INDEPENDENT_CLIP, actor_critic_parameters
from scripts.carla_steering_audit import guard


def prepare(parent, road_geometry_version="carla_driving_64_v1", controlled_budget=640, wall_seconds=900, episode_seconds=20, evaluate=False, feedback_ablation=False, road_candidate_policy=False):
    if not 640<=controlled_budget<=100000 or not 900<=wall_seconds<=7200 or not 20<=episode_seconds<=90:
        raise ValueError("Native development budget outside supported bounds")
    if road_candidate_policy and not feedback_ablation:
        raise ValueError('Structured road policy requires the declared H0/H/B1 feedback comparison')
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError("Resource guard: "+str(reason))
    parent = Path(parent)
    batch = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_native_road_learning")
    batch.mkdir()
    for name in ("infrastructure", "calibration"):
        (batch/name).mkdir()
    shutil.copy2(parent/"calibration/calibration.json", batch/"calibration/calibration.json")
    survey = json.loads((parent/"native_routes.json").read_text(encoding="utf-8"))
    hashes = {}
    for folder in ("navstack", "scripts", "tests"):
        for source in (ROOT/folder).rglob("*.py"):
            relative = source.relative_to(ROOT)
            target = batch/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('start_carla.ps1','stop_owned_carla.ps1'):
        relative=Path('scripts')/name
        shutil.copy2(ROOT/relative,batch/'source'/relative)
        hashes[relative.as_posix()]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    write_json(batch/"protocol.json", {"scope": "Shared native-road real PPO interface training, not convergence or hierarchy advantage",
        "created_epoch": time.time(), "deadline_epoch": time.time()+wall_seconds, "source_hashes": hashes,
        "road_geometry_version":road_geometry_version,
        "deterministic_training_route_evaluation":evaluate,
        "feedback_ablation":feedback_ablation,
        "road_candidate_policy":road_candidate_policy,
        "cleanup_owned_carla":road_candidate_policy,
        "representation_note": "4701shared inputs: existing candidate geometry plus road raster; learned CNN and shared candidate scores/pooling, fixed binary road/candidate scaling. New incompatible models, no masks or route prior." if road_candidate_policy else "Historical flat observation/MLP representation",
        "hypothesis": "Same-size H0/H feedback ablation and equally informed direct B1; unchanged action catalog, reward, controller and PPO settings" if feedback_ablation else "Shared native-road interface development",
        "calibration_sha256": hashlib.sha256((batch/"calibration/calibration.json").read_bytes()).hexdigest(),
        "routes": select_routes(survey), "methods": ["H0", "H", "B1"] if feedback_ablation else ["H", "B1"], "seed": 11,
        "checkpoint_thresholds": sorted({min(controlled_budget, step) for step in (10000, 30000, controlled_budget)}),
        "controlled_budget_per_method": controlled_budget, "reset_ticks_counted_separately": True,
        "rollout_actions": 32, "ppo_epochs": 2, "episode_seconds": episode_seconds,
        "resource_limits": {"wall_seconds": wall_seconds, "disk_min_GiB": 12, "ram_min_GiB": 4, "gpu_stop_temperature_C": 85, "automatic_retries": 0},
        "disk_free_bytes": shutil.disk_usage(ROOT).free})
    write_json(ROOT/"runs/native_road_learning.json", {"batch": str(batch)})
    print(batch, flush=True)


def training_config(protocol, method, vehicle):
    planner_nodes=protocol.get('planner_nodes',1800)
    if type(planner_nodes) is not int or planner_nodes not in (1800,7200):
        raise ValueError('Native training supports the registered1800/7200node budgets')
    feedback = ("ablated" if method == "H0" else "enabled") if protocol.get("feedback_ablation", False) else "off"
    if protocol.get('continuation_feedback', False):
        feedback = 'continuation_ablated' if method == 'H0' else 'continuation'
    config = EnvConfig(method="H" if method == "H0" else method, vehicle=vehicle, road_observation=True,
        execution_feedback=feedback,candidate_observation=protocol.get('road_candidate_policy',False),
        road_geometry_version=protocol.get("road_geometry_version","carla_driving_64_v1"),
        subgoal_mode=protocol.get('subgoal_mode', 'ego_arcs'),
        actuator_observation=protocol.get('actuator_observation', 'off'),
        local_candidate_filter=protocol.get('candidate_head')=='unshared_local_v1',
        road_collision_backend=protocol.get('road_collision_backend', 'scalar'),
        flat_hold=10, deadline=protocol["episode_seconds"], obstacle_num=0,
        flat_longitudinal_mode=protocol.get('flat_longitudinal_mode','acceleration') if method=='B1' else 'acceleration')
    if method=='B1' and protocol.get('flat_action_catalog','continuous')!='continuous':
        config.flat_action_catalog=protocol['flat_action_catalog'];config.__post_init__()
    config.planner.max_seconds=None
    config.planner.max_nodes=planner_nodes
    heuristic=protocol.get('planner_heuristic','euclidean')
    if heuristic not in ('euclidean','dubins_v1'):raise ValueError('Unknown native planner heuristic')
    config.planner.heuristic_mode=heuristic
    config.planner.vectorized_primitives=True
    config.planner.analytic_expansion=True
    # An explicit new task reward applies equally to all methods. Historical
    # protocols keep the original potential-only progress signal unchanged.
    if 'task_progress_weight' in protocol:
        weight=protocol['task_progress_weight']
        if isinstance(weight,bool) or not isinstance(weight,(int,float)):
            raise ValueError('Task progress weight must be an explicit finite number')
        config.reward=RewardConfig(**{**asdict(config.reward),'progress_weight':weight})
    return config


class RecordedRoadEnv(CarlaNavigationEnv):
    def __init__(self, config, client, cases, output):
        self.output = output
        super().__init__(config, client, cases)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        write_json(self.output/("scene_%04d.json" % self.reset_count), {"scene": self.scene_evidence,
            "initial": self.initial_scenario, "frame": self.plant.frame,
            "observation_shape": list(result[0].shape)})
        return result


def restore_training(checkpoint, raw, config, seed, gradient_clip_mode=None, allow_intervention=False,
                     rollout_actions=None, allow_rollout_intervention=False):
    """Restore learning state, deliberately begin a new measured CARLA episode."""
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint/'metadata.json').read_text(encoding='utf-8'))
    metadata = json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    if metadata['signature'] != signature(config):
        raise ValueError('Recovery configuration differs from saved policy semantics')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() != saved[key]:
            raise ValueError('Recovery checkpoint hash mismatch')
    state_file=checkpoint/'resume_state.pkl'
    expected_rng=saved.get('resume_state_sha256')
    if expected_rng is not None:
        if not state_file.is_file() or hashlib.sha256(state_file.read_bytes()).hexdigest()!=expected_rng:
            raise ValueError('Recovery RNG hash mismatch or recorded state missing')
    elif state_file.exists():
        raise ValueError('Recovery RNG file has no recorded hash')
    vec = VecNormalize.load(checkpoint/'normalization.pkl',raw)
    vec.training=True;vec.norm_reward=False;vec.returns[:]=0
    model = load_ppo(checkpoint/'model.zip',env=vec,device='cpu',force_reset=True,
                     gradient_clip_mode=gradient_clip_mode,allow_intervention=allow_intervention,
                     rollout_actions=rollout_actions,allow_rollout_intervention=allow_rollout_intervention)
    if state_file.exists():
        with state_file.open('rb') as stream:state=pickle.load(stream)
        random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
        raw.envs[0].np_random.bit_generator.state=state['env_rng']
        raw.envs[0].case_index=state['next_case_index']
        raw._seeds=[None]
        mode='saved RNG; interrupted physical episode discarded, next route reset'
    else:
        # Legacy inference checkpoints lack RNG and physical state. Account for
        # their lost work explicitly; never claim an uninterrupted continuation.
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);raw.seed(seed)
        raw.envs[0].case_index=saved['episodes']+1
        mode='legacy checkpoint: explicit fresh RNG seed and new measured episode'
    return model,vec,mode


def save_training_checkpoint(output, threshold, model, vec, raw, metadata):
    """Publish a checkpoint only after all files and their hashes exist."""
    checkpoint=output/('checkpoint_%09d'%threshold)
    temporary=output/(checkpoint.name+'.partial')
    temporary.mkdir()
    model.save(temporary/'model.zip');vec.save(temporary/'normalization.pkl')
    shutil.copy2(output/'config.json',temporary/'config.json')
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
           'env_rng':raw.envs[0].np_random.bit_generator.state,'next_case_index':raw.envs[0].case_index}
    with (temporary/'resume_state.pkl').open('wb') as stream:pickle.dump(state,stream)
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        metadata[key]=hashlib.sha256((temporary/name).read_bytes()).hexdigest()
    metadata['note']='Learning state retained; recovery starts a new CARLA episode and counts all reset/lost interactions'
    write_json(temporary/'metadata.json',metadata)
    os.rename(temporary,checkpoint)
    atomic_json(output/'latest_checkpoint.json',{'folder':checkpoint.name,**metadata})


def registered_jobs(protocol):
    """Legacy method lists retain behavior; experimental paired arms are explicit."""
    jobs = protocol.get('jobs')
    if jobs is None:
        return [{'name': method, 'method': method} for method in protocol['methods']]
    clipping=bool(protocol.get('gradient_clipping_experiment'))
    rollout=bool(protocol.get('rollout_length_experiment'))
    distribution=bool(protocol.get('flat_distribution_experiment'))
    head=bool(protocol.get('actor_head_experiment'))
    if sum((clipping,rollout,distribution,head)) != 1 or not jobs:
        raise ValueError('Named native jobs require exactly one registered learning intervention')
    names = [job['name'] for job in jobs]
    if names != protocol['methods'] or len(set(names)) != len(names):
        raise ValueError('Native job names must uniquely match worker scopes')
    for job in jobs:
        if not job['name'].replace('_', '').isalnum() or job['method'] not in ('H0','H','B1'):
            raise ValueError('Invalid native experiment job')
        if job.get('gradient_clip_mode') not in (GLOBAL_CLIP, INDEPENDENT_CLIP):
            raise ValueError('Every experiment arm must declare its gradient clipping mode')
        if distribution and (job['method'] != 'B1' or
                job['gradient_clip_mode'] != GLOBAL_CLIP or
                job.get('flat_action_distribution') not in ('clipped_gaussian','tanh_gaussian_v1') or
                'rollout_actions' in job or protocol.get('recovery')):
            raise ValueError('Distribution pair requires fresh flat policies and unchanged common PPO schedule')
        if not distribution and 'flat_action_distribution' in job:
            raise ValueError('Action distribution changes require their own registered experiment')
        if head and (job['method'] not in ('H0','H') or
                job['gradient_clip_mode'] != GLOBAL_CLIP or
                job.get('candidate_head') not in ('shared','unshared_v1') or
                'rollout_actions' in job or protocol.get('recovery')):
            raise ValueError('Head comparison requires fresh hierarchical policies and the common PPO schedule')
        if not head and 'candidate_head' in job:
            raise ValueError('Candidate head changes require their own experiment')
        if rollout and (job['gradient_clip_mode'] != GLOBAL_CLIP or
                        type(job.get('rollout_actions')) is not int or job['rollout_actions'] not in (32,256)):
            raise ValueError('Rollout study must retain global clipping and declare 32 or 256 actions')
        if clipping and 'rollout_actions' in job:
            raise ValueError('Clipping study cannot change rollout length')
    return jobs


def run(batch):
    batch = Path(batch).resolve()
    protocol = json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    jobs = registered_jobs(protocol)
    for relative, digest in protocol["source_hashes"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Run frozen source only")
    if hashlib.sha256((batch/"calibration/calibration.json").read_bytes()).hexdigest() != protocol["calibration_sha256"]:
        raise ValueError("Calibration mismatch")
    if (batch/"started.json").exists():
        raise RuntimeError("No automatic repeat of completed or failed evidence")
    guard(batch, protocol)
    write_json(batch/"started.json", {"epoch": time.time()})
    vehicle = VehicleConfig(**json.loads((batch/"calibration/calibration.json").read_text(encoding="utf-8"))["vehicle"])
    torch.set_num_threads(1)
    results = []
    try:
        for job in jobs:
            method, job_name = job['method'], job['name']
            rollout_actions=job.get('rollout_actions',protocol['rollout_actions'])
            guard(batch, protocol)
            if time.time() >= protocol["deadline_epoch"]:
                raise RuntimeError("Registered wall budget expired")
            config=training_config(protocol,method,vehicle)
            output = batch/job_name
            output.mkdir()
            write_json(output/"config.json", {"env": config.to_dict(), "signature": signature(config),
                "seed": protocol["seed"], "backend": "CARLA_0.9.15", "scope": protocol["scope"]})
            cases = [{"scenario": "native_curve", "condition": index, "native_route": route} for index, route in enumerate(protocol["routes"])]
            client = BridgeClient(batch)
            vec = None
            recorder = None
            try:
                raw = DummyVecEnv([lambda: RecordedRoadEnv(config, client, cases, output)])
                raw.seed(protocol["seed"])
                recovery=protocol.get('recovery',{}).get(job_name)
                if recovery:
                    model,vec,resume_mode=restore_training(batch/recovery['checkpoint'],raw,config,protocol['seed'],
                        gradient_clip_mode=job.get('gradient_clip_mode'),
                        allow_intervention=bool(protocol.get('gradient_clipping_experiment')),
                        rollout_actions=rollout_actions,
                        allow_rollout_intervention=bool(protocol.get('rollout_length_experiment')))
                    write_json(output/'resume_event.json',{'mode':resume_mode,**recovery,
                        'gradient_clip_mode':getattr(model,'gradient_clip_mode',GLOBAL_CLIP),
                        'rollout_actions':model.n_steps,
                        'registered_rollout_intervention':bool(protocol.get('rollout_length_experiment')),
                        'registered_gradient_intervention':bool(protocol.get('gradient_clipping_experiment'))})
                else:
                    normalizer,policy=VecNormalize,"MlpPolicy"
                    if protocol.get('road_candidate_policy',False):
                        from navstack.road_candidate_policy import RoadCandidateVecNormalize,RoadCandidateActorCriticPolicy,FlatRoadCandidateActorCriticPolicy
                        normalizer=RoadCandidateVecNormalize
                        policy=FlatRoadCandidateActorCriticPolicy if method=='B1' else RoadCandidateActorCriticPolicy
                        if method=='B1' and getattr(config,'flat_action_catalog','continuous')=='speed_steer_3x7_v1':
                            from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
                            policy=DiscreteFlatRoadPolicy
                        if job.get('flat_action_distribution') == 'tanh_gaussian_v1':
                            from navstack.squashed_road_policy import SquashedFlatRoadPolicy
                            policy=SquashedFlatRoadPolicy
                        selected_head=job.get('candidate_head',protocol.get('candidate_head'))
                        if selected_head == 'unshared_v1':
                            from navstack.road_candidate_policy import UnsharedRoadCandidatePolicy
                            policy=UnsharedRoadCandidatePolicy
                        elif selected_head == 'unshared_local_v1':
                            from navstack.road_candidate_policy import LocalWitnessRoadPolicy
                            policy=LocalWitnessRoadPolicy
                    vec = normalizer(raw, norm_reward=False, gamma=config.gamma)
                    algorithm = IndependentClipPPO if job.get('gradient_clip_mode') == INDEPENDENT_CLIP else PPO
                    model = algorithm(policy, vec, n_steps=rollout_actions, batch_size=32,
                        n_epochs=protocol["ppo_epochs"], gamma=config.gamma, gae_lambda=.95**.5,
                        policy_kwargs={"net_arch":{"pi":[64,64],"vf":[64,64]}}, seed=protocol["seed"], device="cpu")
                if model.n_steps != rollout_actions or model.n_epochs != protocol['ppo_epochs']:
                    raise ValueError('Restored learner differs from registered rollout/epoch schedule')
                write_json(output/'policy_architecture.json',{'policy':type(model.policy).__name__,
                    'rollout_actions':model.n_steps,'ppo_epochs':model.n_epochs,'batch_size':model.batch_size,
                    'algorithm':type(model).__name__,'gradient_clip_mode':getattr(model,'gradient_clip_mode',GLOBAL_CLIP),
                    'normalizer':type(vec).__name__,'parameters':sum(p.numel() for p in model.policy.parameters()),
                    'observation_shape':list(model.observation_space.shape),
                    'action_shape':list(model.action_space.shape),'road_candidate_policy':protocol.get('road_candidate_policy',False)})
                inherited_lineage=0 if recovery is None else recovery['lineage_controls']
                inherited_cost=0 if recovery is None else recovery['consumed_controls_lower_bound']
                budget=protocol.get('method_budgets',{}).get(job_name,protocol['controlled_budget_per_method'])
                evaluation_only=bool(protocol.get('evaluation_only'))
                if evaluation_only and (budget!=0 or recovery is None):
                    raise ValueError('Evaluation-only phase requires zero budget and a verified saved checkpoint')
                model.set_logger(configure(str(output/"sb3"), ["csv"]))
                before = actor_vector(model).clone()
                recorder = ResearchRecorder(output, protocol["deadline_epoch"])
                optimizer_steps = [0]
                gradient_samples = []
                gradient_groups = actor_critic_parameters(model.policy) if protocol.get('gradient_clipping_experiment') else None
                def count_step(*unused):
                    optimizer_steps[0] += 1
                    if gradient_groups is not None:
                        # Adam leaves .grad in place until the next zero_grad.
                        # Both arms use the same post-step measurement, after
                        # their respective clipping, without touching gradients.
                        gradient_samples.append([float(torch.sqrt(sum(
                            (p.grad.detach().square().sum() for p in group if p.grad is not None),
                            torch.zeros((),device=model.device)))) for group in gradient_groups])
                hook = model.policy.optimizer.register_step_post_hook(count_step)
                started = time.time()
                saved_thresholds = set()
                while recorder.counts["physics_steps"] < budget and not recorder.stopped_reason:
                    guard(batch, protocol)
                    previous = model._n_updates
                    gradient_samples.clear()
                    model.learn(rollout_actions, callback=recorder, reset_num_timesteps=False)
                    if model._n_updates > previous:
                        recorder.counts["optimization_phases"] += 1
                        gradient_record = {}
                        if gradient_samples:
                            gradient_array=np.asarray(gradient_samples)
                            gradient_record={'postclip_gradients':{
                                'optimizer_steps_sampled':len(gradient_samples),
                                'actor_l2_mean':float(gradient_array[:,0].mean()),
                                'critic_l2_mean':float(gradient_array[:,1].mean()),
                                'scope':'Measured after clipping and Adam step, before next gradient reset'}}
                        append_json(output/"optimization.jsonl", {**recorder.counts, "optimizer_steps": optimizer_steps[0],
                            **gradient_record,
                            "actor_l2_change": float(torch.linalg.vector_norm(actor_vector(model)-before)),
                            "metrics": {k: float(v) for k,v in model.logger.name_to_value.items() if np.isscalar(v) and np.isfinite(v)}})
                    thresholds=sorted(set(protocol.get('checkpoint_thresholds',[]))|set(range(10000,budget+1,10000))|{budget})
                    for threshold in thresholds:
                        if threshold not in saved_thresholds and recorder.counts["physics_steps"] >= threshold:
                            save_training_checkpoint(output,threshold,model,vec,raw,{"threshold": threshold, **recorder.counts,
                                "optimizer_steps":optimizer_steps[0], "world_ticks":client.total_ticks,
                                'lineage_control_steps':inherited_lineage+recorder.counts['physics_steps'],
                                'charged_control_steps_lower_bound':inherited_cost+recorder.counts['physics_steps']})
                            saved_thresholds.add(threshold)
                hook.remove()
                # A completed training trace must survive a later evaluation
                # interruption. Never leave its gzip footer open during driving.
                recorder.close()
                if evaluation_only:
                    origin=batch/recovery['checkpoint']
                    for name in ('model.zip','normalization.pkl'):
                        shutil.copy2(origin/name,output/name)
                    (output/'optimization.jsonl').touch(exist_ok=False)
                    checkpoint=output/'checkpoint_000000000';checkpoint.mkdir()
                    for name in ('model.zip','normalization.pkl','resume_state.pkl'):
                        shutil.copy2(origin/name,checkpoint/name)
                    shutil.copy2(output/'config.json',checkpoint/'config.json')
                    metadata={**recorder.counts,'optimizer_steps':0,'world_ticks':0,
                        'lineage_control_steps':inherited_lineage,'charged_control_steps_lower_bound':inherited_cost,
                        'evaluation_only':True,'origin_metadata_sha256':hashlib.sha256((origin/'metadata.json').read_bytes()).hexdigest()}
                    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
                        metadata[key]=hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
                    write_json(checkpoint/'metadata.json',metadata)
                    write_json(output/'latest_checkpoint.json',{'folder':checkpoint.name,**metadata})
                else:
                    model.save(output/"model.zip"); vec.save(output/"normalization.pkl")
                loaded = load_ppo(output/"model.zip", device="cpu")
                if evaluation_only:
                    if any(not torch.equal(v,loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items()):
                        raise ValueError('Frozen evaluation weights differ from the saved source')
                else:
                    sample = vec.normalize_obs(raw.envs[0]._observation()[None])
                    np.testing.assert_array_equal(model.predict(sample, deterministic=True)[0], loaded.predict(sample, deterministic=True)[0])
                result = {**recorder.counts, "method": method, "job_name":job_name,
                    'rollout_actions':model.n_steps,'ppo_epochs':model.n_epochs,'batch_size':model.batch_size,
                    "gradient_clip_mode":getattr(model,'gradient_clip_mode',GLOBAL_CLIP), "backend": "CARLA_0.9.15",
                    "world_ticks": client.total_ticks, "reset_ticks": client.total_ticks-recorder.counts["physics_steps"],
                    "optimizer_steps": optimizer_steps[0], "actor_l2_change": float(torch.linalg.vector_norm(actor_vector(model)-before)),
                    "reload_exact": all(torch.equal(v, loaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items()),
                    "wall_seconds": time.time()-started, "stopped_reason": recorder.stopped_reason,
                    'lineage_control_steps':inherited_lineage+recorder.counts['physics_steps'],
                    'charged_control_steps_lower_bound':inherited_cost+recorder.counts['physics_steps'],
                    'inherited_controls_not_in_checkpoint':inherited_cost-inherited_lineage,
                    "scope": protocol["scope"]}
                write_json(output/"summary.json", result)
                results.append(result)
                print(json.dumps(result), flush=True)
                if protocol.get("deterministic_training_route_evaluation",False):
                    # These are the same four training geometries: development
                    # endpoint probes, not held-out generalization evidence.
                    vec.training=False
                    mean,var=vec.obs_rms.mean.copy(),vec.obs_rms.var.copy()
                    evaluation_ticks=client.total_ticks
                    evaluation=RecordedRoadEnv(config,client,cases,output) if evaluation_only else CarlaNavigationEnv(config,client,cases)
                    episodes=[]
                    with gzip.open(output/"evaluation_trace.jsonl.gz","wt",encoding="utf-8") as stream:
                        for case_id,case in enumerate(cases):
                            guard(batch,protocol)
                            obs,_=evaluation.reset(seed=7301+case_id,options={"case":case})
                            actions=0
                            while True:
                                if actions%20==0:
                                    guard(batch,protocol)
                                    atomic_json(output/'evaluation_heartbeat.json',{'pid':os.getpid(),'epoch':time.time(),'case_id':case_id,'actions':actions})
                                action,_=loaded.predict(vec.normalize_obs(obs[None]),deterministic=True)
                                obs,_,done,truncated,info=evaluation.step(action[0]);actions+=1
                                stream.write(json.dumps({"case_id":case_id,"policy_action":action[0],"info":info},default=jsonable)+"\n")
                                if done or truncated:
                                    episodes.append({"case_id":case_id,**info["episode_metrics"]});break
                    evaluation.close()
                    np.testing.assert_array_equal(mean,vec.obs_rms.mean)
                    np.testing.assert_array_equal(var,vec.obs_rms.var)
                    write_json(output/"evaluation.json",{"scope":"Frozen deterministic probes on all four TRAINING roads, no extra policy updates or independent-condition claim",
                        "episodes":episodes,"world_ticks":client.total_ticks-evaluation_ticks,
                        "model_sha256":hashlib.sha256((output/"model.zip").read_bytes()).hexdigest(),
                        "normalization_sha256":hashlib.sha256((output/"normalization.pkl").read_bytes()).hexdigest()})
                    print(json.dumps({"method":method,"job_name":job_name,"evaluation_success":sum(e["success"] for e in episodes),"episodes":4}),flush=True)
            finally:
                if recorder is not None:
                    recorder.close()
                if vec is not None:
                    vec.close()
                client.close()
        write_json(batch/"summary.json", {"status": "complete", "methods": results})
    except BaseException as exc:
        write_json(batch/"failure.json", {"error": repr(exc), "epoch": time.time(), "completed_methods": results})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare")
    parser.add_argument("--batch")
    parser.add_argument("--road-geometry-version",choices=["carla_driving_64_v1","carla_driving_refined_union_64_v2","carla_driving_indexed_union_64_v3"],default="carla_driving_64_v1")
    parser.add_argument("--controlled-budget",type=int,default=640)
    parser.add_argument("--wall-seconds",type=int,default=900)
    parser.add_argument("--episode-seconds",type=float,default=20)
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--feedback-ablation",action="store_true")
    parser.add_argument("--road-candidate-policy",action="store_true")
    args = parser.parse_args()
    prepare(args.prepare,args.road_geometry_version,args.controlled_budget,args.wall_seconds,args.episode_seconds,args.evaluate,args.feedback_ablation,args.road_candidate_policy) if args.prepare else run(args.batch)
