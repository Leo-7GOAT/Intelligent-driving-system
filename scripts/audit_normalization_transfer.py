"""Cross saved policies and normalization to diagnose development forgetting.

Swapped pairs are counterfactual diagnostics, never promoted evaluation models.
The physical environment and loader come from the original frozen batch source.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--left-checkpoint", help="Explicit intermediate checkpoint for a diagnostic, not model selection")
    parser.add_argument("--right-checkpoint", help="Explicit intermediate checkpoint for a diagnostic, not model selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-cases", action="store_true", help="Use all declared tasks and parking probes")
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 30 <= args.wall_seconds <= 3600 or args.left == args.right:
        raise ValueError('Require two different endpoints and a 30..3600s budget')
    batch, output = Path(args.batch).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    limits = {'created_epoch': started, 'deadline_epoch': started+args.wall_seconds,
              'source_batch': batch.name, 'left': args.left, 'right': args.right,
              'all_protocol_cases': args.protocol_cases,
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'resource_limits': {'wall_seconds': args.wall_seconds, 'cpu_workers': 1,
                                  'disk_min_GiB': 12, 'ram_min_GiB': 4, 'gpu_stop_temperature_C': 85,
                                  'automatic_retries': 0}, 'worker_scopes': [],
              'disk_free_bytes': shutil.disk_usage(output).free}
    (output/'protocol.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
    protocol = json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    for relative, digest in protocol["source_hashes"].items():
        assert hashlib.sha256((batch/"source"/relative).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(batch/"source"))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, write_json, jsonable, signature
    from navstack.research import resources_ok, atomic_json
    torch.set_num_threads(1)
    last_heartbeat, actual_controls = 0., 0

    def heartbeat(**stage):
        nonlocal last_heartbeat
        healthy, reason = resources_ok(output)
        if not healthy or shutil.disk_usage(output).free < 12*1024**3 or time.time() >= limits['deadline_epoch']-10:
            raise RuntimeError('Normalization diagnostic resource/deadline guard: '+str(reason))
        last_heartbeat = time.time()
        atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'epoch': last_heartbeat,
                                             'actual_controls': actual_controls, **stage})

    heartbeat(stage='loading')
    if 'jobs' in protocol:
        job = protocol['jobs'][0]
        run = batch/protocol['run_group']/job['id']
        development_cases = protocol['development_cases']
    elif ('supervised_epochs' in protocol or 'ppo_milestones' in protocol) and protocol['env']['method'] == 'B1':
        job = {'id': 'B1_demonstration_warmstart'}
        run = batch/'B1'
        development_cases = [c for c in protocol['cases'] if 'label' not in c]
    else:
        raise ValueError('Unsupported normalization-diagnostic protocol')
    pairs = {}
    provenance = {}
    for target in (args.left, args.right):
        explicit = args.left_checkpoint if target == args.left else args.right_checkpoint
        result = None if explicit else json.loads((run/f"validation_{target}.json").read_text(encoding="utf-8"))
        checkpoint = Path(explicit) if explicit else Path(result["checkpoint"])
        model_hash = hashlib.sha256((checkpoint/"model.zip").read_bytes()).hexdigest()
        if result is not None:
            assert model_hash == result["model_sha256"]
        else:
            assert json.loads((checkpoint/'training_summary.json').read_text(encoding='utf-8'))['physics_steps'] == target
        provenance[target] = {"checkpoint":checkpoint.name, "model_sha256":model_hash,
            "normalization_sha256":hashlib.sha256((checkpoint/"normalization.pkl").read_bytes()).hexdigest()}
        model, normalizer = load_run(checkpoint)
        config = EnvConfig.from_dict(json.loads((checkpoint/"config.json").read_text(encoding="utf-8"))["env"])
        pairs[target] = (model, normalizer, config)
    assert signature(pairs[args.left][2]) == signature(pairs[args.right][2])
    if args.protocol_cases:
        cases = [{**case, 'label':case.get('label',case['scenario']), '_probe':'label' in case}
                 for case in protocol['cases']]
    else:
        cases = [{**case, "label":case["scenario"], '_probe':False} for case in development_cases]
        cases += [{"scenario":"empty", "seed":seed, "label":"empty6m", '_probe':True, "options":{"diagnostic_route_length":6.}}
                  for seed in (7301,7302)]
    write_json(output/'inputs.json', {'checkpoints':provenance, 'cases':cases,
                                     'parent_protocol_sha256':hashlib.sha256((batch/'protocol.json').read_bytes()).hexdigest()})
    groups, all_rows = [], []
    with gzip.open(output/"trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for policy_target, (model, _, config) in pairs.items():
            for norm_target, (_, norm, _) in pairs.items():
                mean, variance, count = norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count
                rows = []
                for case in cases:
                    cfg = EnvConfig.from_dict(config.to_dict()); cfg.split = "validation"; cfg.scenario = case["scenario"]
                    env = NavigationEnv(cfg)
                    obs, _ = env.reset(seed=case["seed"], options=case.get("options"))
                    clip_counts = np.zeros(4); counts = np.zeros(4)
                    while True:
                        normalized = norm.normalize_obs(obs[None])
                        for index, (lo,hi) in enumerate(((0,23),(23,144),(163,343),(343,351))):
                            clip_counts[index] += np.sum(np.abs(normalized[0,lo:hi]) >= norm.clip_obs-1e-6)
                            counts[index] += hi-lo
                        action, _ = model.predict(normalized, deterministic=True)
                        obs, _, done, _, info = env.step(action[0])
                        actual_controls += info['executed_substeps']
                        if time.time()-last_heartbeat > 5:
                            stream.flush()
                            heartbeat(stage='evaluation', policy_target=policy_target, normalizer_target=norm_target,
                                      case=case)
                        stream.write(json.dumps({"policy_target":policy_target,"normalizer_target":norm_target,
                            "case":case,"action":action[0],"info":info}, default=jsonable)+"\n")
                        if done:
                            row = {"policy_target":policy_target,"normalizer_target":norm_target, "case":case,
                                "clipped_fraction":dict(zip(("ego_goal","local_grid","known_obstacles","map_corners"),clip_counts/counts)),
                                "terminal_state":info["trace"][-1]["state"], **info["episode_metrics"]}
                            rows.append(row); all_rows.append(row)
                            write_json(output/'progress.json', all_rows)
                            break
                    env.close()
                np.testing.assert_array_equal(mean, norm.obs_rms.mean)
                np.testing.assert_array_equal(variance, norm.obs_rms.var)
                assert norm.obs_rms.count == count
                groups.append({"policy_target":policy_target, "normalizer_target":norm_target,
                    "full_task_success":sum(r["success"] for r in rows if not r["case"]['_probe']),
                    "short6m_success":sum(r["success"] for r in rows if r["case"]["label"] in ('empty6m','retention_6m')),
                    "retention":{str(d):sum(r['success'] for r in rows if r['case']['label']==f'retention_{d}m') for d in (6,12,18,23)},
                    "episode_results":[[r["case"]["label"],r["case"]["seed"],r["termination_reason"]] for r in rows]})
    for _, norm, _ in pairs.values():
        norm.close()
    result = {"scope":"Frozen-source development factorial diagnosis, no model selection or independent-policy claim",
        "batch":batch.name,"job":job["id"],"checkpoints":provenance,"groups":groups,"episodes":all_rows,
        'actual_controls':actual_controls,'optimizer_updates':0,'teacher_queries':0,
        'wall_seconds':time.time()-started}
    write_json(output/"results.json", result)
    write_json(output/'completed.json', {'actual_controls':actual_controls,'groups':len(groups),
                                        'episodes':len(all_rows),'optimizer_updates':0})
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
