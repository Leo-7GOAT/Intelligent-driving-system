"""Cross saved policies and normalization to diagnose development forgetting.

Swapped pairs are counterfactual diagnostics, never promoted evaluation models.
The physical environment and loader come from the original frozen batch source.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--left-checkpoint", help="Explicit intermediate checkpoint for a diagnostic, not model selection")
    parser.add_argument("--right-checkpoint", help="Explicit intermediate checkpoint for a diagnostic, not model selection")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    batch, output = Path(args.batch).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((batch/"protocol.json").read_text(encoding="utf-8"))
    for relative, digest in protocol["source_hashes"].items():
        assert hashlib.sha256((batch/"source"/relative).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(batch/"source"))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.env import NavigationEnv
    from navstack.experiments import load_run, write_json, jsonable, signature
    torch.set_num_threads(1)
    if 'jobs' in protocol:
        job = protocol['jobs'][0]
        run = batch/protocol['run_group']/job['id']
        development_cases = protocol['development_cases']
    elif 'supervised_epochs' in protocol and protocol['env']['method'] == 'B1':
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
    cases = [{**case, "label":case["scenario"]} for case in development_cases]
    cases += [{"scenario":"empty", "seed":seed, "label":"empty6m", "options":{"diagnostic_route_length":6.}}
              for seed in (7301,7302)]
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
                        stream.write(json.dumps({"policy_target":policy_target,"normalizer_target":norm_target,
                            "case":case,"action":action[0],"info":info}, default=jsonable)+"\n")
                        if done:
                            row = {"policy_target":policy_target,"normalizer_target":norm_target, "case":case,
                                "clipped_fraction":dict(zip(("ego_goal","local_grid","known_obstacles","map_corners"),clip_counts/counts)),
                                "terminal_state":info["trace"][-1]["state"], **info["episode_metrics"]}
                            rows.append(row); all_rows.append(row)
                            break
                    env.close()
                np.testing.assert_array_equal(mean, norm.obs_rms.mean)
                np.testing.assert_array_equal(variance, norm.obs_rms.var)
                assert norm.obs_rms.count == count
                groups.append({"policy_target":policy_target, "normalizer_target":norm_target,
                    "full_task_success":sum(r["success"] for r in rows if r["case"]["label"]!="empty6m"),
                    "short6m_success":sum(r["success"] for r in rows if r["case"]["label"]=="empty6m"),
                    "episode_results":[[r["case"]["label"],r["case"]["seed"],r["termination_reason"]] for r in rows]})
    for _, norm, _ in pairs.values():
        norm.close()
    result = {"scope":"Frozen-source development factorial diagnosis, no model selection or independent-policy claim",
        "batch":batch.name,"job":job["id"],"checkpoints":provenance,"groups":groups,"episodes":all_rows}
    write_json(output/"results.json", result)
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
