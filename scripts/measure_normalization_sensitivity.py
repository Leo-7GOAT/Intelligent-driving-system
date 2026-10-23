"""Same-observation factorial inference; no environment steps or training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navstack.experiments import load_run
from scripts.measure_exploration_scale import sha


def measure(batch, dataset, output):
    batch, dataset, output = Path(batch), Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Preserve existing measurement')
    torch.set_num_threads(1)
    with np.load(dataset) as data:
        indices = np.flatnonzero(data['train_mask'])[::16]
        observations = data['observations'][indices]
    loaded = []
    inputs = []
    try:
        for target in (0,25000):
            evaluation = json.loads((batch/'B1'/f'validation_{target}.json').read_text(encoding='utf-8'))
            cp = batch/'B1'/Path(evaluation['checkpoint']).name
            assert sha(cp/'model.zip') == evaluation['model_sha256']
            loaded.append(load_run(cp))
            inputs.append({'target':target, 'model_sha256':sha(cp/'model.zip'),
                           'normalization_sha256':sha(cp/'normalization.pkl')})
        actions = {}
        for weights, (model, _) in enumerate(loaded):
            for statistics, (_, norm) in enumerate(loaded):
                before = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count)
                with torch.no_grad():
                    dist = model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(observations), dtype=torch.float32))
                    actions[weights, statistics] = dist.distribution.mean.numpy().clip(-1,1)
                assert not norm.training
                assert np.array_equal(before[0],norm.obs_rms.mean)
                assert np.array_equal(before[1],norm.obs_rms.var) and before[2]==norm.obs_rms.count
        rows = []
        for name, a, b in [('statistics_only',(0,0),(0,1)),
                           ('weights_with_old_statistics',(0,0),(1,0)),
                           ('weights_with_new_statistics',(0,1),(1,1))]:
            difference = np.abs(actions[a]-actions[b])
            rows.append({'contrast':name,
                'median_abs_action_change':np.median(difference,axis=0).tolist(),
                'p95_abs_action_change':np.quantile(difference,.95,axis=0).tolist(),
                'fraction_abs_change_above_point1':np.mean(difference>.1,axis=0).tolist()})
        result = {'scope':'Fixed training-visit counterfactual inference; not on-policy visits or causal training effects',
                  'dataset_sha256':sha(dataset),'selected_training_visits':len(indices),
                  'interaction_controls':0,'optimizer_updates':0,'teacher_queries':0,
                  'inputs':inputs,'rows':rows,
                  'normalization_counts':[float(norm.obs_rms.count) for _,norm in loaded]}
    finally:
        for _, norm in loaded:
            norm.close()
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    measure(args.batch,args.dataset,args.output)
