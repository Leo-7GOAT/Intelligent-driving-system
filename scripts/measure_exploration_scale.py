"""Conditional gSDE action-scale audit on fixed training observations only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navstack.experiments import load_run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(batch, dataset, output):
    batch,dataset,output=Path(batch),Path(dataset),Path(output)
    if output.exists(): raise ValueError('Preserve existing measurements')
    torch.set_num_threads(1)
    data=np.load(dataset)
    selected=np.flatnonzero(data['train_mask'])[::16]
    observations=data['observations'][selected]
    rows=[]
    for target in (0,25000):
        evaluation=json.loads((batch/'B1'/f'validation_{target}.json').read_text(encoding='utf-8'))
        checkpoint=batch/'B1'/Path(evaluation['checkpoint']).name
        assert sha(checkpoint/'model.zip')==evaluation['model_sha256']
        model,norm=load_run(checkpoint)
        try:
            before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
            with torch.no_grad():
                dist=model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(observations),dtype=torch.float32)).distribution
                def bounds_probability(std):
                    normal=torch.distributions.Normal(dist.mean,std)
                    return (normal.cdf(torch.full_like(dist.mean,-1))+1-normal.cdf(torch.full_like(dist.mean,1))).mean(0).tolist()
                std=dist.stddev.numpy()
                rows.append({'target':target,'model_sha256':sha(checkpoint/'model.zip'),
                    'normalization_sha256':sha(checkpoint/'normalization.pkl'),
                    'log_std_parameter_mean':float(model.policy.log_std.mean()),
                    'conditional_std_median':np.median(std,axis=0).tolist(),
                    'conditional_std_p95':np.quantile(std,.95,axis=0).tolist(),
                    'outside_action_bounds_probability_mean':bounds_probability(dist.stddev),
                    'unexecuted_std_scaled_exp_minus2_probability':bounds_probability(dist.stddev*np.exp(-2))})
            np.testing.assert_array_equal(before[0],norm.obs_rms.mean)
            np.testing.assert_array_equal(before[1],norm.obs_rms.var)
            assert before[2]==norm.obs_rms.count and not norm.training
        finally: norm.close()
    result={'scope':'Offline distribution probe on every16th train-mask visit; same observations for both models, not their on-policy state distributions or success evidence',
        'dataset_sha256':sha(dataset),'selected_training_visits':len(selected),
        'selected_indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
        'interaction_controls':0,'optimizer_updates':0,'teacher_queries':0,'rows':rows}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True)
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();measure(args.batch,args.dataset,args.output)
