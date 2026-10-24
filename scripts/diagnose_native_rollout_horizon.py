"""Measure logged rollout composition and GAE sensitivity to endpoint values.

Larger groups only reorganize recorded transitions from changing policies; they
are not reconstructed on-policy updates or proof of better driving performance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_coefficients(terminals, gamma, gae_lambda):
    """Derivative of GAE returns w.r.t. the final value, holding other values fixed."""
    coefficients = [0.]*len(terminals)
    value = gamma
    for index in reversed(range(len(terminals))):
        if terminals[index]:
            value = 0.
        elif index < len(terminals)-1:
            value *= gamma*gae_lambda
        coefficients[index] = value
    return coefficients


def composition(rows, size, gamma, gae_lambda, dt):
    groups = [rows[i:i+size] for i in range(0,len(rows),size)]
    coefficients = [c for group in groups for c in bootstrap_coefficients(
        [bool(row['termination_reason']) for row in group],gamma,gae_lambda)]
    episodes = [len({row['episode_id'] for row in group}) for group in groups]
    terminals = [sum(bool(row['termination_reason']) for row in group) for group in groups]
    durations = [sum(row['controls'] for row in group)*dt for group in groups]
    return {'decisions_per_full_group':size,'groups':len(groups),'decisions_in_final_group':len(groups[-1]),
        'all_logged_decisions_included':sum(map(len,groups)),
        'groups_without_terminal_fraction':sum(n==0 for n in terminals)/len(groups),
        'mean_episodes_per_group':statistics.mean(episodes),
        'mean_physical_seconds_per_group':statistics.mean(durations),
        'mean_terminal_count_per_group':statistics.mean(terminals),
        'nonzero_final_value_dependency_fraction':sum(c!=0 for c in coefficients)/len(coefficients),
        'mean_final_value_error_multiplier':statistics.mean(coefficients),
        'max_final_value_error_multiplier':max(coefficients)}


def diagnose(batch, method, output):
    batch, output = Path(batch), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous horizon diagnostic')
    folder=batch/method
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=read(folder/'endpoint_raw_audit.json')
    for name,key in [('config.json','config_sha256'),('evaluation.json','evaluation_sha256'),
                     ('optimization.jsonl','optimization_sha256'),('summary.json','summary_sha256')]:
        if sha(folder/name)!=audit[key]:
            raise ValueError('Raw-audited input changed')
    if sha(folder/'model.zip')!=read(folder/'evaluation.json')['model_sha256']:
        raise ValueError('Saved learner changed')
    with zipfile.ZipFile(folder/'model.zip') as archive:
        # These scalar training parameters are JSON; never deserialize policy code.
        data=json.loads(archive.read('data'))
    parameters={key:data[key] for key in ('gamma','gae_lambda','n_steps','batch_size','n_epochs')}
    rows=[]
    for name,digest in sorted(audit['transition_hashes'].items()):
        if sha(folder/name)!=digest:
            raise ValueError('Recorded training stream changed')
        with gzip.open(folder/name,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                rows.append({'episode_id':row['episode_id'],'termination_reason':row['termination_reason'],
                             'controls':len(row['trace'])})
    updates=[json.loads(line) for line in (folder/'optimization.jsonl').read_text().splitlines()]
    size=parameters['n_steps']
    if ([row['actions'] for row in updates]!=list(range(size,len(rows)+1,size))
            or len(rows)%size or sum(row['controls'] for row in rows)!=audit['training_control_frames_verified']):
        raise ValueError('Logged optimizer boundaries do not match saved rollout length')
    dt=read(folder/'config.json')['env']['control_dt']
    groups={str(n):composition(rows,n,parameters['gamma'],parameters['gae_lambda'],dt)
            for n in sorted({size,256})}
    result={'scope':'Logged composition and analytical GAE endpoint-value sensitivity only. Alternative groups span changing behavior policies and are not new on-policy training, counterfactual returns, or driving improvement.',
        'batch':batch.name,'method':method,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'model_sha256':sha(folder/'model.zip'),'learner_parameters':parameters,
        'recorded_updates':len(updates),'composition':groups,'script_sha256':sha(Path(__file__)),
        'new_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    diagnose(args.batch,args.method,args.output)
