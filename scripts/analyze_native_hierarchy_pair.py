"""Audit one complete H/B1 development pair;no seed-level inference claim."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.stats import binomtest
from scripts.audit_native_method_endpoint import audit,read,sha


def training_record(batch,method):
    protocol=read(batch/'protocol.json');summary=read(batch/method/'summary.json')
    origin=batch
    if protocol.get('evaluation_only'):
        recovery=protocol['recovery'][method]
        origin=batch.parent/recovery['parent_batch']
        checkpoint=origin/method/recovery['parent_checkpoint']
        if sha(checkpoint/'model.zip')!=sha(batch/method/'model.zip'):
            raise ValueError('Frozen evaluation differs from original trained checkpoint')
        summary=read(origin/method/'summary.json')
        metadata=read(checkpoint/'metadata.json')
        if metadata['physics_steps']!=summary['physics_steps'] or metadata['optimizer_steps']!=summary['optimizer_steps']:
            raise ValueError('Training origin counters differ from frozen checkpoint')
    if not summary['physics_steps'] or not summary['optimizer_steps']:
        raise ValueError('Require actual PPO training in both methods')
    return {'batch':origin.name,'summary_sha256':sha(origin/method/'summary.json'),
        'physical_controls':summary['physics_steps'],'optimizer_steps':summary['optimizer_steps'],
        'charged_controls_lower_bound':summary['charged_control_steps_lower_bound'],
        'lineage_controls':summary['lineage_control_steps']}


def analyze(h,b1,output):
    batches=list(map(Path,(h,b1)));output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous paired evidence')
    protocols=[read(b/'protocol.json') for b in batches]
    for key in ('seed','routes','native_case_manifest','native_validation','calibration_sha256',
                'episode_seconds','road_geometry_version','task_progress_weight','unsafe_event_penalty'):
        if protocols[0].get(key)!=protocols[1].get(key):raise ValueError('Task differs in '+key)
    if not protocols[0].get('native_validation'):raise ValueError('Require explicit development validation')
    inventory=protocols[0]['native_validation']['manifest']['cases'];count=len(inventory)
    configs=[read(b/m/'config.json')['env'] for b,m in zip(batches,('H','B1'))]
    differences={k:{'H':configs[0].get(k),'B1':configs[1].get(k)} for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)}
    expected={'method':{'H':'H','B1':'B1'},'flat_hold':{'H':10,'B1':1},
              'endpoint_candidate_filter':{'H':True,'B1':None}}
    if differences!=expected:raise ValueError('Unregistered method/environment differences')
    for name in ('env','planning','mpc','trajectory','reward','road_grid','native_route','carla_actuation','config'):
        if len({sha(b/'source/navstack'/f'{name}.py') for b in batches})!=1:
            raise ValueError('Physical/task source differs: '+name)
    endpoints=[];episodes=[]
    for batch,method in zip(batches,('H','B1')):
        folder=batch/method;rebuilt=audit(batch,method);stored=read(folder/'endpoint_raw_audit.json')
        if rebuilt!=stored:raise ValueError('Raw endpoint audit differs')
        evaluation=read(folder/'evaluation.json');rows=evaluation['episodes']
        if evaluation.get('partition')!='validation' or len(rows)!=count or {r['case_id'] for r in rows}!=set(range(count)):
            raise ValueError('Incomplete or wrong evaluation inventory')
        for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
            if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated model differs')
        episodes.append({r['case_id']:r for r in rows})
        endpoints.append({'method':method,'batch':batch.name,'model_sha256':evaluation['model_sha256'],
            'normalization_sha256':evaluation['normalization_sha256'],'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
            'verified_controls':stored['total_applied_controls_verified'],'training':training_record(batch,method),
            'successes':sum(r['success'] for r in rows),'termination_counts':dict(Counter(r['termination_reason'] for r in rows)),
            'unsafe_episodes':sum(r['termination_reason'] in ('collision','out_of_bounds') for r in rows)})
    wins=[i for i in range(count) if episodes[0][i]['success'] and not episodes[1][i]['success']]
    losses=[i for i in range(count) if episodes[1][i]['success'] and not episodes[0][i]['success']]
    families={}
    for family,kind in [('curve_parking',None),('static_obstacle','static_vehicle'),('dynamic_interaction','crossing_walker')]:
        ids=[i for i,c in enumerate(inventory) if (c['hazard'] is None if kind is None else c['hazard'] is not None and c['hazard']['kind']==kind)]
        if len(ids)!=count//3:raise ValueError('Unbalanced core inventory')
        families[family]={'cases':len(ids),**{m:sum(rows[i]['success'] for i in ids) for m,rows in zip(('H','B1'),episodes)}}
    result={'scope':'One trained-seed native-system development comparison;not final confirmation or isolated PPO contribution',
        'seed':protocols[0]['seed'],'cases':count,'endpoints':endpoints,'families':families,
        'H_only_success_cases':wins,'B1_only_success_cases':losses,
        'paired_difference_percentage_points':100*(len(wins)-len(losses))/count,
        'conditional_exact_discordance_p_two_sided':float(binomtest(len(wins),len(wins)+len(losses)).pvalue) if wins or losses else 1.,
        'declared_environment_differences':differences,
        'limits':'One training seed;development conditions already viewed. H includes candidate feasibility filtering and MPC while B1 directly controls each physical step. These are native-system differences,not evidence that PPO beats a strong rule high level. Teacher inputs,tuning and interrupted ancestor costs are not equalized by a common new-training budget. No seed-level95percent interval or formal superiority claim.',
        'new_controls':0,'optimizer_updates':0,'research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('h','b1','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();analyze(a.h,a.b1,a.output)
