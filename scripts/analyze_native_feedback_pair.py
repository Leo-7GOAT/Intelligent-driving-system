"""Describe one audited shared-mask training pair without seed-level inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.stats import binomtest
from scripts.audit_native_method_endpoint import read,sha


def analyze(h,h0,output):
    batches=[Path(h),Path(h0)];output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing pair report')
    protocols=[read(p/'protocol.json') for p in batches]
    for key in ('seed','routes','native_case_manifest','calibration_sha256','shared_endpoint_mask_v1',
                'rollout_actions','ppo_epochs','episode_seconds','road_geometry_version'):
        if protocols[0].get(key)!=protocols[1].get(key):raise ValueError('Pair differs in '+key)
    if protocols[0].get('shared_endpoint_mask_v1') is not True:raise ValueError('Require common endpoint mask')
    critical=('env','planning','mpc','trajectory','reward','road_grid','road_grid_batch',
              'native_route','config','candidate_filter','road_candidate_policy','carla_actuation')
    for name in critical:
        relative='navstack/'+name+'.py'
        if len({sha(p/'source'/relative) for p in batches})!=1:raise ValueError('Physical/policy source differs: '+relative)
    configurations=[read(p/m/'config.json')['env'] for p,m in zip(batches,('H','H0'))]
    if configurations[0]!=configurations[1]:raise ValueError('Shared-mask environments differ')
    architectures=[read(p/m/'policy_architecture.json') for p,m in zip(batches,('H','H0'))]
    if [a['policy'] for a in architectures]!=['EndpointClearRoadPolicy','SharedEndpointAblatedRoadPolicy']:
        raise ValueError('Unexpected learned feedback ablation classes')
    episodes=[];proof=[]
    for batch,method in zip(batches,('H','H0')):
        folder=batch/method;raw=read(folder/'endpoint_raw_audit.json');evaluation=read(folder/'evaluation.json')
        if (sha(folder/'evaluation_trace.jsonl.gz')!=raw['evaluation_trace_sha256']
                or sha(batch/'protocol.json')!=raw['protocol_sha256']
                or sha(folder/'config.json')!=raw['config_sha256']):raise ValueError('Raw audit binding differs')
        for name,key in (('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')):
            if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluated model differs')
        rows={r['case_id']:r for r in evaluation['episodes']}
        if set(rows)!=set(range(12)):raise ValueError('Require all twelve development conditions')
        episodes.append(rows);proof.append({'batch':batch.name,'raw_controls':raw['total_applied_controls_verified'],
            'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'model_sha256':evaluation['model_sha256'],
            'summary':read(folder/'summary.json')})
    wins=[i for i in range(12) if episodes[0][i]['success'] and not episodes[1][i]['success']]
    losses=[i for i in range(12) if episodes[1][i]['success'] and not episodes[0][i]['success']]
    families={}
    manifest=protocols[0]['native_case_manifest']['cases']
    for label,kind in (('curve_parking',None),('static_obstacle','static_vehicle'),('dynamic_interaction','crossing_walker')):
        ids=[i for i,c in enumerate(manifest) if (c['hazard'] is None if kind is None else c['hazard'] is not None and c['hazard']['kind']==kind)]
        if len(ids)!=4:raise ValueError('Require four cases in every core family')
        families[label]={'cases':ids,'H_success':sum(episodes[0][i]['success'] for i in ids),
                         'H0_success':sum(episodes[1][i]['success'] for i in ids)}
    result={'scope':'One independently trained seed pair on twelve known development conditions;exploratory feedback ablation,not final confirmation',
        'seed':protocols[0]['seed'],'H_successes':sum(r['success'] for r in episodes[0].values()),
        'H0_successes':sum(r['success'] for r in episodes[1].values()),'cases':12,
        'H_only_success_cases':wins,'H0_only_success_cases':losses,
        'paired_difference_percentage_points':100*(len(wins)-len(losses))/12,
        'conditional_exact_discordance_p_two_sided':float(binomtest(len(wins),len(wins)+len(losses)).pvalue) if wins or losses else 1.,
        'families':families,'source_and_model_proof':proof,
        'limits':'One training seed;conditions reused in development;actual physical budgets and inherited costs are reported per method,including phase overshoot. No seed-level confidence interval or PPO-vs-rule advantage established.',
        'research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_and_model_proof'}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('h','h0','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();analyze(a.h,a.h0,a.output)
