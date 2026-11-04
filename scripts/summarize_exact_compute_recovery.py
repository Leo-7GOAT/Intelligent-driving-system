"""Summarize only a closed, audited development recovery; never create missing results."""
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from navstack.research import atomic_json


def summarize(execution):
    execution=Path(execution).resolve()
    done=read(execution/'complete.json');registration=read(execution/'registration.json')
    archive=execution.parent;parent=archive/registration['parent']
    prefix_path=parent/'H/exact_compute_recovery_prefix_raw_audit.json';prefix=read(prefix_path)
    retained=read(execution/'retained_prefix.json')
    if (retained['prefix_raw_sha256']!=sha(prefix_path)
            or done['retained_completed_case_ids']!=prefix['completed_case_ids']):
        raise ValueError('Completed prefix binding differs')
    episodes={c['case_id']:c['metrics'] for c in prefix['completed_cases']}
    references={str(prefix_path.relative_to(archive)):sha(prefix_path)}
    starts=read(parent/'H/compute_prefix_start_inference.json')
    static=read(parent/'H/compute_prefix_static_inference.json')
    for report in (starts,static):
        if report['raw_audit_sha256']!=sha(prefix_path):raise ValueError('Prefix inference is not raw-bound')
    controls=prefix['all_recorded_actual_controls_verified'];children=[]
    for item in done['children']['results']:
        batch=archive/item['batch'];folder=batch/'H';raw_path=folder/'endpoint_raw_audit.json'
        raw=read(raw_path);evaluation=read(folder/'evaluation.json');protocol=read(batch/'protocol.json')
        if (sha(raw_path)!=item['raw_sha256'] or raw['evaluation_sha256']!=sha(folder/'evaluation.json')
                or not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']):
            raise ValueError('Closed child evidence changed')
        references[str(raw_path.relative_to(archive))]=sha(raw_path)
        for name in ('measured_start_inference.json','static_action_inference.json'):
            report=read(folder/name)
            if report['raw_audit_sha256']!=sha(raw_path):raise ValueError('Child inference is not raw-bound')
        controls+=raw['evaluation_control_frames_verified']
        children.append(dict(item,controls=raw['evaluation_control_frames_verified'],
            failed_cases=[dict(case_id=e['case_id'],termination=e['termination_reason'],
                controls=e['physics_steps'],interventions=e['interventions'])
                for e in evaluation['episodes'] if not e['success']]))
        ids=protocol['exact_compute_recovery']['original_case_ids']
        if ids is not None:
            if len(ids)!=len(evaluation['episodes']):raise ValueError('Tail inventory differs')
            for entry in evaluation['episodes']:
                original=ids[entry['case_id']]
                if original in episodes:raise ValueError('Duplicate retained case')
                episodes[original]=entry
    if sorted(episodes)!=list(range(30)):raise ValueError('Require all thirty registered conditions')
    cases=[dict(case_id=i,success=e['success'],termination=e['termination_reason'],
        controls=e['physics_steps'],interventions=e['interventions']) for i,e in sorted(episodes.items())]
    return dict(scope='Closed known-development transfer;two disclosed computation profiles;not independent final confirmation',
        research_acceptance='UNMET',execution=execution.name,model_sha256=starts['model_sha256'],
        source_report_script_sha256=sha(Path(__file__)),evidence_sha256=references,
        placements=dict(cases=30,successes=sum(c['success'] for c in cases),
            reasons=dict(Counter(c['termination'] for c in cases)),episodes=cases,
            completed_controls=sum(c['controls'] for c in cases),
            prior_interrupted_controls=prefix['all_recorded_actual_controls_verified']-sum(c['controls'] for c in prefix['completed_cases']),
            verified_NN_decisions=sum(r['decisions'] for r in static['results'])+sum(
                sum(r['decisions'] for r in read(archive/c['batch']/'H/static_action_inference.json')['results'])
                for c in children if read(archive/c['batch']/'protocol.json')['exact_compute_recovery']['original_case_ids'] is not None)),
        children=children,all_actual_evaluation_controls_charged=controls,new_PPO_updates=0,new_final_conditions=0,
        learning_comparison='Previously closed same3 endpoint remained2/3 before and after;no demonstrated learning gain.',
        limits='Single continued policy;known repeated development conditions. Earlier23/30used different low-level/model. Do not attribute differences to PPO, infer independent-seed intervals, claim real-time operation, or treat zero unsafe episodes as proof of general safety.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve existing reports')
    result=summarize(a.execution);atomic_json(output,result)
    print({k:result['placements'][k] for k in ('cases','successes','reasons','completed_controls','prior_interrupted_controls','verified_NN_decisions')})
