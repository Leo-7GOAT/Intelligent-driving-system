"""Explain audited deadline states with the existing distance/speed/yaw gates."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(batch,method,output):
    batch,output=Path(batch),Path(output);folder=batch/method
    if output.exists():raise FileExistsError('Preserve previous deadline diagnosis')
    evidence=read(folder/'endpoint_raw_audit.json');trace=folder/'evaluation_trace.jsonl.gz'
    for path,key in ((trace,'evaluation_trace_sha256'),(batch/'protocol.json','protocol_sha256'),
                     (folder/'config.json','config_sha256'),(folder/'evaluation.json','evaluation_sha256')):
        if sha(path)!=evidence[key]:raise ValueError('Raw endpoint binding differs: '+path.name)
    config=read(folder/'config.json')['env'];evaluation=read(folder/'evaluation.json');rows=[]
    expected={r['case_id'] for r in evaluation['episodes'] if r['termination_reason']=='deadline'}
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['info']['termination_reason']!='deadline':continue
            tick=row['info']['trace'][-1];state=tick['state'];goal=row['info']['initial_scenario']['goal']
            distance=math.hypot(state['x']-goal[0],state['y']-goal[1])
            heading=abs((state['yaw']-goal[2]+math.pi)%(2*math.pi)-math.pi)
            rows.append({'case_id':row['case_id'],'frame':tick['backend_frame'],'state':state,'goal':goal,
                'distance_metres':distance,'speed_abs_m_s':abs(state['v']),'heading_error_abs_rad':heading,
                'distance_gate_met':distance<=config['goal_tolerance'],
                'speed_gate_met':not config['parking'] or abs(state['v'])<=config['goal_speed'],
                'heading_gate_met':not config['parking'] or heading<=config['goal_yaw_tolerance']})
    if len(rows)!=len(expected) or {r['case_id'] for r in rows}!=expected:raise ValueError('Deadline inventory differs')
    result={'scope':'Recorded terminal-state diagnosis only;existing success/timeout definitions unchanged;zero driving/updates',
        'batch':batch.name,'method':method,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'trace_sha256':sha(trace),'thresholds':{k:config[k] for k in ('goal_tolerance','goal_speed','goal_yaw_tolerance','parking')},
        'deadline_cases':rows,'new_physical_controls':0,'optimizer_updates':0}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps([{'case':r['case_id'],'distance_m':r['distance_metres'],'speed':r['speed_abs_m_s'],
        'yaw_rad':r['heading_error_abs_rad'],'met':[r['distance_gate_met'],r['speed_gate_met'],r['heading_gate_met']]} for r in rows]));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('batch','method','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();audit(a.batch,a.method,a.output)
