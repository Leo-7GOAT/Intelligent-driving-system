import gzip
import json
import math

import pytest
from scripts.diagnose_native_action_learning import evaluation_goal_errors, sha


def test_measured_goal_errors_wrap_yaw_and_preserve_failure_priority(tmp_path):
    config={'env':{'goal_tolerance':.8,'goal_speed':.25,'goal_yaw_tolerance':.35,'parking':True}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    row={'case_id':0,'policy_action':0,'info':{'mode':'wait','initial_scenario':{'goal':[0,0,-math.pi]},
        'trace':[{'state':{'x':.1,'y':0,'yaw':math.pi,'v':.1},'intervention':True},
                 {'state':{'x':1,'y':0,'yaw':0,'v':1},'intervention':False}],
        'termination_reason':'collision'}}
    with gzip.open(tmp_path/'evaluation_trace.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    audit={key:sha(tmp_path/name) for name,key in [('config.json','config_sha256'),
        ('evaluation_trace.jsonl.gz','evaluation_trace_sha256')]}
    case=evaluation_goal_errors(tmp_path,audit)['cases'][0]
    assert case['minimum_goal_distance_m']==.1
    assert case['pose_qualified_controls']==1 and case['pose_qualified_without_intervention_controls']==0
    assert case['final_unmet_pose_criteria']==['position','speed','yaw']
    assert case['recorded_termination_reason']=='collision'
    assert case['closest_measured_approach']['absolute_speed_mps']==.1
    assert case['first_position_entry']==case['closest_measured_approach']
    assert case['first_position_entry']['intervention'] is True
    assert case['mode_counts']=={'wait':1} and case['last_ten_policy_actions']==[0]
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='changed after raw audit'):
        evaluation_goal_errors(tmp_path,audit)


def test_goal_entry_at_speed_is_distinct_from_closest_approach_and_final_failure(tmp_path):
    config={'env':{'goal_tolerance':.8,'goal_speed':.25,'goal_yaw_tolerance':.35,'parking':True}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    rows=[]
    for i,(x,v,yaw) in enumerate(((1.,1.,0.),(.8,.9,.4),(.2,.6,.1),(2.,.3,0.))):
        rows.append({'case_id':0,'policy_action':[-.1,.2],'info':{'mode':'flat',
            'initial_scenario':{'goal':[0,0,0]},'termination_reason':'out_of_bounds' if i==3 else None,
            'trace':[{'state':{'x':x,'y':0.,'v':v,'yaw':yaw,'time':(i+1)*.05},'intervention':False}]}})
    with gzip.open(tmp_path/'evaluation_trace.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.writelines(json.dumps(row)+'\n' for row in rows)
    audit={key:sha(tmp_path/name) for name,key in [('config.json','config_sha256'),
        ('evaluation_trace.jsonl.gz','evaluation_trace_sha256')]}
    case=evaluation_goal_errors(tmp_path,audit)['cases'][0]
    assert case['first_position_entry']['distance_m']==.8
    assert case['first_position_entry']['unmet_pose_criteria']==['speed','yaw']
    assert case['closest_measured_approach']['distance_m']==.2
    assert case['closest_measured_approach']['unmet_pose_criteria']==['speed']
    assert case['closest_measured_approach']['policy_action']==[-.1,.2]
    assert case['final_goal_distance_m']==2.
    assert case['recorded_termination_reason']=='out_of_bounds'
    assert case['pose_qualified_controls']==0
