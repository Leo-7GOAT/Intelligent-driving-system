from copy import deepcopy
import pytest
from scripts.summarize_native_runtime import summarize_rows


def tick(seconds,speed,intervention=False):
    return {'control_cycle_seconds':seconds,'state':{'v':speed},'controller_status':'solved',
        'intervention':intervention,'shared_shield_intervention':False,
        'requested_command':{'acceleration':-1.},'backend_evidence':{'vehicle_control':{'brake':.2}}}


def test_failed_case_timings_and_fallbacks_remain_in_pooled_quantiles():
    rows=[{'case_id':0,'policy_action':26,'info':{'trace':[tick(.01,1.)]*2,
        'environment_decision_seconds':.1,'mode':'reference','plan':None}},
        {'case_id':1,'policy_action':6,'info':{'trace':[tick(.25,0.,True) for _ in range(3)],
        'environment_decision_seconds':3.,'mode':'plan_failed',
        'plan':{'status':'search_budget','planning_time':2.9,'expanded_nodes':7200}}}]
    common={'actions':1,'progress':2.,'plan_failures':0,'mpc_failures':0,'interventions':0}
    episodes=[{**common,'case_id':0,'success':True,'termination_reason':'success','physics_steps':2,'simulation_seconds':.1},
        {**common,'case_id':1,'success':False,'termination_reason':'deadline','physics_steps':3,'simulation_seconds':.15}]
    result=summarize_rows(rows,episodes,.05,.5)
    assert result['counts']=={'cases':2,'successes':1,'unsafe_episodes':0,'decisions':2,'controls':5}
    assert result['latencies']['control_cycle']['count']==5
    assert result['latencies']['control_cycle']['p95']==.25
    assert result['latencies']['control_cycle']['over_period_fraction']==.6
    assert result['latencies']['environment_decision']['max']==3.
    assert result['success_only_travel_seconds']==[.1]
    assert result['all_episode_durations_seconds']==[.1,.15]
    assert result['intervention_controls']==result['stationary_controls_abs_v_below_0_05']==3
    assert result['mode_controls']=={'reference':2,'plan_failed':3}
    for fault in ('missing_case','missing_control','nonfinite'):
        broken=deepcopy(rows)
        if fault=='missing_case':broken.pop()
        elif fault=='missing_control':broken[1]['info']['trace'].pop()
        else:broken[1]['info']['environment_decision_seconds']=float('nan')
        with pytest.raises(ValueError):summarize_rows(broken,episodes,.05,.5)
    # Terminal sensor refreshes retain the measured frame but omit the earlier
    # step's control report. Never drop that tick or assume it did not brake.
    terminal=rows[1]['info']['trace'][-1]
    terminal['backend_evidence']={'episode_id':'measured-episode','frame':42}
    missing=summarize_rows(rows,episodes,.05,.5)
    assert missing['vehicle_control_missing_controls']==1
    joined=summarize_rows(rows,episodes,.05,.5,{('measured-episode',42):{'brake':.2}})
    assert joined['vehicle_control_missing_controls']==0
    assert joined['vehicle_controls_recovered_from_raw_same_frame']==1
    assert joined['actual_brake_controls_above_1e_6']==5
