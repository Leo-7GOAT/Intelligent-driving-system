from types import SimpleNamespace
import io
import queue
import pytest
from navstack.carla_actuation import deliver_vehicle_control,vehicle_control_matches
from scripts.audit_native_control_delivery import control_errors
from scripts.prepare_native_control_delivery_probe import replace_once
from navstack.carla_bridge import BridgeClient,InfrastructureFailure
from scripts.diagnose_flat_clock import shifted_clock


def control(**changes):
    return SimpleNamespace(**dict(dict(throttle=.2,brake=.03,steer=-.1,gear=1,
        manual_gear_shift=True,hand_brake=False,reverse=False),**changes))


def test_rule_root_trace_is_classified_as_evaluation_not_training(tmp_path):
    import gzip,json
    from scripts.audit_native_control_delivery import evaluation_episodes
    row={'info':{'trace':[{'backend_evidence':{'episode_id':'rule_episode'}}]}}
    with gzip.open(tmp_path/'evaluation_trace.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    assert evaluation_episodes(tmp_path,{'methods':['B3'],'policy_kind':'rule_only_B3'})=={'rule_episode'}
    assert evaluation_episodes(tmp_path,{'methods':['H']})==set()


def test_sync_submission_uses_explicit_no_tick_barrier_and_no_async_actor_call():
    calls=[]
    actor=SimpleNamespace(id=42,apply_control=lambda value:calls.append(('async',value)))
    client=SimpleNamespace(apply_batch_sync=lambda commands,tick:calls.append((commands,tick)) or [SimpleNamespace(error='')])
    c=control();deliver_vehicle_control(client,actor,c,'batch_sync_v1',lambda aid,value:('control',aid,value))
    assert calls==[([('control',42,c)],False)]


@pytest.mark.parametrize('responses',[[],[SimpleNamespace(error='bad actor')],[SimpleNamespace(error=''),SimpleNamespace(error='')]])
def test_failed_submission_is_not_acknowledged(responses):
    client=SimpleNamespace(apply_batch_sync=lambda commands,tick:responses)
    with pytest.raises(RuntimeError):deliver_vehicle_control(client,SimpleNamespace(id=42),control(),'batch_sync_v1',lambda a,c:(a,c))


def test_legacy_submission_is_byte_semantic_compatible_and_unknown_mode_rejects():
    sent=[];actor=SimpleNamespace(apply_control=sent.append)
    c=control();deliver_vehicle_control(None,actor,c,'actor_async_legacy');assert sent==[c]
    with pytest.raises(ValueError):deliver_vehicle_control(None,actor,c,'unregistered')
    assert sent==[c]


def test_acknowledgement_accepts_wire_rounding_but_rejects_prior_command_and_handbrake():
    assert vehicle_control_matches(control(),control(throttle=.200000003))
    assert not vehicle_control_matches(control(),control(throttle=.1))
    assert not vehicle_control_matches(control(),control(hand_brake=True))
    assert not vehicle_control_matches(control(),control(brake=float('nan')))


def test_raw_delivery_comparison_does_not_confuse_float32_rounding_with_previous_control():
    desired={'throttle':.2,'brake':0.,'steer':-.1}
    assert control_errors(desired,{'throttle':.200000003,'brake':0.,'steer':-.100000001})=={}
    assert set(control_errors(desired,{'throttle':.1,'brake':0.,'steer':-.2}))=={'throttle','steer'}
    with pytest.raises(ValueError):control_errors(desired,{'throttle':float('nan'),'brake':0.,'steer':-.1})


def test_frozen_delivery_overlay_rejects_missing_or_duplicate_sentinels():
    assert replace_once('before marker after','marker','changed')=='before changed after'
    with pytest.raises(ValueError):replace_once('absent','marker','changed')
    with pytest.raises(ValueError):replace_once('marker marker','marker','changed')


def test_main_client_rejects_unconfirmed_sync_frame_and_preserves_response():
    client=BridgeClient.__new__(BridgeClient)
    client.request_audit=io.StringIO();client.process=SimpleNamespace(stdin=io.StringIO())
    client.messages=queue.Queue();client.timeout=1
    result={'ok':True,'request_id':1,'episode_id':'owned','control_delivery':'batch_sync_v1',
        'applied_control_matches_request':False,'frame':73,'state':{'v':1.}}
    client.messages.put(result)
    with pytest.raises(InfrastructureFailure,match='acknowledgement'):
        client.exchange({'op':'step','request_id':1,'episode_id':'owned'})
    assert client.last_failure_response is result
    result={**result,'request_id':2,'applied_control_matches_request':True};client.messages.put(result)
    assert client.exchange({'op':'step','request_id':2,'episode_id':'owned'}) is result


def test_clock_counterfactual_preserves_all_physical_inputs_and_deadline_sum():
    import numpy as np
    x=np.arange(3*4705,dtype=np.float32).reshape(3,4705)
    x[:,22]=[0.,10.,24.];x[:,21]=45-x[:,22]
    z,keep=shifted_clock(x,5.)
    assert keep.tolist()==[True,True,False]
    other=np.ones(4705,dtype=bool);other[[21,22]]=False
    np.testing.assert_array_equal(x[:,other],z[:,other]);np.testing.assert_array_equal(z[:,21]+z[:,22],[45.,45.,45.])
