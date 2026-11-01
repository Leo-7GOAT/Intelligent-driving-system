"""Frozen-policy four-TRAIN-road transfer to explicit synchronous control delivery."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from navstack.research import atomic_json,resources_ok


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Frozen delivery overlay requires exactly one known source sentinel')
    return text.replace(before,after)


def overlays(parent):
    result={}
    helpers=(ROOT/'navstack/carla_actuation.py').read_text(encoding='utf-8').split('CONTROL_DELIVERY_MODES =',1)[1].split('\ndef actuator_state_values',1)[0]
    path='navstack/carla_actuation.py';old=(parent/'source'/path).read_text(encoding='utf-8')
    marker='ACTUATOR_STATE_FIELDS = ("signed_pedal", "filtered_acceleration", "feedback_integral")'
    result[path]=replace_once(old,marker,marker+'\n\nCONTROL_DELIVERY_MODES ='+helpers)
    path='scripts/carla_bridge37.py';text=(parent/'source'/path).read_text(encoding='utf-8')
    before='        self.cal=json.loads(Path(calibration).read_text(encoding="utf-8")); self.lock=Path(lock)'
    text=replace_once(text,before,before+"\n        from navstack.carla_actuation import CONTROL_DELIVERY_MODES\n        self.control_delivery=self.cal['recommended_adapter'].get('control_delivery','actor_async_legacy')\n        if self.control_delivery not in CONTROL_DELIVERY_MODES:raise ValueError('Unknown control delivery version')")
    before='''        self.actor.apply_control(carla.VehicleControl(throttle=throttle,brake=braking,steer=steer,
                                 manual_gear_shift=manual,gear=1 if manual else 0))'''
    after='''        from navstack.carla_actuation import deliver_vehicle_control,vehicle_control_matches
        requested_control=carla.VehicleControl(throttle=throttle,brake=braking,steer=steer,
                                 manual_gear_shift=manual,gear=1 if manual else 0)
        deliver_vehicle_control(self.client,self.actor,requested_control,self.control_delivery)'''
    text=replace_once(text,before,after)
    before='''        applied=self.actor.get_control()
        return {**self.read(),"applied_vehicle_control":'''
    after='''        applied=self.actor.get_control()
        delivery=({'control_delivery':self.control_delivery,'applied_control_matches_request':vehicle_control_matches(requested_control,applied)}
                  if self.control_delivery!='actor_async_legacy' else {})
        return {**self.read(),**delivery,"applied_vehicle_control":'''
    result[path]=replace_once(text,before,after)
    path='navstack/carla_bridge.py';text=(parent/'source'/path).read_text(encoding='utf-8')
    before='''            if result.get("episode_id")!=request["episode_id"]:raise InfrastructureFailure("Mismatched response episode")'''
    after='''            if (request['op']=='step' and result.get('control_delivery')=='batch_sync_v1'
                    and result.get('applied_control_matches_request') is not True):
                self.last_failure_response=result
                raise InfrastructureFailure('Synchronous control differs from last-tick engine acknowledgement')
'''+before
    text=replace_once(text,before,after)
    before='''            if key in result:self.evidence[key]=result[key]

    @property'''
    after='''            if key in result:self.evidence[key]=result[key]
        if result.get('control_delivery')=='batch_sync_v1':
            for key in ('control_delivery','applied_control_matches_request','applied_vehicle_control'):
                if key in result:self.evidence[key]=result[key]

    @property'''
    result[path]=replace_once(text,before,after)
    return result


def prepare():
    pointer=ROOT/'runs/native_control_delivery_probe_registration_20261101.json'
    if pointer.exists():raise FileExistsError('Preserve registered synchronous delivery probe')
    parent=ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension';require_finished_parent(parent)
    pilot=ROOT/'runs/20261101T075805697090Z_native_B1_exploration_PPO';require_finished_parent(pilot)
    diagnosis=read(ROOT/'reports/native_PPO_legacy_control_delivery_20261101.json')
    if (diagnosis['batch']!=pilot.name or diagnosis['counts'].get('mismatches')!=51
            or diagnosis['counts'].get('mismatches_matching_previous_request')!=51):
        raise ValueError('Require the measured discrepancy before this low-level version change')
    old=read(parent/'protocol.json');origin=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    metadata=verified_checkpoint_metadata(origin)
    if (metadata['model_sha256']!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0'
            or metadata['normalization_sha256']!='db467fb35b0f78a989e8e042b0400036cada30cfad36b41fba727a4f432e5df5'
            or old['methods']!=['B1'] or len(old['native_case_manifest']['cases'])!=4):
        raise ValueError('Use the fixed,already evaluated replay reference and all4known TRAINroads')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent source changed')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    changes=overlays(parent);batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_sync_delivery_transfer')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    for name,text in changes.items():(batch/'source'/name).write_text(text,encoding='utf-8')
    for name in ('prepare_native_control_delivery_probe.py','audit_native_control_delivery.py'):
        shutil.copy2(ROOT/'scripts'/name,batch/'source/scripts'/name)
    calibration=read(parent/'calibration/calibration.json');calibration['recommended_adapter']['control_delivery']='batch_sync_v1'
    atomic_json(batch/'calibration/calibration.json',calibration)
    p=deepcopy(old)
    for key in ('jobs','own_success_replay','native_validation','native_camera_evaluation','freeze_observation_statistics'):p.pop(key,None)
    now=time.time();cost=read(pilot/'B1/endpoint_raw_audit.json');summary=read(parent/'B1/summary.json')
    p.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'B1':0},checkpoint_thresholds=[],
        created_epoch=now,deadline_epoch=now+1800,resource_limits={**old['resource_limits'],'wall_seconds':1800,'automatic_retries':0},
        calibration_sha256=sha(batch/'calibration/calibration.json'),
        scope='Frozen-policy low-level transfer to synchronous command delivery,all4original TRAINroads. Zero PPO/BC;not new learning,independent final testing or a matched trained comparison.',
        control_delivery_transfer={'version':'batch_sync_v1','source_model_sha256':metadata['model_sha256'],
            'normalization_sha256':metadata['normalization_sha256'],'new_PPO_controls':0,'new_supervised_updates':0,
            'acknowledgement_check':'Every step must match the engine last-tick control or abort after raw frame preservation.',
            'unchanged':'Vehicle geometry,gains,reward,goal thresholds,NNweights,norm,planner/MPC and20Hzphysical tick.',
            'source_overlays':list(changes),'calibration_change_only':'recommended_adapter.control_delivery',
            'legacy_diagnosis_sha256':sha(ROOT/'reports/native_PPO_legacy_control_delivery_20261101.json'),
            'limits':'Performance is frozen-policy transfer only;any later PPOtraining must explicitly use this new low-level version.'},
        recovery={'B1':{**old['recovery']['B1'],'checkpoint':'resume/B1','parent_batch':parent.name,
            'parent_checkpoint':'B1/'+read(parent/'B1/latest_checkpoint.json')['folder'],
            'lineage_controls':20480,'consumed_controls_lower_bound':cost['charged_controls_lower_bound']+cost['evaluation_control_frames_verified'],
            'retained_optimizer_steps':old['recovery']['B1']['retained_optimizer_steps']+summary['optimizer_steps']}})
    p['extra_supervision']['supervised_optimizer_steps']+=512
    for name in list(changes)+['scripts/prepare_native_control_delivery_probe.py','scripts/audit_native_control_delivery.py']:
        p['source_hashes'][name]=sha(batch/'source'/name)
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(pointer,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),'new_controls_so_far':0})
    return batch


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();print(prepare(),flush=True)
