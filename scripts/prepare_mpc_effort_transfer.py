"""Stage the same closed two-condition learned endpoint with one MPC cost change."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from navstack.mpc_reference_effort import VERSION,settings
from scripts.run_exact_compute_recovery import stage
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def prepare(parent,output):
    parent,output=Path(parent),Path(output);require_finished_parent(parent)
    old=read(parent/'protocol.json')
    if old.get('fixed_lowlevel_endpoint',{}).get('original_case_ids')!=[21,27]:
        raise ValueError('Require the completed same2 learned endpoint')
    if (read(parent/'H/endpoint_raw_audit.json')['protocol_sha256']!=sha(parent/'protocol.json')
            or not read(parent/'strict_delivery.json')['strict_delivery_gate_passed']):
        raise ValueError('Require closed raw/engine verification')
    stage(parent,output,parent/'resume/H',prior_evaluation_controls=read(parent/'H/endpoint_raw_audit.json')['evaluation_control_frames_verified'])
    p=read(output/'protocol.json');p['mpc_reference_effort']=dict(version=VERSION)
    p['mpc_effort_transfer_registration']=dict(original_case_ids=[21,27],source_parent_protocol_sha256=sha(parent/'protocol.json'),
        source_parent_raw_sha256=sha(parent/'H/endpoint_raw_audit.json'),
        change='Center input effort cost on the same timed-reference acceleration and steering feedforward;physical limits and cost weights unchanged',
        frozen_model_sha256=sha(parent/'H/model.zip'),new_PPO_updates=0,new_final_conditions=0,
        scope='Known-condition lower-level engineering transfer;not additional PPO learning or independent confirmation')
    p['scope']='Frozen learned PPO paired MPC reference-effort transfer on the same two known conditions'
    settings(p);atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.parent,a.output)
