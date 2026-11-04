"""Prepare (never launch) a frozen H-to-Grid deployment ablation.

All registered known conditions and the original actor/schema/trajectory/MPC
are retained. The explicit B2 profile replaces only spatial search and its
postprocessor. Use the existing bounded launcher after native queues finish.
"""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from navstack.spatial_planner_transfer import PROFILE, settings
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.run_fixed_lowlevel_adaptation import endpoint


def unchanged_ast(path,names):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    tree.body=[node for node in tree.body if not isinstance(node,(ast.FunctionDef,ast.ClassDef)) or node.name not in names]
    return ast.dump(tree,include_attributes=False)


def prepare(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve()
    require_finished_parent(parent)
    p=read(parent/'protocol.json');raw=read(parent/'H/endpoint_raw_audit.json')
    evaluation=read(parent/'H/evaluation.json')
    if (p.get('methods')!=['H'] or p.get('evaluation_only') is not True
            or p.get('reference_spatial_planner') is not None
            or not read(parent/'strict_delivery.json')['strict_delivery_gate_passed']
            or raw['protocol_sha256']!=sha(parent/'protocol.json')
            or evaluation['model_sha256']!=sha(parent/'H/model.zip')
            or len(evaluation['episodes'])!=24
            or p.get('reference_context')!={'version':'action_aligned_outcome_context_v1'}):
        raise ValueError('Require a closed audited H endpoint on all24known development conditions')
    for filename,key in (('config.json','config_sha256'),('summary.json','summary_sha256'),
                         ('evaluation.json','evaluation_sha256'),('evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(parent/'H'/filename)!=raw[key]:raise ValueError('Parent endpoint evidence changed:'+filename)
    if sum(row['physics_steps'] for row in evaluation['episodes'])!=raw['evaluation_control_frames_verified']:
        raise ValueError('Parent complete condition/control accounting differs')
    if output.drive.lower()!='d:' or shutil.disk_usage(output.parent).free<16*1024**3:
        raise ValueError('Use D-drive storage with16GiB headroom')
    for name,digest in p['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Frozen parent source changed:'+name)
    # No whole-tree upgrade: preserve the plant, geometry, trajectory and MPC.
    overlays={'navstack/planning.py':{'grid_search','_grid_search','GridAStarPlanner'},
              'navstack/steering_transition.py':{'settings','prepare_reference'}}
    for name,functions in overlays.items():
        if unchanged_ast(parent/'source'/name,functions)!=unchanged_ast(ROOT/name,functions):
            raise ValueError('Unrelated physical code differs:'+name)
    endpoint(parent,output)
    p=read(output/'protocol.json')
    p.pop('fixed_lowlevel_endpoint',None)
    for name in (*overlays,'navstack/spatial_planner_transfer.py','scripts/prepare_shared_native_grid.py'):
        shutil.copy2(ROOT/name,output/'source'/name)
    p['reference_spatial_planner']=deepcopy(PROFILE)
    p['scope']='B2 frozen-policy deployment transfer on24known conditions;zero learning,not independently trained B2 or final confirmation'
    p['recovery']['H']['consumed_controls_lower_bound']+=raw['evaluation_control_frames_verified']
    p['shared_grid_transfer']=dict(label='B2',runtime_checkpoint_schema='H',
        parent=parent.name,parent_protocol_sha256=sha(parent/'protocol.json'),
        parent_endpoint_audit_sha256=sha(parent/'H/endpoint_raw_audit.json'),
        model_sha256=sha(parent/'H/model.zip'),normalization_sha256=sha(parent/'H/normalization.pkl'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        overrides={name:sorted(functions) for name,functions in overlays.items()},
        preserved='All24conditions,actor,normalizer,observation/action schema,reward,road refinement,point constraints,retiming,MPC,external safety',
        changed='Grid A*+body_spline_v1 spatial search;no Hybrid fallback,no Hybrid-specific memo or speed-based identical Grid retries',
        limits='Grid spatial path has no measured-steering primitive model;the common retimer anchors measured steering and may reject it. Different geometric certifiers and costs are disclosed.',
        new_PPO_updates=0,new_final_conditions=0,selection='All registered known conditions;no outcome-based selection')
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*')
        if f.is_file() and '__pycache__' not in f.parts}
    settings(p)
    from navstack.steering_transition import settings as steering_settings
    steering_settings(p)
    atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(prepare(args.parent,args.output),flush=True)
