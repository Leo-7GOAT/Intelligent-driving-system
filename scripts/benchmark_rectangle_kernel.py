"""Offline same-query comparison against the immutable actual training kernel."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from navstack.config import EnvConfig
from navstack.planning import HybridAStar
from scripts.audit_native_method_endpoint import read,sha
from scripts.benchmark_road_batch import ExperimentalWorld,plan_record,timing
from scripts.experimental_rectangle_kernel import RectangleRoadBatch


def benchmark(baseline,fixture,output,repeats=3):
    baseline,fixture,output=map(Path,(baseline,fixture,output))
    if output.exists() or not 2<=repeats<=10:raise ValueError('Use fresh output and2..10repetitions')
    started=time.monotonic();protocol=read(baseline/'protocol.json');dependencies={}
    for name in ('config','planning','world','road_grid','road_grid_batch','vehicle','dubins_guide'):
        relative='navstack/'+name+'.py';original=baseline/'source'/relative
        if sha(original)!=protocol['source_hashes'][relative] or original.read_text()!=(ROOT/relative).read_text():
            raise ValueError('Another physical/search dependency differs: '+name)
        dependencies[relative]={'original_sha256':sha(original),'current_sha256':sha(ROOT/relative),'source_text_equal':True}
    original=baseline/'source/navstack/road_grid_batch.py'
    spec=importlib.util.spec_from_file_location('navstack._frozen_rectangle_benchmark',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=read(fixture);config=EnvConfig.from_dict(case['config'])
    if config.planner.max_seconds is not None:raise ValueError('Only fixed-node search allowed')
    pose=tuple(case['state'][k] for k in ('x','y','yaw'))
    def world(kernel,capture=False):
        instance=ExperimentalWorld(case['world'],'batch',capture=capture)
        instance._road_batch=kernel(instance.road);return instance
    def signature(plan):
        result=plan_record(plan)
        if result['cost'] is not None and not np.isfinite(result['cost']):result['cost']=None
        return result
    captured=world(module.ExactRoadBatch,True)
    expected=signature(HybridAStar(config.vehicle,deepcopy(config.planner)).plan(pose,case['goal'],captured))
    for key in ('status','expanded_nodes','failure_reason'):
        if expected[key]!=case['recorded_plan'][key]:raise ValueError('Offline query does not reproduce original '+key)
    if not captured.calls:raise ValueError('No actual planner collision workload captured')
    kernels={'original_cells':module.ExactRoadBatch(captured.road),'rectangle_partition':RectangleRoadBatch(captured.road)}
    flags=[kernels['original_cells'].intersects_many(poses,config.vehicle,margin) for poses,margin in captured.calls]
    for (poses,margin),expected_flags in zip(captured.calls,flags):
        if not np.array_equal(expected_flags,kernels['rectangle_partition'].intersects_many(poses,config.vehicle,margin)):
            raise ValueError('Actual collision flags changed')
    kernel_times={k:[] for k in kernels};search_times={k:[] for k in kernels}
    for repetition in range(repeats):
        for key in (list(kernels) if repetition%2==0 else list(reversed(kernels))):
            t=time.perf_counter()
            for poses,margin in captured.calls:kernels[key].intersects_many(poses,config.vehicle,margin)
            kernel_times[key].append(time.perf_counter()-t)
            instance=world(type(kernels[key]));t=time.perf_counter()
            result=signature(HybridAStar(config.vehicle,deepcopy(config.planner)).plan(pose,case['goal'],instance))
            search_times[key].append(time.perf_counter()-t)
            if result!=expected:raise ValueError('Full nodes/status/cost/path changed')
    report={'scope':__doc__,'baseline_batch':baseline.name,'fixture_sha256':sha(fixture),
        'prototype_sha256':sha(ROOT/'scripts/experimental_rectangle_kernel.py'),'benchmark_sha256':sha(Path(__file__)),
        'dependencies':dependencies,'repeats':repeats,'recorded_query_reproduced':True,
        'actual_planner_batches':len(captured.calls),'actual_planner_poses':sum(len(p) for p,_ in captured.calls),
        'collision_flags_exact':True,'full_search_exact':expected,
        'occupied_rectangles':{k:len(v.rectangles) for k,v in kernels.items()},
        'kernel_timings':{k:timing(v) for k,v in kernel_times.items()},
        'search_timings':{k:timing(v) for k,v in search_times.items()},
        'limits':'Finite saved queries only,not universal floating-point equivalence or full-runner realtime proof. Interleaved while independent frozen CARLAtraining may run.',
        'production_behavior_changed':False,'new_controls':0,'optimizer_updates':0,'wall_seconds':time.monotonic()-started}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('baseline','fixture','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();benchmark(args.baseline,args.fixture,args.output,args.repeats)
