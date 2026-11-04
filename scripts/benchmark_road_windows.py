"""Compare exact road SAT kernels on primitives seeded by saved native states."""
import argparse
from collections import defaultdict
import gzip
import importlib.util
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import steering_transition_primitive
from navstack.road_grid import RoadGrid
from navstack.road_grid_batch import ExactRoadBatch
from navstack.world import body_corners
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def benchmark(batch,legacy_source):
    batch,legacy_source=Path(batch).resolve(),Path(legacy_source).resolve()
    require_finished_parent(batch)
    p=read(batch/'protocol.json');raw=read(batch/'H/endpoint_raw_audit.json')
    trace=batch/'H/evaluation_trace.jsonl.gz';legacy_file=legacy_source/'navstack/road_grid_batch.py'
    if (sha(trace)!=raw['evaluation_trace_sha256']
            or sha(legacy_file)!=p['source_hashes']['navstack/road_grid_batch.py']):
        raise ValueError('Use the bound completed trace and its unchanged original collision kernel')
    spec=importlib.util.spec_from_file_location('navstack._road_batch_benchmark_previous',legacy_file)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=EnvConfig.from_dict(read(batch/'H/config.json')['env']);vehicle=cfg.vehicle
    grids={}
    for file in sorted((batch/'H').glob('scene_*.json')):
        scene=read(file)['scene'];grid=RoadGrid(scene['road_grid']);grids[grid.sha256]=grid
    states=defaultdict(list)
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        import json
        for line in stream:
            row=json.loads(line);info=row['info'];key=info['initial_scenario']['road_sha256']
            states[key].append(info['trace'][0]['state'])
    workload=[]
    for key,values in sorted(states.items()):
        for index in np.linspace(0,len(values)-1,min(24,len(values)),dtype=int):
            s=values[index];start=[s['x'],s['y'],s['yaw']]
            # These are model-generated planner candidates from measured starts,
            # not newly executed or measured vehicle trajectories.
            poses=np.concatenate([steering_transition_primitive(start,s['steering'],delta,
                cfg.planner.primitive_length,vehicle,.25,cfg.planner.collision_step)[0]
                for delta in np.linspace(-vehicle.max_steer,vehicle.max_steer,5)])
            workload.append((key,poses))
    if len(grids)!=8 or set(states)!=set(grids):
        raise ValueError('Require every road of the original complete known24inventory')
    old_check={k:old.ExactRoadBatch(g) for k,g in grids.items()}
    new_check={k:ExactRoadBatch(g) for k,g in grids.items()}
    expected=[];scalar_count=0
    for key,poses in workload:
        a=old_check[key].intersects_many(poses,vehicle)
        b=new_check[key].intersects_many(poses,vehicle)
        scalar=np.array([grids[key].intersects(body_corners(pose,vehicle)) for pose in poses])
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,scalar)
        expected.append(a);scalar_count+=len(poses)
    trials=[]
    for trial in range(5):
        result={}
        arms=[('old',old.ExactRoadBatch),('candidate',ExactRoadBatch)]
        if trial%2:arms.reverse()
        for name,cls in arms:
            begin=time.perf_counter();instances={k:cls(g) for k,g in grids.items()}
            construction=time.perf_counter()-begin
            timings=[]
            for phase in ('cold','warm'):
                begin=time.perf_counter()
                outputs=[instances[key].intersects_many(poses,vehicle) for key,poses in workload]
                elapsed=time.perf_counter()-begin
                for a,b in zip(outputs,expected):np.testing.assert_array_equal(a,b)
                timings.append(elapsed)
            result[name]=dict(construction_seconds=construction,cold_seconds=timings[0],warm_seconds=timings[1],
                cache_hits=sum(getattr(x,'window_cache_hits',0) for x in instances.values()),
                cache_misses=sum(getattr(x,'window_cache_misses',0) for x in instances.values()),
                maximum_cached_cell_rows=max(getattr(x,'_window_cells',0) for x in instances.values()))
        trials.append(result)
    medians={arm:{phase:statistics.median(t[arm][phase] for t in trials)
        for phase in ('construction_seconds','cold_seconds','warm_seconds')} for arm in ('old','candidate')}
    speedups={phase:medians['old'][phase]/medians['candidate'][phase] for phase in ('cold_seconds','warm_seconds')}
    return dict(scope=__doc__,research_acceptance='UNMET',batch=batch.name,
        protocol_sha256=sha(batch/'protocol.json'),bound_trace_sha256=sha(trace),
        legacy_collision_sha256=sha(legacy_file),candidate_collision_sha256=sha(ROOT/'navstack/road_grid_batch.py'),
        benchmark_script_sha256=sha(Path(__file__)),roads=len(grids),measured_seed_states=len(workload),
        generated_primitive_pose_tests=scalar_count,all_boolean_masks_equal_to_old_and_scalar=True,
        fixed_reference_speed=.25,steering_choices=5,trials=trials,medians=medians,speedup_ratios=speedups,
        new_native_controls=0,new_optimizer_updates=0,active_training_source_unchanged=True,
        limits='Offline planner-model primitives seeded by saved measured states;not native execution or full HybridA* replay.All8known roads and up to24uniformly sampled starts perroad retained.Five alternating-order trials on a shared workstation are exploratory kernel timings,not isolated real-time control or new task completion evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    p.add_argument('--legacy-source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    target=Path(a.output)
    if target.exists():raise FileExistsError('Preserve each candidate benchmark')
    result=benchmark(a.batch,a.legacy_source);atomic_json(target,result)
    print({k:result[k] for k in ('generated_primitive_pose_tests','all_boolean_masks_equal_to_old_and_scalar','speedup_ratios')})
