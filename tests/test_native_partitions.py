from copy import deepcopy
import hashlib,json
import pytest
from navstack.native_cases import registered_native_cases,verify_registered_native_begin
from navstack.native_partitions import VERSION,validate_native_partitions
from navstack.experiments import write_json
from scripts.audit_native_method_endpoint import audit,read,read_prefix
from test_native_case_manifest import protocol,begin_for
from test_native_method_endpoint_audit import _batch,_write_gzip


def separate_protocol():
    p=protocol();p['routes']=p['routes'][:1];p['native_case_manifest']['cases']=p['native_case_manifest']['cases'][:1]
    val=deepcopy(p)
    for point in val['routes'][0]['points']:point['y']+=50
    refresh_id(val['routes'][0])
    p['native_validation']={'version':VERSION,'routes':val['routes'],'manifest':val['native_case_manifest'],'minimum_route_separation_m':8.}
    return p,val


def refresh_id(route):
    route['geometry_id']=hashlib.sha256(json.dumps([[round(p[k],3) for k in ('x','y','yaw_degrees_carla')] for p in route['points']]).encode()).hexdigest()


def test_old_inventory_is_byte_compatible_and_new_validation_changes_only_evaluation_roads():
    old=protocol();assert registered_native_cases(old)==registered_native_cases(old,'evaluation')
    p,val=separate_protocol();report=validate_native_partitions(p)
    assert report['training_conditions']==report['validation_conditions']==1
    assert registered_native_cases(p)==registered_native_cases({k:v for k,v in p.items() if k!='native_validation'})
    assert registered_native_cases(p,'evaluation')==registered_native_cases(val)
    verify_registered_native_begin(begin_for(val,0),p,0,partition='evaluation')
    with pytest.raises(ValueError):verify_registered_native_begin(begin_for(val,0),p,0)


@pytest.mark.parametrize('fault',['duplicate','nearby','unknown_field','bad_gap','bad_hash','sparse','wrong_map','final'])
def test_partition_leaks_or_unbound_schemas_fail_before_physical_execution(fault):
    p,_=separate_protocol();v=p['native_validation'];route=v['routes'][0]
    if fault=='duplicate':v['routes']=deepcopy(p['routes'])
    elif fault=='nearby':
        for point in route['points']:point['y']=8.
        refresh_id(route)
    elif fault=='unknown_field':v['chosen_after_results']=True
    elif fault=='bad_gap':v['minimum_route_separation_m']=True
    elif fault=='bad_hash':route['geometry_id']='bad'
    elif fault=='sparse':route['points']=route['points'][::2];refresh_id(route)
    elif fault=='wrong_map':route['map']='another'
    else:v['manifest']['split']='final'
    with pytest.raises(ValueError):registered_native_cases(p)


@pytest.mark.parametrize('fault',[None,'training_road_in_eval','validation_road_in_train','missing_partition'])
def test_raw_frame_audit_binds_training_and_validation_resets_separately(tmp_path,fault):
    batch=_batch(tmp_path);p,val=separate_protocol();old=read(batch/'protocol.json');old.update(p);write_json(batch/'protocol.json',old)
    stream=batch/'infrastructure/bridge_fixture_responses.jsonl.gz';rows,error=read_prefix(stream);assert error is None
    for wrapper in rows:
        response=wrapper['response']
        if 'reset_ticks' not in response:continue
        is_eval=response['episode_id']=='evaluate'
        selected=val if is_eval else p
        if fault=='training_road_in_eval' and is_eval:selected=p
        if fault=='validation_road_in_train' and not is_eval:selected=val
        response.update(begin_for(selected,0))
    _write_gzip(stream,rows)
    path=batch/'B1/evaluation.json';evaluation=read(path)
    if fault!='missing_partition':evaluation['partition']='validation'
    write_json(path,evaluation)
    if fault:
        with pytest.raises(ValueError):audit(batch,'B1')
    else:assert audit(batch,'B1')['total_applied_controls_verified']==2


def test_partition_overlay_changes_only_explicit_worker_case_handling_and_preserves_teacher_indices(tmp_path):
    from pathlib import Path
    from scripts.native_partition_initialization import install_inventory,REPLACEMENTS
    root=Path(__file__).resolve().parents[1];current=(root/'scripts/native_road_learning.py').read_text(encoding='utf-8')
    original=current
    for old,new in reversed(REPLACEMENTS):
        assert original.count(new)==1
        original=original.replace(new,old)
    source=tmp_path/'source';(source/'scripts').mkdir(parents=True);(source/'navstack').mkdir()
    worker=source/'scripts/native_road_learning.py';worker.write_text(original,encoding='utf-8')
    p,val=separate_protocol();inventory=tmp_path/'inventory.json';write_json(inventory,p)
    old=deepcopy(p);old.pop('native_validation')
    result=install_inventory(old,source,inventory)
    assert worker.read_text(encoding='utf-8')==current
    assert result['original_teacher_conditions_preserved']==1
    assert registered_native_cases(old,'evaluation')==registered_native_cases(val)
    before=worker.read_bytes()
    with pytest.raises(ValueError):install_inventory(old,source,inventory)
    assert worker.read_bytes()==before


def test_registered_development_inventory_has_balanced_distinct_routes_and_retains_original_teacher_cases():
    from pathlib import Path
    p=read(Path(__file__).resolve().parents[1]/'experiments/native_development_train36_validation24_v1.json')
    train=registered_native_cases(p);validation=registered_native_cases(p,'evaluation')
    assert len(train)==36 and len(validation)==24
    for cases,count in ((train,12),(validation,8)):
        assert sum('native_hazard' not in c for c in cases)==count
        assert sum(c.get('native_hazard',{}).get('kind')=='static_vehicle' for c in cases)==count
        assert sum(c.get('native_hazard',{}).get('kind')=='crossing_walker' for c in cases)==count
    assert p['registration']['no_policy_results_used_for_road_selection']
    assert 'final confirmation' in p['registration']['scope']


@pytest.mark.parametrize('fault',[False,True])
def test_rule_audit_uses_the_same_registered_validation_roads(tmp_path,fault):
    from scripts.audit_native_rule import audit as rule_audit
    from test_native_rule_audit import fixture
    batch=tmp_path;row=fixture(batch);p,val=separate_protocol()
    old=read(batch/'protocol.json');old.update(p);write_json(batch/'protocol.json',old)
    begin=begin_for(p if fault else val,0)
    scene_path=batch/'scene_0000.json';scene=read(scene_path)
    scene['scene'].update(begin['scene']);scene['initial'].update(dynamic=[],static=[])
    write_json(scene_path,scene);row['info']['initial_scenario']=scene['initial']
    _write_gzip(batch/'evaluation_trace.jsonl.gz',[row])
    stream=batch/'infrastructure/bridge_fixture_responses.jsonl.gz';responses,error=read_prefix(stream);assert error is None
    for wrapper in responses:
        response=wrapper['response']
        if 'reset_ticks' in response:
            response.update(begin);response['scene']=deepcopy(scene['scene'])
    _write_gzip(stream,responses)
    if fault:
        with pytest.raises(ValueError):rule_audit(batch)
    else:assert rule_audit(batch)['evaluation_controls_verified']>0
