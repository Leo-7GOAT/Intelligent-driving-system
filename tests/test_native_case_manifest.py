"""Synthetic case registration/rejection tests; no CARLA result is claimed."""
from copy import deepcopy
import hashlib
import json

import pytest

from navstack.native_cases import VERSION, registered_native_cases, verify_registered_native_begin
from navstack.native_hazards import resolve_native_hazard
from navstack.native_route import local_layout


def protocol():
    routes=[]
    for index in range(4):
        points=[{'x':float(x),'y':float(index*10),'z':0.,'yaw_degrees_carla':0.} for x in range(26)]
        geometry=hashlib.sha256(json.dumps([[round(p[k],3) for k in ('x','y','yaw_degrees_carla')] for p in points]).encode()).hexdigest()
        routes.append({'map':'fixture','opendrive_sha256':'fixture-map','points':points,'geometry_id':geometry})
    cases=[{'route_index':i,'hazard':hazard} for hazard in (None,
           {'version':'native_hazard_v1','kind':'static_vehicle','distance_m':13.},
           {'version':'native_hazard_v1','kind':'crossing_walker','lateral_m':-2.5}) for i in range(4)]
    return {'routes':routes,'native_case_manifest':{'version':VERSION,'split':'development','cases':cases}}


def begin_for(p, index):
    case=registered_native_cases(p)[index];route=case['native_route']
    layout=local_layout(route,route['map'],route['opendrive_sha256'])
    scene={'condition':index,'native_layout':layout,'static':[]}
    begin={'scene':scene,'dynamic':[],'native_dynamic_actors':[]}
    if 'native_hazard' in case:
        observed={**resolve_native_hazard(layout,case['native_hazard']),'actor_id':42,
                  'actor_type':'vehicle.tesla.model3' if index<8 else 'walker.pedestrian.0001'}
        scene['native_hazard']=observed;x,y,_=observed['spawn_pose']
        if index<8: scene['static']=[[x-2,y-1,x+2,y+1]]
        else:
            state={'x':x,'y':y,'vx':0.,'vy':0.,'radius':.5};begin['dynamic']=[state]
            begin['native_dynamic_actors']=[{'actor_id':42,'actor_type':observed['actor_type'],'state':state}]
    return begin


def test_twelve_conditions_preserve_all_four_roads_and_three_explicit_actor_families():
    p=protocol();original=deepcopy(p);cases=registered_native_cases(p)
    assert p==original and len(cases)==12
    assert [c['condition'] for c in cases]==list(range(12))
    assert all('native_hazard' not in c for c in cases[:4])
    assert {c['native_hazard']['kind'] for c in cases[4:8]}=={'static_vehicle'}
    assert {c['native_hazard']['kind'] for c in cases[8:]}=={'crossing_walker'}
    for index in range(12):
        assert cases[index]['native_route']==p['routes'][index%4]
        verify_registered_native_begin(begin_for(p,index),p,index)


@pytest.mark.parametrize('fault',['duplicate','bool_index','omitted_road','global_hazard','final_split','extra_field'])
def test_invalid_or_ambiguous_condition_inventories_fail_before_rpc(fault):
    p=protocol();manifest=p['native_case_manifest']
    if fault=='duplicate': manifest['cases'].append(deepcopy(manifest['cases'][4]))
    elif fault=='bool_index':manifest['cases'][0]['route_index']=True
    elif fault=='omitted_road':manifest['cases']=[c for c in manifest['cases'] if c['route_index']!=3]
    elif fault=='global_hazard':p['native_hazard']={'version':'native_hazard_v1','kind':'static_vehicle'}
    elif fault=='final_split':manifest['split']='final'
    else:manifest['cases'][0]['teacher_action']=0
    with pytest.raises(ValueError):registered_native_cases(p)


@pytest.mark.parametrize('geometry',['axis_aligned_box_v1','projected_box_strips12_v1'])
def test_physically_identical_default_aliases_do_not_count_as_distinct_conditions(geometry):
    p=protocol();item=deepcopy(p['native_case_manifest']['cases'][4])
    item['hazard'].update(distance_m=13,static_geometry=geometry)
    p['native_case_manifest']['cases'].append(item)
    with pytest.raises(ValueError,match='Duplicate physical'):registered_native_cases(p)


@pytest.mark.parametrize('fault',['condition','route','hidden_actor','dynamic_identity'])
def test_measured_reset_cannot_be_reassigned_to_another_case_or_hide_actors(fault):
    p=protocol();index=8 if fault=='dynamic_identity' else 0;begin=begin_for(p,index)
    if fault=='condition':begin['scene']['condition']=1
    elif fault=='route':begin['scene']['native_layout']['geometry_id']='different'
    elif fault=='hidden_actor':begin['scene']['static']=[[1.,1.,2.,2.]]
    else:begin['native_dynamic_actors'][0]['actor_id']=43
    with pytest.raises(ValueError):verify_registered_native_begin(begin,p,index)


def test_legacy_global_hazard_inventory_is_unchanged():
    p=protocol();p.pop('native_case_manifest')
    p['native_hazard']={'version':'native_hazard_v1','kind':'static_vehicle','distance_m':10.}
    assert registered_native_cases(p)==[{'scenario':'native_curve','condition':i,'native_route':r,
                                       'native_hazard':p['native_hazard']} for i,r in enumerate(p['routes'])]


def test_fresh_learning_keeps_the_same_twelve_condition_inventory_for_all_methods(tmp_path):
    from test_native_hazard_learning import parent
    from navstack.experiments import write_json
    from scripts.prepare_native_hazard_learning import learning_protocol
    parent(tmp_path)
    original=json.loads((tmp_path/'protocol.json').read_text())
    mixed=protocol();original.update(mixed);original.pop('native_hazard')
    write_json(tmp_path/'protocol.json',original)
    prepared=learning_protocol(tmp_path,20480,7200,11,['H','H0','B1'])
    assert prepared['native_case_manifest']==mixed['native_case_manifest']
    assert registered_native_cases(prepared)==registered_native_cases(mixed)
    assert len(registered_native_cases(prepared))==12 and 'native_hazard' not in prepared


@pytest.mark.parametrize('kind',['rule','policy'])
@pytest.mark.parametrize('fault',[None,'condition','route','missing_case','hidden_actor'])
def test_raw_auditors_bind_manifest_condition_and_reset_inventory(tmp_path,kind,fault):
    from navstack.experiments import write_json
    from scripts.audit_native_method_endpoint import read,read_prefix,audit as policy_audit
    from scripts.audit_native_rule import audit as rule_audit
    from test_native_method_endpoint_audit import _batch,_write_gzip
    from test_native_rule_audit import fixture
    p=protocol();p['routes']=p['routes'][:1];p['native_case_manifest']['cases']=p['native_case_manifest']['cases'][:1]
    if kind=='policy':
        batch=_batch(tmp_path)
    else:
        batch=tmp_path;row=fixture(batch)
        scene_path=batch/'scene_0000.json';scene=read(scene_path)
        begin=begin_for(p,0);scene['scene'].update(begin['scene']);scene['initial'].update(dynamic=[],static=[])
        write_json(scene_path,scene);row['info']['initial_scenario']=scene['initial']
        _write_gzip(batch/'evaluation_trace.jsonl.gz',[row])
    original=read(batch/'protocol.json');original.update(p)
    path=batch/'infrastructure/bridge_fixture_responses.jsonl.gz'
    responses,error=read_prefix(path);assert error is None
    for wrapper in responses:
        response=wrapper['response']
        if 'reset_ticks' not in response:continue
        reset=deepcopy(begin_for(p,0))
        if kind=='rule':reset['scene']=deepcopy(read(batch/'scene_0000.json')['scene'])
        response.update(reset)
        # The policy fixture has a separate training episode. Corrupt only
        # the actual frozen evaluation reset, leaving training evidence intact.
        if kind=='policy' and response['episode_id']=='train':continue
        if fault=='condition':response['scene']['condition']=1
        elif fault=='route':response['scene']['native_layout']['geometry_id']='wrong-road'
        elif fault=='hidden_actor':response['dynamic']=[{'x':1.,'y':1.,'vx':0.,'vy':0.,'radius':.5}]
    if fault=='missing_case':
        original['native_case_manifest']['cases'].append({'route_index':0,'hazard':{'version':'native_hazard_v1','kind':'static_vehicle'}})
    write_json(batch/'protocol.json',original);_write_gzip(path,responses)
    run=lambda:policy_audit(batch,'B1') if kind=='policy' else rule_audit(batch)
    if fault is None:assert run()['total_applied_controls_verified' if kind=='policy' else 'evaluation_controls_verified']>0
    else:
        with pytest.raises(ValueError):run()


@pytest.mark.parametrize('compatible',[False,True])
def test_manifest_transfer_requires_a_runtime_that_handles_every_registered_condition(compatible):
    from test_native_scene_transfer import inputs
    from scripts.prepare_native_scene_transfer import transfer_protocol
    old,reference,lineage=inputs();mixed=protocol()
    old['routes']=deepcopy(mixed['routes']);reference=deepcopy(mixed)
    if compatible:old['source_hashes']={'navstack/native_cases.py':'registered-fixture-source'}
    if not compatible:
        with pytest.raises(ValueError,match='Saved runtime'):transfer_protocol(old,reference,lineage,100.,1800)
    else:
        changed=transfer_protocol(old,reference,lineage,100.,1800)
        assert len(registered_native_cases(changed))==12 and 'native_hazard' not in changed
        assert old['native_hazard']['kind']=='static_vehicle'
