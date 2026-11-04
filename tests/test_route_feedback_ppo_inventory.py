from copy import deepcopy
from collections import Counter
import hashlib
import json
import pytest
from scripts.queue_route_feedback_ppo import training_inventory
from navstack.native_cases import registered_native_cases
from test_native_case_manifest import protocol


def inventories():
    reference=protocol();inventory=deepcopy(reference)
    inventory['routes']=[]
    for index in range(12):
        route=deepcopy(reference['routes'][0])
        for point in route['points']:point['y']=200.+10*index
        route['geometry_id']=hashlib.sha256(json.dumps([[round(p[k],3) for k in ('x','y','yaw_degrees_carla')]
            for p in route['points']]).encode()).hexdigest()
        inventory['routes'].append(route)
    # Deliberately invalid validation payload: the TRAIN selection never reads it.
    reference['native_validation']={'routes':'must never be read'}
    return reference,inventory


def test_ppo_inventory_retains_sixteen_train_roads_and_interleaves_all_families_without_validation():
    reference,inventory=inventories();before=deepcopy((reference,inventory))
    routes,manifest=training_inventory(reference,inventory)
    cases=registered_native_cases(dict(routes=routes,native_case_manifest=manifest))
    assert len(routes)==16 and len(cases)==48 and (reference,inventory)==before
    assert routes==reference['routes'][:4]+inventory['routes']
    assert manifest['split']=='development'
    assert Counter(c.get('native_hazard',{}).get('kind','curve') for c in cases)=={
        'curve':16,'static_vehicle':16,'crossing_walker':16}
    for index in range(16):
        assert [c['native_route'] for c in cases[3*index:3*index+3]]==[routes[index]]*3
    routes[0]['geometry_id']='mutated'
    assert reference==before[0]


def test_duplicate_physical_train_roads_are_rejected_before_any_simulator_command():
    reference,inventory=inventories();inventory['routes'][0]=deepcopy(reference['routes'][0])
    with pytest.raises(ValueError,match='Duplicate physical'):
        training_inventory(reference,inventory)
