from copy import deepcopy
import pytest
from scripts.native_road_learning import camera_evaluation_mode
from scripts.prepare_native_camera import selected_manifest


def camera_protocol():
    return {'methods':['B1'],'evaluation_only':True,'controlled_budget_per_method':0,
        'method_budgets':{'B1':0},'deterministic_training_route_evaluation':True,
        'recovery':{'B1':{'checkpoint':'resume/B1'}},
        'native_case_manifest':{'cases':[{},{}]},
        'native_camera_evaluation':{'version':'frozen_native_RGB_v1','model_sha256':'a'*64,
            'normalization_sha256':'b'*64,'original_case_ids':[0,3]}}


@pytest.mark.parametrize('change',[
    {'evaluation_only':False},{'controlled_budget_per_method':1280},
    {'method_budgets':{'B1':1280}},{'recovery':{}},{'methods':['H','B1']},
    {'deterministic_training_route_evaluation':False},{'native_validation':True}])
def test_camera_rejects_learning_mixed_methods_or_confirmation(change):
    p=camera_protocol();p.update(change)
    with pytest.raises(ValueError):camera_evaluation_mode(p)


def test_camera_is_explicit_opt_in_and_pins_artifacts_and_case_mapping():
    assert camera_evaluation_mode({}) is False
    p=camera_protocol();assert camera_evaluation_mode(p) is True
    p['native_camera_evaluation']['model_sha256']='unregistered'
    with pytest.raises(ValueError,match='hashes'):camera_evaluation_mode(p)
    p=camera_protocol();p['native_camera_evaluation']['original_case_ids']=[0,0]
    with pytest.raises(ValueError,match='selected known'):camera_evaluation_mode(p)


def test_camera_manifest_is_a_copied_explicit_development_subset():
    manifest={'version':'v1','split':'development','cases':[{'route_index':i,'hazard':None} for i in range(4)]}
    before=deepcopy(manifest);chosen=selected_manifest(manifest,[0,3])
    assert manifest==before and [r['route_index'] for r in chosen['cases']]==[0,3]
    chosen['cases'][0]['route_index']=2
    assert manifest==before
    for ids in ([],[0,0],[-1],[4],[True]):
        with pytest.raises(ValueError):selected_manifest(manifest,ids)
    manifest['split']='final'
    with pytest.raises(ValueError,match='final confirmation'):selected_manifest(manifest,[0])
