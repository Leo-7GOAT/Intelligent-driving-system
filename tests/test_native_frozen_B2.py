import pytest
from navstack.config import EnvConfig
from navstack.experiments import signature
from navstack.env import NavigationEnv
from navstack.planning import GridAStarPlanner,HybridAStar
from scripts.prepare_native_planner_ablation import planner_replacement_record


def test_frozen_B2_changes_only_planning_method_and_selects_actual_grid_backend():
    h=EnvConfig(method='H');record={'env':h.to_dict(),'signature':signature(h),'seed':11}
    changed=planner_replacement_record(record);b2=EnvConfig.from_dict(changed['env'])
    assert changed['signature']==signature(b2) and changed['seed']==11
    assert {k for k in set(h.to_dict())|set(b2.to_dict()) if h.to_dict().get(k)!=b2.to_dict().get(k)}=={'method'}
    assert isinstance(NavigationEnv(h).planner,HybridAStar)
    assert isinstance(NavigationEnv(b2).planner,GridAStarPlanner)
    assert NavigationEnv(h).action_space==NavigationEnv(b2).action_space


@pytest.mark.parametrize('method',['B1','B2','B3'])
def test_planner_replacement_rejects_other_source_action_interfaces(method):
    config=EnvConfig(method=method)
    with pytest.raises(ValueError):planner_replacement_record({'env':config.to_dict(),'signature':signature(config)})


def test_body_spline_replacement_changes_only_method_and_declared_postprocessor():
    from copy import deepcopy
    h=EnvConfig(method='H');record={'env':h.to_dict(),'signature':signature(h),'seed':11};original=deepcopy(record)
    changed=planner_replacement_record(record,'body_spline_v1')
    expected=deepcopy(record['env']);expected['method']='B2';expected['planner']['grid_postprocess']='body_spline_v1'
    assert changed['env']==expected and record==original
    assert changed['signature']==signature(EnvConfig.from_dict(expected))


def test_spline_runtime_configuration_matches_registered_saved_semantics():
    from test_native_case_manifest import protocol
    from navstack.config import VehicleConfig
    from scripts.native_road_learning import training_config
    p=protocol();p['episode_seconds']=45;old=training_config(p,'B2',VehicleConfig()).to_dict()
    p['grid_postprocess']='body_spline_v1'
    actual=training_config(p,'B2',VehicleConfig()).to_dict()
    old['planner']['grid_postprocess']='body_spline_v1'
    assert actual==old
    with pytest.raises(ValueError):training_config(p,'H',VehicleConfig())
    p['grid_postprocess']='unknown'
    with pytest.raises(ValueError):training_config(p,'B2',VehicleConfig())


def test_config_override_cannot_replace_training_or_evaluation_loop(tmp_path):
    from scripts.prepare_native_planner_ablation import copy_planning_config_harness
    old=tmp_path/'old.py';new=tmp_path/'new.py';out=tmp_path/'out.py'
    old.write_text('def training_config(): return 1\ndef execute(): return 4\n')
    new.write_text('def training_config(): return 2\ndef execute(): return 4\n')
    copy_planning_config_harness(old,new,out)
    assert out.read_bytes()==new.read_bytes()
    new.write_text('def training_config(): return 2\ndef execute(): return 5\n')
    with pytest.raises(ValueError):copy_planning_config_harness(old,new,out)
