import ast
from pathlib import Path
import pytest
from navstack.config import EnvConfig
from navstack.experiments import signature
from scripts.prepare_native_control_cadence import cadence_record,environment_overlay,worker_overlay


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('method',['H','B1'])
def test_resolved_record_discloses_true_cadence_without_changing_physics(method):
    config=EnvConfig(method=method,flat_hold=1);record={'env':config.to_dict(),'signature':signature(config)}
    actual=cadence_record(record,method,5)
    assert record['env']==config.to_dict() and actual['env']['control_dt']==.05
    assert actual['deployment_control_cadence']['command_update_seconds']==.25
    assert actual['deployment_control_cadence']['MPCdesign_retuned'] is False
    assert actual['env']=={**record['env'],**({'flat_hold':5} if method=='B1' else {})}


def test_environment_overlay_adds_only_controller_reset_wrapper():
    original=(ROOT/'navstack/env.py').read_text();changed=environment_overlay(original)
    def other(text):
        tree=ast.parse(text)
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='NavigationEnv':
                node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name!='reset']
        return ast.dump(tree,include_attributes=False)
    assert other(original)==other(changed)
    with pytest.raises(ValueError):environment_overlay(changed)


def test_worker_overlay_changes_only_registered_config_and_metadata():
    original=(ROOT/'scripts/native_road_learning.py').read_text();changed=worker_overlay(original)
    assert 'configure_cadence(config,protocol,method)' in changed
    assert '"deployment_control_cadence": protocol["deployment_control_cadence"]' in changed
    with pytest.raises(ValueError):worker_overlay(changed)


@pytest.mark.parametrize('hold',[1,True,3])
def test_only_the_two_registered_new_cadences_are_prepared(hold):
    config=EnvConfig();record={'env':config.to_dict(),'signature':signature(config)}
    with pytest.raises(ValueError):cadence_record(record,'H',hold)
