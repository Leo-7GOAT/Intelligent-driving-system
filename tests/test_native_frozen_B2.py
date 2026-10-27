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
