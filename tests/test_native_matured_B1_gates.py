import json
import pytest
from scripts.run_native_matured_B1_cohort import require_all_basic_gates
from scripts.prepare_native_actor_maturation import prepare


def test_all_five_valid_basic_gates_release_the_common_training_phase():
    require_all_basic_gates([{'seed':s,'passed':True} for s in (11,22,33,44,55)])


def test_one_failed_seed_prevents_any_cohort_training():
    rows=[{'seed':s,'passed':s!=22} for s in (11,22,33,44,55)]
    with pytest.raises(RuntimeError,match='no PPOseeds started'):
        require_all_basic_gates(rows)


@pytest.mark.parametrize('rows',[
    [],[{'seed':s,'passed':True} for s in (11,33,44,55)],
    [{'seed':s,'passed':True} for s in (11,22,33,44,44)],
    [{'seed':s,'passed':'true'} for s in (11,22,33,44,55)]])
def test_missing_duplicated_or_implicit_gate_results_do_not_qualify(rows):
    with pytest.raises(ValueError,match='all five'):
        require_all_basic_gates(rows)


@pytest.mark.parametrize('change',[{'additional_actor_steps':512},{'actor_learning_rate':.0001},
    {'parents':[{'seed':s} for s in (11,33,44,55)]}])
def test_maturation_cannot_change_the_frozen_fit_recipe_or_omit_seeds(tmp_path,change):
    plan={'parents':[{'seed':s} for s in (11,22,33,44,55)],'additional_actor_steps':1024,'actor_learning_rate':.00005,**change}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));output=tmp_path/'fit'
    with pytest.raises(ValueError,match='fixed disclosed'):
        prepare(path,output)
    assert not output.exists()
