import pytest
from scripts.analyze_native_feedback_ppo import adam_step_increments


def test_learning_proof_requires_all_existing_parameter_states_to_advance():
    old={0:{'step':348.},1:{'step':340.}}
    final={0:{'step':668.},1:{'step':660.}}
    assert adam_step_increments(old,final,320)=={'parameter_states':2,'step_increment_each':320}
    with pytest.raises(ValueError):adam_step_increments(old,{0:{'step':668.},1:{'step':340.}},320)
    with pytest.raises(ValueError):adam_step_increments(old,{0:{'step':668.}},320)
    with pytest.raises(ValueError):adam_step_increments(old,final,0)
    with pytest.raises(ValueError):adam_step_increments({0:{'step':.5}},{0:{'step':320.5}},320)
