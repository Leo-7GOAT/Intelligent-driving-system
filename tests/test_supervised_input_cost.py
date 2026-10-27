import pytest
from scripts.prepare_native_supervised_ppo import supervised_input_cost


@pytest.mark.parametrize('teacher,prior,expected',[(4329,16717,16717),(4329,4329,4329),(4329,0,4329)])
def test_PPO_continuation_keeps_prior_physical_input_cost(teacher,prior,expected):
    protocol={'extra_supervision':{'teacher_controls':teacher},'recovery':{'B1':{'consumed_controls_lower_bound':prior}}}
    assert supervised_input_cost(protocol,'B1')==expected


@pytest.mark.parametrize('prior',[-1,True,1.5])
def test_invalid_physical_cost_rejected(prior):
    with pytest.raises(ValueError):
        supervised_input_cost({'extra_supervision':{'teacher_controls':4329},'recovery':{'B1':{'consumed_controls_lower_bound':prior}}},'B1')
