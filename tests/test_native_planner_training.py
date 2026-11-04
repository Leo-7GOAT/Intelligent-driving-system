from copy import deepcopy
import pytest
from navstack.config import EnvConfig
from navstack.experiments import signature
from scripts.prepare_native_planner_ablation import planner_replacement_record
from scripts.prepare_native_planner_training import training_protocol


def inputs():
    protocol = {'methods': ['H', 'H0'], 'controlled_budget_per_method': 20480,
                'method_budgets': {'H': 20480, 'H0': 20480},
                'extra_supervision': {'dataset_sha256': 'same'},
                'freeze_observation_statistics': True,
                'resource_limits': {'wall_seconds': 7200, 'automatic_retries': 0},
                'recovery': {'H': {'lineage_controls': 0, 'retained_optimizer_steps': 0,
                                   'consumed_controls_lower_bound': 9243}}}
    cfg = EnvConfig(method='H')
    record = planner_replacement_record({'env': cfg.to_dict(), 'signature': signature(cfg)}, 'body_spline_v1')
    metadata = {'physics_steps': 0, 'optimizer_steps': 0, 'lineage_control_steps': 0}
    return protocol, metadata, record


def test_same_priors_budget_and_frozen_statistics_without_reusing_trained_H():
    protocol, metadata, record = inputs()
    before = deepcopy(protocol)
    result = training_protocol(protocol, metadata, record, 'reference', 20480, 7200)
    assert protocol == before
    assert result['methods'] == ['B2'] and result['evaluation_only'] is False
    assert result['method_budgets'] == {'B2': 20480}
    assert result['recovery']['B2']['lineage_controls'] == 0
    assert result['recovery']['B2']['consumed_controls_lower_bound'] == 9243
    assert result['recovery']['B2']['checkpoint'] == 'resume/B2'
    assert result['freeze_observation_statistics']
    assert result['extra_supervision'] == before['extra_supervision']
    assert result['grid_postprocess'] == 'body_spline_v1'


@pytest.mark.parametrize('field', ['physics_steps', 'optimizer_steps', 'lineage_control_steps'])
def test_rejects_warm_learning_state_even_when_other_counters_are_zero(field):
    protocol, metadata, record = inputs()
    metadata[field] = 1
    with pytest.raises(ValueError, match='pristine'):
        training_protocol(protocol, metadata, record, 'reference', 20480, 7200)


def test_rejects_hidden_inherited_PPO_updates():
    protocol, metadata, record = inputs()
    protocol['recovery']['H']['retained_optimizer_steps'] = 1
    with pytest.raises(ValueError, match='pristine'):
        training_protocol(protocol, metadata, record, 'reference', 20480, 7200)


@pytest.mark.parametrize('controls', [1280, 40960])
def test_rejects_unmatched_physical_budget(controls):
    protocol, metadata, record = inputs()
    with pytest.raises(ValueError, match='match'):
        training_protocol(protocol, metadata, record, 'reference', controls, 7200)


def test_rejects_eval_only_source_and_implicit_online_extra_learning():
    for extra in ({'evaluation_only': True}, {'own_success_replay': {'steps_per_phase': 10}},
                  {'jobs': [{'method': 'H'}]}, {'freeze_observation_statistics': False}):
        protocol, metadata, record = inputs()
        protocol.update(extra)
        with pytest.raises(ValueError, match='canonical'):
            training_protocol(protocol, metadata, record, 'reference', 20480, 7200)


@pytest.mark.parametrize('controls,wall', [(True, 7200), (20480, True), (20480, 100), (20480, 20000)])
def test_rejects_unbounded_or_boolean_budgets(controls, wall):
    protocol, metadata, record = inputs()
    with pytest.raises(ValueError, match='bounded'):
        training_protocol(protocol, metadata, record, 'reference', controls, wall)
