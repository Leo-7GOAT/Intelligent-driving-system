import pytest


def test_current_fixed_controller_uses_centered_cost_without_changing_legacy():
    from navstack.config import VehicleConfig
    from scripts.native_curve_control import make_controller
    vehicle=VehicleConfig();p=dict(dt=.05,mpc_horizon=20,comparison_controller='mpc20')
    assert make_controller(vehicle,p).reference_centered_effort is False
    profile=dict(version='current_fixed_reference_control_v1',reference_centered_effort=True,new_training_controls=0,new_optimizer_updates=0)
    assert make_controller(vehicle,{**p,'current_fixed_reference_control':profile}).reference_centered_effort is True
    with pytest.raises(ValueError):make_controller(vehicle,{**p,'dt':.1,'current_fixed_reference_control':profile})


def test_current_fixed_pursuit_retains_registered_parameters_and_rejects_hidden_tuning():
    from navstack.config import VehicleConfig
    from navstack.pursuit_controller import TimedPursuitPI,PARAMETERS
    from scripts.native_curve_control import make_controller
    p=dict(dt=.05,mpc_horizon=20,comparison_controller='pursuit_pi',controller_parameters=PARAMETERS,
        current_fixed_reference_control=dict(version='current_fixed_reference_control_v1',reference_centered_effort=True,new_training_controls=0,new_optimizer_updates=0))
    assert isinstance(make_controller(VehicleConfig(),p),TimedPursuitPI)
    with pytest.raises(ValueError):make_controller(VehicleConfig(),{**p,'controller_parameters':{}})


def test_fixed_reference_delivery_episodes_are_not_mislabeled_training(tmp_path):
    import gzip,json
    from scripts.audit_native_control_delivery import evaluation_episodes
    folder=tmp_path/'case_00';folder.mkdir()
    with gzip.open(folder/'transitions.jsonl.gz','wt') as stream:
        stream.write(json.dumps({'measurement':{'episode_id':'controller_episode','request_id':2}})+'\n')
    assert evaluation_episodes(tmp_path,{'comparison_controller':'mpc20'})=={'controller_episode'}
