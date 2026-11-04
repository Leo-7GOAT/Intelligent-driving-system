from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import probe_native_execution_prediction as probe


def test_validation_withholds_both_the_entire_route_and_both_policy_seeds():
    seed=np.repeat([11,22,33,44,55],2);route=np.tile(['heldout','other'],5)
    train,test=probe.validation_masks(seed,route,'heldout')
    assert np.flatnonzero(train).tolist()==[1,3,5]
    assert np.flatnonzero(test).tolist()==[6,8]
    assert not np.any(train&test)


def test_history_baseline_does_not_read_validation_labels():
    feedback=np.zeros((6,6));feedback[[2,3,5],1]=1
    y=np.array([0,1,1,1,0,1]);train=np.array([True]*4+[False]*2);test=~train
    predicted=probe.history_only_probabilities(feedback,y,train,test)
    np.testing.assert_array_equal(predicted,[.5,.75])
    changed=y.copy();changed[test]=1-changed[test]
    np.testing.assert_array_equal(predicted,probe.history_only_probabilities(feedback,changed,train,test))


def test_action_switches_do_not_cross_episode_or_method_boundaries():
    method=np.array(['H','H','H','H0','H0'])
    episode=np.array([0,0,1,1,1]);action=np.array([2,3,4,5,5])
    assert probe.action_switches(method,episode,action).tolist()==[False,True,False,False,False]


def test_selected_features_use_only_current_measured_state_and_previous_feedback(monkeypatch):
    state=SimpleNamespace(x=0.,y=0.,yaw=0.,v=1.5,steering=.1,acceleration=.5,time=2.)
    env=SimpleNamespace(state=state,goal=[10.,0.,0.],execution_outcome=np.array([0.,1.,0.,0.,0.,.2],dtype=np.float32),
        plant=SimpleNamespace(previous_command=SimpleNamespace(acceleration=.3,steering=.1)),
        config=SimpleNamespace(vehicle=object()),world=SimpleNamespace(dynamic=[],free=lambda *args:True),
        decode_action=lambda action:((6.,1.,.2),1.5))
    before=deepcopy(vars(state));past=env.execution_outcome.copy()
    monkeypatch.setattr(probe,'analytic_connection',lambda *args:np.zeros((2,3)))
    monkeypatch.setattr(probe,'candidate_goal_witness',lambda *args:False)
    geometry,feedback=probe.features(env,2)
    assert geometry.shape==(35,) and feedback.shape==(6,)
    assert geometry[0]==.5 and geometry[11]==pytest.approx(6/15)
    np.testing.assert_array_equal(geometry[17:20],[1.,1.,0.])
    np.testing.assert_array_equal(feedback,past)
    feedback[:]=9
    assert vars(state)==before
    np.testing.assert_array_equal(env.execution_outcome,past)
    assert not hasattr(env.plant,'step')


def test_same_seeded_fit_cannot_use_validation_labels():
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        x=np.random.default_rng(8).normal(size=(12,41)).astype(np.float32)
        y=np.array([0,1]*6);train=np.array([True]*10+[False]*2);test=~train
        before=probe.fit(x,y,train,test)
        altered=y.copy();altered[test]=1-altered[test]
        np.testing.assert_array_equal(before,probe.fit(x,altered,train,test))
    finally:
        torch.set_num_threads(previous_threads)


def test_probability_metrics_account_for_ties():
    assert probe.metrics(np.array([.5,.5]),np.array([0,1]))['AUC']==.5
    result=probe.metrics(np.array([.1,.8]),np.array([0,1]))
    assert result['AUC']==1. and result['Brier']==pytest.approx(.025)
