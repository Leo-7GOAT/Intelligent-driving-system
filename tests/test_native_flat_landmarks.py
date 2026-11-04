from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.diagnose_native_flat_landmarks import decision_landmarks, require_same_observation_normalization


def test_landmarks_exclude_unexecuted_decision_after_terminal():
    rows=[]
    for x in (.8,.3,0.):
        rows.append({'info':{'initial_scenario':{'goal':[0,0,0],'initial':{'x':2.,'y':0.}},
                             'trace':[{'state':{'x':x,'y':0.}}]}})
    result=decision_landmarks(rows,.8)
    assert result=={'initial':0,'closest_observed_decision':2,
                    'last_observed_decision':2,'first_position_entry_decision':1}


def test_landmarks_do_not_invent_goal_entry():
    rows=[{'info':{'initial_scenario':{'goal':[0,0,0],'initial':{'x':2.,'y':0.}},
                   'trace':[{'state':{'x':0.,'y':0.}}]}}]
    assert decision_landmarks(rows,.8)=={'initial':0,'closest_observed_decision':0,'last_observed_decision':0}


def test_return_statistics_may_differ_but_observation_transform_must_be_exact():
    norm=SimpleNamespace(obs_rms=SimpleNamespace(mean=np.zeros(3),var=np.ones(3),count=20.),
                         norm_obs=True,clip_obs=10.,epsilon=1e-8,ret_rms='original')
    other=deepcopy(norm);other.ret_rms='updated'
    require_same_observation_normalization(norm,other)
    other.obs_rms.mean[1]=.001
    with pytest.raises(AssertionError):
        require_same_observation_normalization(norm,other)
    other=deepcopy(norm);other.clip_obs=5.
    with pytest.raises(ValueError,match='transform differs'):
        require_same_observation_normalization(norm,other)
