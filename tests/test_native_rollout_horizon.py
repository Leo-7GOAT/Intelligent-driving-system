import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from scripts.diagnose_native_rollout_horizon import bootstrap_coefficients


@pytest.mark.parametrize('terminals', [[False]*4,[False,False,False,True],[False,True,False,False]])
def test_endpoint_sensitivity_matches_actual_sb3_gae(terminals):
    def returns(tail_value):
        buffer=RolloutBuffer(4,spaces.Box(-1,1,(1,)),spaces.Discrete(2),
                             device='cpu',gamma=.9,gae_lambda=.8)
        for i in range(4):
            buffer.add(np.array([[0.]],dtype=np.float32),np.array([0]),np.array([1.]),
                       np.array([i==0 or terminals[i-1]]),torch.tensor([0.]),torch.tensor([0.]))
        buffer.compute_returns_and_advantage(torch.tensor([tail_value]),np.array([terminals[-1]]))
        return buffer.returns[:,0].copy()
    actual=returns(1.)-returns(0.)
    np.testing.assert_allclose(actual,bootstrap_coefficients(terminals,.9,.8),atol=2e-7)
