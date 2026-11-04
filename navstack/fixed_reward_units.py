"""Fixed PPO reward units; physical environment rewards remain unchanged."""
import numpy as np
from .road_candidate_policy import RoadCandidateVecNormalize


class FixedRewardUnitsNormalize(RoadCandidateVecNormalize):
    def __getstate__(self):
        # Detached checkpoint normalizers have no class_attributes/returns
        # until set_venv(). Preserve SB3's transient-field exclusions in both
        # the detached preparation and attached training states.
        state=self.__dict__.copy()
        for name in ('venv','class_attributes','returns'):state.pop(name,None)
        return state

    def normalize_reward(self,reward):
        if self.fixed_reward_scale!=.05 or self.norm_reward:
            raise ValueError('Require registered fixed1/20reward units without adaptive normalization')
        return np.asarray(reward).copy()*self.fixed_reward_scale

    def unnormalize_reward(self,reward):
        if self.fixed_reward_scale!=.05 or self.norm_reward:
            raise ValueError('Require registered fixed1/20reward units without adaptive normalization')
        return np.asarray(reward).copy()/self.fixed_reward_scale


def initialize_fixed_reward_units(normalizer,scale):
    if type(normalizer) is not RoadCandidateVecNormalize or scale!=.05 or normalizer.norm_reward:
        raise ValueError('Reward units require the unchanged road normalizer and explicit fixed scale.05')
    normalizer.__class__=FixedRewardUnitsNormalize
    normalizer.fixed_reward_scale=scale
    return {'version':'fixed_physical_reward_units_v1','PPO_reward_scale':scale,
            'unsafe_event_in_PPO_units':-4.5,'raw_environment_reward_unchanged':True,
            'observation_statistics_unchanged':True,
            'scope':'Positive constant scale of all PPO rewards,including terminal events; no clipping,adaptive return statistics,extra labels or physical-reward change. Untrained initial critic tensors remain identical.'}
