"""Keep actual policy inputs from successful on-policy episodes for actor replay."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class OwnSuccessMemory(BaseCallback):
    def __init__(self,observations,actions,groups,condition,qualify=None):
        super().__init__()
        self.initial=(observations.copy(),actions.copy(),groups.copy())
        self.condition=condition;self.episode_index=0;self.pending=[];self.safe=True
        self.extra_observations=[];self.extra_actions=[];self.extra_groups=[]
        self.successes=0;self.qualify=qualify;self.rejected_quality=0

    def _on_step(self):
        if len(self.locals['infos'])!=1:raise ValueError('Own replay currently requires one measured environment')
        info=self.locals['infos'][0]
        self.pending.append((self.model._last_obs[0].copy(),self.locals['actions'][0].copy()))
        self.safe=self.safe and not any(t['intervention'] or t['controller_status']=='failed' for t in info.get('trace',[]))
        if self.locals['dones'][0]:
            if info.get('success',False) and self.safe:
                if self.qualify is not None and not self.qualify(info):self.rejected_quality+=1
                else:
                    group=self.condition(self.episode_index)
                    for observation,action in self.pending:
                        self.extra_observations.append(observation);self.extra_actions.append(action);self.extra_groups.append(group)
                    self.successes+=1
            self.pending=[];self.safe=True;self.episode_index+=1
        return True

    def arrays(self):
        if not self.extra_groups:return self.initial
        return tuple(np.concatenate((initial,np.asarray(extra,dtype=initial.dtype)))
            for initial,extra in zip(self.initial,(self.extra_observations,self.extra_actions,self.extra_groups)))

    def save(self,path):
        observations,actions,groups=self.arrays()
        np.savez_compressed(path,normalized_observations=observations,actions=actions,groups=groups)
        return {'samples':len(groups),'new_own_success_decisions':len(self.extra_groups),
                'new_own_success_episodes':self.successes,
                'successful_episodes_rejected_by_quality':self.rejected_quality,
                'scope':'Actual frozen-normalizer policy inputs; failed episodes remain in raw logs; no teacher or new simulator execution'}
