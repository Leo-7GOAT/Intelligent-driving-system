"""Structured road-policy inputs with explicitly versioned optional filters.

The 605 existing state/candidate features are followed by the same 64x64 binary
road raster supplied to every method. Road and candidate geometry retain fixed
physical encodings. H scores each candidate with shared weights; B1 pools the
same candidate set and emits direct actuator commands. The encoder adds no route
prior; a versioned route catalog can supply one through shared candidate geometry.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from .candidate_policy import CandidateVecNormalize,SharedActionScores


class RoadCandidateVecNormalize(CandidateVecNormalize):
    def _normalize_obs(self, obs, obs_rms):
        normalized=super()._normalize_obs(obs,obs_rms)
        if obs.shape[-1] not in (4701,4705):
            raise ValueError('Road candidate normalization requires4701 or4705inputs')
        normalized[...,605:4701]=obs[...,605:4701]
        return normalized

    def _unnormalize_obs(self, obs, obs_rms):
        original=super()._unnormalize_obs(obs,obs_rms)
        if obs.shape[-1] not in (4701,4705):
            raise ValueError('Road candidate normalization requires4701 or4705inputs')
        original[...,605:4701]=obs[...,605:4701]
        return original


def road_encoder():
    return nn.Sequential(nn.Conv2d(1,8,5,stride=2),nn.Tanh(),
        nn.Conv2d(8,16,3,stride=2),nn.Tanh(),nn.AdaptiveAvgPool2d((8,8)),
        nn.Flatten(),nn.Linear(16*8*8,64),nn.Tanh())


class RoadCandidateExtractor(nn.Module):
    def __init__(self, flat=False, actuator_inputs=0):
        super().__init__()
        self.flat=flat
        self.latent_dim_pi=64 if flat else 31*32
        self.latent_dim_vf=64
        # Names retain the existing actor-vector accounting convention.
        self.policy_net_road=road_encoder()
        self.policy_net_context=nn.Sequential(nn.Linear(357+actuator_inputs+64,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.policy_net_candidate=nn.Sequential(nn.Linear(8 if flat else 72,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
        if flat:
            self.policy_net_fusion=nn.Sequential(nn.Linear(64+64,64),nn.Tanh())
        # Actor and critic have separate road encoders; no hidden parameter tie.
        self.value_net_road=road_encoder()
        self.value_net_candidate=nn.Sequential(nn.Linear(8,32),nn.Tanh())
        self.value_net_fusion=nn.Sequential(nn.Linear(357+actuator_inputs+64+64,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())

    @staticmethod
    def parts(features):
        return (torch.cat((features[:,:351],features[:,599:605],features[:,4701:]),dim=1),
                features[:,351:599].reshape(-1,31,8),
                features[:,605:4701].reshape(-1,1,64,64))

    def forward_actor(self, features):
        state,candidates,road=self.parts(features)
        context=self.policy_net_context(torch.cat((state,self.policy_net_road(road)),dim=1))
        if self.flat:
            encoded=self.policy_net_candidate(candidates)
            return self.policy_net_fusion(torch.cat((context,encoded.mean(1),encoded.amax(1)),dim=1))
        joined=torch.cat((context[:,None,:].expand(-1,31,-1),candidates),dim=2)
        return self.policy_net_candidate(joined).flatten(1)

    def forward_critic(self, features):
        state,candidates,road=self.parts(features)
        encoded=self.value_net_candidate(candidates)
        return self.value_net_fusion(torch.cat((state,self.value_net_road(road),encoded.mean(1),encoded.amax(1)),dim=1))

    def forward(self, features):
        return self.forward_actor(features),self.forward_critic(features)


class RoadCandidateActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        if self.observation_space.shape not in ((4701,),(4705,)) or not isinstance(self.action_space,spaces.Discrete) or self.action_space.n!=31:
            raise ValueError('Road candidate policy requires4701 or4705inputs and31categorical actions')
        self.mlp_extractor=RoadCandidateExtractor(actuator_inputs=self.observation_space.shape[0]-4701).to(self.device)
        self.action_net=SharedActionScores().to(self.device)
        self.value_net=nn.Linear(64,1).to(self.device)
        if self.ortho_init:
            for module,gain in ((self.mlp_extractor,np.sqrt(2)),(self.action_net,.01),(self.value_net,1.)):
                module.apply(partial(self.init_weights,gain=gain))
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class FlatRoadCandidateActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        if self.observation_space.shape not in ((4701,),(4705,)) or not isinstance(self.action_space,spaces.Box) or self.action_space.shape!=(2,):
            raise ValueError('Flat road policy requires4701 or4705inputs and2direct continuous controls')
        self.mlp_extractor=RoadCandidateExtractor(flat=True,actuator_inputs=self.observation_space.shape[0]-4701).to(self.device)


class OrderedFlatRoadExtractor(RoadCandidateExtractor):
    """Direct control retains every fixed-catalog candidate instead of pooling."""
    def forward_actor(self, features):
        state,candidates,road=self.parts(features)
        context=self.policy_net_context(torch.cat((state,self.policy_net_road(road)),dim=1))
        encoded=self.policy_net_candidate(candidates).flatten(1)
        return self.policy_net_fusion(torch.cat((context,encoded),dim=1))


class OrderedFlatRoadCandidatePolicy(FlatRoadCandidateActorCriticPolicy):
    def _build_mlp_extractor(self):
        if self.observation_space.shape not in ((4701,),(4705,)) or not isinstance(self.action_space,spaces.Box) or self.action_space.shape!=(2,):
            raise ValueError('Ordered flat policy requires4701 or4705inputs and2direct continuous controls')
        self.mlp_extractor=OrderedFlatRoadExtractor(flat=True,actuator_inputs=self.observation_space.shape[0]-4701).to(self.device)

    def _build(self,lr_schedule):
        # Build the original encoders, critic and head first. Only the actor
        # fusion changes; its initialization must not shift their seeded bytes.
        super()._build(lr_schedule)
        with torch.random.fork_rng(devices=[]):
            fusion=nn.Sequential(nn.Linear(64+31*32,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
            if self.ortho_init:fusion.apply(partial(self.init_weights,gain=np.sqrt(2)))
        self.mlp_extractor.policy_net_fusion=fusion.to(self.device)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class UnsharedRoadCandidatePolicy(RoadCandidateActorCriticPolicy):
    """Versioned fixed-catalog head; shared encoders/critic, independent logits.

    A shared score applied to identical candidate embeddings cancels common
    context in softmax. This head can use that context directly and also sees
    all candidate embeddings. It sacrifices candidate permutation equivariance;
    the registered route catalog therefore retains its fixed slot semantics.
    """
    def _build(self, lr_schedule):
        # Construct the original network first, preserving identical seeded
        # encoders, critic and global RNG state for the matched fresh control.
        super()._build(lr_schedule)
        with torch.random.fork_rng(devices=[]):
            self.action_net=nn.Linear(31*32,31).to(self.device)
            if self.ortho_init:
                self.action_net.apply(partial(self.init_weights,gain=.01))
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class LocalWitnessRoadPolicy(UnsharedRoadCandidatePolicy):
    """PPO still chooses wait/target/speed among sufficient local references.

    Global continuation is not required. All-unknown local states retain the
    full action set so Hybrid A* can search beyond analytic connectors.
    """
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from .candidate_filter import LocalWitnessExtractor,LocalWitnessHead
        self.mlp_extractor=LocalWitnessExtractor(self.mlp_extractor)
        self.action_net=LocalWitnessHead(self.action_net)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class EndpointClearRoadPolicy(UnsharedRoadCandidatePolicy):
    """PPO chooses among necessary endpoint-clear actions, including wait.

    No local/global connector witness is required. Unknown feedback is unmasked.
    """
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from .candidate_filter import EndpointClearExtractor,LocalWitnessHead
        self.mlp_extractor=EndpointClearExtractor(self.mlp_extractor)
        self.action_net=LocalWitnessHead(self.action_net)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class SharedEndpointAblatedRoadPolicy(EndpointClearRoadPolicy):
    """H0: same physical endpoint gate as H,zero feedback in learned networks."""
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        from .candidate_filter import SharedEndpointFeedbackAblationExtractor
        self.mlp_extractor=SharedEndpointFeedbackAblationExtractor(self.mlp_extractor.backbone)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


def endpoint_policy_class(protocol,method):
    shared=protocol.get('shared_endpoint_mask_v1',False)
    if type(shared) is not bool:raise ValueError('Shared endpoint mask must be an explicit boolean')
    return SharedEndpointAblatedRoadPolicy if shared and method=='H0' else EndpointClearRoadPolicy
