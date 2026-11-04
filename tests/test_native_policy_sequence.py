"""History reconstruction is checked against production observation updates."""
from dataclasses import asdict

import numpy as np
import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World
from scripts.audit_native_policy_sequence import observation_env, advance_observation


class FixtureEnv(NavigationEnv):
    def _create_episode(self, options=None):
        self.grid = sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'sequence'},
                                    lambda x, y: 12 < y < 20)
        return World(32., 32., road=RoadGrid(self.grid)), VehiclePlant(
            self.config.vehicle, VehicleState(6., 16.)), (24., 16., 0.), 'fixture'


@pytest.mark.parametrize('feedback', ['enabled', 'ablated'])
def test_reconstructed_history_matches_every_production_observation(feedback):
    config = EnvConfig(execution_feedback=feedback, candidate_observation=True,
                       road_observation=True, deadline=5., planner=PlannerConfig(
                           max_seconds=None, max_nodes=32, analytic_expansion=True))
    env = FixtureEnv(config)
    obs, _ = env.reset(seed=11)
    scene = {'width': 32., 'height': 32., 'goal': list(env.goal), 'static': [], 'road_grid': env.grid}
    initial = {'initial': asdict(env.state), 'goal': list(env.goal), 'static': [],
               'dynamic': [], 'road_sha256': env.world.road.sha256}
    replay = observation_env(config, scene, initial)
    assert not hasattr(replay.plant, 'step')
    np.testing.assert_array_equal(replay._observation(), obs)
    modes = set()
    try:
        for action in (5, 30, 0, 6):
            obs, _, done, _, info = env.step(action)
            modes.add(info['mode'])
            advance_observation(replay, {'policy_action': action, 'info': info})
            np.testing.assert_array_equal(replay._observation(), obs)
            assert replay.state == env.state
            if done:
                break
        assert {'wait', 'reference', 'plan_failed'} <= modes
    finally:
        env.close()


def test_reconstruction_rejects_unrecorded_input_queue():
    config = EnvConfig()
    config.vehicle.input_delay_steps = 1
    with pytest.raises(ValueError, match='no input queue'):
        observation_env(config, {}, {})


def dynamic_macro_fixture():
    from copy import deepcopy
    from navstack.world import MovingObstacle
    config=EnvConfig(execution_feedback='enabled',candidate_observation=True,road_observation=True,
        deadline=5.,planner=PlannerConfig(max_seconds=None,max_nodes=32,analytic_expansion=True))
    live=FixtureEnv(config);live.reset(seed=11)
    live.world.dynamic=[MovingObstacle(12.,18.8,0.,.8,.5)]
    scene={'width':32.,'height':32.,'goal':list(live.goal),'static':[],'road_grid':live.grid}
    initial={'initial':asdict(live.state),'goal':list(live.goal),'static':[],
             'dynamic':[asdict(o) for o in live.world.dynamic],'road_sha256':live.world.road.sha256}
    begin={'scene':scene,'state':initial['initial'],'dynamic':initial['dynamic'],'frame':10,'episode_id':'macro-fixture',
           'native_dynamic_actors':[{'actor_id':42,'actor_type':'walker.pedestrian.0001','state':initial['dynamic'][0]}]}
    replay=observation_env(config,scene,initial,begin=begin)
    _,_,_,_,info=live.step(5);info=deepcopy(info)
    for index,tick in enumerate(info['trace'],1):
        # An arbitrary measured displacement, deliberately not velocity*dt.
        state={'x':12.+.123*index,'y':18.8+.066*index,'vx':0.,'vy':.8,'radius':.5}
        tick['backend_frame']=10+index
        tick['backend_evidence']={'episode_id':'macro-fixture','frame':10+index,
            'native_dynamic_actors':[{'actor_id':42,'actor_type':'walker.pedestrian.0001','state':state}]}
    live.world.dynamic=[MovingObstacle(**state)]
    return live,replay,{'policy_action':5,'info':info}


def test_macro_views_copy_measured_dynamic_frames_without_actor_or_vehicle_integration(monkeypatch):
    live,replay,row=dynamic_macro_fixture()
    def forbidden(*args):raise AssertionError('Measured macro view attempted integration or planning')
    monkeypatch.setattr(replay.world,'advance',forbidden);monkeypatch.setattr(replay.planner,'plan',forbidden)
    try:
        advance_observation(replay,row)
        assert not hasattr(replay.plant,'step')
        assert replay.world.dynamic[0].x==live.world.dynamic[0].x
        np.testing.assert_array_equal(replay._observation(),live._observation())
        assert replay._measured_frame==10+len(row['info']['trace'])
    finally:live.close();replay.close()


@pytest.mark.parametrize('fault',['actor','missing','episode','frame','time','nonfinite'])
def test_macro_view_rejects_invalid_middle_or_terminal_frames_before_any_mutation(fault):
    live,replay,row=dynamic_macro_fixture()
    try:
        before=(asdict(replay.state),asdict(replay.world.dynamic[0]),replay.last_behavior.copy(),replay._measured_frame)
        trace=row['info']['trace'];tick=trace[len(trace)//2];evidence=tick['backend_evidence']
        if fault=='actor':evidence['native_dynamic_actors'][0]['actor_id']=43
        elif fault=='missing':evidence.pop('native_dynamic_actors')
        elif fault=='episode':evidence['episode_id']='other'
        elif fault=='frame':evidence['frame']+=1
        elif fault=='time':tick['state']['time']+=.05
        else:evidence['native_dynamic_actors'][0]['state']['vx']=float('nan')
        with pytest.raises(ValueError):advance_observation(replay,row)
        assert (asdict(replay.state),asdict(replay.world.dynamic[0]))==before[:2]
        np.testing.assert_array_equal(replay.last_behavior,before[2]);assert replay._measured_frame==before[3]
    finally:live.close();replay.close()
