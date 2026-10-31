from types import SimpleNamespace
import pytest
from navstack.control_cadence import HeldMPC,configure_cadence,VERSION
from navstack.config import EnvConfig
from navstack.mpc import ControlResult,MPC
from navstack.vehicle import Command,VehicleState


class Controller:
    dt=.05
    def __init__(self):self.calls=[];self.status='solved'
    def step(self,state,reference,previous):
        self.calls.append((state.time,reference,previous))
        return ControlResult(Command(float(len(self.calls)),.1),self.status,
                             {'control_seconds':.02,'qp_seconds':.01})


@pytest.mark.parametrize('hold,expected',[(1,6),(2,3),(5,2)])
def test_true_recompute_cadence_does_not_skip_measured_physics_calls(hold,expected):
    inner=Controller();controller=HeldMPC(inner,hold);reference=object();previous=Command()
    results=[controller.step(SimpleNamespace(time=i*.05),reference,previous) for i in range(6)]
    assert len(inner.calls)==expected and len(results)==6
    assert sum(r.diagnostics['controller_recomputed'] for r in results)==expected
    for i,r in enumerate(results):
        assert r.command.acceleration==i//hold+1 and r.status=='solved'
        assert r.diagnostics['control_seconds']==(.02 if i%hold==0 else 0.)
    assert all(call[2] is previous for call in inner.calls)


def test_new_reference_forces_fresh_control_before_hold_expires():
    inner=Controller();controller=HeldMPC(inner,5)
    controller.step(SimpleNamespace(time=1.),object(),Command())
    controller.step(SimpleNamespace(time=1.05),object(),Command())
    assert len(inner.calls)==2


@pytest.mark.parametrize('time',[0.,2.])
def test_clock_rewind_or_long_gap_cannot_reuse_stale_command(time):
    inner=Controller();controller=HeldMPC(inner,5);reference=object()
    controller.step(SimpleNamespace(time=1.),reference,Command())
    controller.step(SimpleNamespace(time=time),reference,Command())
    assert len(inner.calls)==2


def test_held_solver_failure_and_its_diagnostics_are_not_reported_as_a_new_solve():
    inner=Controller();inner.status='failed';controller=HeldMPC(inner,2);reference=object()
    actual=controller.step(SimpleNamespace(time=0.),reference,Command())
    held=controller.step(SimpleNamespace(time=.05),reference,Command())
    assert actual.status==held.status=='failed' and actual.command==held.command
    assert actual.diagnostics['controller_recomputed'] is True
    assert held.diagnostics['controller_recomputed'] is False and held.diagnostics['qp_seconds']==0
    assert inner.calls and actual.diagnostics['qp_seconds']==.01


@pytest.mark.parametrize('hold',[True,0,3,10])
def test_only_registered_native_cadences_are_allowed(hold):
    with pytest.raises(ValueError):HeldMPC(Controller(),hold)


def test_control_hold_cannot_change_the_actual_physics_grid():
    inner=Controller();inner.dt=.1
    with pytest.raises(ValueError,match='physics'):
        HeldMPC(inner,2)


@pytest.mark.parametrize('method',['H','B1'])
def test_only_control_updates_change_while_actual_physics_and_Hdecision_stay_fixed(method):
    config=EnvConfig(method=method,flat_hold=1);before=config.to_dict()
    protocol={'evaluation_only':True,'controlled_budget_per_method':0,
              'deployment_control_cadence':{'version':VERSION,'hold_steps':5}}
    configure_cadence(config,protocol,method)
    assert config.control_dt==.05 and config.decision_dt==.5
    if method=='H':
        assert config.to_dict()==before and config.native_control_hold_steps==5
    else:
        assert config.to_dict()=={**before,'flat_hold':5} and config.substeps==5


@pytest.mark.parametrize('change',[{'evaluation_only':False},{'controlled_budget_per_method':1},
    {'deployment_control_cadence':{'version':VERSION,'hold_steps':True}},
    {'deployment_control_cadence':{'version':VERSION,'hold_steps':3}}])
def test_unregistered_or_training_cadence_changes_fail_before_driving(change):
    protocol={'evaluation_only':True,'controlled_budget_per_method':0,
              'deployment_control_cadence':{'version':VERSION,'hold_steps':2},**change}
    with pytest.raises(ValueError,match='zero-update'):
        configure_cadence(EnvConfig(),protocol,'H')


def test_one_tick_wrapper_preserves_actual_MPC_commands_on_the_same_measured_states():
    from navstack.config import VehicleConfig
    from navstack.trajectory import parameterize
    vehicle=VehicleConfig();reference=parameterize([[4,8,0],[16,8,0]],1.,0.,vehicle,stop=True)
    original=MPC(vehicle);wrapped=HeldMPC(MPC(vehicle),1);previous=Command()
    for i in range(8):
        state=VehicleState(4+i*.05,8,v=1.,time=i*.05)
        before=original.step(state,reference,previous);after=wrapped.step(state,reference,previous)
        assert before.status==after.status=='solved'
        assert after.command.acceleration==pytest.approx(before.command.acceleration,abs=1e-12)
        assert after.command.steering==pytest.approx(before.command.steering,abs=1e-12)
        assert after.diagnostics['controller_recomputed'] is True
        previous=before.command
