"""Deployment-only zero-order command hold;CARLA physics stays at20Hz."""
from dataclasses import replace
import math

VERSION='native_control_hold_v1'


def configure_cadence(config,protocol,method):
    profile=protocol.get('deployment_control_cadence')
    if (not isinstance(profile,dict) or profile.get('version')!=VERSION
            or type(profile.get('hold_steps')) is not int or profile['hold_steps'] not in (2,5)
            or protocol.get('evaluation_only') is not True
            or protocol.get('controlled_budget_per_method')!=0
            or not math.isclose(config.control_dt,.05,rel_tol=0,abs_tol=1e-12)
            or config.shared_shield or method not in ('H','B1')):
        raise ValueError('Registered zero-update native20Hz cadence probe required')
    if method=='B1':
        config.flat_hold=profile['hold_steps']
        config.__post_init__()
    else:
        config.native_control_hold_steps=profile['hold_steps']
    return config


class HeldMPC:
    """Keep the existing20Hz prediction model;change recomputation cadence only.

    This is a deployment stress probe,not a retuned lower-frequency MPC design.
    Reference changes,clock rewind and long gaps invalidate the cached command.
    Native braking outside MPC and per-frame event checks remain with the caller.
    """
    def __init__(self, controller, hold_steps):
        if type(hold_steps) is not int or hold_steps not in (1,2,5):
            raise ValueError('Explicit1/2/5tick controller hold required')
        if not math.isclose(controller.dt,.05,rel_tol=0,abs_tol=1e-12):
            raise ValueError('The native physics/prediction grid must stay.05seconds')
        self.controller=controller
        self.hold_steps=hold_steps
        self.reference=None
        self.last_result=None
        self.last_time=None

    def step(self,state,reference,previous_command):
        now=float(state.time)
        if not math.isfinite(now):raise ValueError('Measured finite simulation time required')
        recompute=(self.last_result is None or reference is not self.reference
            or now<self.last_time-1e-8 or now-self.last_time>=self.hold_steps*.05-1e-6)
        if recompute:
            self.last_result=self.controller.step(state,reference,previous_command)
            self.last_time=now
            self.reference=reference
        diagnostics=dict(self.last_result.diagnostics)
        if not recompute:
            diagnostics.update(control_seconds=0.,qp_seconds=0.)
        diagnostics.update(controller_recomputed=recompute,
            controller_hold_steps=self.hold_steps,controller_period_seconds=self.hold_steps*.05,
            prediction_grid_seconds=.05,
            cadence_scope='deployment-only zero-order hold;unchanged native MPCdesign,no retraining')
        return replace(self.last_result,diagnostics=diagnostics)


def apply_controller_cadence(controller,config):
    hold=getattr(config,'native_control_hold_steps',1)
    return HeldMPC(controller,hold)
