"""Single-cycle linear time-varying QP MPC. All predictions are local arrays."""

from dataclasses import dataclass
import math
import time
import numpy as np
import osqp
from scipy import sparse
from .vehicle import Command, brake


@dataclass
class ControlResult:
    command: Command
    status: str
    diagnostics: dict


class MPC:
    def __init__(self, vehicle, dt=0.05, horizon=20):
        self.vehicle, self.dt, self.horizon = vehicle, dt, horizon
        self.solver = None
        self.pattern = None
        self.solution = None

    def step(self, state, reference, previous_command):
        started = time.perf_counter()
        c = self.vehicle
        dt = self.dt
        n = self.horizon

        def failed(reason, qp=0.0):
            return ControlResult(
                brake(state, c),
                "failed",
                {
                    "reason": reason,
                    "qp_seconds": qp,
                    "control_seconds": time.perf_counter() - started,
                },
            )

        if reference is None:
            return failed("no_reference")
        desired = reference.sample(state.time + np.arange(n + 1) * dt)
        desired[:, 3] += (
            2 * math.pi * round((state.yaw - desired[0, 3]) / (2 * math.pi))
        )
        z0 = np.array([state.x, state.y, state.v, state.yaw])
        nx = 4 * (n + 1)
        nu = 2 * n
        size = nx + nu
        qweight = np.array([4.0, 4.0, 2.0, 3.0])
        rweight = np.array([0.08, 0.3])
        rd = np.array([0.08, 2.0])
        P = np.zeros((size, size))
        q = np.zeros(size)
        for k in range(n + 1):
            sl = slice(4 * k, 4 * k + 4)
            weights = qweight * (2 if k == n else 1)
            P[sl, sl] = np.diag(2 * weights)
            q[sl] = -2 * weights * desired[k]
        prev = np.array([previous_command.acceleration, previous_command.steering])
        for k in range(n):
            sl = slice(nx + 2 * k, nx + 2 * k + 2)
            P[sl, sl] += np.diag(2 * (rweight + rd))
            if k == 0:
                q[sl] -= 2 * rd * prev
            else:
                last = slice(nx + 2 * k - 2, nx + 2 * k)
                P[last, last] += np.diag(2 * rd)
                P[sl, last] -= np.diag(2 * rd)
                P[last, sl] -= np.diag(2 * rd)
        rows = []
        cols = []
        values = []
        lo = []
        hi = []

        def row(entries, low, high):
            rid = len(lo)
            for col, value in entries:
                rows.append(rid)
                cols.append(col)
                values.append(value)
            lo.append(low)
            hi.append(high)

        for j in range(4):
            row([(j, 1.0)], z0[j], z0[j])
        for k in range(n):
            # Nominal velocity has a small floor only for Jacobian controllability at rest.
            v = max(0.1, desired[k, 2])
            yaw = desired[k, 3]
            kappa = float(
                np.interp(
                    state.time + k * dt - reference.start_time,
                    reference.t,
                    reference.kappa,
                )
            )
            delta = float(
                np.clip(math.atan(c.wheelbase * kappa), -c.max_steer, c.max_steer)
            )
            A = np.eye(4)
            A[0, 2] = dt * math.cos(yaw)
            A[0, 3] = -dt * v * math.sin(yaw)
            A[1, 2] = dt * math.sin(yaw)
            A[1, 3] = dt * v * math.cos(yaw)
            A[3, 2] = dt * math.tan(delta) / c.wheelbase
            B = np.zeros((4, 2))
            B[2, 0] = dt
            B[3, 1] = dt * v / (c.wheelbase * math.cos(delta) ** 2)
            C = np.array(
                [
                    dt * v * math.sin(yaw) * yaw,
                    -dt * v * math.cos(yaw) * yaw,
                    0,
                    -B[3, 1] * delta,
                ]
            )
            for j in range(4):
                # Keep explicit zeros so sparse structure stays identical across cycles.
                entries = (
                    [(4 * (k + 1) + j, 1.0)]
                    + [(4 * k + i, -A[j, i]) for i in range(4)]
                    + [(nx + 2 * k + i, -B[j, i]) for i in range(2)]
                )
                row(entries, C[j], C[j])
            row([(4 * (k + 1) + 2, 1.0)], 0, c.max_speed)
            row([(nx + 2 * k, 1.0)], -c.max_decel, c.max_accel)
            row([(nx + 2 * k + 1, 1.0)], -c.max_steer, c.max_steer)
            if k == 0:
                row(
                    [(nx + 1, 1.0)],
                    prev[1] - c.steer_rate * dt,
                    prev[1] + c.steer_rate * dt,
                )
            else:
                row(
                    [(nx + 2 * k + 1, 1.0), (nx + 2 * k - 1, -1.0)],
                    -c.steer_rate * dt,
                    c.steer_rate * dt,
                )
        Ap = sparse.csc_matrix((values, (rows, cols)), shape=(len(lo), size))
        Pp = sparse.triu(sparse.csc_matrix(P), format="csc")
        qp_start = time.perf_counter()
        try:
            if self.solver is None:
                self.solver = osqp.OSQP()
                self.solver.setup(
                    P=Pp,
                    q=q,
                    A=Ap,
                    l=np.array(lo),
                    u=np.array(hi),
                    verbose=False,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    max_iter=3000,
                    polishing=False,
                    warm_starting=True,
                )
            else:
                self.solver.update(q=q, Ax=Ap.data, l=np.array(lo), u=np.array(hi))
            answer = self.solver.solve(raise_error=False)
        except Exception as exc:
            self.solver = None
            return failed(
                f"solver_exception:{type(exc).__name__}", time.perf_counter() - qp_start
            )
        qp = time.perf_counter() - qp_start
        if (
            answer.info.status_val not in (1, 2)
            or answer.x is None
            or not np.isfinite(answer.x).all()
        ):
            return failed(f"qp:{answer.info.status}", qp)
        # Reject materially violated solutions even if OSQP reports inaccurate success.
        residual = Ap @ answer.x
        violation = float(
            max(np.max(np.array(lo) - residual), np.max(residual - np.array(hi)), 0)
        )
        if violation > 0.002:
            return failed("constraint_residual", qp)
        self.solution = answer.x.copy()
        command = Command(
            float(answer.x[nx]),
            float(
                np.clip(
                    answer.x[nx + 1],
                    prev[1] - c.steer_rate * dt,
                    prev[1] + c.steer_rate * dt,
                )
            ),
        )
        return ControlResult(
            command,
            "solved",
            {
                "solver_status": answer.info.status,
                "iterations": answer.info.iter,
                "qp_seconds": qp,
                "control_seconds": time.perf_counter() - started,
                "constraint_violation": violation,
                "objective": float(answer.info.obj_val),
            },
        )
