# Controller comparison and native execution recovery

This is development evidence. The original thesis objective and all acceptance
thresholds remain unchanged and unmet.

## Fixed-reference control comparison

The completed lightweight physical-integrator benchmark has80episodes:
four measured road grids,five paired initial-error seeds,two dynamics settings
and two controllers. Both MPC20and timed Pure Pursuit/PI complete40/40.
[Full state/metric verification](fixed_reference_controllers_verified_20261028.json)
replays38962commands and independently checks tracking samples,termination,
goal distances,jerk and computation-time aggregates. These are not CARLA
steps or independent trained-policy seeds.

| Dynamics | Controller | Success | Mean lateral RMSE,m | Mean jerk RMS,m/s³ | Mean episode P95 computation,ms |
|---|---|---:|---:|---:|---:|
| Nominal | MPC20 | 20/20 | .04951 | 1.29458 | 2.19678 |
| Nominal | Pursuit/PI | 20/20 | .02447 | 1.92192 | .14268 |
| Lag,delay,scale mismatch | MPC20 | 20/20 | .05811 | 1.13921 | 1.97852 |
| Lag,delay,scale mismatch | Pursuit/PI | 20/20 | .02867 | 1.58459 | .13512 |

Pursuit has lower lateral error and CPU cost;MPC has lower jerk. This supports
a control tradeoff,not uniform MPC superiority. Times exclude CARLA,RPC and
planning,and were collected during other research work;no real-time
certification is inferred. [Figure](figures/fixed_reference_controllers_20261028.png).
The pursuit implementation uses conventional geometric tracking described by
[Coulter,CMU-RI-TR-92-01](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm),
with independently implemented timed speed/progress PI and shared limits.

An earlier centreline-reference batch completed20lightweight episodes/10190
controls before the next route's full-body reference check failed. Its partial
summary and all20traces remain under023315. It was incorrectly described as
zero execution in an intermediate update;the measured accounting is corrected
here. Total measured integrator consumption is49152controls across the two
versions;their scores are not pooled. Two incomplete reference-export prototypes
also remain;those planner queries added no physical controls.

The replacement references use the original frozen B3planner and initial
measured states. One local reference gets one global extension query from its
predicted endpoint;no actual vehicle state is changed. All four complete-goal
references pass full-body checks before either controller runs.

Two native batches024600457573/MPC20and024604590505/Pursuit/PI are prepared and
ValidateOnly passes390source files each. Calibration,vehicle limits,all frozen
source hashes and fixed references match exactly. No new native-controller
score exists yet. A raw-RPC binding helper also verified1938historical native
control frames against their actual saved bridge requests/responses;
[receipt](native_control_rpc_binding_20261028.json). The helper rejects unknown
measurements,changed demands,wrong reset geometry and unaccounted controls.

## Interrupted shared-mask PPO

Batch013316completed Htraining with20674real controls/650Adam updates.
Its partial evaluation contains10completed cases,7successes/3deadlines and
4864controls. Reset of the following dynamic condition failed the unchanged
unknown-actor guard. This is not a complete12-condition score. H0never started.
The owned server independently probed empty and was stopped;the original
training,partial evaluation and failure remain intact.

The actor identity responsible was absent from the old diagnostic,so its cause
is unknown. New diagnostics record actor IDs/types/aliveness/snapshot presence
while still refusing to tick or destroy any unknown actor,including one absent
from the current snapshot. Fresh-server reevaluation is permitted only after
exact failure classification,matching empty shutdown probe,closed ports and
verified ended process identities. No ownership protection is bypassed.

New Hreevaluation001732retains the trained checkpoint and adds zero updates.
H0-only recovery001825retains its untouched original initialization and teacher
provenance;it does not retrain H. Recovery also preserves the legacy one-argument
case API when the old frozen runtime has no validation partition.

## Actually resumed execution,31October

The old queue PIDs were absent and CARLA/RPC ports empty before activation.
Previously unstarted B1fixed-reward-unit batch020728is now actually training,
worker31216/guard33992. Only its activation clock changed;weights,cases and
20480-control budget remain registered. It is followed by these actually
launched bounded queue programs:

1. H0-only training001825.
2. Hfrozen reevaluation001732.
3. Native MPC20and Pursuit/PI fixed-reference probes.
4. Previously unstarted seed22PPO013440.
5. Previously unstarted train36/validation24batch014801.

The exact launch identities and queue receipts remain in
runs/resumed_owned_queue_launches_20261031.json. Each child requires normal
predecessor completion,owned-empty server stop,released lifecycle lease and
unchanged prepared protocol. One simulator runs at a time,with fixed wallcaps,
12GiB disk/4GiB RAM/85C GPU limits and no automatic retry.

Controller-core full regression passed650/1skip/4CARLA deselections162.29s.
Later recovery/ownership/partition targeted checks pass58/58in6.24s;the current
complete regression result is recorded separately. Publication is a development
delivery and does not establish healthy B1,PPOvalue,five trained seeds or final
confirmation.

Current complete regression passed657tests,1skip,4CARLA deselections in172.19s.
At08:25local,the actual B1worker31216had exceeded7800controls;its matching
creation identity and guard33992were verified live. Results remain pending.
