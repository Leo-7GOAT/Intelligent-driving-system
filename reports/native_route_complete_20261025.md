# Complete native route development batch — 2026-10-25

Batch `20261025T085433Z_native_route_learning` has normally completed all
registered H0/H/B1 training and all four frozen training-road probes per method.
This is **one development seed on four reused native-road geometries**, not
independent confirmation, a healthy final B1, or a hierarchical-advantage claim.
The result is H0 **3/4**, H **2/4**, B1 **0/4**. The previously audited
rule-only B3 completes4/4. The current H feedback version does not improve
completion over H0 or establish learned decision value over B3.

## Training and fairness accounting

All methods start fresh at seed11, with no teacher, inherited checkpoint or
normalizer. The frozen execution source is b495e71. All share the four sampled
CARLA roads, measured vehicle/actuator calibration,4705available input slots,
road/candidate information, physical reward, termination rules and a nominal
20,480physical-control budget. H0 zeroes the99witness/outcome feedback values
while retaining the four measured actuator states. H0/H choose31categorical
goals/speeds followed by the same Hybrid A*/trajectory/MPC; B1 directly chooses
two normalized actuator commands, held for10physics ticks. Physics/MPC period
is0.05s and high-level/direct action-hold period0.5s. These architectural and
control-frequency differences remain part of the comparison and need separate
formal sensitivity analysis.

| Method | Training controls | Charged overshoot | Decisions | PPO phases/Adam steps | Parameters | Training wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 20,702 | 222 | 2,080 | 65/130 | 208,034 | 5,312.48 |
| H | 20,665 | 185 | 2,080 | 65/130 | 208,034 | 8,630.72 |
| B1 | 20,621 | 141 | 2,080 | 65/130 | 212,293 | 3,062.88 |

Standard SB3 PPO/global gradient clipping,32-decision rollouts, batch32 and
two epochs are retained throughout. Full rollouts cause overshoot; every actual
control is charged. All final model reloads are exact and stopped reasons null.
The different wall costs are disclosed, rather than treating equal policy
decision counts as equal computation.

Training outcomes are H0:19success/10deadline; H:20success/8deadline/**1bounds**;
B1:1success/6deadline/**39bounds**. These are104completed training episodes,
with partial final episode controls also charged. Training successes are not
test scores. B1's extensive failures require further baseline development;
this20k-control version cannot serve as the sole sufficiently trained comparator.

## Every frozen evaluation outcome

Each cell reports outcome, actual controls and progress in meters. All measured
states come from the original CARLA bridge; failed cases remain in totals.

| Known road | H0 | H | B1 |
|---|---|---|---|
| 0 | Success;337;24.1104 | Success;341;24.3164 | Bounds;289;21.1850 |
| 1 | Deadline;900;0 | Deadline;900;17.0442 | Bounds;134;2.8195 |
| 2 | Success;329;22.5262 | Success;329;22.4616 | Bounds;250;13.8798 |
| 3 | Success;321;20.2868 | Deadline;900;17.8448 | Bounds;271;5.4769 |

H0's road1 fails all90plans and intervenes at all900controls without motion.
H has no evaluation plan/MPC failures or interventions, but selects65successive
waits on road1 and48waits in total on road3. Low collision counts and tracking
error cannot replace task completion. The [H input diagnostic](native_feedback_inputs_20261025.md)
finds persistent initial-weight wait rankings and available queried references
at five failed-road states; none of those alternatives was executed.

B1 exits the shared road/body bounds in all four probes after14.45/6.70/12.50/
13.55simulated seconds. Those early failures must not be described as faster
successful traversal. Its reference tracking errors and MPC failure rates are
correctly missing (`flat_policy_has_no_reference`), not zero. No evaluation
method records a collision in these empty-road probes; this does not establish
safety in independent obstacle or interaction conditions.

Full B1 decision P95 is1.7651/4.6786/1.8721/2.4973s, with maxima1.8094/4.7224/
1.9719/3.3950s. H also frequently exceeds0.5s. These full cycles include
observation/geometry work, rather than reporting QP or network inference alone.
No general realtime or comfortable-driving claim follows; measured jerk and
terminal braking costs remain in the raw metrics.

No seed-level confidence interval can be estimated from one training seed.
The observed2/4versus0/4 is not the requested formal superiority test. All
five-seed, fresh120-condition, baseline-health and PPO-versus-strong-rule
acceptance gates remain open.

## Complete original-frame and policy evidence

| Method | Train controls | Eval controls | Verified total | World ticks | Deterministic decisions reconstructed |
|---|---:|---:|---:|---:|---:|
| H0 | 20,702 | 1,887 | 22,589 | 23,642 | 190 |
| H | 20,665 | 2,470 | 23,135 | 24,188 | 248 |
| B1 | 20,621 | 944 | 21,565 | 23,145 | 96 |
| Total accounting | 61,988 | 5,301 | **67,289** | **70,975** | **534** |

World ticks include3,686reset ticks. Raw audits bind every control to the
corresponding original state, applied command, IMU frame, actuator history,
episode identity and terminal fence. The three bridge streams are distinct.
All534frozen policy decisions reconstruct from saved models and measured
history, with frozen observation statistics and zero additional execution.
Independent fixed-environment raw audits and the complete12-panel figure are
byte-identical. The final figure was visually inspected and retains every case.

Local complete summary: `analysis/complete_evidence_summary.json`, SHA256
`e59d76cac7870d0b4b8e0c9a44003641fe937f364770977b68d4d0d95065d180`.
Full figure: `analysis/frozen_endpoints.png`, SHA256
`3b71cfc60d213c2d729959d1ce05ef0db71948edcbfcba089e21e7cd47fd2eae`.
Per-method audits, all-policy reconstruction receipts and their frozen helpers
are retained in each method folder. The [checkpoint report](native_route_checkpoint_20261025.md)
contains intermediate integrity checks and the completed H/H0 analysis.

## Lifecycle and publication status

The original worker and guard exit0, with guard reason null. After a matching
0.9.15empty-world probe, the owned server is deliberately stopped and ports
are clear. `server_cleanup.json` returns0 and `server_stop.json` records
`owned_empty_server_stopped`. The foreground server process records exit-1
during that deliberate owned stop; this record is retained, not rewritten as0.
There is no automatic retry or discarded failed evaluation.

All registered methods/failures will be exported together with unchanged model,
normalizer and raw control bytes. The current stage has not yet uploaded that
complete asset; actual ZIP hash, publication and downloaded-asset reproduction
must be recorded after they occur. Existing development prereleases do not
constitute research acceptance. The original source remains immutable while
the separately declared scalar/batchSAT B3 computation contrast proceeds.
