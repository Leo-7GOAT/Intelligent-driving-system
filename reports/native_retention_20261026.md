# Native development: action representation and skill retention

These are seed-11 development experiments on already known roads. The original
research acceptance requirements remain open. In particular, a three-metre
curriculum success is not a healthy full-task B1 or evidence for hierarchy.
Exact batch identifiers, model/normalizer hashes, raw audits, failures and costs
are in [the measured ledger](native_retention_summary_20261026.json).

| Completed variant | Task | Frozen successes |
|---|---|---:|
| H unshared head, sufficient local-reference filter | Original full roads | 2/4 |
| H unshared head, physical-progress reward1 | Original full roads | 1/4 |
| B1 categorical direct speed/steering | Six-metre prefixes, 4,105 training controls | 1/4 |
| Same B1 continued to20,510 lineage controls | Six-metre prefixes | 0/4 |
| Own-success actor fit, 1,000 auxiliary steps | Six-metre prefixes | 3/4 |
| Ordinary PPO after that fit, moving statistics | Six-metre prefixes, 4,372 new controls | 0/4 |
| Matched continuation, frozen statistics | Six-metre prefixes, 4,106 new controls | 2/4 |
| Three-metre curriculum, frozen statistics | Three-metre prefixes, 4,398 new controls | 2/4 |
| Pooled own-success fit | Three-metre / original six-metre prefixes | 2/4 / 3/4 |
| PPO plus interleaved own-success replay | Three-metre prefixes, 4,350 new controls | **4/4** |
| Same saved model, no updates, original goals restored | Six-metre prefixes | 3/4 |
| Continue that learner on4.5m nominal prefixes | 4.5m prefixes, 4,366 new controls | 3/4 |
| Updated own-success pool,100replay steps/phase | 4.5m prefixes,4,376 new controls | 3/4 |
| Same saved model/data,10replay steps/phase | 4.5m prefixes,4,109 new controls | 1/4 |
| Online addition of newly successful policy inputs | 4.5m prefixes,8,445 new controls | 3/4 |

The last failed road advances4.08m in the4.5m curriculum probe. Training includes
one successful episode on that road; the deterministic endpoint still times
out. All four goals must pass before the curriculum staging tool permits the
next registered distance. No formal test data have been inspected or relabeled.

The failed road exposes two terminal-control errors. The first4.5m continuation
ends0.544m/0.059rad from the goal, within the unchanged pose limits, but speed
0.749m/s exceeds the0.25m/s stopping limit. The online-buffer endpoint is nearly
stationary but0.831m away,31mm outside the0.8m position limit. Even the3m road1
success stops0.792m away. Positive replay can reinforce fragile boundary
successes. These are measured late/early braking failures, not evidence that
the road is blocked; evaluation tolerances and vehicle state are unchanged.

## What changed and what the evidence supports

The local-reference filter is optional, shared with B3, leaves wait available
and falls back to all actions when every moving connector is unknown. It uses
local sufficient witnesses, not absent goal-continuation witnesses. Nevertheless,
the score worsens from the prior unfiltered3/4 to2/4; it is not adopted.

The progress-reward H trial completes4,133 training controls and130 Adam steps,
then its900-second guard interrupts evaluation. Only4,073 training controls are
readable from the original unclosed gzip stream: the60-record gap is retained.
The saved model/normalizer are recovered byte-for-byte for a separate complete
evaluation with zero training controls and zero optimizer steps. That1/4 result
does not retroactively make the original batch complete or its missing trace
verified. Future training streams close before evaluation.

B1's new optional MultiDiscrete[3,7] directly selects speed0/.75/1.5m/s and
normalized steering[-1,-.5,-.2,0,.2,.5,1]. A disclosed proportional speed servo
maps requested speed minus measured speed to acceleration within the common
vehicle limits. There is no planner, MPC, goal-based braking rule or external
teacher. The representation changes the prior/distribution and must receive a
properly tuned comparator before any hierarchy claim. Legacy controls remain
the default. Encoder/critic initialization and global construction RNG match
the continuous control, apart from the declared action head/distribution.

The first auxiliary fit uses360 audited decisions from17 own successful PPO
training episodes, with no success on road1. The second fit pools922 decisions
from49 successes across five correlated development segments; it includes only
17 decisions from one road1 success in the shorter curriculum. This is
reward-selected replay from the same policy family, including sibling branches,
not independent data or ordinary on-policy PPO. Failures remain at their sources.
Both fixed1,000-step fits leave critic weights, critic Adam state, observation
statistics and the PPO update counter unchanged. Their optimization cost is
charged separately. Dataset and source hashes are retained.

Moving observation statistics lose all3/4 fitted successes; freezing the same
statistics retains2/4. The frozen arm's mean/variance/count exactly equal the
saved source; moving count changes2082.0001 to2531.0001. This supports a partial
retention problem, not a complete causal explanation or solved baseline.

The interleaved variant adds100 condition-balanced actor fitting steps after
each PPO phase, with a fixed own-success dataset and frozen observation
statistics. Its3m run has140 PPO optimizer steps and1,400 auxiliary actor steps;
the4.5m continuation has the same split. The total Adam counter includes both,
while the PPO update counter counts only PPO. Every auxiliary fit asserts that
the critic and its Adam state are unchanged. This augmented objective remains
experimental and needs equal optimization/tuning opportunities for controls.

## Infrastructure and validation boundaries

The first3/4 auxiliary evaluation finishes before the process guard can attach.
The worker tool session returns exit0 and the803 control frames pass the raw
audit; the server is explicitly probed empty and stopped. The failed guard
attachment log remains, and no normal guard receipt is fabricated. Other scored
phases in the ledger have actual exit0 guard receipts and owned-server shutdown.

Unstarted preparation failure, missing client environment, inherited training
flag rejection and absent-server connection failure are preserved separately.
Closed bridge evidence verifies zero controls in the last two launch failures.
Explicit recovery registers new immutable phases rather than overwriting them.
The launcher now ignores PID0 TCP TIME_WAIT entries while still refusing any
actual CARLA process or TCP/UDP owner. A matching-version empty probe is required
before dependent worker launch.

The third fixed1,000-step actor fit uses1,565decisions from86own successes,
including94decisions on road1. The matched100/10-step continuations start from
identical fitted weights, Adam state, frozen normalization, dataset and saved RNG;
they retain3/4 versus1/4. Reducing replay is not adopted.

The separate online-buffer phase keeps exact normalized inputs that PPO actually
received, with executed categorical actions, only after a successful episode
without intervention/controller failure. It adds691decisions from35successes
to1,565retained inputs. Failed and intervened episodes remain in raw logs and do
not enter that positive buffer. This8,445-control phase has270PPO and2,700actor
replay optimizer steps; its3/4 endpoint still fails road1. Its doubled nominal
interaction budget prevents treating it as an equal-budget superiority result.
The normalized input buffer, hash and sample costs are preserved. No further
identical continuation is launched merely because this endpoint remains below
the four-case curriculum gate.

The16completed scored phases in the ledger contain80,581new frame-verified
controls, without duplicating inherited training. Interrupted H training and
partial old evaluations are separately retained and are not silently included
as fully verified segments.

Independent-checkout CPU regression atcbaa76epasses317/1skip/4CARLA deselections;
targeted training, continuation, recovery and raw-audit checks pass. Hosted CI at
the same source passes315/3platform-or-opt-in skips/4deselections:
[actual run](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40308103444).
Newer online/export changes require their own full/hosted result; CPU tests are not
CARLA competence, independent-seed replication or formal confirmation.

Current local online/export regression subsequently passes320/1skip/4CARLA
deselections in93.22s. The initial NPZ privacy test exposed Unicode array
strings that evade raw-byte searching; export now scans decompressed arrays,
decodes strings and refuses object arrays without loading pickle. The corrected
regression passes. MultiDiscrete model inference also reconstructs all four
measured start actions with frozen statistics.

The new raw models and traces are retained locally. The older complete public
assets remain available; this report does not claim the new evidence has already
been published as a downloadable model release.
