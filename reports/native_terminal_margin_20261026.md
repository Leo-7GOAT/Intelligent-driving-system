# Matched own-success quality selection and longer-goal adaptation

Status: the seed-11 matched4.5m probe improves from3/4 to4/4 with stricter
training-example selection. Frozen6m transfer and subsequent real-PPO6m
adaptations still score3/4. No independent confirmation, complete-task B1
health or original research acceptance is claimed. All task thresholds remain
0.8m position /0.25m/s speed with the existing heading requirement.

## Selection experiment

The source pool contains147successful training episodes from ten correlated
native PPO development segments (59,032new training controls). They are actual
measured own-policy trajectories, with no rule/external teacher. Road1has14
successes; four have distance<=0.5m and speed<=0.15m/s. Thus fragile boundary
examples are present but are not the only available successes.

Both fits start from the same completed online-PPO model, value network, Adam,
saved RNG and frozen normalizer. Both use exactly1,000actor optimizer steps,
condition-balanced64-sample minibatches, the same learning rate and no new PPO
updates or simulator interactions. The only selection difference is:

- Control: all147successes,2,765decisions.
- Margin:47successes/895decisions with terminal distance<=0.5m,
  speed<=0.15m/s and absolute heading error<=0.2rad. The100unselected successes
  and all failed episodes remain at their sources and in provenance metadata.

The original four4.5m evaluation goals, task/controller parameters, physical
limits and success gates are identical. All four measured initial scenarios
match exactly. Raw frame audits and normal owned-server cleanup complete.

| Endpoint | Frozen result | New train / eval controls | Added PPO / actor steps |
|---|---:|---:|---:|
| All-success fit | 3/4 | 0 /801 | 0 /1,000offline |
| Margin fit | **4/4** | 0 /798 | 0 /1,000offline |
| Same margin model, original6m goals | 3/4 | 0 /745 | 0 /0 |
| Real PPO with fixed margin replay,6m | 3/4 | 8,463 /779 | 270 /2,700 |
| Same initialization/schedule with qualified online additions,6m | 3/4 | 8,413 /804 | 270 /2,700 |
| Fixed replay with speed-only label smoothing0.3,6m | 3/4 | See measured ledger | See measured ledger |

The strictness applies only to selecting training examples; it does not relax
or replace task evaluation. The selector changes supervision and data volume,
so this small development contrast is not proof of causality or a formal RL
advantage. It supports retaining the quality-selection hypothesis for further
development rather than repeating undifferentiated successful-example replay.

## Actual longer-goal learning and exploration

The curriculum staging tool advances only after all four current goals pass
under the same model, normalized inputs, task/controller settings, native
execution implementation and calibration. It verifies the completed raw audit
and rejects changed models, missing goals or easier task tolerances.

Fixed replay advances failed-road progress from3.55m under frozen transfer to
5.05m after8,463real training controls. The matched online variant adds122
decisions from7qualified successes and rejects23other successful episodes only
from the auxiliary buffer; their true task successes and raw data remain. It
still reaches only4.76m in the failed deterministic case. Neither phase passes
the next curriculum gate, and there is no plain-PPO or new-seed claim.

Measured-start reconstruction of the fixed-replay model finds road1speed
probabilities approximately[0.0000275,0.9999725,0.000000000313] for0/.75/1.5m/s.
The actual training road1speed-action counts are95/126/18across239decisions.
This is evidence of start-state exploration collapse, not proof that a faster
request alone would solve the curve.

A separately registered contrast smooths only auxiliary speed labels to
0.7one-hot+0.3uniform; steering labels remain hard. The actor still samples
from its actual categorical policy and PPO records its exact likelihoods.
The physical controller, steering limits, observation, task reward and terminal
gates are unchanged. Its completed endpoint is3/4: road1advances5.81m but exits
bounds. Greater speed exploration is therefore insufficient for safe longer
curve execution. The endpoint and its failures are retained rather than adopted
as a healthy baseline. Exact costs and probabilities are in the JSON ledger.
Its measured-start1.5m/s probability on road1rises to0.137279, versus
0.000000000313under hard-label replay; the new steering/bounds failure remains.

## Evidence, checks and scope

[Measured ledger](native_terminal_margin_summary_20261026.json) records source
episodes, used/unselected examples, artifacts, models, normalizers, protocols,
raw audits, original outcomes and phase costs. Existing public model releases
remain available; the new margin/adaptation models in this report are local
development artifacts until a separately verified publication is recorded.

Current CPU regression passes324/1skip/4CARLA deselections after tests cover
terminal-margin selection, unchanged critic/Adam/PPO counters, strict curriculum
gating, qualified online collection, and speed-only smoothing gradients under
collapsed probabilities. Synthetic tests are not CARLA evidence. Actual phases
retain bridge acknowledgments, matching measured states/IMU frames, closed raw
traces, guard exit receipts and owned-server shutdown.

Actual hosted1f403bcregression passes322/3skips/4CARLA deselections in122.79s:
[run40322362420](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40322362420).
Independent checkout at the same commit passes324/1skip/4deselections in124.82s.
It reproduces the margin4.5m model's four measured-start actions and byte-identical
raw audit with zero new controls/updates; [receipt](native_terminal_margin_reproduction_20261026.json).
This uses local retained model bytes and is not a new public asset download.
The six completed scored phases retain30,038new raw-verified controls; original
source-pool interactions are not charged again as new training.

The original exploratory registration allowed two matched4.5m evaluations.
After the all-four margin result, the announced conditional6m transfer and
separately registered training contrasts were added. They are development
extensions, not independent final tests or repetitions used to claim significance.

The next distinct development issue is safe higher-frequency flat steering,
with physical-time rollout/discount/update budgets accounted for. Repeating
the same short-goal actor fits or ignoring the new bounds failure is unsupported.
Native static/dynamic competence, full-road health, five independent training
seeds,120fresh conditions and formal H/H0/B1/B2/B3 attribution remain open.
