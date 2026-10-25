# Fresh-route H0 optimizer reconstruction — 2026-10-25

The completed H0 policy scores3/4known training roads but remains close to a
uniform categorical distribution. This experiment reconstructs its **actual
last two PPO rollouts** to distinguish a broken update from a weak functional
change. It does not alter the still-running H/B1 queue or its frozen low level.

## Exact replay, including a real episode boundary

Starting at `H0/checkpoint_000020000` (20,063controls/2,016decisions), the script
uses the saved policy, Adam state, normalization, RNG and raw-audited recorded
CARLA states. It reconstructs all earlier observable history without executing
a plant, including route-relative last behavior and all four actuator states.
The checkpoint's normalized observation must equal its saved `_last_obs` exactly.

The next64sampled actions reproduce exactly. The first32-decision rollout has
319controls and one terminal/reset transition; the second has320controls.
Physical-step reward discounting, GAE, updating observation statistics and
terminal handling reproduce both actual PPO optimization log rows. The final
policy/value tensors, full Adam state, observation moments **and return moments
are bit-exact** with the saved20,702-control endpoint. Learning counters also
match. Thus this is a reconstruction of actual on-policy learning, not an
optimizer fed fabricated transitions.

VecNormalize omits its transient discounted-return accumulator from pickle.
The replay derives it from the already audited prefix rather than setting it
to zero mid-episode. This is offline reconstruction only: actual CARLA recovery
still starts an explicitly new physical episode and counts discarded work.

Evidence is `H0/last_two_updates_replay_verified.json`, bound to raw audit SHA
`c533950446a5ae79a6fa600c6ed82c0ffc3ab42308aa2ff568ebe211f0d5fdeb`
and checkpoint metadata SHA
`9da561cdaf87ff9aaa2937f4fa18587311ae843b04e6a3995830a9936922a238`.
Earlier v1/v2 JSON and their original source copies are retained separately.

## Measured gradient scales

These are pre-update gradients at each of the two reconstructed rollouts,
not claims about every update, method or training seed.

| Start decision | Actor gradient L2 | Weighted critic gradient L2 | Shared clipping multiplier | Actor components below Adam epsilon after clipping |
|---|---:|---:|---:|---:|
| 2,016 | .023115 | 86.00015 | .00581394 | 99.9800% |
| 2,048 | .019111 | 16.73883 | .02987065 | 99.9724% |

In both samples, every road/context/candidate-encoder gradient component falls
below epsilon1e-5 after shared clipping. The33-parameter shared score head has
36.36%/12.12% below epsilon. This is a scale diagnostic: **Adam epsilon is not a
hard cutoff**, moments persist and encoder parameters still move. It does not
mean gradients are zero or that standard PPO is implemented incorrectly.

## Discarded standard-PPO optimizer branches

Each branch starts from the identical saved checkpoint and uses only the first
actual32-decision buffer. The recorded sampled data stay fixed. No alternative
branch is saved as a policy or executed, and no performance comparison follows.
Only the named setting differs; baseline has2epochs, epsilon1e-5 and vf_coef.5.

| Branch | Actor parameter L2 change | Centered logit change RMS | Mean exact categorical KL | Mean maximum action probability |
|---|---:|---:|---:|---:|
| Baseline | .00382321 | .00012374 | 7.6153e-9 | .03240403 |
| Adam epsilon1e-8 | .06539545 | .00017669 | 1.5529e-8 | .03240602 |
| vf_coef.01 | .00857793 | .00020513 | 2.0977e-8 | .03240632 |
| 10epochs | .01367727 | .00045960 | 1.0510e-7 | .03241627 |

Reducing epsilon increases parameter displacement about17times while KL only
roughly doubles. Most of the extra parameter movement is in encoders; score-head
change is.00119092 versus baseline.00114336. Common shifts across all logits
also exceed their centered changes (baseline RMS.00067098 versus.00012374).
Parameter distance alone therefore overstates functional policy change here.
Larger epoch counts or reduced critic weight do not prove better decisions.

The earlier independent actor/critic clipping experiment's all0/4results remain
retained; this replay does not erase that failure or justify silently adopting
another optimizer. Any future selected learning intervention needs a separately
registered real interaction trial and comparable treatment of relevant baselines.

The process performs20counted, discarded in-memory optimizer steps (four for
actual-rollout reconstruction and16across the one-buffer branches). It performs
**zero new environment controls and zero live-worker optimizer updates**.
Current H/B1 training parameters, weights and normalization files are untouched.

## Reproduction

```powershell
$env:NAVSTACK_AUDIT_SOURCE_ROOT=(Resolve-Path 'runs/20261025T085433Z_native_route_learning/source').Path
python scripts/replay_native_checkpoint_updates.py --batch runs/20261025T085433Z_native_route_learning --method H0 --checkpoint checkpoint_000020000 --output runs/H0_updates_rebuilt.json
```

Use a new output path, the original trusted checkpoint, and exact archived
runtime source. The script refuses changed raw-audit inputs, incomplete selected
rollouts, mismatched RNG/actions or differing final tensors/statistics. It never
connects to CARLA and reconstructed observation holders have no plant step.
This is single-seed optimizer evidence, not unseen task success or PPO value.
