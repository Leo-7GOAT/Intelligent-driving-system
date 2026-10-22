# Camera evidence and matched curriculum development, 2026-10-22

Research acceptance remains incomplete. These are single-seed development experiments, not new independent confirmation. Earlier inspected final conditions remain historical.

## Completed training, including counterexamples

| Batch | Actual budget | Development endpoint |
|---|---|---|
| v11 mixed-task flat B1, seed 11 | 1,000,960 physical steps; 391 PPO phases; 15,620 optimizer steps | 0/10; zero training successes in 1,505 episodes |
| v12 shared-candidate H | 1,002,490 physical steps | 8/10; both unseen cases fail |
| v12 same-network H0 without feedback | 1,000,376 physical steps | 6/10; narrow and unseen cases fail |
| v12 CARLA H, adapted from the fixed 100k lightweight model | 101,048 world ticks; 38 phases; 1,520 optimizer steps | 4/6, down from 5/6 at 26,564 ticks |
| v12 CARLA H0, adapted from its fixed 100k model | 100,939 world ticks; 38 phases; 1,520 optimizer steps | 3/6 at both milestones |

CARLA budgets include reset ticks. The six-case evaluations have no unresolved infrastructure failures and no observed collision/bounds episodes; deadlines and intervention rates prevent interpreting that as adequate navigation or road safety. Strong B3 previously achieved 10/10 on the shared lightweight cases. The v11 baseline lacks v12's explicit candidate geometry; v13 supplies the matched 605-dimensional input to B1. None of these comparisons establishes the final causal claim.

Source data: `runs/20261022T054701Z_matched_flat_v11_development`, `runs/20261022T060649Z_candidate_v12_development`, and `runs/20261022T061639Z_carla_feedback_adaptation`. Full logs, checkpoints, optimizer accounting and unsuccessful episodes are retained locally.

## Common task staging: v13

Batch `20261022T093852Z_staged_v13_development` starts H, H0 and B1 from scratch with seed 11. Each spends its first 300k physical steps on the full 23 m empty task, then enters the original mixed distribution at the next episode boundary. Pretraining is included in the same 1M budget. Reward, vehicle, low level and endpoint cases are unchanged. H and B1 receive the same 605 features; H0 removes only the declared feedback features. B1 still controls the actual actuators directly.

The test is motivated by v9's successful full-length empty-task B1 versus v11's zero mixed-task training successes. It does not assume that 300k is sufficient. The common schedule is frozen in `protocol.json`, not adjusted separately after seeing one method's results. The two-hour supervisor window ends at approximately 19:38 local; three CPU workers use one Torch thread each. Guards require at least 12 GiB disk, 4 GiB available memory, GPU below 85 C and advancing work within 300 seconds. No failed-job retry is automatic.

At the 18:06 inspection all three OS workers were live. H's 100k development score was 10/10, but fell to 8/10 at 300k; H0 was 6/10 at both. B1's 100k score was 0/10, with one training success by the inspected 253 episodes. This is preliminary and explicitly retains H's regression rather than selecting its earlier peak as a final result.

Recovery for this existing batch (do not initialize a duplicate):

```powershell
$batch = (Get-Content runs/staged_development.json -Raw | ConvertFrom-Json).batch
.\.venv\Scripts\python.exe scripts/development_watchdog.py --batch $batch
```

For a new independent development batch on a fresh checkout, `scripts/initialize_staged_development.py` creates the protocol and source snapshot. It does not itself start training.

## Real camera evidence

The selected illustrative source is the H checkpoint at 26,564 CARLA world ticks, after 101,923 lightweight physical steps. Model SHA256: `b03aba60b8ca0c35ff4a39a30183f0504fd2009193929bd3bb6914298b1979c2`. This is a CARLA-adapted PPO, not a rule controller or unadapted transfer model. The checkpoint and two inspected conditions were declared before capture to show both a success and a failure, not to estimate generalization.

Final capture batch `20261022T095316Z_carla_camera_evidence`, run `20261022T095338Z_PPO_H_development_low_level_recalibrated_frozen_policy_866bccf4`:

| Condition | Outcome | Progress | Real frames | Encoded duration |
|---|---|---:|---:|---:|
| static / 7301 | success | 22.2118 m | 362 | 18.10 s |
| curve / 7301 | deadline | 12.1026 m | 901 | 45.05 s |

The run contains 1,261 control steps and 1,322 world ticks including initialization/cleanup accounting, no unresolved infrastructure failures and frozen normalization. The owner verified an empty world and stopped its server. Every encoded frame is linked to the RGB sensor frame, timestamp, PNG hash and matching world timestamp; no frames were invented or interpolated. Full MP4 decoding confirmed all 362/901 frames at 20 fps. A first/middle/last contact sheet was visually checked: the orange ego remains in frame; blue cars are task obstacles. Rendering colors and camera images never enter PPO input. The existing generic run label does not replace the explicit adapted-model provenance above.

Camera fixes were evidence-driven: Unicode-safe PNG output replaced native saves that failed on a non-ASCII path; duplicate callbacks are audited; conflicting duplicates fail; exact sensor/world frame fences include reset warmup. Setting synchrony before reloading the recording world produced 30/30 matching frames in an isolated live diagnostic. Recording reload is optional and off in training. Camera runs can have slightly different physical outcomes from non-rendered runs and are separately identified.

Failed captures are preserved: `20261022T093209Z_carla_camera_evidence` (native image save), `20261022T093618Z_carla_camera_evidence` (duplicates), and `20261022T094031Z_carla_camera_evidence` (conflicting timestamps). They are infrastructure failures, not policy failures or successes. The first valid portrait-framed capture, `20261022T094729Z_carla_camera_evidence`, is retained alongside the final landscape capture.

The evidence package contains the exact model/normalization, frozen source, both complete evaluation traces, calibration, frame manifests and encoded videos. Raw PNGs remain local. Verification checks all 145 file hashes, links videos to episodes/model, and loads the policy in both the project environment and the independently installed clean-checkout CPU environment. This is artifact integrity verification, not another CARLA replay or five-seed confirmation.

## Validation and remaining work

The completed regression at 17:50 was 74 passed, 5 conditional skips. Subsequent video tools have been executed on real recordings; their integrity checks do not substitute for planner/control/PPO tests. Final publication checks are recorded in the release record.

Remaining research gates include healthy full-task B1, repeatable feedback contribution, PPO decision value over strong B3, unseen geometry, native-road curves and actuator coverage, five independently trained policies, full B2/frequency/shared-safety/control comparisons and a newly frozen 120-condition CARLA confirmation. Videos and a pre-release do not satisfy those gates.
