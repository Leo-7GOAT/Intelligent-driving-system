# Completing the broader B1 development cohort

Seed22's previously registered single pilot completed 20,480 new actual PPO
controls on 36 existing TRAIN conditions and 630 Adam steps. Validation is
**7/24**: curve/parking 5/8, static 0/8, walkers 2/8, with 11 bounds failures,
five collisions and one deadline. All 31,480 controls and 24 start actions verify;
observation RMS is unchanged. It fails the development screen. Its original
protocol and selection history remain unchanged. [Completed audit](../reports/native_B1_broader_seed22_20261102.json).

Before observing that pilot's validation outcomes, four further fixed courses
were registered for **all remaining seeds 11/33/44/55**, including the two seeds
that failed the basic gate. Each resumes its fixed final basic checkpoint,
retains PPO/Adam/RNG/normalization, and receives 20,480 additional controls with
the same frozen worker, physical modules, parking aid, calibration and scene
inventory. No new supervised steps or variance/statistics reset is introduced.
Seed55's earlier interruption and discarded work remain disclosed.

[Registered batches, hashes, budgets and reporting rules](../protocols/native_B1_broader_additional_four_20261102.json).

Each staged course passed actual frozen-source import, configuration-signature
comparison, saved PPO hyperparameter comparison, 36 TRAIN / 24 VAL parsing,
all 391 source hashes, dataset hashes, and launcher validation. These checks
perform no new physical controls. Repeated preparation and launch requests are
rejected without modifying the existing preparation or observer.

The new observer holds an exact read-only Windows handle to the current pilot
observer, verified by PID, creation time and executable. It waits for normal
completion **and the pilot's full raw/start/Adam/RMS/parking audit**, then runs
the four courses serially. A failed performance screen remains in the results;
an infrastructure failure stops the queue without automatic retry. Existing
resource guards and finite native/observer deadlines apply.

The prospective development screen uses the original engineering thresholds:
≥80% overall success, ≥70% in each core class, collision-or-bounds ≤5%. Every
seed is reported regardless of its screen result. This does not convert known
development conditions into new final conditions or waive the basic failures.

This combined cohort completes the development coverage of five independently
warm-started policies. It is not registration of all five before any training,
from-scratch learning, a budget-matched H/B1 comparison, an added-PPO effect,
or final confirmation. Seed11 is now complete at11/24:curve4/8,static4/8,
walkers3/8,with five deadlines,five bounds failures and three collisions.
It also fails the development screen. [Complete audit](../reports/native_B1_broader_seed11_20261102.json).
Seed33 also completes at11/24:curve6/8,static0/8,walkers5/8,seven bounds and six
collisions. All29,621controls verify;619actualAdamupdates and unchangedRMS.
[Complete seed33 audit](../reports/native_B1_broader_seed33_20261102.json).
It fails the development screen. Recovered44now completes15/24;55is running.
The full five-seed outcome remains pending. An independent source checkout
reconstructs seed22's complete audit field-for-field using existing local data;
its raw/model capsule has not yet been published. This uses a different initial
model from the earlier 14/24 probe, so no causal gain or regression is computed.
[Local reconstruction receipt](../reports/native_B1_broader_seed22_independent_verification_20261102.json),
[live execution handoff](handoff.md).

[Separate TRAIN-only teacher preparation](broader_TRAIN_teacher.md) is queued
after this locked cohort. It does not change these courses or their outcome rules.

Seed44 subsequently stopped at the disk guard, saving11,520actualcontrols and
360Adamsteps in all35states. All11,520frame/engine joins verify. Its explicit
child resumes only8,960remainingcontrols with unchanged source and saved
model/optimizer/RMS/RNG;55remains the original unstarted course. This is a
recovered course,not an uninterrupted replicate. Completed old artifacts were
relocated with complete SHA256 verification to restore disk headroom.
[Interruption evidence](../reports/native_B1_broader_seed44_interruption_20261102.json),
[recovery registration](../protocols/native_B1_broader_disk_recovery_20261102.json).

The recovered44course is now audited:15/24successes (curve6/8,static4/8,
walker5/8),four collisions,three bounds and two deadlines. It fails the same
screen. Saved11520/360Adam plus8960/280Adam retain the original20480/640
learning budget;all31097original+child TRAIN/VALcontrols verify. Unknown tail
uncertainty is retained. Seed55 has started,so the fullfive outcome is pending.
[Recovered segment](../reports/native_B1_broader_seed44_recovered_20261102.json),
[all four completed outcomes and costs](../reports/native_B1_broader_partial_cohort_20261102.json).
The summary rebuilds with `python scripts/summarize_broader_b1_cohort.py
--allow-partial --output <fresh-output.json>`;without `--allow-partial`,a missing
seed is an error. Six integrity tests cover incomplete-cohort reporting,
recovery costs and altered evidence. Do not count96policy/condition evaluations
as96unique physical conditions or treat a partial mean as the fullcohort mean.
