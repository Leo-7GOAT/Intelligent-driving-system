# Completing the broader B1 development cohort

Seed22's previously registered single pilot is running on 36 existing TRAIN
conditions, with 24 known development validation conditions after 20,480 new
actual PPO controls. Its original protocol and selection history remain unchanged.

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
or final confirmation. Outcomes remain pending. [Live execution handoff](handoff.md).
