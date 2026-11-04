# Matched new-PPO window for the flat baseline

The five original zero-PPO B1actors use the same inherited native TRAIN dataset
and4,329teacher controls. Their saved optimizer recipes differ,so they cannot
be launched as a uniform cohort without an explicit initialization version.

The new version preserves actor means,critic tensors,observation RMS and saved
RNG. It initializes all five learners with the earlier tested flat recipe:
1,280decision rollouts,320minibatches,10epochs,3e-6learning rate,0.01target KL,
and Gaussian log standard deviation−4. Learner rewards use the disclosed fixed
0.05scale;physical rewards and evaluation metrics remain in original task units.
No new supervised optimization or teacher interaction is performed.

Each seed receives30,720new actual controls on the same TRAIN48and64-exposure
rehearsal schedule as the current H/H0cohort. B1controls acceleration/steering
at20Hz;H chooses candidate pose/speed at2Hz and uses planning/MPC. The categorical
endpoint action mask has no direct continuous-control analogue. These algorithm
priors,decisions,network sizes and computation are separate reported costs.
Vehicle limits,task rewards,termination,calibration and shared parking aid agree.

Per seed,the queue retains the zero-PPO basic4probe,actual TRAIN48learning,
post-training basic4qualification and all24known validation entries. Qualification
uses the existing gate:at least3of4basic successes with zero collision/bounds.
All five seeds run irrespective of scores. Failure of this gate is retained;
it prevents claiming a healthy baseline,it does not trigger a selective retry.

The initializers have2,304inherited supervised updates and39,189charged prior
controls each,versus896supervised updates and the separately recorded H costs.
Thus the new PPO window and common teacher interaction budget are matched;
total historical development cost is not claimed equal. Extra supervision,
counterfactual TRAIN views and prior unsuccessful branches remain disclosed.
These are development conditions;no final-confirmation conditions are generated.

Prepare into a fresh directory from the repository root:

```powershell
.venv/Scripts/python.exe scripts/run_matched_flat_cohort.py prepare --output D:/ThesisNativeArchive_20261104/20261104_matched_flat_TRAIN48 --h-cohort D:/ThesisNativeArchive_20261104/20261104_independent_outcome_TRAIN48
```

Run its frozen copy with the existing Python runtime:

```powershell
.venv/Scripts/python.exe D:/ThesisNativeArchive_20261104/20261104_matched_flat_TRAIN48/source/scripts/run_matched_flat_cohort.py run --output D:/ThesisNativeArchive_20261104/20261104_matched_flat_TRAIN48
```

The coordinator first holds a read-only process handle for the registered H
cohort and waits for all25closed jobs. It verifies their protocols and normal
owned cleanup. A missing process without complete evidence or a recorded H
failure stops this observer before any B1native job. It never restarts or kills
the dependency. Twenty B1native jobs then run sequentially on2110,with four-hour
job guards,resource admission,no automatic retries and a bounded total window.
Local registration files contain runtime paths and must be sanitized before
public asset packaging. Prepared or waiting actors are not trained policies.
