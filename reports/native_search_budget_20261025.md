# Search-budget diagnosis and prospective contrast — 2026-10-25

Final update:074939Z completes3/4 normally,1,919controls+123reset; road3deadline
with21reference/69wait. Whole raw audit passes; no PPOgatepass. Increasing nodes
alone does not fixcoverage. See [heading-guide evidence](native_dubins_guide_20261025.md).
The live execution status paragraphs below are historical snapshots.

The full research goal is still incomplete. No new PPO training is claimed.

## Interrupted feedforward B3

`20261025T070545Z_native_rule` is stopped by its original30-minute guard at
15:35:48, exit1/reason `wall_deadline`. The owned bridge/server are cleaned up;
an empty-world0.9.15 probe and `owned_empty_server_stopped` receipt are saved.
Roads0/1/2 have terminal success records at341/350/317controls. Road3 reaches
82decisions in heartbeat;1,828controls are reported overall. It does not reach
its900-control task endpoint before the wallguard. It is an interrupted run,
not a completed3/4 evaluation, and cannot pass the4/4training coverage gate.

Both gzip files have incomplete tails. The bridge prefix retains1,568 applied
control responses; the policy trace prefix retains1,638controls. These differ
from the heartbeat and are not a full raw audit. Unknown/unflushed tails remain
unknown; no fabricated final case metrics or full comparison figure is written.
The stricter existing whole-run comparator correctly cannot accept this run.

## Bound first-wait state

`scripts/diagnose_rule_wait.py` requires a stopped worker and normal owned
cleanup, checks frozen navigation sources and binds the case prefix through the
first wait to real bridge frames, including initial actuator/reset state.
It never steps a plant. In the actual run it verifies200selected prefix controls
for road3; it does not claim to verify the entire interrupted run.

Before first wait atdecision19, measured pose is
(17.9368642,14.6196354,-0.6703852rad), speed1.464334m/s at9.5s.
The three preceding reference decisions have maximum lateral errors
.000465/.002859/.003788m and heading errors .001420/.015933/.005959rad;
all30 MPC cycles solve. This does not support blaming gross tracking failure.
The first wait's rule searches three targets, exhausts5,400nodes total and
spends about19.78s before choosing wait.

Unchanged production decoding/search/time-parameterization queries at that
measured state produce valid references for actions7/8/9/10, with26failed
references and one wait. Thus wait is not evidence that every candidate fails:
the bounded rule searches only three optimistic targets. A separate direct-goal
query fails at1,800nodes; the preregistered fourfold7,200-node diagnostic finds a
path after2,277expansions (cost9.9225). This establishes a finite-search-budget
failure at this state, not successful execution of the alternative path.

The artifact is `runs/20261025T070545Z_native_rule/wait_diagnostic.json`; it binds
source, protocol, scene and the stopped raw prefixes. Search failures elsewhere
are not global infeasibility proofs. Independent reconstruction is required.

## Prospective change, fixed before execution

Change only the planner node budget from1,800 to7,200 relative to the preceding
feedforward/continuation route variant. Retain all four known training roads,
45s task deadline, same rule score/three-search limit, native-score speeds,
candidate geometry, feedback, reward, MPC, physical limits and exact measured
feedforward calibration. No action mask or policy substitution. Use the same
budget for newly trained H/H0/B1 configurations; B1 remains direct actuation.
Planner computation is an explicit engineering cost, not PPO learning benefit.

Register a60-minute wallguard before launch to accommodate the expanded search;
12GiB disk/4GiB RAM minimum,85°C GPU stop,no-progress300s,zero retries and owned
cleanup remain. The interrupted run is retained and not resumed or overwritten.
Only a new completed/audited4/4coverage result can stage fresh H0/H/B1seed11 at
20,480controls each; use new model signatures and no inherited weights. Do not
change the coverage gate, final task gates, five-seed requirements or final
confirmation protocol based on this diagnostic.

```powershell
python scripts/diagnose_rule_wait.py --batch runs/20261025T070545Z_native_rule --case 3 --output runs/wait_diagnostic_rebuilt.json
python scripts/evaluate_native_rule.py --prepare runs/20261024T075110Z_native_rollout_pair --source-method H_rollout32 --catalog route_frame_terminal --continuation-feedback --calibration runs/20261022T122427Z_native_curve_control/calibration/calibration.json --planner-nodes 7200 --wall-seconds 3600
```

## Execution and reproduction status

Source1d3c9a9 is pushed. Independent checkout reproduces the selected-state
diagnostic byte-identically, SHA256
`dd99aa8261f2579472b794fb2b116a576e0535e483837e61055501d5a01a67d9`, and passes33
targeted tests in10.40s. The stored larger-budget goal path also passes the
unchanged timeparameterizer and sampled full-body check:81samples,6.7020s,
terminal speed0,maxcurvature.181823/m. This does not execute that alternative.

Actual new B3 batch `20261025T074939Z_native_rule` starts after the independent
checks and0.9.15 empty-world probe. At15:59, firstthree roads succeed in
341/361/317controls; fourth is live atdecision25. Normalcompletion/rawaudit
and coverage remain pending. No fresh PPO has started. The60-minute deadline
is16:49:39; owned lifecycle and resource guards are recorded inhandoff.

The selected-state figure is generated locally and visually checked, SHA256
`0e332c466467943ff82d6c2ca07537dd0a98f0dea3bc633dd207c95b22be092e`:

```powershell
python scripts/plot_rule_wait.py --batch runs/20261025T070545Z_native_rule --diagnostic runs/20261025T070545Z_native_rule/wait_diagnostic.json --output runs/first_wait_rebuilt.png
```

This figure labels the measured prefix and unexecuted offline path separately.
