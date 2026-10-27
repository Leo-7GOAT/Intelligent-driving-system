# CPU regression CI

Source5ababeb actually passes **518tests,15skips,4CARLA exclusions**in190.91s:
[run40496081032](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40496081032).
Root33new fit-split/ordered/view/gate checks pass16.43s;independentcheckout5ababeb
passes33in10.20s. These do not establish driving quality or final acceptance.

Latest actual hosted source36f5297 passes **511 tests,15 skips,4 CARLA exclusions**
in92.08s: [run40491222582](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40491222582).
Gate/ordered source8a1451b separately passes511/15/4in123.40s:
[run40490208868](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40490208868).
Root and independent locked Windows environments pass60targeted gate,ordered
actor,registered-view,case-manifest and recovery checks in16.06/18.00s. The
legacy public B1 figure remains byte-identical after twelve-case condition labels.
Hosted CPU verification is separate from CARLA training and research acceptance.
Older version-specific records follow.

Latest local full regression:324passed/1skip/4CARLA deselections,168.19seconds,
with raw log/XML retained under runs/terminal_margin_smoothed_cpu_20261026.*.
This covers the terminal-margin selector, strict curriculum gate, qualified
online replay and speed-only label smoothing. The hosted results below describe
their exact older source versions until a new run is recorded.

Source1f403bcnow has actual hosted success:322passed/3skips/4CARLA deselections
in122.79s, [run40322362420](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40322362420).
Independent fixed Windows checkout1f403bcalso passes324/1skip/4deselections in
124.82s. [Receipt](../reports/native_terminal_margin_reproduction_20261026.json)
also verifies the local margin model's four measured-start actions and identical
raw audit with zero new driving or optimization. Inference explicitly uses the
recorded frozen-source root; invoking the checkout root directly was safely
rejected by its source fingerprint before inference.
Nineteen targeted actor-replay, continuation, recovery and endpoint-audit checks
also pass. Actual hosted source5cf9297passes313/3skips/4deselections in66.40s:
[run40304264696](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40304264696).
Later local changes do not inherit that hosted result; inspect their actual CI.

The subsequent online/export source5a300e9also passes actual hosted regression:
318passed/3skips/4CARLA deselections in97.21s,
[run40310891811](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40310891811).
The fixed Windows environment passes320/1skip/4deselections at this version.
Independent checkout5a300e9also passes320/1skip/4deselections in97.33s; its fixed
dependency environment is reused, rather than claiming a newly installed runtime.

`.github/workflows/cpu-regression.yml` runs the repository's CPU tests on source,
test, dependency or workflow changes, pull requests, and manual dispatch. It
uses Python3.11, the exact dependency lock, CPU-only PyTorch and a standard
`ubuntu-24.04` runner. Official checkout/setup actions are pinned to full commit
IDs; the job has read-only repository contents permission and a20-minute limit.

The repository is public. Standard hosted runners are free for public
repositories under [GitHub's billing rules](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
This workflow does not request larger runners, upload artifacts or configure a
persistent cache. Its test report stays in the job's ephemeral filesystem; the
console log and job summary expose results and scope.

The CPU command is:

```powershell
python -m pytest -q -m 'not carla' --junitxml=runs/cpu-regression.xml
```

It includes actual lightweight vehicle integration, forward Hybrid A* search,
QP/MPC, reward/terminal handling, PPO training/reload, observation and evidence
audit tests. It excludes the `carla` marker and explicitly disables all live
CARLA opt-ins, including the separately guarded continuation test. It never
starts the simulator or connects to the locally owned training server. A green
CPU job is not a CARLA training result, a realtime result or research acceptance.

Local Windows runs additionally cover retained Windows process-handle ownership;
Linux skips those platform-specific cases. Conditional skips/deselections must
stay visible in the logs. Raw CARLA audits, policy/model evidence and live
resource guards remain separate local deliverables.

## Latest actual hosted execution

[CPU regression run40192669405](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40192669405)
actually succeeds at `7605ba07408a161558b9350ff76660aea06dc96a`, including
the new initial/trained-weight evaluator and evidence rejection tests.
Locked installation and `pip check` pass on CPython3.11.17.

| Environment | Passed | Skipped | CARLA deselected | Test seconds |
|---|---:|---:|---:|---:|
| Local Windows, project environment | 295 | 1 | 4 | 216.08 |
| Hosted Ubuntu24.04, fresh Python setup/install | 293 | 3 | 4 | 92.85 |

The same platform/opt-in skip distinction and three warnings described below
remain. Local full output/XML is `runs/native_weight_cpu_regression_20261025.*`;
remote metadata/logs are `runs/cpu_ci_github_40192669405.json` and `.log`.
The separate clean checkout passes32targeted tests with its fixed environment.
Actual H0 seeded initialization also reproduces there with all30saved policy
tensors exact; see [weight-contrast scope](native_weight_contrast.md). No live
CARLA contrast ran in CI or during these offline initialization checks.

## Historical first hosted execution

[CPU regression run40188376038](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40188376038)
completes successfully at source
`c77314f8b5d82f9aae985423fc4bd69203c7f808`. The standard Ubuntu24.04 runner
installs the locked CPU dependencies, reports no broken requirements, and runs
with CPython3.11.17. The job is actually executed; this is not merely a workflow
file or a proposed CI command.

| Environment | Passed | Skipped | CARLA deselected | Test seconds |
|---|---:|---:|---:|---:|
| Local Windows, project environment | 278 | 1 | 4 | 186.17 |
| Hosted Ubuntu24.04, fresh Python setup/install | 276 | 3 | 4 | 67.42 |

The two additional Linux skips are Windows-only process ownership/infrastructure
tests. Both environments retain the opt-in live continuation skip. Three
warnings remain visible: two SB3 deprecations and nonfinite-pose arithmetic in
the invalid-input equivalence test. The smaller Linux pass count is not a
different scene-selection rule or a removed failure.

Remote run metadata and console logs are saved locally as
`runs/cpu_ci_github_40188376038.json` and `.log`. CI success supports portable
CPU regression and locked dependency installation. It adds no CARLA control
steps, independent training seeds or evidence of hierarchical advantage.

The exact CPU selection command passes locally on Windows: **278 passed,
one skipped, four CARLA tests deselected**,186.17s. The skip is the opt-in live
continuation test. Two existing SB3 deprecations and the intentional nonfinite
pose arithmetic warning remain visible. Output is
`runs/cpu_ci_local_20261025.log` with JUnit XML beside it. This local result is
not a hosted Linux CI result.
# Current verified revision — 2026-10-26

At sourcee0f9099, [actual hosted run40281946359](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40281946359)
passes302tests/3platform-or-opt-in skips/4CARLA deselections in145.38s.
Independent fixed Windows checkout passes304/1skip/4deselections in168.97s.
This is CPU verification; CARLA training and research gates are separate.
The earlier counts above retain their historical revision scope.
