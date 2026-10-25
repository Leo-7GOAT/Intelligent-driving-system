# CPU regression CI

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
