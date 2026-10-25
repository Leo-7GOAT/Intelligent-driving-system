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

Initial publication/run status will be recorded after actual execution. The
presence of this workflow file alone is not reported as CI success.

The exact CPU selection command passes locally on Windows: **278 passed,
one skipped, four CARLA tests deselected**,186.17s. The skip is the opt-in live
continuation test. Two existing SB3 deprecations and the intentional nonfinite
pose arithmetic warning remain visible. Output is
`runs/cpu_ci_local_20261025.log` with JUnit XML beside it. This local result is
not a hosted Linux CI result.
