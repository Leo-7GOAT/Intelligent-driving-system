# Measured direct-control representation diagnostic (2026-10-27)

A continuous direct network initialized from actual B3controls completes **3/4**
known full static CARLA roads. The remaining road times out without moving.
All1,976frozen controls verify; collision/bounds/interventions are zero in these
four probes. This is **extra supervised initialization, zero PPO updates**.
[Raw/model/cost ledger](native_measured_bc_20261027.json).

The1,926teacher controls include all four cases, including900deadline controls.
No successful-case subset is used. State, previous command, actuator evidence
and static-road inputs are reconstructed from measured frames; the holder has
no plant step method. Direct-policy history matches production-step observations
in tests. Small QPrequest residuals up to1e-3use the same physical action clipping;
raw commands/acknowledgements stay unchanged, larger residuals are rejected.

The actor receives896supervised optimizer steps. Every fifth teacher control is
excluded from fitting and normalization. Its final action MSE is0.001272on those
correlated controls; this is not independent validation. The value network is
not fitted. The saved PPO optimizer and interaction/update counters remain zero.

Compared with the earlier categorical2Hz B1, this diagnostic also changes to
continuous acceleration/steering and a0.05s simulated action period. No Hybrid A*
or MPCproduces its execution commands. Shared geometric witness inputs still
incur computation. Full decision P95is0.075–0.104s and over-period fractions are
34.6–100%; do not claim wall20Hz. This cannot isolate a single cause or establish
hierarchical advantage. A formal comparison must share teacher access and
separately charge teacher, supervised and on-policy costs.

The rejected arc-catalog experiment is also complete: H/H0/B1all0/4after
20,480/20,480/20,680training controls. All69,514training/evaluation controls verify.
H's30initial tensors and postinitialization RNG exactly match the earlier
route-offset H, whose endpoint was2/4. Final rollout counts differ explicitly.
[Complete rejected-version ledger](native_arc_catalog_20261027.json). Better
initial geometric connections did not translate into whole-task learning.

Real PPO continuation `20261027T024727Z_native_supervised_ppo` now starts from the
3/4supervised network: nominal20,480new controls, frozen teacher-training-only
normalization, Gaussian log_std−2.3. Only log_stdchanges at initialization; every
mean/critic parameter and original checkpoint remains preserved. There is no
additional teacher query or replay in that phase. Its learning and frozen
endpoint must be evaluated separately; the3/4initial result is not credited to
PPO. It is one correlated seed, not a new independent strategy.

The first data-conversion failure and the first worker preflight failure are
retained. The latter occurred before started.json/bridge/control; its complete
original protocol/logs remain under preflight_attempt_01. Adding the missing
calibration hash reused identical fitted weights with zero added updates.

```powershell
python -m pytest -q tests/test_measured_demo.py tests/test_supervised_ppo.py
python scripts/prepare_native_measured_bc.py --parent <complete_audited_B3_batch>
python scripts/prepare_native_supervised_ppo.py --parent <complete_audited_supervised_batch> --controls 20480 --wall-seconds 7200
```

Preparation is not execution. Never repeat registered/started evidence. Use the
owned server, version/empty-world probe, identified worker and resource guard
from the handoff. Original research acceptance, five independent trained seeds,
new paired confirmation conditions and PPO value over strong B3remain unmet.
