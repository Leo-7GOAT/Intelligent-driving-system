# Development model card

## Separate v12 CARLA camera example model

`candidate_v12_H25k_camera_examples.zip` is a separate, single-seed illustrative asset. It contains the shared-candidate H policy with 605 inputs and 31 unmasked actions, trained for 101,923 lightweight physical steps and then 26,564 actual CARLA world ticks (25,511 controlled and 1,053 reset ticks). CARLA training performed 10 PPO phases and 400 optimizer steps, with measured actor L2 change 2.5423 and exact checkpoint reload. Model SHA256: `b03aba60b8ca0c35ff4a39a30183f0504fd2009193929bd3bb6914298b1979c2`.

The full six-case development evaluation was 5/6 at this checkpoint and regressed to 4/6 after 101,048 CARLA world ticks; both results are retained. This asset's two predeclared inspected camera cases illustrate static success and curve timeout. They are not an independently selected final model/test or evidence that 5/6 generalizes. The package includes both full evaluation traces, frozen source/configuration/normalization, measured calibration, sensor/world manifests and actual RGB videos. Absolute metadata paths are replaced with non-resolving origin identifiers; model and normalization bytes are unchanged. Complete training transition archives remain local. Reproduction commands and limits are in [the CARLA guide](CARLA_bridge.md#real-camera-development-evidence-2026-10-22).

## Original v8 lightweight development models

The `development_v8_seed11.zip` asset contains **development-only**, single-seed H and H0 policies. It is not the final research model, a safety-certified controller, or an autonomous-road-driving product.

| Model | Physical training steps | PPO optimization phases | Development completion |
|---|---:|---:|---:|
| H | 201,117 | 79 | 8/10 |
| H0 | 201,115 | 79 | 6/10 |

Both policies use SB3 PPO, separate two-layer64-unit actor/value networks,31 joint local-goal/speed actions,450 observation values,0.05s physical controls and0.5s high-level decisions. Training samples empty/static/narrow/dynamic scenarios; the configured extra obstacle count is0, but each nonempty family still generates its defining obstacles. Evaluation includes an additional unseen-layout family. The two development conditions per family are not a diverse independent confirmation set.

H receives99 additional features:31 candidate geometric witnesses/travel proxies and six previous-execution outcomes. H0 receives zeros in the same slots; network shapes and low level match. Candidate analytic witnesses are sufficient geometric checks for that connection, not guarantees that Hybrid A* succeeds, that tracking is safe, or that dynamic interactions succeed. No policy action is masked or replaced by a rule.

The asset includes frozen source, weights, observation normalization, configuration/signature, optimizer and training-episode summaries, and all ten200k evaluation traces per policy. Exact model and file hashes are in `manifest.json`. The original full training transition logs remain in the local experiment archive; this bounded asset does not contain every training transition or CARLA binaries/videos.

Known limitations: both200k policies fail the unseen lightweight layout; the200k H/H0 difference disappears at300k (8/10 each, with different failed families). Frozen200k transfer to the calibrated CARLA vehicle gives2/6 for each policy. Failure cases repeatedly select unplannable goals and brake. The asset demonstrates reproducible learned closed-loop execution, not an established feedback advantage or generalization claim.

Later development results must not be omitted when interpreting this asset: continuation to1,000,551/1,001,774 physical steps yields H8/10 and H0 9/10. The same200k source policies separately adapted for100,890/100,985 actual CARLA world ticks (including resets) each reach3/6. Each CARLA run completes38 PPO optimization phases and1,520 optimizer steps with verified parameter changes and exact reload. These single-seed observations do not demonstrate feedback benefit. The published200k weights remain unchanged; they are a reproducible development snapshot, not the best final model selected after these results.

Only load serialization files from a trusted, hash-verified source. The reproduction command verifies every listed file, uses the bundled frozen source, freezes normalization, rebuilds the development endpoint figure and checks all requested replay outcomes against saved raw results.

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_development_asset.py --asset runs/development_v8_seed11 --output runs/reproduced --full
```

The project retains its MIT license and original attribution. Dependency software is installed separately under its respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
