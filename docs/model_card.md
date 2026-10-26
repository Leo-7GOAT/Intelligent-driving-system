# Development model card

## Local terminal-margin models (2026-10-26 late evening)

[Matched experiment](../reports/native_terminal_margin_20261026.md) and
[artifact/raw ledger](../reports/native_terminal_margin_summary_20261026.json):
all147own-success episodes give4.5m3/4;47qualified episodes give4/4with identical
source weights/Adam/normalizer/RNG and1,000actor steps each. Actual starting
scenarios match. Frozen6m transfer and three new real-PPO adaptations remain3/4;
the speed-smoothed phase fails by bounds. No evaluation tolerance was loosened.
These models and raw traces are retained locally, not included in older public
capsules below. They are neither a healthy full-task baseline nor independent
confirmation. Current local CPU324/1skip/4CARLA exclusions.

## Local curriculum and retention models (2026-10-26 evening)

Publication addendum: the complete3m training/evaluation and unchanged-policy6m
transfer capsules below are now PUBLIC and actually downloaded/reproduced in
independent checkout5a300e9. Their794files,5,894rawcontrols,8measured-start actions
and2byte-identical figures verify, with zero new driving/updates. Other recent
development branches remain local. [Receipt](../reports/native_retention_public_reproduction_20261026.json).

| Complete development asset | Result | SHA256 |
|---|---|---|
| [3m PPO plus own-success replay](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_self_replay_3m_seed11_20261026.zip) | 4/4known3m roads;4,350train+710eval controls | `ebd6be337dbd439b5fea38f3fac7d8de7fe9d4d3e1ccf3d4c10bc717df3d5f05` |
| [Frozen original6m transfer](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_self_replay_6m_transfer_seed11_20261026.zip) | 3/4known6m roads;834eval controls, zero updates | `c2cf7c0725e139d5ab66cc0dd4e24d2565aebb6a89499b043975a6649a4b9feb` |

Archive sizes are10,203,688/7,836,349bytes and inventories413/381files. Model and
normalizer bytes match across the two phases. The3m phase uses140PPO and1,400
auxiliary actor optimizer steps, with inherited/pool costs separately retained.
Complete raw failures, frozen sources, saved/resume states, numeric replay data
and metadata transformation ledgers are included. Six auxiliary JSON path
redactions per capsule preserve every model/normalizer/raw byte. Ancestor raw
traces are not all duplicated in each capsule; the fixed replay input dataset is
included. This is an honest addition to the unchanged historical prerelease,
not formal research acceptance or a healthy complete-task baseline.

```powershell
python scripts/reproduce_native_asset.py --archive native_self_replay_3m_seed11_20261026.zip --sha256 ebd6be337dbd439b5fea38f3fac7d8de7fe9d4d3e1ccf3d4c10bc717df3d5f05 --output runs/replay_3m_reproduction
python scripts/reproduce_native_asset.py --archive native_self_replay_6m_transfer_seed11_20261026.zip --sha256 c2cf7c0725e139d5ab66cc0dd4e24d2565aebb6a89499b043975a6649a4b9feb --output runs/replay_6m_reproduction
```

See [full development outcomes](../reports/native_retention_20261026.md) and
[model/normalizer/raw-audit ledger](../reports/native_retention_summary_20261026.json).
Direct categorical B1 with actual PPO plus separately counted own-success actor
replay completes4/4three-metre known-road goals. Frozen six-metre transfer and
the4.5m continuation remain3/4. These are local, correlated seed-11 development
models, with the two capsules above now public; not a healthy full-task baseline or formal
hierarchy result. Preserve successful and failed endpoints and charge all
auxiliary actor steps and pooled sibling-branch interactions.

## Published matched H decision-head models (2026-10-26)

[Complete shared/unshared head asset](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_actor_head_seed11_20261026_full.zip):
402files/16,826,597bytes, SHA256
`8e4dac0babed85e9365158ef75200634968acfe24cbee58c3ad91f7095a37170`.
Export sourcef38f38a, exact execution sourcee0f9099 plus individually hashed
frozen sources. Both fresh seed11 models use the same original full four
development roads; shared208,034parameters scores1/4, unshared238,784scores3/4.
The latter fails road1with90search failures/900interventions and about8sdecision
P95, so this is not realtime, full acceptance or independent generalization.

Actual public download in independent checkoutf38f38a verifies all402files,
13,259rawcontrols across both audits,8measured-start actions with frozen
normalization, and the byte-identical complete figure (SHA256
`c0bf64b27d15720943480ee8fce0d7768cae5428749e91f0b95a2ae5dcbc6231`).
Zero new controls/updates. [Receipt](../reports/native_actor_head_public_reproduction_20261026.json)
and [complete outcomes](../reports/native_actor_head_20261026.md) are retained.
Six auxiliary metadata redactions preserve original model/normalizer/raw bytes;
the historical prerelease tag is unchanged.

```powershell
python scripts/reproduce_native_asset.py --archive native_actor_head_seed11_20261026_full.zip --sha256 8e4dac0babed85e9365158ef75200634968acfe24cbee58c3ad91f7095a37170 --output runs/head_reproduction
```

## Published B1 sanity negatives (2026-10-26)

These four complete assets retain five B1 endpoints, all frozen scores0/4 on
known approximately6m tasks; no healthy/full-task baseline is claimed. Together
they contain37,637newverified controls (inherited continuation counts are not
summed twice). All public downloads actually reproduce in independent checkout
70059ca:1,672filehashes,5rawaudits,20initial actions and4byte-identical figures,
with frozen normalizers and zero new driving/updates. [Exact receipts](../reports/B1_public_reproduction_20261026.json)
and [outcomes](../reports/native_flat_distribution_20261026.md) are retained.

| Complete development asset | SHA256 |
|---|---|
| [Clipped/tanh distribution pair](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_flat_distribution_seed11_20261026_full.zip) | `7fd051bf9762a51959c58c0ac9d51a1bc62653622f3d98b57d269a9e53283666` |
| [Progress reward](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_progress_sanity_seed11_20261026_full.zip) | `f2e968a76163294914689b01e2dfdbeaaeab5240357916dd94b8ec94bfc6db10` |
| [Exact continuation](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_progress_continuation_seed11_20261026_full.zip) | `e9f89de197e9eea46c863c9b25b263175806f441315c686ee74880a32bccbe88` |
| [Target-speed interface](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_speed_sanity_seed11_20261026_full.zip) | `18be359e984a0fe2ea7981956bebc7fb5018b89f63614d43605ff1dd24ebb243` |

Export source70059ca; exact execution sources/configs are inside each archive.
Each asset has six explicit auxiliary metadata redactions, with original
models/normalizers/raw controls unchanged. The historical prerelease tag is
unchanged. Local no-plot packaging drafts were not published. Reproduce any
complete asset with `scripts/reproduce_native_asset.py --archive FILE --sha256
HASH --output NEW_DIRECTORY` from the current source and fixed environment.

## Published complete fresh native-route models (2026-10-26 status)

[H0/H/B1 models and full development evidence](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_route_seed11_20261025.zip):570files,
63,447,215bytes, SHA256 `c28f506192b977d70e0f74edd16aa1f21d7c8c6b3475b4880adbae2ff06a9385`; export source `4f7e09a`.
Four known training roads, seed11: H0 **3/4**, H **2/4**, B1 **0/4** (all bounds).
These are failed research gates, not superiority or unseen-generalization proof.
All67,289controls are audited; all534evaluation decisions reconstruct. The
public-download independent reproduction verifies all files, three raw audits,
12initial actions and a byte-identical complete figure without new driving or
updates. Full checkpoint/normalization/config/source correspondence is inside
the asset. The prerelease tag is historical and unchanged; use current source.
See [complete report](../reports/native_route_complete_20261025.md) and
[reproduction commands](current_status_20261026.md).

## Published CPU collision-compute evidence (no model)

[road_collision_batch_20261025.zip](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/road_collision_batch_20261025.zip)
contains122manifestfiles/298,907bytes, SHA256
`cc58e501af5df3f4dec8f1a0a38f884bd36853900e5ac05923f08ce0ea926ba6`,
exported from `b36da0c`. All three offline prototypes, including the first
scattered-pose slowdown, retain their raw timings and frozen sources. The public
download's complete manifest/source hashes are verified independently. This
asset contains no learned model or additional CARLA execution, and supports no
hierarchy, learned-decision or realtime claim. Reproduction and independent
checkout measurements are in the [collision report](../reports/road_collision_batch_20261025.md).

## Published six-arm clipping development evidence

[native_clipping_seed11_20261024.zip](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_clipping_seed11_20261024.zip)
contains639manifestfiles/110,064,724bytes, SHA256
`bbea000c380e202c1801770679413dcf616b59dfc593423a5e57a31005f2bcfc`.
Export and independent reproduction source is `d35572bba8f4ce320e6838d32c0f66d1d13bd79e`.
The historical prerelease tag predates these tools; it has not been rewritten.

H0/H/B1 each fork the same parent into global and separate actor/critic clipping,
adding about20k controls per arm. All six score0/4, every probe is a deadline,
and independent clipping lowers mean progress for all three methods. These are
correlated seed11 forks; no independent-seed or superiority claim is supported.
Full accounting and failures are in the [clipping report](../reports/native_clipping_pair_20261024.md).

The independently downloaded archive verifies every file before loading models.
All six raw audits (142,446controls),24measured-start actions, the complete
paired report and the trajectory figure reproduce. This is offline evidence
reconstruction, not additional vehicle execution. Resume checkpoints are included;
upstream raw traces are not duplicated, but historical losses/unknown tails stay
charged. Twelve auxiliary metadata files have explicit path-redaction receipts.
CARLA/runtime installations and credentials are excluded.

```powershell
python scripts/reproduce_native_asset.py --archive native_clipping_seed11_20261024.zip --sha256 bbea000c380e202c1801770679413dcf616b59dfc593423a5e57a31005f2bcfc --output runs/clipping_reproduction
python scripts/analyze_native_gradient_pair.py --batch runs/clipping_reproduction/asset/20261024T064940Z_native_recovery --output runs/clipping_reproduction/paired.json
```

## Published native 100k development evidence

[native_structured_100k_seed11_20261024.zip](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_structured_100k_seed11_20261024.zip)
contains 895 manifest files, 175,728,217 bytes; SHA256
`94ef2147966e191d89aa4d42fba9efa9e2d053fee3b863a31421bc31c207305b`.
It was exported from clean source `1494af311632622bc6c462f8ec7e19e8be0533c9`
and uploaded to the existing development prerelease. The historical release tag
predates this asset; use the current source branch and recorded commits.

H0/H/B1 all score **0/4** on four training roads, with 100,160 / 100,160 /
100,050 retained training controls. All 250,019 segment training/evaluation
controls match CARLA frames. This is one development seed and failed task
completion, not research acceptance or evidence of superiority.

A separate checkout at `2040d3f4b560125cb7ff1001a35739ee1e400578` downloaded the
release and verified the archive plus all 895 file hashes before loading models.
All three raw audits and the twelve-panel figure rebuild byte-for-byte; all
twelve measured-start actions reproduce exactly with frozen normalization,
zero new controls and zero optimizer updates. Full CPU regression there is
**183 passed, five conditional skips**, with two existing SB3 deprecation warnings.
The receipt is
`runs/prerelease_20261022/native_road_checkout/runs/native_100k_downloaded_reproduction/reproduction.json`;
the regression XML is `runs/native_100k_release_regression.xml` in that checkout.

One proxy download ended with EOF and was rejected before extraction. A range
request retrieved the missing suffix from GitHub and restored the published hash;
an independent full retry also downloaded successfully with the identical hash.
Failed/partial files and the range-download receipt are retained locally.

```powershell
gh release download v0.2.0-dev.20261022 --repo Leo-7GOAT/Intelligent-driving-system --pattern native_structured_100k_seed11_20261024.zip --dir runs/native_download
python scripts/reproduce_native_asset.py --archive runs/native_download/native_structured_100k_seed11_20261024.zip --sha256 94ef2147966e191d89aa4d42fba9efa9e2d053fee3b863a31421bc31c207305b --output runs/native_reproduction
```

Use new output directories. This reconstructs saved evidence and inference;
it does not perform or claim a new CARLA closed-loop drive.

## Native continuation export format and earlier local validation

`scripts/package_native_continuation.py` exports every registered endpoint of a
normally completed native batch only after guard completion, verified owned-server
cleanup and rebuilding each raw endpoint audit. It retains all segment control
traces, failed/unfinished episodes, models, normalization, checkpoints, frozen
source and inherited resume checkpoints. Historical ancestor raw traces are not
duplicated; their charged costs, recovery chains and unknown tails remain explicit.

Model, normalization, raw traces and hash-bound audit inputs keep their bytes.
Auxiliary JSON paths (including JSON embedded in lifecycle stdout) are replaced by
non-resolving origin identifiers; original/export hashes and changed JSON pointers
are listed in the manifest. Nested ZIP/gzip and SB3 base64 serialization are scanned
without executing pickles. A private path in a critical input blocks export instead
of silently breaking its audit. Partial or existing archives are never overwritten.

A local validation export of the completed `20261024T044910Z_native_recovery`
B1 segment contains303 files and10,500 verified controls. Unpacking it reproduces
the original raw audit byte-for-byte and all four measured-start inference actions
with frozen normalization and zero new controls. Three exporter tests pass. This
is a local export check, not a newly trained policy, downloaded release asset or
complete H/H0/B1 comparison. The complete native 100k export and downloaded
reproduction are recorded above separately from this earlier local check.

```powershell
python scripts/package_native_continuation.py --batch COMPLETED_NATIVE_BATCH --output runs/NEW_NATIVE_ASSET.zip
```

The archive README contains the raw-audit and frozen-source inference commands.
Use its manifest to verify extracted file hashes before loading serialized models.

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
