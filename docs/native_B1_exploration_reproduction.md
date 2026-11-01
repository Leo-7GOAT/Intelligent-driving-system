# Actual PPO exploration pilot: complete and rejected

The seed22pilot performs10,240new actual CARLA PPOcontrols and320Adam updates,
with no teacher actions or supervised updates. PPOcontrol lineage reaches30,720.
Only the two initial Gaussian log_std values change(std~.018to.10)from the
fixed4f205708...replay reference;all other initial weights,Adam state,counters,
RMS,configuration and saved RNG are retained. The source and intervened initial
checkpoints are both included,so this intervention can be reconstructed exactly.

All12,219training/evaluation controls are audited. The four original TRAINroad
endpoint outcomes are:

| Case | Pure-policy outcome | Controls |
|---|---|---:|
| 0 | out_of_bounds | 420 |
| 1 | deadline | 900 |
| 2 | success | 318 |
| 3 | success | 341 |

The unchanged gate≥3/4successandzero collision/bounds fails. This model is not
promoted. The15bounds/7success terminal TRAINepisodes and one unfinished tail
are also retained. These are development failures,not independent final tests.
Both the guard and owned-empty-server cleanup complete normally;no training
process remains active for this batch. No additional budget is queued for it.

[Complete report](../reports/native_B1_exploration_PPO_pilot_20261101.json),
[four raw-bound endpoint panels](../reports/figures/native_B1_exploration_PPO_seed22_endpoint_20261101.png).
The frozen frame-level dataset,final/initial/original models,normalizers,
checkpoints,configuration,raw logs and retained failures are published as
7parts plus a manifest on the existing
[development prerelease](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101).
Its target remainsba932787113168ad5781ee640c6f06d09bbee0e1;the capsule's frozen
source manifest,not that older tag,defines this new experiment.

Archive:56,422,535bytes,563files,SHA256
`438778411ba8cba4faf2708c3e07daa39b70c8597927f6db18fc55a1118bdbcf`.
Eight auxiliary metadata files have path-only portability transformations;
all raw controls,source,models and normalization bytes remain protected.
The transformations and original hashes are retained in the archive manifest.
The endpoint figure is an explicit versioned Gitreference outside this already
published archive;the reproduction command must include `--reference-figure`.

From a checkout containing this document and the locked environment:

```powershell
.venv/Scripts/python.exe scripts/download_native_asset.py --manifest-url https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261101/native_B1_exploration_PPO_seed22_20261101_repro.zip.parts.json --output runs/public_B1_exploration_download --sha256 438778411ba8cba4faf2708c3e07daa39b70c8597927f6db18fc55a1118bdbcf
.venv/Scripts/python.exe scripts/reproduce_native_asset.py --archive runs/public_B1_exploration_download/native_B1_exploration_PPO_seed22_20261101_repro.zip --sha256 438778411ba8cba4faf2708c3e07daa39b70c8597927f6db18fc55a1118bdbcf --output runs/public_B1_exploration_reproduction --reference-figure reports/figures/native_B1_exploration_PPO_seed22_endpoint_20261101.png
.venv/Scripts/python.exe scripts/audit_native_exploration_initialization.py --batch runs/public_B1_exploration_reproduction/asset/20261101T075805697090Z_native_B1_exploration_PPO --output runs/public_B1_exploration_initialization.json
```

The first command downloads without authentication. The second verifies every
file,all control frames and four measured-start actions,and rebuilds the figure
byte-for-byte. The third reapplies the declared std-only intervention in memory,
then compares every policy/optimizer tensor,counter,normalizer,RNG and config.
No simulator steps,model overwrite or optimizer updates occur in reproduction.
Use fresh output directories;failed or completed artifacts are preserved.

The three preceding recovery profiles also failed;their9,010physical controls
are charged in this pilot's cost lower bound. This development sequence adds
21,229actual controls overall,1,536auxiliary Adam proposals in rejected recovery
fits,and320actual PPOAdam updates in this separate pilot. The five-seed healthy
B1gate,PPOvalue,stable feedback benefit and120new final conditions remain unmet.
