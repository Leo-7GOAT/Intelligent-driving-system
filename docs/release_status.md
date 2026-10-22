# Development release record

## Shared native-road development addition

Code`7a82fbc83f7b23ec27806b1e13ed1ee5f0dbe343`is pushed and remote-verified. Fresh`native_road_checkout`passes107tests/5skips, loads all four actually downloaded models with frozen normalization, and rebuilds the second batch's plot/summary byte-identically. Complete proxy download matches all396hashes; a separate interrupted direct download remains partial and is not counted as verified. Actual independent-checkout CARLA training from downloaded calibration/routes completes640controlspermethod with realupdates/reload, Hdeadline/B1rasterfailures retained;1280framesverified. Ownedserver stopped withreceipt in thatcheckout's`runs/20261022T132946Z_native_road_learning`. No CI claim.

`native_road_learning_development.zip` is actually uploaded to the existing pre-release:396manifestfiles,14,053,181bytes,SHA256`b54fdd0f142f1944878a9e095363ffde716162f2dd772dcaafd9cb9b01c34757`. GitHub uploaded state/digest match. It retains both new H/B1 interface-training batches, all failed/unfinished episodes, models, normalization, road grids, frame ledgers and the read-only map-query audit. There are no task successes and these repeats share seed11. Code publication, downloaded verification and independent reproduction are recorded after execution below. Earlier assets/tag remain unchanged.

## Acceleration feedback evidence addition

Code commit `abb2b1a7498452de72fe9e0118b42883f30f11e7` is pushed and remote-verified. Independent checkout passes105 tests with5conditional skips, reloads the published145-file PPO camera asset, and reconstructs both downloaded diagnostic plots and verified summaries byte-identically. Actual CARLA feedforward replay from this checkout completes4/4 routes,1920 control ticks, speed RMS .105/.148/.105/.166m/s and jerk68.53/77.73/67.94/71.95m/s³. Deceleration still reaches approximately−24m/s². Detailed trajectories vary; this is repeatability evidence, not another independent condition or seed. The owned server was verified empty and stopped. Raw replay evidence is in that checkout's `runs/20261022T124930Z_native_curve_control`.

The existing pre-release additionally contains `acceleration_feedback_diagnostics.zip`:377 manifest files,2,487,456bytes, SHA256`c1f4cedb03d085b476c540e69e53a501e93dff0c3087186e4fb3d82c4226d6ba`. GitHub reports uploaded with that digest; an actual fresh download matches the archive and every manifest member. Both new four-route experiments are included, with PI failures and feedforward residual braking spikes. No older asset or tag was replaced. Code publication and independent checkout verification for this addition are recorded after actual execution, not inferred from the upload.

## Native actuation diagnostics, 2026-10-22

Code067db54b2803d8cc26fd9b63fbc4db62900606c2was pushed to the existing feature branch and matched remote SHA. Independent checkout at that commit passes102tests/5conditional skips and reloads the published camera model. It reconstructs all three native diagnostic plots and verified summaries byte-identically, including from an actual fresh release download. ActualCARLA replay from this checkout completes4/4native routes,1924controlticks, no observed collision or sampled road departure, but retains deceleration near−25m/s²andjerk131–139m/s³. Detailed trajectories differ. Owned server was verified empty and stopped; receipts live in its`runs/20261022T120642Z_native_curve_control`.

The existing pre-release now additionally contains [native_actuation_diagnostics.zip](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_actuation_diagnostics.zip),3,909,935bytes/559manifestfiles, SHA256`5f8822bc6caca88429e5830f829b916352c8e6588f0debe637b69399a65160a5`. GitHub uploaded state/digest and downloaded content hashes match. No existing tag/asset was replaced. All12control episodes,20raw brake phases and12throttle phases are retained, plus frozen source and calibrations. No CARLA binaries/maps/runtime directories are bundled. The failedcoast/manualgear hypotheses remain visible and disabled by default. This artifact is developmental control evidence, not PPO superiority or completed research acceptance.

## Camera evidence addition, 2026-10-22

The existing pre-release now also contains `candidate_v12_H25k_camera_examples.zip` (2,218,569 bytes), SHA256 `c24ad7dec570f7becaae46f0014ce5320f5e9a2ae87849416d327d5f887aac44`. GitHub reports `uploaded` with the same digest; a fresh download matched the local hash. The old tag and two old assets were not replaced. This separate package contains the actual CARLA-adapted H25k model, frozen source/config/normalization, measured calibration, both success/failure traces, sensor/world frame manifests and real videos. See the [model card](model_card.md) for the later 100k regression and [current evidence report](../reports/camera_staged_20261022.md).

All 145 manifest files were checked; the actual ZIP has exactly those files plus its manifest. A scan of the immutable archive, including decoded SB3 metadata, found no credential patterns or concrete user-home paths. Local CPU verification created bytecode caches outside the ZIP; those were detected, not uploaded, and the verifier now disables bytecode generation. Independent clean-checkout environment model loading passed. Both videos fully decode at 20 fps (362 and 901 frames), with hashes and actual sensor/world linkage checked. The new full CPU regression passed 74 tests with 5 conditional skips. This does not constitute another independent CARLA experiment, hosted CI, or completed research acceptance.

## Original development release

The camera/curriculum source commit `02f75a8148fbae1fb1cbcc9ac00a4f22b4c5f5ac` was pushed to the existing feature branch and matched `git ls-remote`. A new clone (`camera_checkout`) at that commit is clean. Using the independently installed pinned environment from the first clean checkout, it passed 74 tests with 5 conditional skips, loaded the freshly downloaded camera model, verified all 145 manifest files and decoded both downloaded videos. Optional imageio-ffmpeg 0.6.0 was installed only into that isolated environment.

Actual CARLA camera replay from this new checkout and downloaded model also completed: static7301 success at 22.3754 m progress, curve7301 deadline at 16.3914 m, 1,242 control/1,303 total world ticks, frozen normalization and no unresolved infrastructure failures. The same success/failure outcomes reproduce; detailed progress and timing differ and are not claimed bitwise deterministic. New 343/901-frame recordings were encoded and fully decoded. The owned server was verified empty and stopped. Evidence is in `runs/prerelease_20261022/camera_checkout/runs/20261022T101242Z_carla_camera_evidence`. This is repeatability evidence for the two declared examples, not another independent seed or a new confirmation set.

Target repository: `https://github.com/Leo-7GOAT/Intelligent-driving-system`, existing public visibility retained. Development branch: `feature/physical-hierarchical-ppo`. Intended pre-release tag: `v0.2.0-dev.20261022`. Publication is verified by remote SHA and release metadata, not by this document alone.

Before publication:

- Complete local CPU regression and the20-episode asset replay are checked.
- All152 asset files are hashed. Source release scanning found no credential patterns or concrete home-directory paths; the generic path-detection literals in the packaging script were reviewed as expected code.
- The asset SHA256 is `ad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c`.
- The asset contains two development models and all associated200k evaluation cases, not a selected successful subset.
- Dependency runtimes, CARLA binaries/maps and local environments are excluded. Original MIT attribution remains intact.

## Verified fresh-checkout reproduction

Source commit `bbaf3037ea5e69754e6d1ea126332020c2e385af` was pushed to the development branch and the remote SHA matched. A new clone from GitHub installed `requirements.lock.txt` into a new isolated Python3.11 virtual environment. `pip check` reported no broken requirements. CPU regression:66passed,5conditional skips. The extracted asset verified152 files and replayed all20 episodes with recorded outcomes matching; normalization stayed frozen and the figure was rebuilt.

The clone exposed a line-ending-only normalization issue in the unchanged legacy `dialog_parser.py`. Its index was normalized to the repository LF policy; `git diff --ignore-space-at-eol` confirmed no semantic changes. The12 original archived files retain their exact original SHA256 bytes.

The Git HTTPS client required the already configured local system proxy for this machine; no global Git settings were changed. No hosted CI result is claimed. Publication of this development snapshot does not satisfy the research acceptance criteria.

## Published and downloaded

Development pre-release: https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261022

Tag and remote release commit: `9bcaa79d34a6985a3d12764d3bf01383c7f06d83`. GitHub reports `isPrerelease=true` and the3220328-byte asset as uploaded. A fresh download produced the sameSHA256 listed above. A second fresh checkout of the actual release tag is clean; using the independently installed pinned environment it passed66tests with5conditional skips and replayed all20downloaded-asset cases with matching outcomes. No force push or tag rewriting was used.

The first reproduction clone remains retained, including its line-ending-only dirty file, as evidence of the issue. The clean release-tag clone verifies the correction.

Representative CARLA replay from the fresh release checkout is complete. The downloaded H200k model reproduces static7301 deadline (2.786m progress) and dynamic7302 success (22.508m progress), with frozen normalization,1,369 control ticks and1,430 world ticks including reset. This reproduces an expected failure as well as a success; it is not new independent validation. The first server-readiness attempt timed out before policy execution and remains archived; a bounded retry connected to an empty0.9.15 world. The owned server was confirmed empty and stopped afterward. Evidence: `runs/prerelease_20261022/release_checkout/runs/carla_replay_retry1`.

The release also contains `carla_replay_support.zip` (3,602 bytes), with measured calibration, the two representative cases, license and manifest. GitHub reports it uploaded with SHA256 `f28ce6c76a5ad45584a2f971c60a5cd220567362712c8a1198c19667009f0657`. It contains no CARLA binaries or runtime. Later1M lightweight and100k CARLA negative results are disclosed in the release notes; the release remains a development pre-release.
