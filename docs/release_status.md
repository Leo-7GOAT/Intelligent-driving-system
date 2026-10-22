# Development release record

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

The first reproduction clone remains retained, including its line-ending-only dirty file, as evidence of the issue. The clean release-tag clone verifies the correction. Representative CARLA replay specifically from a fresh release checkout remains pending; existing development CARLA evidence is not mislabeled as that verification.
