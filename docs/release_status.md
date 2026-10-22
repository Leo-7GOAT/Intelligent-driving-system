# Development release record

Target repository: `https://github.com/Leo-7GOAT/Intelligent-driving-system`, existing public visibility retained. Development branch: `feature/physical-hierarchical-ppo`. Intended pre-release tag: `v0.2.0-dev.20261022`. Publication is verified by remote SHA and release metadata, not by this document alone.

Before publication:

- Complete local CPU regression and the20-episode asset replay are checked.
- All152 asset files are hashed. Source release scanning found no credential patterns or concrete home-directory paths; the generic path-detection literals in the packaging script were reviewed as expected code.
- The asset SHA256 is `ad656334aa8386d1efcf0afbd14b5ef45f3fc4fae1da763b0dc8542469cc6b1c`.
- The asset contains two development models and all associated200k evaluation cases, not a selected successful subset.
- Dependency runtimes, CARLA binaries/maps and local environments are excluded. Original MIT attribution remains intact.

Clean-checkout installation, tests and model replay must be recorded after actual execution. No hosted CI result is claimed. Publication of this development snapshot does not satisfy the research acceptance criteria.
