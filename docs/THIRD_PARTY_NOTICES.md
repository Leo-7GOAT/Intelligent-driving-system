# Third-party notices

The original project MIT license and copyright attribution in `LICENSE` are preserved. The development asset contains project-generated models/data and project source. It does not redistribute installed dependency runtimes, CARLA server/client binaries, map assets, or third-party video assets.

The following inventory was read from the installed distribution metadata used for the local experiments. License expressions may include bundled components; the complete distribution notices remain authoritative.

| Dependency | Installed version | Declared license |
|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.17.1 | BSD-style; see installed distribution notices |
| osqp | 1.1.3 | Apache-2.0 |
| gymnasium | 1.4.0 | MIT |
| stable-baselines3 | 2.9.0 | MIT |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| matplotlib | 3.10.8 | Matplotlib/PSF-style; bundled components have additional notices |
| pytest | 9.1.1 | MIT |

Project URLs from package metadata are preserved in `dependency_licenses.json`. Install dependencies independently rather than copying environment directories into Git or release assets. CARLA installation and map redistribution are outside this release; users must obtain the matching simulator under its own terms.

Algorithm attribution: PPO is implemented by Stable-Baselines3; QP solving uses OSQP; the environment API uses Gymnasium; tensor learning uses PyTorch. The project implements its own forward bicycle-model Hybrid A* search, trajectory handling and MPC assembly. It does not claim these established algorithms as original inventions.
