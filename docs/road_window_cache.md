# Exact road-window cache: development measurement

The opt-in `batch_sat_v1` collision backend now caches occupied-cell rectangles,
centers and radii by exact integer raster bounds. Each grid retains at most
256 windows and 65,536 cell rows (about 4 MiB of cached float64 arrays, plus
index and object overhead). Public raster edits or cell-size changes invalidate
the cache. Pose-dependent collision decisions, full vehicle geometry, contact
tolerance and scalar fallback remain unchanged. The default backend remains
the scalar implementation.

The retained measurement in
`reports/road_window_cache_development_20261104.json` covers all eight saved
roads from the completed seed11 zero-PPO known24 development evaluation.
Up to 24 uniformly selected measured starts per road give 186 starts and
37,200 model-generated steering-transition poses. Every boolean mask matches
both the previous batched kernel and scalar oracle. Five alternating-order
trials yield median old/new ratios of 1.106 cold and 1.214 warm; constructor
costs are recorded separately. These are exploratory kernel timings on a shared
workstation, not whole-planner speedups or real-time control evidence.

Two additional offline Hybrid A* searches on actual saved static/curve road
fixtures preserve status, expanded nodes, cost and every path bit. They do not
exercise the complete native refinement/execution pipeline. Nine collision
regressions and 35 related road/planning/trajectory regressions pass locally.

Reproduce on the corresponding complete local evidence batch and its frozen
legacy source, writing to a new output file:

```powershell
.venv/Scripts/python.exe scripts/benchmark_road_windows.py --batch <completed-native-batch> --legacy-source <its-frozen-source> --output <new-report.json>
```

The CLI verifies the completed trace and legacy collision source hashes before
using them. Original measurements and their script hash remain retained; the
current script corrects only the report wording from "fixed24" to "up to24".
No timing trial was rerun for selection.

The active H/H0 cohort and waiting B1 cohort retain their previously frozen
sources. This change adds no native controls or optimizer updates and supplies
no new evidence of PPO, feedback or final research acceptance.
