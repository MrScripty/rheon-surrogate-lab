# Eight-cohort pressure-method campaign

Completed results: [summary](results.md), [all measurements and hashes](results.json).
All eight bounded cohorts and selected follow-ups are complete; no model was promoted.

The user's attachment is the scope authority, captured by hash in
[campaign.json](campaign.json). The objective is to execute and test all eight
cohorts, make follow-up tests for promising results, and record the results.
Completion of one diagnostic does not close this campaign.

| Cohort | Required experiment | Completion evidence |
| --- | --- | --- |
| 1 | Add 32³ and 64³ domains, retaining small grids and 8/16 cells per metre | Qualified native captures, mixed-size training/restore, per-size accuracy/time and small-grid regressions |
| 2 | Numerical correction before/after frozen C53/C44/C37 | Same-input measurements for all declared policies, numerical qualification, retention diagnostics, promising-policy follow-up |
| 3 | Two/four teacher substeps per fixed model interval | Native timestep refinement at matched physical times, correctly aligned pressure labels, trained comparison and evaluation |
| 4 | Numerical coarse solve and trained neural fine correction | Consistent coarse operator/transfers, completed fit/restore, fine-equation and timing comparison |
| 5 | Sparse numerical anchors and neural infill | Deployable anchor computation, completed fit/restore, full anchor+infill cost and accuracy |
| 6 | Revised fully learned coarse/fine architecture | Explicit difference from prior learned coarse branch, matched control fit, qualification and timing |
| 7 | Broader spacing training, then explicit spacing conditioning | Counts/spacing separated, matched trained ablation, scale consistency and per-spacing evaluation |
| 8 | Periodic numerical correction | Matched independently evolved trajectories, numerical intervals, first tolerance crossings and total cost |

The preceding [design and evidence](../pressure-hybrid-options-v1/plan.md) inform
implementation. The user has now authorized proceeding with every cohort, so
earlier conditional prioritization is not a reason to omit one. Negative results
remain valid outcomes; qualification thresholds are preserved. Reserved scenes
are not tuning data. Production-size 256³–1024³ qualification remains deferred.

## First executing slice

Cohort 2 freezes the three original checkpoints. It evaluates 30 neural/hybrid
policies and two numerical-only controls on the exact 128 consumed requests,
240 existing manufactured cases, and 32 mixed-mode cases specified before
inspection. Single-thread CPU workers run serially with bounded lifetimes and a
12 GiB host reserve. No optimizer or simulation steps occur in this slice.

Timing covers prepared physical tensors through final residual, including all
numerical transforms and neural recurrence. It excludes loading, oracle metrics,
IPC and simulation. Original chart timings are not directly comparable. Native
full-policy and trajectory follow-up remains required for a promising result.
Accuracy is recorded separately for the three populations and intermediate
corrections; no raw Pareto improvement alone qualifies deployment.

For a fixed budget, a promising policy must improve the retained neural control's
worst residual without increasing median policy time, or reduce time while
preserving its worst residual and success count. An all-128 `1e-6` success with
less work also triggers follow-up. Apply the same comparison within every
population, report conflicts, and keep candidates with a tradeoff as exploratory.
Do not select ordering or rank using unused scenes. Follow-ups include repeated
isolated timing and independent operator/trajectory checks before any promotion.

Status and per-worker process receipts are in [cohort-02-execution.json](cohort-02-execution.json).
Full numeric arrays and per-case measurements remain in `runs/pressure-methods-v1`.
Final completion requires the full table above, including drift/cost evidence;
missing training or rollout evidence must stay explicitly incomplete.
