# Pressure inference budget and scaling

Status: Active. Admission: start, authorized by the user's request to continue
the proposed sequence after comparing model parameter counts and file sizes.
Canonical plan: docs/plans/water-inference-budget-v1/plan.md. Owner: this task.

Objective: measure accuracy versus CPU inference work, compare scaling against
Rheon's native solver, and fit the two prepared single-pass CUDA models. Retain
methods, successful and unsuccessful results, source and checkpoint lineage.

Branch experiment/water-inference-budget-v1 starts at
5cac4c4016256525df5c49bf57d8aedc0a2737ef; target is the retained stability
experiment lineage. Retain for experiment review/continuation, synchronize by
new commits only, retire after accepted integration or explicit retirement.
No worktree, history rewrite, merge or publication. No Rheon/Tuldok source edits.

Binding decisions:
- Reuse the normal existing Tuldok catalog, CUDA trainer, captured sources,
  managed CPU restores, native evaluator and independent trajectory engine.
- Neural parameters/output remain float32; physical arithmetic and gates remain
  unchanged. Previous pressure, native warm-up, cubic spacing and sealed water
  assumptions stay explicit. Never count native refinement as AI success.
- Compare the fixed 32-evaluation reference with fewer bounded proposals and
  residual-based stopping, checking unchanged native physical gates afterwards.
  Inference consumes no pressure labels. Report exact evaluations and CPU time.
- Benchmark 16,24,32,40 cells per axis within the existing 65,536-cell contract;
  larger grids are unsupported here, not silently admitted. Separate fixed
  physical volume from fixed cells/metre. Native controls qualify each scene.
  Repeat timings on the same CPU core without competing CPU diagnostics;
  distinguish solve/callback time from whole-track timing and CPU/GPU contention.
- Train stability-modes-single-pass and stability-combined-single-pass with the
  already prepared 5,000-update configurations, one managed GPU job at a time.
  Retain the combined single-pass non-finite-gradient failure; a separately
  named lr1e4 follow-up lowers the entire learning-rate schedule tenfold while
  preserving initialization, sampling, objective, analytical bank and exposure.
  CPU work can overlap GPU fitting, but not the isolated timing repetitions.
- Broader cohort training and architectural replacement remain conditional on
  fitting/stability evidence. No fresh final test is consumed in this tranche.

Write set: experiments/water-stability-learning-v1/learned_iteration.py and its
focused test; experiments/water-residual-learning-v1/rollout.py;
experiments/water-conditioning-v1/native_parallel.py if adaptive receipts need
an explicit diagnostic contract; new experiments/water-inference-budget-v1;
docs/plans/water-inference-budget-v1; README.md and tuldok-project.json links.
Additional write set: experiments/water-conditioning-v1/configs/
stability-combined-single-pass-lr1e4.json for the bounded numerical follow-up.
No historical reports or checkpoints are overwritten. New captures/results live
under ignored runs/water-inference-budget-v1. Prepared training configurations
are reused without changing their scientific specification.

Composed-design review: not-applicable; existing trainer, inference arithmetic,
native engine, catalog and process lifecycle owners remain authoritative. New
measurement/configuration drivers compose those owners without deployment.

Milestones and acceptance:
1. Parameter/file comparison and immutable capture of both single-pass fits.
   C1 artifact/automated/local CPU: satisfied (reports/model-sizes.json). Exact numeric inventories, checkpoint
   bytes/hashes and source semantics; no optimizer during qualification.
2. Managed CUDA train/CPU restore/development inference for completed fits, plus the drained failed combined Run and its separately
   configured tenfold-lower-learning-rate follow-up.
   C2 system/automated/required-real CUDA+Tuldok: pending. Completed/drained Runs,
   restored bit parity, actual updates and independent native accuracy retained.
3. Fewer-pass and adaptive CPU policy. C3 system/automated/required-real CPU:
   satisfied for scoped development/short trajectories (reports/adaptive-evidence.json). Focused residual/budget tests, independent native gates, actual paired
   trajectories and pass counts; failures remain diagnostics, not acceptance.
4. Controlled scaling and synthesis. C4 performance/automated/required-real CPU:
   pending. Same inputs per comparison, repeated timings, all physical outcomes,
   supported-grid limits and variability retained. C5 system/automated/required-real
   Tuldok: pending. Methods/results linked in the user's existing project UI.

Current acceptance: partial. Current phase: qualify the completed single-pass model and capture the bounded
lower-learning-rate follow-up; preserve numerical failure as evidence.
Exactly one next slice: capture and execute the separately configured lower-rate
fit while finishing the first model CPU checks; then resume serial scaling.
Blockers: none. Replan on invalid native controls, exhausted resource limits,
failed known-case fits or evidence that the supported composition cannot deliver
the comparison. A negative scientific result still completes an experiment.

Budgets: 600 seconds/32 MiB per managed Run; one GPU optimizer; 2 GiB new local
retention plus 1 GiB reserve. CPU throughput work uses distinct cores and one
Torch thread per worker; timing work is isolated. Standards snapshot:
snapshot:v1:05188082-022e-4ed4-8473-6b86e0c441cc.

See [ledger](execution-ledger.md), [issues](issues.md), [results](results.md).
