# Multi-grid pressure learning

Status: Active. Admission: start, authorized by the user's request to continue
and record the next steps proposed after the accepted inference-budget tranche.
Canonical plan: docs/plans/water-multigrid-learning-v1/plan.md. Owner: this task.

Objective: isolate learned-weight gains from stopping granularity, qualify the
latest weights on larger grids, and test whether multi-grid analytical feedback
reduces inference work at the unchanged physical targets. Record unsuccessful
experiments alongside successful ones. CUDA trains; CPU performs inference.

Branch experiment/water-multigrid-learning-v1 starts at
5bf5c2d2f8bcef1c4f3d73b8b9e0090e92faafa4, targeting the retained
experiment/water-inference-budget-v1 lineage. Task owner integrates serially.
Retain this local branch for experiment review and continuation, synchronize by
new commits only, retire after accepted integration or explicit retirement.
No rewrite, merge, remote publication, or Rheon/Tuldok source changes.

Binding decisions:
- Preserve historical configurations, captures, checkpoints and result files.
  Reuse the existing Tuldok catalog, managed CUDA/CPU stages and native oracles.
- Preserve one-channel linear 370,016-parameter network, float32 learned values,
  cubic voxels, fixed water properties, previous pressure and native warm-up.
  No native refinement counts as learned success. Existing physical gates and
  1e-6 learned equation target remain unchanged; native target remains 1e-8.
- New analytical-feedback schema explicitly declares grid counts, scalar cubic
  spacing, uniform grid sampling and bounded resident cache. Labels come from
  independent Neumann eigenvalues. These are operator examples, not new CFD
  simulations. Train-only normalization stays fitted to the real fitting frame.
- Start with an exact single-grid control and a controlled 5,000-update multi-grid
  one-evaluation fit. A two-evaluation trial is conditional on the first result;
  record exposure and neural work separately. One optimizer at a time, 600 s and
  32 MiB outputs per managed Run, 2 GiB new local retention plus 1 GiB reserve.
- CPU correctness jobs may overlap GPU training on distinct cores with one
  thread per worker. Isolate repeated CPU timing after training/diagnostics drain.
  Do not assume using more threads minimizes latency.
- Broader 16-simulation training follows credible known-case fitting at a named
  inference budget and retained physical acceptance on prior development/grids.
  Per-grid work regressions remain explicit and do not authorize default promotion. Otherwise retain
  the failed scientific hypothesis and prepare a concrete follow-up. Larger-grid
  capture contract, architecture replacement and deployment remain separate.

Composed-design review: not-applicable. Existing managed execution, source
capture, neural implementation and native process/oracle owners remain in use;
this tranche adds a bounded training population and measurement policy.

Write set: experiments/water-conditioning-v1/training/{policy,model,analytical,
train,test_analytical,test_multigrid}.py; new training configuration files;
experiments/water-stability-learning-v1/{learned_iteration,test_learned_iteration}.py;
experiments/water-residual-learning-v1/rollout.py;
experiments/water-conditioning-v1/native_parallel.py;
new experiments/water-multigrid-learning-v1; this plan directory; README.md and
tuldok-project.json links. Additional exact write sets require a recorded reason.
New material lives under ignored runs/water-multigrid-learning-v1.

Milestones and acceptance claims:
1. C1 automated/artifact: versioned multi-grid contract rejects invalid grids and
   budgets, preserves legacy bank behavior, independent eigenvalue/operator tests
   cover admitted shapes and units. Single-evaluation inference preserves four
   uninterrupted evaluations and counts actual work. State: Accepted.
2. C2 required-real CUDA+Tuldok: smoke then full fit complete/drain, CPU restore
   preserves weights, managed artifacts reopen/redownload. Failed attempts are
   retained and do not masquerade as completed fits. State: Active.
3. C3 required-real CPU: matched stopping checkpoint comparison, development
   native gates and larger-grid trajectories retain accuracy/work counts. Native
   controls must qualify before interpreting candidate failures. State: Planned.
4. C4 performance/required-real CPU: repeated isolated timings and bounded profile
   separate convolution, operator work and whole-track overhead. State: Planned.
5. C5 artifact/system: evidence, methods, limits, custody and next decisions are
   retained and linked in the user's existing Tuldok project. State: Planned.

Current acceptance: pending. Current phase: 16-simulation CUDA fit and final CPU qualification.
Exactly one next slice: complete the admitted cohort fit/CPU qualification, then run serial matched-resource timings.
Blockers: none. Replan on nonfinite training, Run lifecycle failure, invalid
native controls, exceeded budgets, or a regression that blocks broader training.
No untouched final population is consumed for development decisions.

Standards snapshot: snapshot:v1:0cae5bc9-2ef6-4312-ae7e-8b8f07187eb9.
See [ledger](execution-ledger.md), [issues](issues.md), and reports/.
