# Residual pressure learning sequence

Status: Accepted (bounded experiments and records; model accuracy unqualified).
Admission operation: start, authorized by the user's instruction to proceed
with the sequence and retain methods and results. Canonical plan:
`docs/plans/water-residual-learning-v1/plan.md`.

Objective: measure whether physically conditioned residual inputs, linear
correction, coarse/repeated correction and trajectory feedback improve the
learned sealed-water pressure solution. Retain unsuccessful runs as evidence.

Scope: versioned laboratory model/configuration, normal Tuldok managed CUDA
training and fresh CPU restore/inference, independent native physical evaluation,
CPU diagnostics, persistent reports and existing Tuldok project observation.
No changes to Rheon, Tuldok's scheduler, historical captures or deployed models.

Constraints: CUDA float32 optimization; CPU deployment; cubic cells with one
scalar spacing; sealed uniform water and existing boundary operator. Keep the
1e-6 equation target, divergence/flux/Courant gates and native oracle unchanged.
One-frame fitting is a capacity diagnostic, not generalization qualification.
Prior validation/test results remain visible; do not reuse evaluated populations
as an untouched final test. CPU jobs use distinct available cores with one Torch
thread per worker, following the retained throughput measurements.

Binding design: extend the existing measured training owner with an explicitly
versioned experimental contract. Preserve old configuration semantics and source
captures. Physical residual preprocessing and reconstruction have one model owner;
recipes capture every executable source and exact configuration before admission.
Training, checkpoints and lifecycle remain owned by the existing Tuldok catalog.
Analytic coarse work is labelled separately from learned inference. Failures do
not count as successful AI or successful deployment.

Composed-design review: not-applicable; existing owned trainer, managed runtime,
reader and native evaluator remain the composition. This is a scientific model
intervention inside those boundaries, not a new independently operated system.

Branch: experiment/water-residual-learning-v1, based on retained conditioning
branch 18159af220005a3eb2e8262ed6d1be5ce6d7d5eb. Owner/integrator: this task.
Target: experimental continuation of experiment/water-conditioning-v1. Retain
for review and subsequent qualified adapter integration; no history rewrite,
merge, worktree creation or branch retirement in this tranche.

Milestones:
1. Physical residual versus retained seven-channel no-normalization fit.
   Write set: training model/policy/preparation/project source, new model source,
   focused tests, configurations, plan and reports. Gate: algebraic/operator and
   zero/shape/scaling checks; committed source; managed CUDA/CPU smoke; bounded
   10,000-update fit and native development evaluation. State: Accepted (experiment evidence; model remains unqualified).
2. Fixed-representation linear versus nonlinear correction. Same seed, loss,
   frame and update schedule; positive/negative/superposition diagnostics and
   actual fitted/held-out equation and physical errors. State: Accepted (experiment evidence; model remains unqualified).
3. Coarse control and shared recurrent correction with recomputed residual.
   Retain convergence/error-scale and full CPU costs. State: Accepted (experiment evidence; model remains unqualified).
4. Pressure-gradient/divergence feedback and independent trajectory evaluation.
   Write set also includes probes/residual-rollout-v1 and the laboratory CPU
   trajectory controller. Reuse the existing Rheon paired engine by compiled
   source inclusion with measured source/build receipts; no application edits.
   Parent owns bounded child stdin/stdout, deadline, termination/reaping and
   failure receipts. Native/native control must qualify after warm-up first.
   Preserve physical oracle; learning loss changes are separate comparisons.
   State: Accepted (experiment evidence; model remains unqualified).
5. Conditional small varied cohort, then larger population and several seeds.
   Admit expansion only after known-case fitting is credible; reserve a new
   untouched final test before candidate selection. State: Deferred: all fitted
   candidates miss the unchanged 1e-6 target and independent simulations fail.
   The 16-simulation/32-state view is prepared; revisit after fitting/stability
   evidence supports admission. Larger populations/seeds were not trained.

Objective acceptance: completed and drained normal managed Runs, exact source,
config/input/checkpoint lineage, fresh optimizer-free CPU restore, unchanged
native physical checks and durable per-trial method/result records visible in
the existing project. Model qualification additionally requires physical gates
on unseen cases; no guaranteed accuracy improvement is assumed.

Current phase: completed bounded experiments and durable review records.
Exactly one next slice: none (terminal tranche). Prepared subsequent experiments
are in next-steps.md; they have not been executed. Acceptance: all 17 managed
stages completed/drained/reopened/redownloaded, 40,002 CUDA updates, fresh CPU
restores,645 native teacher-state checks, 15 actual model trajectories and three
native controls. All learned physical checks fail and all 15 model trajectories
hit unchanged hard stops; the scientific negative results are retained. The
model/deployment accuracy objective is not accepted. No execution blocker remains.
Conditional cohort expansion is deferred under its existing evidence gate.

The retained branch supports continued laboratory experiments and later adapter
qualification, owned by this task/future maintainer. Synchronize by reviewed new
commits without rewriting captured history. Supported consumers are retained Run
reconstruction and experimental continuation; retire after accepted integration
or explicit retirement of that continuation. No task worktree was created.

Retention: committed configurations, methods, summaries and small receipts;
Tuldok owns checkpoints and artifacts, with verified local downloads under
runs/water-residual-learning-v1. No weights/datasets in Git. Initial local
retention budget 2 GiB plus 1 GiB free-space reserve, adequate given 228 GiB free.
Per-run limit remains 600 seconds and 32 MiB output. Use established live
activity and project evidence links; optimizer updates are not simulation time.

Standards: Coding Standards MCP snapshot
snapshot:v1:00b6738f-09b4-4388-9cda-42bf949b1d52. Atomic conventional commits,
explicit source/contracts, independent native evidence, claim-scoped metrics.

See [execution ledger](execution-ledger.md), [issues](issues.md),
[results](results.md), [prepared next experiments](next-steps.md) and [previous prepared sequence](../water-conditioning-v1/next-steps.md).
