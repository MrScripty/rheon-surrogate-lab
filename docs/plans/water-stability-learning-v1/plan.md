# Stability-aware pressure learning

Status: Verifying (experiment evidence; no one-pass or deployment qualification).
Admission: start, authorized by the user's “proceed” following
the retained residual-learning next sequence. Previous tranche remains accepted
as experiment evidence, with no qualified learned solver.

Objective: determine whether analytical mode feedback prevents measured error
amplification while improving pressure accuracy. CUDA float32 training, fresh CPU
restore/inference, cubic cells and unchanged native physical gates remain binding.

Canonical plan: docs/plans/water-stability-learning-v1/plan.md. Owner/integrator:
this task. Branch experiment/water-stability-learning-v1 descends from
95122213b5d9e52c5cffb1cd46b65af4ec25616e. It is retained for experimental
continuation/review, synchronized by new reviewed commits; supported consumers
are Run reconstruction and later qualified integration. Retire after accepted
integration or explicit retirement. No worktree, rewrite, merge or publication.

Binding design: extend the existing model/trainer owner with a closed v3
configuration. Analytical samples are training-only compatible sealed Neumann
pressure fields with exact eigenvalue labels, not additional CFD trajectories.
Keep recorded real-frame training exposure and initialization fixed; record
additional analytical exposure and compute separately. Keep all historical
configuration semantics, captures and checkpoints. Model scale selection is a
separate controlled intervention. Tuldok retains normal Run/artifact custody;
live activity remains visible in the existing project. No Rheon/Tuldok edits.
Composition review: not applicable; existing training, custody and native
evaluation boundaries remain unchanged.

Write set: experiments/water-conditioning-v1/training/{policy,model,train,
project,physical_residual}.py; new analytical training owner and focused test;
experiments/water-residual-learning-v1/{record,rollout}.py and
experiments/water-conditioning-v1/native_parallel.py for explicitly labelled
CPU repeated-correction diagnostics;
new reusable configurations; experiments/water-stability-learning-v1;
docs/plans/water-stability-learning-v1; README.md and tuldok-project.json links.
Ignored runs/water-stability-learning-v1 owns new captures and local results.

Milestones:
1. Implement and qualify analytical auxiliary feedback. Exact Neumann mode and
   mixed-mode labels; signed amplitudes; non-amplification plus nonzero accuracy
   loss; deterministic train-only generation. Regression checks must prove the
   operator against analytical eigenvalues and reject invalid configurations.
   State: Accepted (focused checks and real managed CUDA/fresh CPU smoke).
2. Matched bounded 5,000-update mode-feedback comparison against the retained
   four-pass recurrent control, then a separately configured correction-RMS
   output-scale comparison. Combine the two isolated changes in the preconfigured
   factorial comparison when their fitting/development evidence supports it.
   Source/config/input/initial-state lineage, CUDA
   updates, fresh CPU parity, all 128 development states, mode probes and three
   independent 64-step simulations retained. No assumption of improvement.
   State: Accepted (all three full fits and CPU/native comparisons retained).
3. Test fixed eight-proposal CPU inference after measured convergence: 32 neural
   passes, unchanged checkpoint/physical arithmetic, no native refinement. Qualify
   the fitted state, 128 development states and three full independent trajectories.
   Keep single-proposal qualification separate. State: Accepted (128/128
   development states and 854/854 AI trajectory steps with 32 neural evaluations).
4. Admit full-trajectory feedback and the prepared 16-parent cohort only when
   known-case fitting and stability are credible. Replan with measured evidence;
   otherwise retain the next experiment and explicitly defer expansion.
   State: Deferred. Iterative-policy success does not establish one-pass
   accuracy. Single-pass follow-up configurations are prepared, not trained;
   broader data and deployment need an explicit inference-budget decision.

Acceptance claims (all automated): focused analytical/configuration checks
(deterministic environment); real managed CUDA train and fresh optimizer-free
CPU restore (required-real GPU/Tuldok); native teacher-state and three independent
trajectory comparisons after 15 native warm-up steps (required-real native CPU);
durable method/results/checkpoint custody and project evidence (system).
Current acceptance: partial; final project comparison links are pending.
C1 focused/deterministic/automated: satisfied (18 model checks, two residual-feedback
checks, two process checks, two real-source rejection checks, report CLI check).
C2 system/required-real GPU and Tuldok/automated: satisfied (11 completed/drained
stages, 15,002 CUDA updates, fresh CPU parity and whole redownloads).
C3 system/required-real native CPU/automated: satisfied for the scoped comparison
and separately qualified iterative policy; see results.md and per-trial reports.
C4 system/required-real Tuldok/automated: verifying final comparison/evidence index.
Scientific completion does not imply model
qualification. Numerical gates stay 1e-6 equation residual plus existing physical
limits. A sampled mode pass is not a convergence guarantee. Final candidate is
fixed by terminal update, not selected from validation. Existing evaluated cases
are development evidence, not a fresh final test population.

Resource limits: one managed GPU trainer, 600 seconds and 32 MiB output per Run;
2 GiB new local retention plus 1 GiB reserve. CPU independent diagnostics use
available distinct cores, one Torch thread each, following measured throughput.
Analytical generation/cache is bounded and captured with configuration/source.

Current phase: publish and verify the final comparison/evidence index.
Exactly one next slice: register final project links, retain its receipt, resolve
all task-owned controllers, and seal the completed experiment record.
 Standards MCP snapshot
snapshot:v1:1a9db05b-79b2-439d-b423-d316efeeb5a0.

See [ledger](execution-ledger.md), [issues](issues.md), and [results](results.md).
