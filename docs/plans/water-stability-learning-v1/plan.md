# Stability-aware pressure learning

Status: Active. Admission: start, authorized by the user's “proceed” following
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
new reusable configurations; experiments/water-stability-learning-v1;
docs/plans/water-stability-learning-v1; README.md and tuldok-project.json links.
Ignored runs/water-stability-learning-v1 owns new captures and local results.

Milestones:
1. Implement and qualify analytical auxiliary feedback. Exact Neumann mode and
   mixed-mode labels; signed amplitudes; non-amplification plus nonzero accuracy
   loss; deterministic train-only generation. Regression checks must prove the
   operator against analytical eigenvalues and reject invalid configurations.
   State: Accepted (focused implementation checks; managed hardware smoke pending).
2. Matched bounded 5,000-update mode-feedback comparison against the retained
   four-pass recurrent control, then a separately configured correction-RMS
   output-scale comparison. Source/config/input/initial-state lineage, CUDA
   updates, fresh CPU parity, all 128 development states, mode probes and three
   independent 64-step simulations retained. No assumption of improvement.
   State: Planned.
3. Admit full-trajectory feedback and the prepared 16-parent cohort only when
   known-case fitting and stability are credible. Replan with measured evidence;
   otherwise retain the next experiment and explicitly defer expansion.
   State: Conditional.

Acceptance claims (all automated): focused analytical/configuration checks
(deterministic environment); real managed CUDA train and fresh optimizer-free
CPU restore (required-real GPU/Tuldok); native teacher-state and three independent
trajectory comparisons after 15 native warm-up steps (required-real native CPU);
durable method/results/checkpoint custody and project evidence (system).
Current acceptance: pending. Scientific completion does not imply model
qualification. Numerical gates stay 1e-6 equation residual plus existing physical
limits. A sampled mode pass is not a convergence guarantee. Final candidate is
fixed by terminal update, not selected from validation. Existing evaluated cases
are development evidence, not a fresh final test population.

Resource limits: one managed GPU trainer, 600 seconds and 32 MiB output per Run;
2 GiB new local retention plus 1 GiB reserve. CPU independent diagnostics use
available distinct cores, one Torch thread each, following measured throughput.
Analytical generation/cache is bounded and captured with configuration/source.

Current phase: qualify real managed CUDA/CPU execution.
Exactly one next slice: run two-update managed CUDA/CPU smoke, then the fixed
5,000-update mode-feedback comparison. Source is committed before capture. Replan if numerical or lifecycle evidence
invalidates the comparison. Standards MCP snapshot
snapshot:v1:1a9db05b-79b2-439d-b423-d316efeeb5a0.

See [ledger](execution-ledger.md), [issues](issues.md), and [results](results.md).
