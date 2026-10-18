# Pressure loss-component diagnosis

Status: Completed. Canonical operation: the admitted diagnostic sequence is closed.
Jeremy's follow-up to continue improving the model authorizes the recorded
CPU-only next proposal from pressure-shape-decay-v1/next-steps.md. That completed
study remains immutable. C53 remains deployed; C68 remains skipped.

Objective: locate local gradient disagreement in the captured four-pass loss,
separating canonical manufactured-pressure loss, final-pass equation feedback,
and weighted early-pass equation feedback at C53, C66 and C67. No optimizer,
new simulations, training admission, model changes or promotion. This bounded
study chooses the next proposal; a new training objective is not assumed useful.

Binding design: same eight X-fundamental cases, train-only scalar amplitude,
cubic spacing, float32 network and float64 physical calculation as C66/C67.
The captured numerical sources and normalization must match. Reconstruct the
three terms from their captured definition; independently call the captured
full objective and verify scalar and joint-gradient equivalence. Retain the
separate-gradient additivity discrepancy and original precision check result.
Reproduce retained C53/C66 full-gradient measurements. Report zero-norm cosine as undefined rather
than manufacturing a direction. Interpret geometry as local Euclidean gradients,
not causal evidence about Adam or the complete training trajectory.

Resource contract: three independent CPU workers, one Torch/BLAS thread each,
affinity 21/22/23, host reserve 12 GiB plus 2 GiB per worker, no CUDA initialization.
Stored fitting tensors <=8 MiB; live vector workspace <=16 MiB. Stream gradient
vectors through ignored local files (<=64 MiB per checkpoint), loading at most
two vectors for geometry; never commit raw vectors. Each child has a 120-second
cap and process-group cancellation/TERM/KILL/reap owned by its parent. Leave user
applications running. Inspect the retained checkpoint inventory; Adam-direction
analysis is unavailable if exact moments are absent.

One implementation/acceptance slice: implement and run the three diagnostics,
verify evidence, and publish an interactive offline report linked from cohort
progress. Exact write set: experiments/pressure-loss-components-v1/, matching
docs directory, and only four_pass_link plus its owned HTML link region in
four_pass_study.py and cohort-progress.html. Runtime outputs stay in runs under
the same study name. Preserve all unrelated dirty files and historical results.

Acceptance: (1) captured-source/normalization/case identity, train-only amplitude,
finite CPU-only values and unchanged inventories; (2) independent captured full
loss matches term sum (relative tolerance 1e-12), and the gradient of the
reconstructed joint sum matches the captured full gradient exactly; retain the
original separate-gradient relative-L2 5e-6 check as a precision limitation when
it fails; (3) analytic literal vector/chain-rule
oracle for geometry/additivity; (4) retained C53/C66 full-gradient replay;
(5) completed/reaped workers and recorded elapsed time, memory and runtime;
(6) generated report with retained-scalar browser checks and narrow-screen layout.
All satisfied under the recorded numerical-oracle replan; 14 separate-gradient
checks remain explicitly limited, not passed. Evidence: reports/verification.json,
reports/browser.json, reports/math-check.json and reports/joint-guard.json.
The joint-equivalence check proves agreement with the captured objective,
not physical solver conformance. No claim of model accuracy improvement is made.

Composed-design applicability: no new production architecture. Existing verified
checkpoint/inference, physical operator and fitting-bank owners remain unchanged.
This study owns only diagnostic decomposition, streamed geometry and its report.
Replan if captured identity/joint-equivalence/replay fails, finite results cannot be
obtained, bounds cannot be met, or diagnosis requires model or training changes.

Branch experiment/pressure-loss-components-20261018 is a private review proposal,
based on bd80c7d8fd10b4fba9523d22cdf509d07e88d6be. Jeremy owns integration into the
preceding experiment work. No automatic push, merge, synchronization or rewrite.
Retain for Jeremy's review; retire after integration or explicit disposition.
No worktree is created. Coding-standards snapshot:
snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711.

Current phase: complete. No next execution slice is admitted under this plan.
The bounded balanced-shape training proposal is in next-steps.md. Final measured
results and limitations: index.html and results.json. Commands: commands.md.
Ledger: execution-ledger.md; issues: issues.md; evidence: reports/.
