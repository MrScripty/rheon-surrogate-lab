# Aggregate gradient balance diagnosis

Status: Completed — CPU diagnosis and evidence qualification. Jeremy's “continue with
the next step” admits pressure-balanced-shapes-v1/next-steps.md. C53 remains
deployed. C69/C66 and all historical measurements remain unchanged.

Objective: at C53, C66 and C69, measure which cases influence the shared mean
gradient and whether it opposes the failed 16×7×11 case. Use the same eight
X-fundamental samples, train-only amplitude, normalization, captured four-pass
full objective and canonical/final/weighted-early decomposition. Compare only
uncapped averaging with a per-case L2 cap of 1 before averaging, retaining a final
global cap of 1. No optimizer, model perturbation, CUDA, new CFD, holdout use,
training admission or promotion. Selecting another training policy is a later
proposal, not part of this diagnosis.

Stream each float32 gradient vector to ignored local files; dot products and
aggregate geometry use float64. Keep direct full-gradient, reconstructed joint
gradient and separate-component sum checks distinct. Require exact captured versus
joint gradients and scalar sums within 1e-12. Retain separate-gradient discrepancies
above the existing 5e-6 threshold as precision limitations, not universal passes.
Reuse the unchanged captured numerical/component owners, verify source identities,
operator labels (<1e-10), all sample tensors and fitted-pressure replay, and require
unchanged parameter/buffer inventories with no teacher work.

Report raw norm and its fraction of summed norms separately from signed projection
onto the mean (which may be negative). Report cosine, dot product and local
directional derivative for every case's full, final and early objectives under
each proposed common direction. Canonical and jointly differentiated final-plus-
canonical terms remain available for interpretation. Distinguish first-order
Euclidean loss descent from pressure-error improvement and Adam's actual update.
Float64 diagnostic norm capping is not a bitwise CUDA optimizer replay. Exact
Adam moments are absent from retained archives; report that unavailable.

Literal vector/zero-mean and cap oracles must verify the mean, signed contributions,
cross-component derivatives and one counterexample where per-case capping reverses
a target alignment. Detect a value-preserving derivative mutation at the actual
captured joint-gradient guard. Exercise cancellation of the actual CPU supervisor;
all owned process groups must terminate and be reaped. Run three independent
numerical workers on distinct CPUs with one Torch/BLAS thread each and CUDA hidden.
Require 12 GiB host reserve plus 2 GiB per worker, <=8 MiB fitting tensors,
<=16 MiB live vector workspace, <=64 MiB vector disk per checkpoint, and a
120-second pool deadline. Never stop user applications or read private Run outputs.

Acceptance: correct/no-update execution and source/label/replay evidence, exact
joint guard, explicitly retained precision limitations, literal oracle and
process-reaping evidence, time/peak RSS/runtime records, case-level report and
offline interactive visualization verified against retained scalars. Replan on
identity, joint-gradient, finite-value or resource failure. Do not weaken pressure
targets or substitute invented optimizer history. A negative diagnosis is retained.

Write set: experiments/pressure-aggregate-gradients-v1/ and matching docs directory;
only the report-link function/owned HTML region in four_pass_study.py and
cohort-progress.html. Preserve existing dirty chart/model-import edits. Runtime
vectors and scratch outputs stay under ignored runs/pressure-aggregate-gradients-v1/.
No new product architecture: the study owns diagnostic aggregation/interpretation;
existing captured checkpoint/loss/operator and inference owners stay unchanged.

Private branch experiment/pressure-aggregate-gradients-20261018, based on 04b5c39.
Jeremy owns integration into the previous experiment line. No automatic push,
merge, synchronization or history rewrite; retain for review and retire after
accepted integration or explicit disposition. No worktree created. Coding standards
snapshot: snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711. Acceptance satisfied: literal oracle, captured derivative mutation, actual-pool
cancellation/reaping, 24 exact joint gradients, source/pressure replay and
interactive scalar checks. Retain 14 separate-gradient precision limitations.
No next execution slice remains; next-steps.md proposes optimizer instrumentation. Ledger and
findings: execution-ledger.md and issues.md; evidence in reports/.
