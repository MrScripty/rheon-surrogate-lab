# Variable-shape pressure model with cubic physical cells

Status: active. Acceptance: pending.
Owner authorized model and training changes after the accepted cubic-cell capture
setup. Root agent owns this serial implementation; no delegated agents.

Private branch feat/variable-cubic-pressure-model from 2b740bc, retained locally
for review and continued integration. Preserve every historical model, capture,
reservation and result. No push, merge, GPU change or external dirty-tree cleanup.
Use the previously adopted Coding Standards 188beda and codebase-design skill.

Write set: new experiments/variable-pressure-v1 and probes/variable-pressure-v1;
this plan and its acceptance evidence; current repo project/README pointers.
Reuse the unchanged cubic-cell producer, qualified Rheon dependency, independent
MAC oracle and Tuldok generic runtime/recipe/Run/artifact owners. New model and
training source are separate from earlier UI and physical-grid commits.

Model: preserve non-affine GroupNorm blocks and parameter inventory; make pooling
handle odd dimensions and dimensions below 16, preserving actual output shape.
No physical field padding, cell stretching, object-array/pickle packets or shape
coercion. Inputs and physical operator/loss reject unequal per-axis spacing;
coordinate features use actual cell centres in metres. Shape capability is bounded
by explicit CPU memory/cell budgets, not a promise of accurate pressure at every
possible resolution.

Training: qualified whole-parent histories at multiple densities and shapes;
train-only normalization and heterogeneous per-frame gradient accumulation, with
no validation participation. Fresh managed CPU training and zero-update restore
in the user's existing Tuldok catalog, exact numeric prediction/weight parity,
and actual native pressure diagnostics on every selected first/final snapshot.
Publish capture, training and diagnostic activity to the real project/egui viewer.

Required evidence: forward/backward on cubic, rectangular, odd, singleton axes
and unseen shapes; same-shaped per-cell outputs; spacing rejection before neural
inference and optimizer; independent operator values/units and zero-residual
bypass; safe heterogeneous packet lineage; real one-frame fixtures before capture;
full qualified reference captures; finite managed training, exact restore and
native accuracy/fallback/iteration results with honest teacher-state scope.
Run affected tests, Rust formatting/Clippy, review exact staged scope and commit.

Implementation qualification: nine focused model/training checks pass, including
exact original 16³ feature-forward/initialization, odd and singleton dimensions,
a 257-cell axis, mixed-shape mean-gradient updates and spacing rejection before
numeric loading/inference. Six native checks, Rustfmt and Clippy pass. Source
commit will precede material freezing; scientific Run acceptance remains pending.

Runtime prefix runs/variable-pressure-runtime-1 failed private-directory checks
before publication; the idle user's Tuldok server was restored. Correct the
caller umask to 0077 and retain the failed prefix. No optimizer/Run was admitted.
