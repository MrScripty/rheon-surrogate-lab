# Fresh Rheon / AI pressure comparison

Status: Accepted for the native diagnostic comparison and context correction. Owner: root, serial; no delegation. This follows the completed
sealed-water baseline, whose trained checkpoint and all scientific results remain
frozen. User requests fresh evolving side-by-side simulations, an optional
AI-minus-algorithm voxel color view, and a pressure-style legend.

Write set: new live-water-comparison-v1 CPU deployment bridge/export material and
this plan/evidence in the lab; dedicated desktop comparison modules, menu/CLI
composition and focused tests/docs in isolated Rheon worktree
runs/live-ai-comparison/Rheon, branch feat/live-ai-pressure-comparison from latest
fetched origin/main (43ed26b). Lab branch feat/sealed-water-pressure-training.
Original dirty Rheon/Tuldok checkouts and dirty prior serving viewer are excluded.
No push/merge. Generic Tuldok/model training code is outside scope.

Accepted interpretation (revised for the user's context correction): tracks share
15 qualified native warm-up steps before the first AI call at step 16. The minimum
is derived from the frozen training reservation's earliest retained zero-based
snapshot index, 15. Training snapshots were steps 16 and 64, each with preceding
native pressure; cold-start inference was outside that measured distribution.
Unsupported cold-start comparisons are testing-setup failures, not evidence of
model accuracy. Preserve those raw records with an audit disposition. The earlier
128 held-out snapshot checks already supplied preceding native pressure and remain
valid accuracy evidence. After the common native prefix the reference and AI evolve
their own velocity/tracer/base-pressure histories. CPU checkpoint is the predeclared 2000-update water candidate.
Native pressure, finite/gauge, full original equation, corrected divergence,
Courant and wall/flux diagnostics remain measured. Native track requires its
original gates. AI track is explicitly a diagnostic preview when pressure fails;
its states are never stamped as qualified simulation snapshots. Finite/gauge,
resource/lifecycle, net wall flux and Courant failure stop the paired run. No silent
fallback or residual clipping. This choice supports the user's requested visible
raw deviation while preserving the qualification contract.

Composition review: applicable. Core Rheon owns advection, viscosity, operator,
projection and reference solve. A desktop diagnostic runner owns two independent
scratch states and computes per-step verdicts; it is separate from accepted core
Simulation sessions. A bounded CPU subprocess adapter owns checkpoint handoff and
inference protocol; one joinable worker owns run state and subprocess teardown.
UI owns immutable frames, synchronized camera/time/slice, display-only differences
and legends. No provider-plus-viscosity core API change is needed: this dedicated
lab composes the existing native stages and assesses every diagnostic result.

Deploy only measured existing captured source, weights, train-only normalization
and typed Run provenance, from the normal Tuldok-owned frozen artifacts. CPU bridge
verifies source/hash/material/runtime/device and restores without an optimizer.
No new training, normalization, checkpoint choice, or held-out tuning.

UI contract: new comparison entry in the Rheon desktop; editable scene seed,
per-axis integer counts, scalar cells/m and force controls; Start/Cancel, retained
paired timeline and playback. Same geometry/time and camera in both panes.
AI pane offers pressure or signed pressure difference p_AI - p_reference in Pa.
Diverging colors center exactly at zero; labeled scale/units/zero, full-volume
range shared across slices, current-frame range by default, optional manual or growing whole-run range;
all-zero fields defined. Show 3D voxel colors with a synchronized cut plane and
numeric voxel probe. Pressure uses a shared range across both tracks. Cubic cells
stay cubic; all grids satisfy declared cell/history budgets. Record both track
verdicts and checkpoint/scene provenance. Activity and fresh fields are observable
through the existing Tuldok project; retain/export concrete diagnostic material.

Resource/lifecycle: grids bounded to 65,536 cells and 96 MiB retained numeric
preview history; timestep 1/60 s, bounded steps. Reuse installed Pumas CPU runtime;
no runtime download. Model child has bounded JSON records, startup/reply deadline,
capped stderr, cancellation kill/drain/join and no secret environment handoff.
Worker cancellation/Drop joins and retains existing immutable frames. At most one
active paired run. No new optimizer or private catalog.

Verification: exact CPU checkpoint parity on existing retained witness, zero
residual/gauge/shape rejection and malformed protocol; native independent-state
and identical initial forcing tests, reference gates and deliberate bad-AI verdict;
signed error/zero-centered legend/cubic geometry/full-volume range/picking tests;
actual desktop fresh runs and cancel/restart, matching paired time/camera and
visible legend/error cells. Native build/tests/fmt/clippy, focused Python tests,
actual UI screenshot/interaction and normal catalog evidence. Honesty about raw
unqualified AI dynamics is part of acceptance.

Accepted slice: qualified the native warm-up handoff and published the fresh
comparison and preserved cold-start disposition. CPU parity and cold request
rejection pass. A 24-step native run confirms the first AI call at step 16 with
solved step-15 pressure; 0/9 AI predictions meet the unchanged gates. The completed
64-step GUI run has 15 native warm-up steps and 0/49 qualified AI predictions. This
is observed learning error with context, not a cold-start result. The shared-prefix
steps never count as model passes. Whole-run color scaling hid later deviations
behind early transients, so current-frame full-volume scaling is now the default.
Blockers: none. Standards selected from adopted Core/Router: implementation,
code-design/contracts, concurrency/lifecycle, interop, verification, Rust/frontend
and commit. Keep source commits for application/runtime integration separate from
model/scientific experiment changes.

Completion evidence: reports/acceptance.json and results.md. The real measured CPU
checkpoint also passed cancellation after two AI predictions, terminal child/reader
drain and immediate restart into a distinct retained run. Strict Clippy includes
test targets. Tuldok collection record 899f4cc68ea64a91a4abc4ad9ac31b7e reopened
byte-exact as a whole bundle and numerically exact for the first AI error slice.
Exactly one next slice: none required for this accepted diagnostic feature; future
model accuracy/training changes need their own experiment and CUDA budget.

Commit authority: user explicitly requested local commits in separate repositories
and Coding Standards MCP commit guidance. Read canonical core, workflow.commit and
workflow.implementation through the Standards Engine, commit-guidance snapshot
snapshot:v1:353b5714-c22b-49cc-a983-ff7a0b45a266. Review exact staged scope, focused
evidence and sensitive-file risk; use conventional atomic commits; preserve history.
Rheon feature commit 6c38078 already exists on its isolated branch. The separate
real-checkpoint lifecycle regression and docs are committed as 66e5038 on the same Rheon feature branch. Lab deployment/export source is d6a9292;
measured results and project pointers form the final documentation slice.
No rewrite, push or merge. Original dirty checkouts remain outside these commits.

Final staged review bound native-prefix selection to the exact split-artifact hash
in the frozen training receipt. Regression proves derivation with a different
snapshot schedule and rejects modified reservation bytes. A fresh export produces
the already-qualified deployment descriptor byte for byte. Four focused Python
boundary/deployment tests pass. No model/runtime or acceptance-policy change.
