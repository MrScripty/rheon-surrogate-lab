# Seven-channel pressure model

Owner authorized removing the voxel-size X/Y/Z network channels. Root agent owns
this serial slice on private feat/variable-cubic-pressure-model, retained locally
for review; no push, merge or unrelated worktree cleanup. Preserve v1 and all
historical sources, weights, reservations and results.

Write set: new experiments/variable-pressure-v2, matching current viewer/export
profiles and documentation. Reuse unchanged cubic captures, prepared Rheon, native
probe, Tuldok catalog and captured CPU runtime. V2 is a deliberately separate
experiment snapshot, not a modification of the accepted ten-channel experiment.

Contract: seven actual input channels residual, previous pressure, X/Y/Z physical
coordinates, rho and dt. No voxel-size channel, hidden broadcast conditioning or
ten-channel intermediate. One positive scalar h per sample supplies physical
coordinates and is expanded to three equal scalar spacings only at the existing
physical-operator boundary. Preserve arbitrary positive shapes and cubic cells.

Training: reuse the exact qualified development fields and splits, explicitly
rebinding the packet to v2 without regenerating or relabelling physical results.
Fresh 200-update CPU training plus exact zero-update restore in the existing
Tuldok catalog, followed by the unchanged native gates on first/final snapshots.
No GPU, autonomous rollout or accuracy guarantee. Publish progress/diagnostic
frames through the existing Tuldok project and Rheon observer.

Acceptance: prove first convolution takes seven channels, input preprocessing
never builds spacing maps, operator scalar h correctness, irregular/singleton
shapes, zero-residual/gauge invariants and heterogeneous training. Qualify
material before optimizer admission; use clean committed source for recipe
freezing. Verify managed Run custody, artifact/prediction parity and actual native
results, then export real seven-channel inputs and exercise the browser.
Apply previously adopted Coding Standards 188beda implementation, frontend,
verification and commit workflows. Keep model and viewer changes in separate
reviewable commits.

Status: complete. Ten model/training checks, six native tests, Clippy and nine
real-browser checks pass. Existing 144-frame cubic captures were reverified;
all physical array hashes and surviving input channels match v1 exactly. Fresh
managed training completed 200 updates; restore completed zero updates with
exact recomputed predictions and 16 shared byte-identical artifacts. Both Runs
are completed/drained in the existing catalog. Raw pressure remains 0/12 under
the unchanged gates; conventional fallback qualifies all 12 cases. The actual
seven-channel tensor is open in the viewer and current sources are registered
as a Tuldok project. Completion.md and acceptance.json record measured evidence.
