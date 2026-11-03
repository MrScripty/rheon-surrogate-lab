# Cubic-cell setup qualification

The current experiment capture uses 16 cells per metre on each axis. Every cell
has side length 0.0625 m. A 1 m cube uses 16³ cells; the 1.5 × 1 × 0.75 m domain
uses 24 × 16 × 12 cells. Domain lengths must be whole cell multiples; invalid
lengths fail instead of stretching or rounding the physical grid.

The new configurable native capture binds the selected configuration and measured
sources to a qualified prepared Rheon dependency. It retains the original solver,
forcing, pressure tolerances and independent finite-volume/MAC physics oracle.
The frozen historical namespaces, models, weights and results remain unchanged.
The capture adapter is a separate version because the previous source bindings
and hard-coded 16³ capture shape must remain reproducible.

Actual qualification: one-frame fixture first, then three complete fresh-seed
development trajectories, 24 frames each. All 72 frames pass independent RHS,
pressure residual, f32 velocity correction, wall, divergence, flux and Courant
checks. Maximum relative residual is 9.9924073e-9, maximum actual divergence is
1.4901161e-7 s⁻¹ and maximum Courant number is 0.0638537. The accepted build is
runs/cubic-pressure-v1-build-4; capture is runs/live-project/cubic-capture-3.
Four Python grid/model checks, four affected observer checks and four native
checks pass; Rust formatting and Clippy with warnings rejected pass.

The unchanged GroupNorm model rejects 24 × 16 × 12 input. Its actual forward
contract and the new preflight both reject it before optimizer creation.
No training, shape padding, resampling, model arithmetic change or GPU work was
performed for this configuration slice. Rectangular-grid training support is
the next separate model concern; these are qualified reference solver captures,
not new learned pressure results.

The current repo project in the user's existing Tuldok catalog is
6ed9bfccb7ec4d85ac5849b6c2dc0673, named Rheon Surrogate Lab · cubic cells.
It shows the new configuration and live completion activity. Draft Data record
77971da0b73d45a29c373a8461133761 contains one complete rectangular trajectory,
24 frames and five fields, with original f32/f64 payloads preserved. The full
Tuldok bundle parser and actual authenticated catalog import both pass; no
review/release or managed training Run was admitted.

The actual egui experiment viewer was inspected at frame 23 with counts
[24,16,12], extents [1.5,1,0.75] m and the verified tracer field. It is a separate
read-only window; the user's simulation and generated recordings remain intact.
The final screenshot and authenticated activity receipt are retained under
runs/live-integration. Full measured evidence is summarized in acceptance.json.

Retained development prefixes: build-1 failed compilation from a usize/u64
geometry constructor mismatch, before producing a binary or scientific capture.
Build-2/cubic-capture-1 qualified the initial implementation. Build-3/capture-2
requalified the final payload-budget and fixture-receipt/publication changes.
Build-4/capture-3 requalified a final EOF formatting correction. Every initial
and frame payload matches capture-2 exactly, including the imported draft
fields. The imported bundle retains its original capture-2 source evidence;
its measured source snapshot is retained under build-3/source-snapshot.
No failed prefix was deleted or relabelled as a successful numerical attempt.

Changes remain local on feat/live-rheon-tuldok-project, following the separate
UI portability commit be6e67c. Branches and ignored source/runtime worktrees are
retained for review and continued user work. Nothing was pushed or merged.
