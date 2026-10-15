# Fresh comparison and previous-pressure context

The live comparison now performs **15 qualified native warm-up steps** before
calling the frozen CPU AI model. Its first prediction is physical step 16, with
step-15 solved pressure and the newly computed equation residual as input. During
the prefix both panes show the same native state and **AI not evaluated**. After
handoff each track evolves its own velocity, tracer and pressure history.

The original cold-start comparisons attempted AI inference at step 1. Training
retained native zero-based snapshots 15 and 63 (physical steps 16 and 64), each
with preceding reference pressure. Cold-start behavior was not qualified. The two
old runs remain intact and are classified **unsupported cold-start testing setup**;
their 0/8 and 0/64 checks are excluded from supported model-accuracy evidence.
This is a correction to the test protocol, not a change to historical numeric data.

A warm-up is necessary for this comparison's declared checkpoint context, but it
does not establish that the model solves pressure. The network is not recurrent:
it consumes the current residual and previous-pressure fields, not a hidden history
of all preceding frames. A zero previous-pressure field is mathematically meaningful,
but this checkpoint was not trained or qualified for that cold-start distribution.

| Fresh run | Native prefix | AI checks | AI passes |
| --- | ---: | ---: | ---: |
| Bounded native verification, 24 steps | 15 | 9 | 0 |
| Desktop evolving comparison, 64 steps | 15 | 49 | 0 |

At the first supported AI step, relative equation residual is **0.0372545** against
the unchanged **1e-6** target. Maximum corrected divergence is **0.00141691 s⁻¹**;
maximum pressure difference from the native solution is **2.21311 Pa**. This first
handoff starts from the same solved state as the reference, before accumulated AI
trajectory drift. All native reference steps pass. The model still has an accuracy
gap after supplying preceding solved pressure. No weights, normalization, physical
gates or training choices were changed.

The earlier reserved-test **0/128** result already supplied preceding reference
pressure for every snapshot. It remains valid accuracy evidence. The fresh live
force setup is a diagnostic experiment, not a new controlled held-out benchmark.
See [the machine-readable audit](reports/acceptance.json) for individual preserved
runs, provenance hashes and dispositions.

The CPU deployment reproduces the retained CPU prediction byte for byte, preserves
an exact zero-residual bypass, initializes no CUDA context and performs no optimizer
updates. The new bridge rejects an early request with `unsupported_context` before
accessing the network. The native worker independently enforces the required prefix. Context export is
bound to the exact split-artifact hash in the frozen training receipt; changing
that reservation is rejected. The final exporter reproduces the previously
verified deployment descriptor byte for byte.
Regression tests verify that no inference occurs during warm-up and that the first
callback receives exactly the preceding solved pressure.

The native desktop has synchronized 3D panes, frame/time, camera, Z cut, signed
`AI − Rheon` pressure colors and a labeled blue/white/red Pa scale. Exact matched
voxel probes are available. Current-frame full-volume scales keep late errors
visible; whole-run and manual ranges are optional. Failed AI pressure states are
raw diagnostics, never accepted Simulation publications or silent native fallbacks.
Hard Courant/finiteness/gauge/wall-flux failures still stop a run.

The completed 65-frame sequence is also a normal draft in the existing Tuldok
catalog: **Fresh water · native warm-up 15 · AI vs Rheon · 65 frames**, record
`899f4cc68ea64a91a4abc4ad9ac31b7e`. It contains native pressure, AI pressure, signed
pressure difference and both tracers, plus scene/checkpoint/per-step verdicts.
Its content hash was reopened and its step-16 error slice exactly matches the
native retained JSON. This diagnostic draft makes no scientific release claim.
The root project observes fresh activity/latest fields; future interactive runs
can be explicitly exported/imported with the reusable field exporter.

Verification: seven focused native comparison tests, 46 ordinary-desktop tests
(including those seven), four Python boundary/deployment tests, one additional
real-checkpoint cancel/restart test, both desktop entry points under strict Clippy, real CPU protocol parity and two actual warm-start runs.
The existing camera input/worker lifecycle tests pass after extracting one shared
camera. Actual window observation confirms warm-up controls, paired cubic geometry,
legend and retained timeline. The user operated that window; automated mouse input
was deliberately kept out of their active controls. A separate environment test
with the actual checkpoint cancelled after two AI predictions, verified terminal cancellation and retained frames, then completed a
fresh restart in the same project directory; both owned child slots were drained.
No simulated subprocess or model stood in for that test. Existing accepted core
solver/provider behavior and unrelated dirty checkouts were preserved.

Implementation is local on Rheon branch `feat/live-ai-pressure-comparison`, based
on freshly fetched `origin/main` 43ed26b, and lab branch
`feat/sealed-water-pressure-training`. No push or merge. The active Rheon worktree
is retained to serve the binary and Tuldok source pointers; its explicit lifecycle
and synchronization policy are documented in Rheon's `docs/ai-pressure-comparison.md`.
