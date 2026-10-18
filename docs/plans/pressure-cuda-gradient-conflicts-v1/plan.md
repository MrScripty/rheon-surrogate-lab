# CUDA case and pass gradient conflicts

Approved continuation of the partial first-moment damping study. Collect
diagnostic backward passes at saved pre-update states 100, 150 and 200. Keep
C53 deployed. No training trajectory, optimizer policy change or new simulation.

Use the existing published Tuldok replay Project/source/data/normalization and
trace identities; select CUDA using its Pumas-managed Python/Torch manifest.
One local diagnostic process owns CUDA, reports through the existing project
activity, and retains bounded output under runs/. This is a backward-pass
diagnostic, not a new Tuldok training Run or a separate Project/catalog.

All eight existing train-only fundamental cases are included. For each, measure
the captured full objective gradient and the equation-loss gradient after each
of passes 1, 2, 3 and 4. Each term gets a fresh forward graph, freed immediately
after backward. Keep float32 network/derivatives and existing float64 physical
loss calculations. The equation term is log1p(relative residual squared/1e-12),
exactly the current feedback term. Original full objective and early weight .05
are unchanged. No teacher and no holdouts.

Before interpretation, reconstruct the historical complete cycle in its exact
saved sample order with loss/8 accumulation and clipping. Compare raw/clipped
gradients to captured CUDA vectors with <=5e-6 relative L2 tolerance and compare
full loss values with <=1e-10 relative tolerance. Failure is retained and stops
the scientific claim; no CPU derivative fallback. Reconstruct sum of individual
full-gradient contributions and disclose any aggregation gap separately.

Report normalized dots of each case/pass gradient with weak case 7 full and
four-pass equation gradients, the full aggregate and observed optimizer delta.
Retain layerwise dot/norm contributions for weak four-pass versus cases 0, 1
and 4 at passes 1 and 2. A negative gradient dot indicates opposing local
first-order descent signals; it does not prove finite-step harm or long-run
benefit. A positive gradient·actual_delta predicts increasing that loss locally.
Retain per-case losses, residuals, gradient norms and clip measurements.

Bound one CUDA worker, 180s elapsed, 4GiB device allocation cap, 4GiB RSS,
12GiB host reserve, 64MiB host vector workspace and 256MiB vector disk. Retain
scalar receipts and source hashes in Git, vectors in ignored runs/ only.
Test independent vector/layer geometry fixtures; verify actual CUDA gradient
replay, graph sequentiality, unchanged parameters, resource/lifecycle receipts
and report values. Retain startup and active-worker cancellation evidence.

Private branch experiment/pressure-cuda-gradient-conflicts-20261018 starts at
92c5043d7e6134ade707b7c7dd5bb833d388b598; owner is the lab agent for the user,
integration target the prior momentum-damping line. Retain for user review until
integrated or explicitly retired. No worktree, push or history rewrite. Write
set: new experiment/docs directories and exact discovery links. Preserve all
concurrent graph/spectrogram/THIRD_PARTY edits.

Resource revision before accepted measurement: attempt 01 reached 2.03GiB RSS
while initializing CUDA and was stopped/reaped by the 2GiB cap in 3.23s.
No snapshot measurement was accepted. Its config/receipts remain retained.
Raise the host RSS cap to 4GiB for attempt 02, retaining the 12GiB reserve,
4GiB allocated-device cap and single sequential worker. This accommodates
CUDA runtime residency rather than allowing concurrent graphs.
