# Next proposal: isolate Adam memory and step magnitude on CPU

Keep C53 deployed and all physics/precision targets unchanged. C69's unchanged
replay reproduced its weights and CPU pressure results exactly. Direct CUDA
before/after measurements show worsening on case 7 at four selected earlier
updates. This justifies an optimizer-focused test, not more training updates or
an immediate new early-loss/capping policy. Five observations cannot establish
that Adam memory alone caused the full trajectory.

Use the captured pre-states, exact accumulated/clipped CUDA gradients and actual
Adam moments for a bounded **CPU-only counterfactual step screen**. No new CUDA
training yet. Compare preserved Adam state with freshly reset moments/step on the
same captured gradient and rate. Include actual observed delta and fractions
0, 1/8, 1/4, 1/2, 1 of each proposed delta. Evaluate the unchanged captured loss
and 1/2/4/8-pass pressure residual on all eight cases before/after each proposal.
This tests finite-step behavior directly rather than relying on CPU/CUDA gradient
cosines, whose precision qualification failed.

Qualify the preserved-state reconstruction against the five observed deltas with
the existing tolerance; update 1's initialized-zero/reset controls should agree.
Keep the explicit old-gradient constraint: a reset is a hypothetical one-step
change from the captured state, not a retrained history. Calibrate CPU/GPU value
agreement on these pre/post states and retain numerical limits; CPU-only success
does not establish CUDA training benefit.

Bound one model/graph at a time per worker, at most three independent CPU workers
if the measured 12 GiB host reserve plus 2 GiB per worker is available. Cap total
wall time at 120 seconds, stream archives, retain scalar outcomes and reaping/RSS
receipts. Preserve the four-pass target and short-pass retention guards. No new
simulation capture, validation/holdout consumption, or unbounded vector history.

Choose one subsequent matched CUDA arm only if this screen identifies a useful
change: reset/alter moment handling if it fixes weak-case ascent without harming
other cases, or a smaller/guarded step if magnitude dominates. Otherwise propose
additional CUDA derivative evidence or explicit case-balancing research. Do not
combine moment changes, loss reweighting, capping and schedule changes in one
training run; do not expand to 24 cases before the eight-case fitting gate passes.
