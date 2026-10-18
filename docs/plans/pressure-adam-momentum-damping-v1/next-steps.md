# Next proposal

Keep C53 deployed. Neither full reset, step shortening nor partial first-moment
damping produced a proposal satisfying the declared retention guards. Do not
start a long damping training run from weak-case gains alone.

The useful finding is a feedback tradeoff: later weak-case four-pass gains can
coincide with worse early-pass performance elsewhere. The next bounded step is
to collect **per-case and per-pass CUDA gradient evidence** at an existing later
snapshot before changing the loss or optimizer again. Compare the weak case's
four-pass loss gradient with gradients from the short-pass cases that regressed
(especially 0, 1 and 4). Record normalized gradient dot products, layerwise
contributions, clipping and individual losses, using managed CUDA float32 and
bounded sequential gradient capture to avoid retaining multiple graphs.

This would distinguish conflicting training signals from a poor momentum
direction more directly than another reset grid. Only then propose one matched
CUDA arm with explicit case/early-pass balancing, unchanged data, step budget,
pressure target and CPU qualification. An alternative objective needs its own
predeclared guards and comparison; do not silently relax the existing ones.

No gradient collection or CUDA training is started by this completed CPU test.
