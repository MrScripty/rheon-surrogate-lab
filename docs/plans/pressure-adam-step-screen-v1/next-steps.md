# Next proposal: isolate the first moment before another CUDA arm

Keep C53 deployed. Full Adam reset helped the weak case at several saved states,
but every proposal violated at least one local retention guard. At update 100,
the 19.43% weak-case residual gain cost two other four-pass qualifications. At
10/100/150, merely reducing the observed step did not improve the weak residual.
There is no qualified reset/smaller-step training arm from this screen.

The next bounded experiment should isolate **first-moment damping** on CPU:
retain the captured second moment, step counter, clipped CUDA gradient and LR,
while multiplying the saved first moment by predeclared factors 0, 1/4, 1/2,
3/4, 1 before the same Adam update. This separates momentum memory from the
second-moment preconditioner and bias-correction reset that changed together in
the full-reset screen. It remains an independent one-step probe, not retraining.

Use the existing five snapshots, literal/preserved-state controls, all-eight-case
loss and 1/2/4/8-pass measurements, empirical value-gap disclosures, strict
retention guards and bounded three-worker CPU lifecycle. Record endpoint lengths
so a changed magnitude is not mistaken for a proven directional improvement.
Do not add capping, new loss weights, schedule changes or 24-case expansion.

Recommend one matched CUDA policy only if damping gives a reproducible useful
tradeoff across later snapshots while preserving the declared guards. Otherwise
stop the optimizer-only shortcut and propose explicit case-balanced feedback or
additional CUDA per-case derivative evidence. CPU one-step gains alone cannot
qualify training, long-run stability or deployment. No further training was
started under this completed screen.
