# Next proposal

Keep C53 deployed. C67 improved the number of four-pass fitting successes from
five to six, but worsened the worst case and still regressed on consumed real
requests. Delaying decay alone is insufficient under this bounded one-seed test.
Do not run the gated C68, extend training or expand CFD data on this evidence.

The next inexpensive step is CPU-only loss-component diagnosis on the same eight
cases at C53, C66 and C67. Split the existing objective into canonical pressure
loss, final-pass equation feedback, and weighted early-pass feedback. Retain each
component's value, gradient norm, pairwise alignment across shapes, and alignment
with the final-pass gradient. Verify the component values/gradients sum back to
the original full objective; keep parameters/buffers unchanged and perform no
optimizer updates. Use the same memory/thread caps and record time and sources.

This addresses a specific uncertainty: the current 16/28 opposing pairs describe
the whole C66 objective. They do not tell us whether shapes disagree on the final
solve or whether earlier-pass objectives compete with the final solve. Euclidean
cosines also do not measure Adam's preconditioned update, so include a frozen
optimizer-state directional analysis only if that exact state is retained and
verified; otherwise report it unavailable.

If early feedback is responsible for most conflict, propose one matched CUDA
screen that penalizes early-pass regressions against frozen C53 training-case
results rather than continually pushing already acceptable early passes toward
zero. Check the hinge/guard semantics independently before training. If the
final-pass gradients themselves conflict strongly, instead propose a balanced
shape-aggregate update screen with explicitly matched example work; distinguish
its changed optimizer-update count. Choose one after diagnosis, not both at once.

Retain the all-case four/eight-pass gate, one/two-pass guards, fresh CPU restore,
128 real requests, 240 manufactured regressions and full time/memory records.
A later successful fitting screen would still need a separate real-frame
retention experiment before deployment. No claim of architectural impossibility,
noise floor or time-step limitation follows from this study.

These are proposals. The authorized gradient/later-decay sequence is complete;
no additional CUDA training has been started.
