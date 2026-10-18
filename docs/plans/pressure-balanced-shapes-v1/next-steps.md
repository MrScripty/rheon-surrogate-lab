# Next proposal

Keep C53 deployed. Do not lengthen C69 or run the 24-case screen: the current
matched-example aggregation test failed its full gate. Preserve C66/C67/C69
results; a higher count of four-pass successes did not improve the worst case.

The next cheap step is CPU-only diagnosis of the failed shape at C69, compared
with C66 and C53 on these exact eight cases. Reuse the captured full loss and
split final/early terms. Record each raw gradient norm, its share of the mean,
and projection/alignment with the common mean; also compare a proposed per-case
norm cap before averaging, without optimizer steps. Stream vectors and retain
case-level scalar evidence, runtime and memory under existing bounds.

This distinguishes poor gradient balance from the untested assumption that more
aggregate updates are needed. Equal loss weights do not imply equal gradient
influence. In particular, check whether already well-fitted cases dominate the
common direction and oppose case 7 (16×7×11 at 16 cells/metre), and whether early
feedback or final-solve feedback supplies that conflict. Do not infer the answer
from error magnitude alone or from the previous checkpoint's gradient geometry.

Treat Euclidean projections as local diagnostics. Exact C69 Adam moments were
not retained, so an Adam-direction replay remains unavailable; do not fabricate
it. A later optimizer-policy test would need explicit qualification and a matched
example/exposure budget with update counts retained. If a per-case cap makes
shared gradients more balanced, propose one bounded CUDA comparison. Otherwise
use the diagnostic to select an early-pass guard or objective-weighting proposal.
Do not launch several mechanisms together or consume untouched scenes.

No additional training is started by this proposal. Architecture, cubic-cell
physics, float32 inference and the 1e-6 pressure target remain unchanged.
