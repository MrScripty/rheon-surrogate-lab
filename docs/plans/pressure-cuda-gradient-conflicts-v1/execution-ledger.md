# CUDA gradient-conflict results

Completed 2026-10-18. C53 remains deployed. This diagnostic performed backward
passes on frozen states, with **zero optimizer updates**, no new simulations,
no holdout use, no new training trajectory and no model promotion.

The CUDA calculation reproduced **exactly** the captured raw gradients, clipped
gradients and eight full loss values at each pre-update state 100, 150 and 200.
Both gradient relative L2 gaps and maximum full loss relative gaps were zero.
This supplies the derivative evidence unavailable from the earlier CPU replay.
It does not validate a new training policy or an entire simulation rollout.

All **18/18** comparisons between the weak case 7 four-pass equation gradient
and cases 0, 1 and 4 one/two-pass equation gradients had negative cosine:

| Other case | Pass | Update 100 | Update 150 | Update 200 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | −0.8465 | −0.8790 | −0.8731 |
| 0 | 2 | −0.6669 | −0.9386 | −0.9432 |
| 1 | 1 | −0.8822 | −0.8943 | −0.8910 |
| 1 | 2 | −0.8719 | −0.9615 | −0.9561 |
| 4 | 1 | −0.8433 | −0.8452 | −0.8409 |
| 4 | 2 | −0.8212 | −0.8094 | −0.8091 |

These are strong local conflicts. A step along the negative weak-case gradient
predicts increased short-pass equation loss in the selected other cases. All
eight cases and passes 1/2/3/4, plus the unchanged full loss, remain available in
the scalar records and interactive viewer, including norms and observed-delta
projections. The selected table was declared before measurement from the
previous damping regressions, not chosen after viewing its signs.

The result supports conflicting feedback as one contributor to the observed
tradeoff. It does not establish that these derivatives alone explain every
finite-step regression. For example, at update 100 the weak four-pass gradient
aligns with the captured raw mean (cosine +0.5780), while its dot with the actual
Adam parameter delta is positive (+0.0758402), predicting a local loss increase.
At update 150 those values are +0.5963 and +0.0123319. The optimizer state and
preconditioner therefore remain relevant alongside the conflicting losses.
At update 200 that actual-delta dot is negative (−0.000121053), consistent with
the earlier observation of very small weak-case gains at its tiny LR.

For case 1/pass 2, the largest negative layer contribution is `head.weight` at
all three states (about −1.185e6, −1.074e6 and −1.057e6). Decoder and first
encoder weights also contribute; this is not evidence for changing only the
head. Full layerwise dot/norm records are retained for each of the 18 selected
comparisons. The separate unscaled per-case full-gradient mean differs from
the original scaled-before-backward CUDA accumulation by relative L2
5.7034e-8, 7.6322e-8 and 8.4459e-8. This disclosed floating-point aggregation
gap is distinct from the exact original accumulation replay.

## Timing, memory and failure record

Accepted attempt 02 took **8.3948s controller elapsed** and **6.8327s numerical
worker elapsed**. Snapshot measurements took 2.1763s, 1.8053s and 1.7544s.
There were **144 sequential backward graphs**, never multiple retained forward
graphs, and 120 float32 gradient vectors (40 per state). No AMP or TF32;
deterministic CUDA policy matches the original source. Physical loss/operator
calculations retain their float64 owner, and model parameters/derivatives are
float32.

Peak allocated CUDA tensors: **17.2MiB**; peak CUDA allocator reservation:
**34.0MiB**. Peak worker self-reported RSS: **2.052GiB**; parent `/proc` sampling
observed **2.053GiB**. Minimum host available memory: **46.60GiB**. This CUDA
runtime host residency is separate from live device tensor allocation. Vector
files total **177,623,040 bytes**, within 256MiB; conservative host vector
workspace 47,362,048 bytes is within 64MiB. GPU allocation is capped at 4GiB,
host RSS at 4GiB, with a 12GiB host reserve and one worker.

Attempt 01 was stopped by its initial **2GiB host RSS guard** during CUDA
startup, after 3.2329s (sampled 2,180,313,088 bytes). No snapshot measurement
was accepted. Every owned child was terminated/reaped. The frozen config and
receipt are retained. The accepted attempt raised only the host RSS budget to
4GiB; it did not relax physics, gradient/value replay tolerances, device cap or
graph sequentiality. The sandbox could not communicate with the GPU driver;
the authorized diagnostic ran outside it with the existing Pumas runtime.

Cancellation qualification terminated the controller during qualifier startup
(0.1048s) and after the first **actual CUDA snapshot** completed (4.5277s total
fixture time). Both controllers and owned children were reaped; that incomplete
measurement is not included in scientific results. No automatic retry occurs.

Independent literal geometry fixtures cover orthogonal/opposed/zero-norm
vectors, exact layer partitions and typed invalid inputs. CPU verification
recomputes all 120 vector norms/projections and 18 layer partitions with a
separate scalar accumulation, confirms identities, resource/custody receipts
and exact replay records. Browser checks all three states, eight cases, five
terms, two comparison anchors and all layer selectors, keyboard labels and
narrow layout; no page exceptions. The final screenshot was visually checked.

Only source, scalar records and small receipts are committed. Raw gradients
remain ignored under the accepted attempt. Existing published weights,
datasets, trace archives and historical reports remain unchanged.
