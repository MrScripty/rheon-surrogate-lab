# Frozen model diagnosis

The accepted PR4 merged normally at61bf281479ce62bcb7e2e51c774b63e9090ad4ed.
This diagnosis uses original packet2 and checkpoint SHA256
f46aec5e0de59bca9a2a7a701e85c4904782c11b7ac6c113cfd3f7625c8391fc. It performs
forward/derivative observations only. Original source, configuration, trained bytes,
Run receipts and all negative evidence remain unchanged. No optimizer, training
Run, new physics trajectory or final-test access occurred.

## Population and reproduction

All384 existing development frames (288train/96validation,16whole histories) were
measured. Initial weights were reconstructed with the original seed and code;
there is no retained initial checkpoint. The original batch2 validation path
reproduced both receipt floats exactly:0.5236181201626715 initial and
0.5214263489529538 final. Original32-witness prediction bytes also match exactly.
The same qualified local CPU Torch2.8.0/NumPy2.3.5 environment was used; this is
not a new captured Tuldok Run. Observed process peak RSS459,161,600B and work
timer16.310773s include admission/forward/hooks/autograd/precision/witness checks,
but exclude final JSON serialization/publication. The180s cooperative wall and
4GiB observed high-water limits are scoped accordingly.

## Measured forward failure

| Observation | Reconstructed initial | Frozen final |
| --- | ---: | ---: |
| Exactly zero correction frames | 0/384 | 384/384 |
| Spatially constant raw head frames | 0/384 | 384/384 |
| Head input maximum within-channel spatial range | 0.05089935 | 0 |
| Final decoder first-convolution pre-SiLU range | -0.26503 to0.25633 | -1,028,147.9375 to-95,590.0234375 |
| Following SiLU zero elements | See per-row evidence | 12,582,912/12,582,912 |

The final raw head is0.15240299701690674 everywhere across all384 frames.
The extreme negative last-decoder input produces exact zero SiLU output. The
following convolution receives zeros, so its bias and next SiLU give spatially
constant channel features. The linear head is spatially constant; subtracting
its first cell removes the entire correction. This is a measured frozen forward
chain, not a claim about which training update caused it.

Earlier trained decoder convolution outputs reach about±399,000; input channels
are much smaller (maximum absolute residual14.7738, previous pressure9.82157,
geometry1.18818, density1.25). Hidden activation amplification is directly visible.
A1³ bottleneck/global-pool has zero within-channel spatial variation by design;
that alone is not evidence of collapse. The decisive tensors are the16³ last
decoder and head input, where spatial variation disappears.

## Derivatives and precision

Eight fixed validation states per model (frames0/23 of all4 validation histories)
were probed without updating any parameters. Final loss gradient with respect to
raw head remains nonzero (L2about0.292–0.750), while the global-pool and deepest
activation gradients are exactly zero. Parameter-gradient norms shrink from
about0.0459–0.1029 initially to1.03e-8–1.10e-7 finally. Tiny f32 residual parameter
gradients should not be treated as useful transmitted learning signal after a
spatially constant feature/gauge chain; finite precision reductions can leave
cancellation residuals. The original operator/loss remains sensitive to nonconstant
raw pressure: fixed-direction central differences agree within1.81e-10 initially
and7.25e-11 finally. Constant-head gauge finite differences are zero up to1.11e-12.

The manual float32 replay first matched production output exactly. On the eight
final derivative cases, both a bias-free higher-precision head on the same f32
features and a full f64 copied-stage replay still produced zero spatial variation
and correction. Promoting just the head, removing its bias, or using f64 alone
therefore does not repair the observed frozen failure on those cases. These probes
are arithmetic observations, not new trained or restored candidate models.

Pressure and residual raw-head gradients are both nonzero. Initial pressure-to-
residual parameter-gradient L2 ratios span0.0266–0.7536 on these8 cases; final
ratios0.2930–4.9571 describe tiny downstream cancellation terms after saturation.
This supports measuring component gradients; it does not establish composite-loss
imbalance as the historical cause or justify changing loss weights from this sample.

## Retained history and uncertainty

Every20-update logs show sampled train loss0.05075 and gradient norm0.5279 at100,
then loss0.52654 and gradient5.107e-8 at120; subsequent gradients remain about1e-8.
Those are different sampled batches and are not a full evaluation curve. Final
weights exhibit the saturated chain, but no intermediate weights or Adam moments
are retained, so precise onset, responsible updates and optimizer causality are
unresolved. Do not replay training to invent missing states.

The final all-row loss is lower than the random initial model on all384 frames;
that is compatible with learning a zero correction that avoids the random initial
output's residual error. It is not useful pressure correction. The existing32native
cases still document0/32raw qualification and32full temporal fallbacks; this
slice neither changes nor reruns their physics evidence.

## Conclusion and proposed next scope

Activation amplification and negative-tail saturation explain the present zero
output. Float32-head rounding alone is unsupported as the remedy. Input scaling,
architecture gain, optimization and loss interaction remain possible contributors
to the path into this state. The proposed next experiment changes only per-case
feature normalization before SiLU, with the original optimizer/loss/data fixed,
so it tests the observed amplitude problem without pretending its cause is proven.
The original full17,921,455-byte report is preserved in ignored task storage with
SHA25627a820e6486e1eb455d0006b5e7975ada689014e9e1d79b54721165f558e7d89.
The3,519,977-byte compact publication retains all384per-row decisive stats and all
16derivative records, with other stages aggregated; postprocessing required no
forward pass and every retained field was checked against the full report. Its
metadata labels timing exclusions and records the complete archive identity.
The archive availability phrase about regeneration means rerunning measurements;
it would produce a NEW observation/hash, not recover the exact original27a820
bytes. The current original archive is present and hash verified. Both report
bytes are preserved; this clarification corrects their publication interpretation.

See [the single proposed ablation](../next-experiment.md), primary-source context
in[research.md](research.md), and compact all384-row evidence in[diagnosis.json](diagnosis.json).
