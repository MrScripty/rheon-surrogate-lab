# Couple learned transport, pressure and momentum in full-state generation

The combined decoder improves velocity and tracer and removes the previous
explosive rollout, but remains far outside0.004. Exact anchor pressure still
does not fix cold decoding. Offline native controls now expose an additional
issue: endpoint velocity depends on accumulated pressures, while the model
approximates that with the last pressure times the horizon. The pressure-sum
cold control has worst errors4.96e-5/1.61e-4/3.38e-4 for K=2/3/4, versus
0.0969/0.2128/0.3310 for the endpoint approximation. These are reference-pressure
controls, not usable future predictions.

## Qualify reuse of the learned pressure representation

The frozen pressure CNN is linear, and an offline bound probe places every
learned modal gain times its forward-operator eigenvalue strictly between 0 and 2.
This supports testing a complete neural unit-response map for the small grids,
with all coefficients derived from trained weights. No analytical inverse or
future field may construct its coefficients. Floating-point casts and gates
require explicit qualification; no equivalence or speed is assumed.

Register the actual construction and tolerances before execution. Compare its
pressure, equation classifications, zero/gauge behavior and batched gradients
against the unchanged learned component on all four grids and both densities.
Preserve failed variants. Report dense-map setup, memory, cache limits and
larger-grid fallback. Dense storage is quadratic in cell count and does not
establish a production-scale solution. Keep the independently confirmed TrackA
selection/source/weights unchanged.

Also inspect the coupled residual operator on independent compatible fields.
Accuracy on native reference requests alone does not prove damping of arbitrary
divergence errors produced during prediction. Any operator stability analysis is
offline qualification, not a numerical pressure correction in candidate inference.

## Build an actual causal full-state model

Retain one shared fluid encoding and use cheap internal latent updates for
later states. Each state must include pressure, staggered velocity and tracer;
known prescribed forcing is available, future native states are not.

One concrete neural factorization is:

1. A learned local transport decoder generates the provisional velocity from
   the current predicted state, retained features and known force.
2. The pretrained neural flux convolution generates the pressure-request field
   from that same provisional velocity. This ties the request to the velocity
   representation instead of fitting two unrelated forecasts.
3. A qualified learned pressure readout generates pressure. Reuse fixed learned
   spatial work rather than calling independent full spatial models at every
   timestep. Count every internal readout.
4. Couple velocity to pressure using coefficients shared with the negative
   spatial transpose of that trained flux layer. The new offline primitive
   qualification matches native momentum within 2.70e-8 relative, with zero new
   updates. Inference consists of learned tensor layers; numerical pressure
   solve or external projection routines remain excluded.
5. Generate tracer through fluid-conditioned learned weights on current tracer
   donors. Preserve linear dependence on passive tracer, zero/constant fields
   and donor range. Native tracer transport uses the entering velocity, so
   qualify temporal staging rather than feeding future reference velocity.

Internal causal updates naturally retain each pressure's effect instead of
replacing a changing pressure history with K times its endpoint. A joint pressure
decoder with prefix-conditioned velocity, or bounded cheap neural refinement
after intermediate predictions, remains a separate possible comparison.

Offline provisional-velocity targets can be derived from existing native
pressure/velocity fields using an independently qualified teacher. They may
supervise or diagnose transport; they cannot enter inference. If adding an
auxiliary target, register its objective and matched control explicitly. Do not
silently change the established full-state objective or normalization.

## Actual admission and success criteria

Implement the real constructors, then qualify CPU/CUDA two-block gradients,
physical units, staggering, causal inputs, masks, feature lifetime and counted
encoder/local-update/readout calls. A one-step control may diagnose dynamics,
but cannot replaceK=2/3/4. Start a bounded full-state fit only after those checks
pass, preserve every failed trial, and evaluate all development horizons and
12/48/240-step own-history before selecting a recipe.

All equation/P/V/T errors must still be at most0.004. Complete CPU block budgets
remain below 4/6/8 ms for K=2/3/4, with no separate internal-stage cap. Include every
operator, donor gather, force feature, output and history update. Measure CPU
latency from actual qualifying weights; old component timings cannot establish
new full-state speed. Replicate and freeze full-state finalists before opening
the eight reserved scenes. No model is promoted by this plan or its controls.

[Complete decoder result](temporal-decoder-results.md) ·
[Learned momentum and pressure-sum controls](tied-momentum-results.md) ·
[Pressure reuse proof preparation](learned-pressure-reuse-note.md) ·
[Original immutable plan](plan.md)
