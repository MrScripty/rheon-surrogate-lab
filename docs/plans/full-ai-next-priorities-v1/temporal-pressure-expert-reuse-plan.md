# Reuse the learned pressure expert in full-state multi-step models

The C107 successor has independently reached the pressure benchmark objective:
2.133ms median CPU inference and4.6963e-4 worst equation residual on768 new
small-grid requests. The endpoint/intermediate and retained-feature models meet
small-grid median speed budgets but fail accuracy. Conditioning, increment-MSE
and a generative curl representation have not resolved that failure.

Reuse the successful learned pressure representation rather than asking another
shared encoder to learn all pressure structure from scratch. This remains a
full AI candidate: no numerical pressure solve or post-corrective projection
may enter inference. Existing fixed neural representation and reconstruction
layers must be disclosed and included in timing.

1. **Qualify the frozen pressure port.** Reconstruct the existing fine CNN from
   exported weights and cache geometry responses of its learned heads. Retain
   its two learned corrections, learned tail and acceptance decisions. Compare
   with the unchanged parent on all four grids and both spatial densities;
   verify batched outputs and deterministic input gradients on CPU and CUDA.
   Unsupported geometries must reject. No fit or temporal claim follows from
   this component check.
2. **Integrate causal full-state prediction.** Predict pressure requests from
   the current state and prescribed forcing, then use the frozen learned
   pressure component. Learn velocity and tracer evolution with all physical
   outputs at every horizon. For this reference contract, tracer must not
   influence pressure or velocity. True future requests, pressure and endpoints
   remain offline targets only. Qualify gradients through the actual deployed
   feedback path before fitting.
3. **Compare the requested sharing paths.** Start with bounded K2 fitting and
   measured cold/warm training diagnostics. One path spends most capacity on an
   accurate predicted endpoint and uses a smaller decoder for intermediate
   states. The other retains features from its first prediction and uses cheap
   latent updates for later states. Include K3 and K4 in the controlled screen;
   count encoder calls and predicted states, and preserve failed outcomes.
4. **Judge complete useful blocks.** Require equation and P/V/T errors at every
   horizon, including240-step own-history, at or below0.004. Then measure the
   complete CPU block: below4ms forK2,6ms forK3 and8ms forK4. No internal stage
   has a separate2ms limit. Include encoding, known forcing, the pressure core,
   every decoder and physical reconstruction; report amortized and total cost.
5. **Replicate qualifying recipes and evaluate reserved scenes.** Keep the
   eight reserved temporal scenes unopened until a full-state finalist qualifies.
   Freeze model choice and recipe before reserved capture. Application drift
   limits remain diagnostic; prepared-pressure success is not native acceptance.

Do not add old component timings to old decoder timings to claim a new block
result. Do not treat an offline future-request oracle as causal generation.
Retain the independent pressure result without reopening its frozen selection.

[Frozen port qualification registration](temporal-pressure-core-port-registration-02.json) · [Current evidence and decisions](next-decisions.md) · [Original immutable plan](plan.md)

The first item is now complete: [CPU/CUDA component qualification passes](temporal-pressure-core-port-results.md)
on128 native requests and32 gradient cases per device. It makes zero optimizer
updates and no temporal prediction or latency claim. Causal full-state integration
and its actual feedback qualification remain next.

## Actual full-state follow-up

Both endpoint/intermediate and retained-feature paths are now implemented for
K2/3/4. All six pass actual-constructor CPU and CUDA qualification: four grids,
cold/warm starts, deterministic two-block gradients, one fluid encoder and one
frozen pressure-expert call per block. Tracer and previous pressure cannot alter
fluid outputs; prescribed forcing does alter predictions. Maximum CPU/CUDA
physical output difference in the probes is2.39e-7. Qualification makes zero
optimizer updates and uses diagnostic copies for activating new heads.

The first bounded fit is K2 endpoint. Its model has111,650 parameter elements,
of which19,009 are trainable; cached learned pressure-head responses also carry
frozen parent information. Initialize the request branch from the managed
11,000-update factored fit, retaining its failed accuracy result. Freeze its
learned flux and all92,560 pressure spatial coefficients. New fluid/tracer
encoders and zero output heads are seeded. This is an architecture/initialization
follow-up, not a single-factor comparison or matched-total-training-work claim.

Predict the anchor pressure request from current velocity and prescribed forcing;
the frozen learned pressure expert generates the anchor pressure. A learned
convolution conditioned on that predicted pressure produces part of the velocity
readout. Its coefficients start at zero and are learned from full-state labels;
no analytical gradient stencil or numerical post-projection is installed.
Prescribed force increments and fixed unit conversions enter reconstruction and
must be included in complete timing. The smaller intermediate decoder consumes
predicted endpoint fluid fields, and its tracer branch consumes predicted
endpoint tracer. The reuse path retains fluid/tracer features for later states.

Use the original16 training scenes, normalization, sampling, increment-MSE terms,
10,000-update schedule and2,000single/8,000two-block curriculum. Count inherited
training separately. Exact CPU restoration, every development horizon and
12/48/240-step own-history evaluation follow the fit. No reserved scene is
admitted and no prediction accuracy or CPU latency result is claimed yet.

[Frozen full-state registration](temporal-pressure-expert-registration-01.json) ·
[Admission](temporal-pressure-expert-admission-01.json) ·
[CPU qualification](../../../runs/full-ai-next-priorities-v1/temporal-pressure-expert-trainer-cpu-qualification-01.json) ·
[CUDA qualification](../../../runs/full-ai-next-priorities-v1/temporal-pressure-expert-trainer-cuda-qualification-01.json)

## Completed first fit and resulting priority

The K2 endpoint fit completes10,000 actual managed updates, exact CPU restoration
and every development horizon/240-step continuation. Declared controls pass.
Worst small-grid single-block P/equation/V errors improve to0.2564/0.4727/0.3911,
but accuracy still fails and the long rollout is severely unstable. CPU latency
remains unmeasured for these new weights. No reserved scenes are consumed.

Train-only oracle controls show that exact anchor pressure does not fix cold
velocity decoding. Pressure-readout lead-time gains differ, a constant pressure
gauge shift alters velocity, and zero current tracer produces nonzero tracer.
The next priority is to compare decoder representations addressing these
measured limitations before extending or replicating this failed recipe.
Every full-state objective and complete4/6/8ms budget remains unchanged.

[Completed result and controls](temporal-pressure-expert-results.md) · [Specific next decisions](next-decisions.md)
