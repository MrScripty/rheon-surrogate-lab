# Next bounded architecture: learned recurrent sampling

Status: next prototype specification; **no new parameters fitted under this specification**. The controlling full-generator targets, 32-cubed reference, smaller regressions, fresh whole-scene verification and later phase gates remain unchanged.

## Why this change

Replacing the frozen pressure branch with the managed learned modal branch removes most pressure/velocity error in causal short blocks, but the generated pressure RHS still misses the 1e-5 criterion. Keeping physical carrier arithmetic helps modestly. With an explicitly offline exact-displacement control, double force/viscosity preparation brings all tested K=1/2/3/4 short blocks below 1e-5; the worst 32-cubed equation error is approximately 7.14e-6 at K=4. This control is diagnostic and **cannot** be deployed as a full-AI candidate.

The present displacement head approximates nonlinear transport using a spatial encoder and a restricted polynomial readout. A more expressive learned sampling composition could remove that approximation while eliminating the expensive fluid encoder. This is a falsifiable architectural hypothesis, not an established accuracy or speed result.

## Proposed neural graph

1. Apply the already managed, learned point-projection coefficients to the current physical staggered velocity. Produce four groups of three coordinate features, one group for each velocity/tracer carrier lattice.
2. A learned twelve-component gate predicts an intermediate query displacement from those features and the known timestep/cell spacing. Initialize the gate to zero; do not insert a native integration coefficient.
3. Sample current velocity features at those neural query locations with a fixed differentiable staggered-field sampling layer. Fixed lattice offsets describe the input representation.
4. A learned component mixing map predicts final displacement features from the sampled values. Initialize it only through a recorded transformation of the existing managed point coefficients, or train it in the managed CUDA recipe. Do not assign an analytical velocity/backtrace stencil.
5. Read out physical carrier fields using the neural displacements. Apply prescribed external forcing in double before float32 storage. Apply the existing learned viscosity coefficients with double accumulation and one float32 store, then the learned flux, learned global pressure branch and tied learned momentum readout. Return separate double pressure and float32 staggered velocity/tracer arrays.

This uses the general neural pattern of predicting sampling locations and applying a fixed sampler. Spatial Transformer Networks establishes that pattern as a differentiable neural layer; it does not establish this fluid task's 1e-5 accuracy, rollout behaviour or CPU budget. [Original paper](https://arxiv.org/abs/1506.02025).

Every dynamic coordinate proposal must depend on learned parameters. Candidate inference must contain no call/import of `displacement_labels`, `NativeClient`, native backtrace/viscosity/pressure code, or an analytical pressure inverse. The offline oracle remains a separate diagnostic. A wrapper around native transport with only a calibrated scalar is not an acceptable replacement; audit the full neural graph and coefficient provenance before admission.

## Cost prototype before training

Use native staggered arrays directly where possible. Avoid common-grid normalization, repeated padding, a large encoder, and intermediate shift arrays that exist only to be consumed by the next layer. A fused CPU implementation may compute learned coordinate features and fixed sampling in one pass, but must reproduce the declared neural graph. All required input checks, known forcing, every intermediate state and output materialization belong in complete block time.

Prototype exactly one two-stage sampling graph first. Use seeded untrained gates and recorded inherited learned coefficients for execution checks; no hand-set native midpoint coefficient may masquerade as a trained result. Check both neutral and nonzero seeded gates. Compare a tensor reference with the deployed CPU implementation across retained shapes, boundaries and both densities.

For sparse execution of inherited linear layers, a deterministic general uniform absolute quantization at 2^-40 is a proposed numerical policy. Preserve raw source weights, quantized weights, their exact hashes/dtypes and maximum coefficient/output differences. Derive sparse support only from those quantized learned values. Do not round to a geometry-specific stencil or analytically supplied coefficient lattice. This policy needs its own actual output qualification before use; it is not yet admitted.

The core-23 benchmark remains the comparable reference. Exploratory core-6 measurements must retain matched baseline timings and remain separately labelled because these cores have different execution characteristics. If a future full-generator contract chooses another core, freeze that choice before its fits and rerun comparable baselines; never attribute a core change to model architecture.

Stop this prototype before an expensive complete-generator fit if mandatory complete work exceeds the registered block budget. Preserve failed controls. A cheaper polynomial/factorized learned coordinate representation is the next bounded mechanism only if the two-stage sampler fails cost qualification; do not open an unrestricted architecture search.

## Learning and independent verification

After cost qualification, freeze exact sources, precision, input envelopes, training work and resource limits. Prepare intermediate-coordinate and final-displacement labels from qualified offline native teachers using original training scenes only. Their numerical routines stay outside candidate inference. Deriving training labels is not permission to assign their coefficients to the model. All parameter learning remains in managed Tuldok CUDA, with exact typed CPU restores and full physical probes.

Freeze fresh whole-scene reservations before fitting. Old development and reserved scenes have already been consumed and cannot provide new blind confirmation. Fit and test actual K=1/2/3/4 configurations, score all intermediate pressure/velocity/tracer fields and native equation residuals, then test 240 own-history steps without resets. A common learned recurrent component may share weights, but report the actual per-block work and do not describe sequential internal readouts as parallel speedup.

Pressure tables currently cover only the four registered geometries. That limitation must remain explicit. Higher-density/random-shape work cannot be declared complete using an unsupported-shape lookup or teacher-generated coefficients at inference. A geometry-conditioned learned branch or additional registered supported shapes and an appropriate shape-generalization study belong to the later phase, after all reference generators pass.
