# Full AI: tighter targets, larger domains, then long rollouts

The preceding study is closed with [29 verified requirements](../full-ai-next-priorities-v1/completion-audit.json). This program implements the user's subsequent ordered objectives without changing its evidence. The controlling records are the [original conditional program](../full-ai-next-priorities-v1/deferred-target-program-01.json) and [activation](activation-01.json). No model has yet passed the new targets.

## 1. Reference models

Build and independently verify separate generators for **1, 2, 3 and 4 complete simulation steps per pass**, using 32 × 32 × 32 provisionally as the reference domain from the user's worked example. Keep the smaller and odd rectangular domains as regressions. This choice is recorded explicitly; the example did not establish a measured 32-cubed success.

| Steps per pass | Maximum complete block time | Maximum block time divided by steps | Maximum worst error |
| --- | --- | --- | --- |
| 1 | 1.2 ms | 1.2 ms | 1e-5 |
| 2 | 2.4 ms | 1.2 ms | 1e-5 |
| 3 | 3.6 ms | 1.2 ms | 1e-5 |
| 4 | 4.8 ms | 1.2 ms | 1e-5 |

The time budget applies to the combined pass. An individual intermediate head may consume more or less than 1.2 ms provided the **complete block** meets its budget. Score every intermediate state, including pressure, staggered velocity, tracer and equation residual. An accurate endpoint alone does not pass.

Measure one CPU thread, batch one, fixed recorded affinity, five balanced serial repeats across a frozen finite case population. Count input encoding, prescribed forcing, neural work, physical output materialization and required history updates. Publish median, p95, maximum, full block time and time per generated state. Acceptance uses the observed maximum, not the median; this finite measurement is not a hard real-time guarantee. Report setup, cache preparation, memory and complete application costs separately.

Reference-grid accuracy must also survive 240 actual own-history steps, without teacher resets. Freeze training-derived field scales, near-zero denominators and absolute tolerances appropriate to 1e-5 before fitting. The earlier 4e-3 tolerances cannot establish this tighter pass. Keep forcing changes, walls and small fields in the evaluation population.

### Work sequence

1. **Measure the existing bottleneck.** Run the [registered frozen-component profile](frozen-component-profile-protocol-01.json) on the actual deployed K=1/2/3/4 checkpoints and already-consumed development states. Check bitwise unchanged outputs and unchanged learned tensors. These inclusive instrumented costs diagnose work; they are not a new frontier or acceptance benchmark.
2. **Locate error floors.** Register offline component controls that separate the learned pressure response, transport/viscosity readout, field encoding/precision and autonomous accumulation. Teacher-derived components may be used only in explicitly labelled offline diagnostics. Replace or retrain a frozen component whose floor already exceeds the target; more training on unrelated temporal heads will not establish its adequacy.
3. **Qualify the cost of a bounded architectural change before fitting.** Use the [primary-source architecture review](architecture-research-01.md). Prioritize a shared learned endpoint/intermediate decoder and a narrow learned spectral trunk according to the measured bottleneck. Compare a coarse latent processor if those fail. Cross-block dynamic feature carry stays conditional on a successful stateless candidate. Register exact tensor layouts, precision, finite alternatives and timing population first.
4. **Freeze new data and learning contracts.** Reuse consumed scenes for development, not as fresh independent confirmation. Reserve new whole scenes and shapes before selection. Register all-horizon supervision, rollout-aware training, seeds, optimizer work, memory and finite stopping budgets. Train through managed Tuldok CUDA and verify typed exact CPU restores. No local optimizer substitutes.
5. **Select on development, confirm independently.** Freeze selected weights, inference sources, normalization, arithmetic policy and cache/reset rules before fresh held-out tests. Verify all four generators' complete outputs, own-history accuracy and observed maximum latency. Retain failures and additional-seed results. Do not activate the next phase on a median-only or endpoint-only success.

Full AI remains the inference contract. A learned transform-domain operator is admissible; an analytic pressure inverse, iterative numerical solve, corrective projection or hidden solver fallback is not. Fixed transforms and learned coefficients need a clear provenance audit. Native Rheon remains an offline teacher and separately timed control.

## 2. Density 50/100 and larger, irregular domains

**Start only after all four reference generators pass phase 1.** Add both 50 and 100 divisions per axis per metre. Keep this distinct from logical grid dimensions: at density d, an Nx × Ny × Nz grid has physical extents Nx/d, Ny/d and Nz/d metres. Neither quantity is fluid mass density.

Generate random aspect ratios in three groups at each density: all axes below d, all axes above d, and mixed axes straddling d. Retain old sizes and densities for regression. Keep whole scenes together in splits and reserve unseen shapes, including odd dimensions. Pilot native capture cost, memory, disk and numerical stability before freezing finite dimension ranges and dataset counts; do not assume that a density change is only a metadata change.

Build and test K=1/2/3/4 models at the new densities and registered sizes with the same 1e-5 worst-error criteria. For V = Nx × Ny × Nz at or above 32³, the maximum per-step time is:

`1.2 ms × V / 32768`

The complete block allowance is K times that quantity. Smaller regression domains retain the 1.2 ms cap.

| Domain | Voxels relative to 32³ | Maximum time per generated step |
| --- | --- | --- |
| 32 × 32 × 32 | 1 | 1.2 ms |
| 64 × 64 × 64 | 8 | 9.6 ms |
| 128 × 128 × 128 | 64 | 76.8 ms |

Use the actual voxel product for rectangular grids. Measure complete-output latency and memory rather than extrapolating from small domains. Seek growth below this ceiling by comparing matched-aspect families, time per voxel and Rheon's native solver on comparable cases. Vary aspect ratio separately so a favourable shape is not mistaken for better scaling. Dense V-by-V compiled maps are unsuitable as the scaling strategy.

## 3. Drift over 600 steps

**Start after the reference and larger-domain targets succeed for every K.** Minimize drift for K=1/2/3/4 over 600 real integration steps at both 60 Hz and 30 Hz: respectively 10 and 20 seconds of simulated time. Use matched native references and independently qualify the timestep input, forcing and native capture contracts. Showing every other frame from a 60 Hz simulation is not the 30 Hz experiment.

Feed each generator its own state throughout, without reference resets or corrective solves. Score all physical states, including every intermediate output. Preserve prior time and accuracy constraints. Report worst and time-varying pressure/velocity/tracer error, equation residual, divergence, boundary flux, tracer mass/bounds, energy behaviour and time to the predeclared diagnostic drift levels. Application-specific drift limits remain unset by user choice.

At activation, freeze finite refinement budgets and stopping rules. “Least possible drift” means the best independently measured result among the registered finite trials; do not claim a global optimum. Retain full curves, failing cases and scene/seed variability so accuracy gains cannot hide late divergence.

## Current evidence

The prior 32-cubed temporal models take about 89–105 ms per generated state by median and have worst errors around 1e-3. Their observed maxima are higher. They do not meet this program's 1.2 ms/1e-5 gate. The small-grid results belong to the preceding contract and do not establish a 32-cubed pass.

The [component profile](frozen-component-diagnosis-01.md) is complete: at 32 cubed the repeated learned pressure path costs about 64 ms in a one-step block and 262 ms in a four-step block. The current readout and preparation also need replacement or substantial execution changes. Instrumentation preserved every output bitwise and changed no neural tensors.

The [offline pressure diagnostic](pressure-floor-diagnosis-01.md) is also complete. Correct native RHS inputs still produce about 6e-4 worst equation residual through the frozen learned pressure component. Passing even an exact native pressure field through the current float32 pressure codec reaches 2.32e-5 residual. The next candidate must therefore qualify a better learned pressure response and a precision-preserving pressure output; this is stronger evidence than assuming temporal head tuning will suffice.

The [execution preflight](execution-path-qualification-01.md) is complete. A separately bounded managed CUDA pressure-component fit and exact typed double CPU restore then completed. The [learned modal pressure branch](modal-pressure-component-results-01.md) passes pressure/residual criteria on all 1,920 consumed development states; independent recomputation of every saved physical output confirms a worst residual of about 1.07e-9. This supplies correct native RHS and therefore does not establish causal full-state prediction or autonomous accuracy. No full-generator target has passed.

The [causal composition and precision comparisons](causal-transport-results-01.md) are complete. Their generated-RHS error now dominates pressure approximation error. Double known-force/learned-viscosity preparation helps the numerical floor; an explicitly offline exact-displacement control passes all tested short-block criteria. That oracle is not a candidate. Current learned displacement still fails, particularly on the smaller retained grids. Fixed neural field-readout vectorization also passed exact-output controls and reduced its core-23 median from about 2.25 to 1.10 ms at 32 cubed, but that component alone nearly exhausts the complete step budget.

Next is the [bounded learned recurrent-sampling prototype](recurrent-sampler-prototype-01.md), with direct staggered-array execution and complete cost qualification before fitting. Every coordinate proposal must come from learned parameters; no native backtrace may enter inference. Freeze fresh whole-scene reservations, exact arithmetic and K=1/2/3/4 training/evaluation contracts after that cost gate. Higher-density capture/training and the 600-step campaign remain gated. New registrations and evidence live in this directory and the corresponding `experiments/full-ai-tight-targets-v1` and `runs/full-ai-tight-targets-v1` directories; the completed study remains preserved.
