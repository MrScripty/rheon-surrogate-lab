# Possible reuse of the full learned pressure map

A read-only offline probe examines the frozen learned modal responses, without
changing any selected pressure model or the active full-state decoder fit.
For all four exported geometries, every retained nonconstant mode has
`0 < learned_gain * forward_operator_eigenvalue < 2`. Observed products range
from 0.5444 to1.1107. Analytical eigenvalues enter this diagnostic proof only;
no inverse or new candidate coefficient is constructed from them.

The frozen spatial CNN is bias-free and linear. With fixed shape-head responses,
each modal/fine stage is also linear in real arithmetic when its contraction
proposal is accepted. The observed gain bounds imply contraction on the retained
nonconstant modal subspace in that arithmetic. Unrepresented modes remain
unchanged by that proposal. This suggests that the complete fixed learned
pressure map could be constructed from neural unit responses and reused as a
linear readout, avoiding repeated spatial CNN work for later generated states.
It would derive coefficients from trained neural weights, not a numerical inverse
or future reference labels.

This is preparation for a possible full-state feature-reuse architecture, not a
qualified implementation. Floating-point conversions and gates still matter.
Before using it, register and qualify the actual unit-response construction,
CPU/CUDA gradients, geometry limits, parent pressure/physical-equation agreement,
zero/gauge cases and every target classification. Preserve all failed variants.
Keep setup time, dense-map memory and cache limits separate and visible. A dense
map grows quadratically in voxel count and is unsuitable as a production-scale
representation; any32³/larger fallback remains separately measured.

If qualified, a future model could retain one fluid encoding and use cheap
internal latent updates and learned pressure readouts to produce all physical
horizons, including their accumulated pressure effect on velocity. Count every
readout/update and reject a claim based on repeating independent full spatial
models. Such a design must still pass the original equation/P/V/T and240-step
criteria plus complete 4/6/8 ms budgets. This note supplies no temporal accuracy,
CPU latency, production performance or model-promotion claim.

[Offline bound probe](../../../runs/full-ai-next-priorities-v1/learned-pressure-gate-bounds-01.json) ·
[Immutable source/result/asset bindings](../../../runs/full-ai-next-priorities-v1/learned-pressure-gate-bounds-01.source.json)

## Completed implementation qualification

The complete learned unit-response map now passes the registered CPU and CUDA
checks: 128 native requests and 32 gradient cases per device, with zero optimizer
updates. Maximum CPU pressure difference from the unchanged parent is 1.14e-8
relative; target classifications remain unchanged. The three small-grid matrices
occupy 147,852,296 bytes, with approximately 76.5 seconds of CPU matrix construction
in that qualification. 32³ uses the original component. These are component
setup/parity measurements, not full-state inference timing.

An additional registered audit probes arbitrary compatible disturbances and
the residual operator on all three small grids. The probes show no persistent
growth. Iterated singular-value estimates are lower bounds, approximately
0.122, 0.116 and 0.071; they do not certify a universal operator norm or nonlinear
rollout stability. The actual coupled model still requires trained every-field
and own-history evaluation.

[CPU qualification](../../../runs/full-ai-next-priorities-v1/full-learned-pressure-map-cpu-qualification-01.json) ·
[CUDA qualification](../../../runs/full-ai-next-priorities-v1/full-learned-pressure-map-cuda-qualification-01.json) ·
[Residual-operator observations](../../../runs/full-ai-next-priorities-v1/full-learned-pressure-map-operator-audit-01.json)

## Accumulated pressure in a future decoder comparison

The native preparation and velocity update also identify a temporal issue to
measure separately. Across K steps, the linear pressure contribution to endpoint
velocity depends on the sum of the K pressures, alongside each step's nonlinear
transport and force contribution. Multiplying only the last pressure byKis an
approximation to that sum when pressure changes with time. The active decoder
comparison deliberately tests lead-time conditioning; it does not claim exact
accumulated dynamics or resolve this issue by construction.

A future comparison could predict all pressure horizons through shared learned
readouts and condition velocity on their prefixes, or use bounded cheap neural
refinement after intermediate predictions become available. Another option is
a shared latent dynamical model with cheap internal learned pressure readouts,
rather than repeated full spatial encoders. All stages and provisional work
must be counted. Learned interpolation, retained features and a pressure-only
component cannot substitute for complete causal P/V/T outputs or own-history
evaluation. This is a hypothesis for a later registered full-state comparison,
not an unrecorded change to the active model.

The local source for this identity is the captured native
[preparation and pressure-conditioned velocity update](../../../runs/full-ai-next-priorities-v1/temporal-build-02/rheon/src/diagnostic/engine.rs),
used here for offline architecture reasoning only.
