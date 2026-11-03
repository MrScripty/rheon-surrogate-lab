# Multi-step generation: combined budget and architecture comparisons

User clarification: the 2 ms target is the total generation cost divided by the
number of physical states generated. It imposes no separate latency limit on
an internal step, endpoint model, interpolation network or decoder stage.

| Generated states | Complete block budget |
| --- | ---: |
| 2 | Below 4 ms |
| 3 | Below 6 ms |
| 4 | Below 8 ms |

Count initial state encoding, known forcing construction, the endpoint or first
prediction, retained-feature processing, every intermediate decoder, physical
field reconstruction and required cache/history updates. Report complete block
and per-generated-step timing together. These definitions agree with the
original registered amortized timing criterion.

For example, a three-state block could spend 4.5 ms predicting the endpoint and
1.0 ms generating the intermediate states and completing reconstruction. Its
5.5 ms total would be 1.83 ms per generated step and pass the timing target.
This is an illustrative allocation, not a measured result. Accuracy must still
pass separately at all three physical times, including the intermediate states.

## Accurate endpoint plus a small neural intermediate decoder

Predict the state at t+K·dt from the current state, available history and
prescribed forcing. Feed that predicted endpoint, the current state and shared
features into a smaller network that produces every intermediate physical
state. The model may spend most of the block budget on the endpoint. It does
not need to make each submodule run in less than 2 ms.

Use the predicted endpoint during training of the deployed decoder, including
its error. A true future reference endpoint may supervise training but must
never become an inference input or replace the endpoint during evaluation.
Supervise pressure, staggered velocities and tracer at every horizon. Retain
own-history block continuation and qualify gradients through the predicted
endpoint. Endpoint accuracy alone does not establish the full block criterion.

This learned intermediate-state decoder is an explicit candidate architecture.
It is compatible with the original prohibition on reference interpolation or
endpoint-only results: it receives no future reference state at inference and
returns all K independently evaluated states.

## Reuse computation from the first generated state

Encode the current state and geometry once. Generate the first future state
while retaining learned spatial/global features. Produce later states using
smaller latent updates or refinement heads that consume those retained features,
the relevant known forcing and any preceding predicted state needed by the
architecture. Expensive feature extraction should not be repeated as an
independent full-model call for every physical step.

Internal sequential latent updates are allowed within a single block call;
physical outputs still include every requested horizon. Count all latent updates
and reconstructions in total latency. Train the same deployed reuse path across
two or more own-history blocks and verify feature lifetime, invalidation and
history encoding. An unexpected event must invalidate all unused outputs and
any retained features that depend on superseded inputs.

## Controlled order of work

The registered one-step shared-global capacity fit and its full evaluation are
complete. The stronger representation did not resolve the accuracy failure;
preserve that negative result as the comparison control.

Then qualify endpoint-conditioned and retained-feature temporal decoders as
separate architecture comparisons. Use common training scenes, normalization,
forcing and per-horizon metrics. Start their bounded fitting comparisons at
K=2, then include both K=3 and K=4 in the architecture screen; a K=2 failure does
not establish the other horizons' limits. Track encoder calls, latent updates,
predicted physical states and processed cells rather than equating optimizer
updates with identical arithmetic work.

Compare against the existing joint-output family and sequential one-step control
on the same consumed development scenes. Select and replicate accuracy-qualified
recipes before reserved evaluation. All intermediate states remain subject to
the original 4e-3 equation/field criterion and own-history rollout checks. No
numerical pressure solve, corrective projection or reference endpoint may be
inserted into candidate inference.

These options extend the architecture choices; they do not reduce the original
outputs, horizons, error requirements or combined time budgets.

## Pressure learning diagnostic before choosing the endpoint core

The completed shared-global model still copies most pressure information from
history. On warm development windows, resetting only the pressure input reduces
its median predicted pressure amplitude to approximately 1.3% of the reference;
removing known forcing changes the output by only approximately 1.2% of that
reference norm. Startup pressure error remains close to 100%.

An offline native control reset pressure at step eight while retaining velocity,
tracer, clock and forcing. Over the next eight reference steps on each of two
training grids, velocity and tracer remain bitwise identical; maximum pressure
difference is about 1.1e-14. This is evidence for the current closed-grid reference
contract, not a claim for arbitrary boundary conditions or integrators.

The endpoint/reuse comparisons should therefore include fresh pressure prediction
rather than an unchecked identity path from previous pressure. One concrete
all-neural endpoint factorization is a small causal network predicting the
pressure-request field, followed by the successful learned C107 pressure
component and cheap learned velocity/tracer heads. The pressure-request network
must receive only current physical state, available history and known forcing;
true future RHS or pressure may supervise it offline but cannot enter inference.
All component work must fit the same combined block budget.

Before fitting that factorization, measure the learned pressure component using
true future RHS as an explicitly labeled offline oracle. If that component
already misses the pressure-field criterion, improve it before attributing the
gap to a new causal predictor. This oracle is not a simultaneous-state model,
never joins the temporal frontier, and does not satisfy any temporal accuracy or
latency objective. Endpoint-conditioned interpolation and retained-feature
updates remain the actual candidate architecture comparisons.

The oracle comparison is now complete for the new learned eight-mode final
pressure block. All 384 train/development queries pass the pressure-field and
equation 4e-3 thresholds. Worst small-grid development pressure error is
1.0677e-4, versus 8.9014e-3 for the earlier component. This removes a measured
component limitation but supplies no temporal accuracy claim: the new causal
predictor must generate the pressure request itself, without future reference
data. Its error and cost, plus all intermediate-state decoding, remain unmeasured.

The train-only causal request data view is prepared. It retains all 16 training
scenes and all 129 states per scene, with original float32 staggered velocities
and float32 request targets converted from authoritative native RHS. Its archive
is 213,993,244 bytes; maximum target conversion relative error is 3.74e-8.
Neither development nor reserved scenes enter it. Current velocity and known
forcing will be model inputs; future request arrays are supervision only.
This component dataset is preparation for the endpoint/reuse comparisons and
does not supply a full-state temporal prediction or a temporal target pass.

## Registered endpoint and reuse screen

Both full-state paths are implemented for K=2,3,4. CPU and CUDA qualification
cover all four grid groups, active and repeatable gradients through two deployed
own-history blocks, CPU/device parity, one encoder call per block, prescribed
forcing dependence and invariance to previous pressure. Endpoint perturbation
also changes its intermediate predictions, verifying that the small decoder
consumes the predicted endpoint. Qualification uses diagnostic parameter copies
and makes zero optimizer updates.

The six managed fits use the original train-only full-state view, scene split,
normalization, sampled windows, loss and 10,000-update schedule. They share the
same seeded width12 global encoder within each K comparison, and use zero
output heads with freshly predicted pressure. They do not inherit weights from
the still-inaccurate causal request component. The prior persistence shortcut
is removed by excluding previous pressure from the encoder. Both still return
all pressure, velocity and tracer fields, with 8,000 of their updates trained
through two own-history blocks. This controls training conditions, not total
FLOPs; all encoder, endpoint, decoder and latent-update work is counted.

The existing `temporal-sharing-batch-01` owns fitting, exact CPU restoration and
the complete development evaluations. Its dependent postprocessing batch waits
for completion and child reaping before control audits, fixed-window fitting
diagnostics and five balanced serial CPU timing repeats. It will not resubmit
training after a failure. Eight reserved temporal scenes remain untouched.
See [the frozen training registration](temporal-sharing-registration-01.json)
and [evaluation protocol](temporal-sharing-evaluation-protocol-01.json).

## Current evidence and next comparison

All six endpoint/reuse fits and five balanced CPU repeats now finish. Their
small-grid median complete blocks take2.86–3.64ms, meeting the combined budgets,
but every model fails full-state accuracy. Later conditioning, increment-loss
and curl-readout trials also fail accuracy; preserve their outcomes rather than
interpreting faster inaccurate states as a successful model.

The pressure-only successor independently reaches2.133ms median and4.6963e-4
worst equation residual on768 new small-grid requests. It meets the pressure
benchmark objective, with larger-grid, setup and native-acceptance limitations
reported separately. Reusing that learned component is the next full-state
factorization to qualify. The endpoint decoder and retained-feature options
remain separate comparisons with the same complete4/6/8ms budgets and all
intermediate output requirements. See [the next implementation plan](temporal-pressure-expert-reuse-plan.md).

The later pressure-expert and decoder fits improve parts of the full-state
prediction but still fail the accuracy target. The next coupled model retains
one shared encoding and cheap internal state updates, with pressure requests
and velocity outputs derived from the same predicted transport. Its actual
two-block CPU and CUDA qualification passes for K=1,2,3,4; those zero-update
checks establish implementation behavior, not fitted accuracy or CPU speed.
See [the coupled comparison plan](temporal-coupled-dynamics-plan.md).

The pressure-only successor has already met both requested benchmark targets.
That result must remain separate from a full simulation-step claim: its median
is across small grids, its 16³ and 32³ requests are slower, and native trajectory
acceptance still fails. Multi-step accuracy, rather than the aggregate warm
pressure-inference speed, is the main outstanding target.

The bounded coupled K2 fit and exact CPU restore now complete. All eight scenes
remain finite through 240 steps. Worst small-grid single-block velocity improves
from 0.1789 to 0.005251, while pressure/equation/tracer remain
0.1041/0.4291/0.01013. Long-rollout accuracy still fails. The width changes from
16 to 12 along with the dynamics architecture; this comparison does not isolate
one cause. No new CPU speed is claimed. See [the complete coupled result](temporal-coupled-results.md).

## Latest status under the combined budget

The user's clarification remains the selection rule: total generation divided
by K must be below 2 ms. There is no requirement that each internal stage or
individual horizon take less than 2 ms. A more expensive endpoint and cheap
intermediate decoder can qualify within the same total budget. Returning every
intermediate state and testing its accuracy remain required.

The initial endpoint/intermediate-decoder and retained-feature trials already
met the combined small-grid speed budgets for K=2,3,4, but failed accuracy.
Their [complete results](temporal-sharing-results.md) remain valid negative
accuracy evidence, rather than evidence against the amortized timing approach.

Later coupled K2 models improve velocity accuracy but take approximately
5.6 ms per complete two-state block, exceeding its 4 ms budget. More training
with provisional-velocity supervision does not resolve their accuracy failures.
See [the matched comparison](temporal-provisional-results.md) and
[actual CPU costs](temporal-provisional-cost-results.md).

The next representation comparison targets learned transport and smaller field
readouts while keeping shared computation. Training-only displacement labels
and fixed readouts now pass 80 native cases per device on CPU and CUDA. This
[qualification](displacement-transport-results.md) establishes no learned
prediction or inference speed. A new predictor must generate its own transport
parameters, then pass every field, intermediate horizon and own-history rollout.

The C107 pressure successor has independently met both warm small-grid targets:
2.133 ms median and 4.6963e-4 worst equation residual. Its stricter native
acceptance and larger-grid limitations are reported in
[the confirmation](pressure-confirmation-v2/results.md). That pressure-request
result supplies no complete simulation-step timing claim.


## Neural viscosity accuracy pass and measured fused cost

The managed viscosity calibration and exact CPU restore are complete. Only
243 coefficients change; all other full-state tensors remain byte-identical.
The calibration uses manufactured native labels, with no native primitive in
candidate inference. This zero-update assembly is bound to its two managed
parents and is not represented as a new full-model training Run.

The repaired K2 model passes every field and horizon on all eight development
scenes, including all 240 own-history steps. Worst combined error is 0.0033941.
The fused fixed readout preserves all 3,856 physical outputs byte for byte.
Five balanced serial repeats measure 4.7793 ms per complete small-grid block,
or 2.3896 ms per generated step. Thus accuracy passes but the 4 ms block budget
still fails; tensor-only 3.7526 ms does not satisfy the complete timing scope.
32³ costs 183.0 ms per block and is reported separately.

The next execution comparison targets forcing construction and unused pressure
input conversions while preserving all physical outputs and learned weights.
Qualify it against the complete development/own240 population before timing.
Then assess remaining encoder/displacement computation reuse. K3/K4 in the
repaired family, replicated controls, reserved240 and application/event timing
remain open. Eight reserved temporal scenes are still untouched.

[Managed calibration](viscosity-calibration-results.md) ·
[All-field accuracy](neural-viscosity-results.md) ·
[Exact fused execution and CPU costs](neural-viscosity-cost-results.md)


## First combined K2 benchmark pass and corrected-family qualification

The qualified execution path preserves all 3,856 physical outputs byte for
byte. Five balanced serial CPU repeats measure 3.9522 ms median per complete
small-grid two-state block, or 1.9761 ms per generated state. Worst all-field
and equation error remains 0.0033941 through own240. This narrowly meets the
registered median budget; repeat medians straddle 4 ms and the full spread is
retained. Slower individual grids, latency tails and production domains remain
outside this claim. Sustained application/event costs remain unmeasured.

The finite corrected-family admission batch is now running actual constructor
and gradient checks, with no new fits yet admitted. Both CPU/CUDA must pass all
64 cases before registration, four managed fits, exact CPU restores and complete
K1/2/3/4 development/own240 evaluations. The learned 243-coefficient viscosity
stage is frozen from initialization. Every fit supervises 36,000 physical
states; differing optimizer work is reported. No numerical primitive enters
candidate inference. No reserved scene is consumed and no application model
is promoted.

[Exact execution costs](fast-preparation-cost-results.md) ·
[Next family and finite gates](fixed-viscosity-family-plan.md)


## Latest combined-budget results and next comparison

The2 ms target applies to total generation time divided by K; no individual
internal stage has a2 ms limit. The corrected family retains learned spatial
features after a shared encoder, with internal updates and every intermediate
physical readout included in these measured costs.

| Generated states | Complete small-grid median ms | Amortized ms/state | Worst own240 error | Combined median target |
| --- | ---: | ---: | ---: | --- |
| 2 | 4.1240 | 2.0620 | 0.00339665 | Misses speed |
| 3 | 5.5742 | 1.8581 | 0.00353177 | Passes |
| 4 | 7.4335 | 1.8584 | 0.00336772 | Passes |

All eight development scenes pass every original equation/pressure/velocity/
tracer criterion, every output and own-history intervals through240steps.
Latency is the warm median over three small geometries; per-grid/tails,32³,
setup/RSS, reserved generalization and sustained application costs are separate.
The C107 successor independently passes its prepared-pressure benchmark at
2.133 ms and4.6963e-4 worst residual; stricter native acceptance remains unmet.

The earlier-composition paired precision comparison completes: float64 measures
4.1750 ms/block and float32 learned-map arithmetic3.7069 ms/block, with actual
float32 own240 worst error0.00339527. Every numerical worker reaps; the retained
read-only publication-order exception is recovered without replaying measurements.

Next qualify float32 arithmetic on the actual corrected K2weights across all
eight scenes/own240 and time a new five-repeat paired comparison. That evaluation
is running under its own frozen protocol; old-composition or short interface
results cannot establish the new model's accuracy. Then select up to two
bundled recipes for two additional seeds, including a replicated one-step
control. Preserve the original primary seed and freeze weights/arithmetic
before reserved capture and application/event timing. Endpoint-plus-small-decoder
remains a separate architecture follow-up if a measured gap persists; its early
speed pass and accuracy failure remain reported.

[Actual family costs](fixed-viscosity-cost-results.md) ·
[Earlier-composition precision costs](float32-map-cost-results.md) ·
[New corrected K2 precision protocol](fixed-viscosity-k2-float32-evaluation-protocol-01.json)


## Corrected K2 precision pass; concrete seed confirmation begins

The actual corrected K2float32-map policy passes every original criterion on
all eight scenes, every intermediate state and own240, with worst error
0.0034031815. Its five-repeat complete small-grid median is3.5445 ms/block,
or1.7723 ms/generated step. The same-campaign float64 baseline is4.0838 ms.
K3/K4retain their qualified float64 development passes at5.5742/7.4335 ms/block.
Thus all requested K now pass development accuracy and the combined median
budget. Per-grid/tail/32³, setup/RSS, reserved evidence and application cost
remain separate; this is not a final whole-plan success or model promotion.

The first corrected-K2publisher failed before execution because its control
interpreter lacks NumPy. The registered read-only repair runs the unchanged
reporter under the qualified numerical runtime, validates the original records
and repeats no training or timing. The original timing and observer failures
remain retained. A fresh admission controller proceeds only after publication.

The concrete selection isK2float32-map andK4float64, plusB1float64 as matched
control. Add initialization seeds2026102046/2026102047 and sampling seeds
2026101923/2026101924. Each of the six managed fits keeps shared frozen learned
components, data, loss and36,000physical-target work. These are independent
cell initializations with shared component pretraining. Qualify actual CPU/CUDA
constructors and two-block gradients before fitting. The ordinary managed CUDA
fits, typed exact CPU restores and every-output/own240 evaluations are one
finite serial command list; no failed/unknown Run is automatically resubmitted.
The original primary seed is retained. K3remains a measured primary even though
its seed coverage is smaller. Reserve final weights/arithmetic before capturing
the eight untouched reserved temporal scenes and measuring application events.

[Actual K2 precision result](fixed-viscosity-k2-precision-results.md) ·
[Concrete seed selection](fixed-viscosity-replication-selection-01.json) ·
[Publication repair](fixed-viscosity-k2-precision-publication-repair-01.json) ·
[Finite admission continuation](fixed-viscosity-replication-admission-batch-specification-02.json)
