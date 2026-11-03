# Full AI experiment progress

Latest status: the fixed learned-tail primary independently confirms the
prepared-pressure target at **2.133 ms median CPU inference and 4.6963e-4 worst
equation residual**. All 768 new small-grid requests pass; both continuation
seeds independently pass too. C107 receives the same execution optimization and
misses accuracy. Source, weights and primary selection froze before capture.
See [the independent result](pressure-confirmation-v2/results.md).

The [causal full-state pressure-expert follow-up](temporal-pressure-expert-reuse-plan.md)
now implements both requested sharing architectures forK2/3/4. Actual CPU/CUDA
constructors pass all48 cold/warm four-grid cases per device, including repeated
two-block gradients and one fluid encoder/pressure-expert call per block. The
first bounded K2 endpoint managed fit, exact CPU restore and full
development/240-step evaluation complete under the owned
`temporal-pressure-expert-batch-01` controller. It has19,009 trainable parameter
elements plus frozen learned components. Prior request/pressure training is
explicitly additional. The [completed result](temporal-pressure-expert-results.md)
improves small-grid worst single-block pressure/equation/velocity errors to
0.2564/0.4727/0.3911, but remains far outside0.004 and becomes severely unstable
by240 steps. The declared comparison audit passes; all frozen coefficients
remain unchanged. Predeclared train-only oracle/gauge/zero-tracer controls reveal
decoder limitations, including cold velocity error with exact anchor pressure.
No new CPU latency, model promotion or reserved evaluation is claimed. See
[the next decoder priorities](next-decisions.md).

The [combined decoder comparison](temporal-decoder-results.md) also completes
its managed fit, exact CPU restore, every-horizon development/240-step evaluation
and declared control audit. It reduces worst small-grid single-block velocity
to0.1789 and tracer to0.009518, with finite own-history outputs throughout240
steps. Pressure/equation remain0.2688/0.6178 on single blocks and all accuracy
criteria still fail. Trained gauge/zero-tracer probes pass exactly. Its new
weights have no measured CPU latency. Eight reserved scenes remain unopened.

Independent [offline learned-momentum and pressure-sum controls](tied-momentum-results.md)
now identify accumulated pressure as a specific temporal representation issue.
They use true future pressures only as diagnostic labels and supply no causal
prediction claim. The [next architecture plan](temporal-coupled-dynamics-plan.md)
keeps all requested horizons and targets while coupling learned transport,
flux, pressure and momentum.

This is warm inference on the heterogeneous small-grid mix. Per-grid results,
32³, cold setup, memory and unchanged native rejection remain separate. Every
model rejects the first native step on all 16 scenes under the stricter gates;
there is no application-throughput or production-readiness claim.
The first causal pressure-request fit and a factored follow-up are complete,
with exact CPU restores and all 1,984 reference-history queries each. Factoring
reduces worst small-grid development pressure error from 5.2160 to 0.64678 and
passes cold starts, but still fails later evolution. These pressure-only
components do not satisfy a full-state temporal target. See the
[factored result](factored-pressure-request-results.md).

Both requested sharing architectures completed all six10,000-update managed
fits, exact CPU restores, full development evaluations and240-step own-history
checks. The full comparison-control audit passes, including common sampling,
physical normalization, objective, feedback curriculum, paired encoder
initialization and counted work. All seven timing workers completed and were reaped.

Every model fails accuracy. All six meet the registered small-grid median
per-generated-step speed budget, with complete block work included:

| Model | Median block ms | Median ms/step | Worst single-block error | Worst240-step error |
| --- | ---: | ---: | ---: | ---: |
| b2-endpoint | 2.949 | 1.475 | 1.63989 | 2.51743 |
| b2-reuse | 2.864 | 1.432 | 1.74979 | 2.43611 |
| b3-endpoint | 3.113 | 1.038 | 1.91647 | 2.57936 |
| b3-reuse | 3.202 | 1.067 | 1.89237 | 2.24286 |
| b4-endpoint | 3.419 | 0.855 | 1.68954 | 2.22623 |
| b4-reuse | 3.645 | 0.911 | 1.75717 | 2.27101 |

Worst error above is the maximum equation/pressure/velocity/tracer relative
error across every scene and physical output horizon. All must be at most0.004.
These are heterogeneous small-grid medians, not per-call guarantees. Across the
six models, full-block p95 values range from4.88 to6.37ms and observed maxima
from4.99 to9.26ms; some calls exceed their combined budget. The separate32³
medians remain11.44–19.66ms per generated step, outside the2ms target.
See [all errors, timings and controls](temporal-sharing-results.md).

The [step-unit conditioning comparison](temporal-step-scaled-results.md) also
completes both managed fits, exact CPU restores and full development checks.
Its audits verify identical initial tensors, architecture, sampled stream,
training work, physical normalization and objective against each corresponding
unscaled control. Worst small-grid pressure error improves to 1.23846 for the
endpoint model and 1.03753 for reuse, but velocity error stays near 1.0 and
240-step tracer error remains approximately 2.59. Neither recipe qualifies.
CPU latency remains unmeasured for these two new fits.

The predeclared gate admits [matched supervision experiments](temporal-increment-supervision-plan.md)
because no conditioning recipe qualifies and the fixed cold-training velocity
median remains 0.9991. Both new arms use the same scaled K2 endpoint model,
initial tensors, data, sampling and curriculum. One uses established full-state
relative MSE; the other changes only velocity/tracer prediction-error units to
fixed aggregate native-step RMS. Their pressure, equation and physical terms
match. This distinguishes increment supervision from replacing robust log loss
with MSE. Actual-constructor CPU and CUDA qualifications pass all four grids,
cold/warm starts and deterministic two-block gradients with zero updates.
Both [matched MSE supervision fits](temporal-increment-supervision-results.md)
complete 10,000 updates, exact CPU restores and all-horizon development/240-step
checks. Their audits verify identical initial tensors, architecture, sampled
windows, counted work and training loop apart from objective construction.
Increment weighting improves cold training velocity median from 0.8418 to
0.3664 and warm tracer median from 0.00190 to 0.000677. However, worst small-grid
single-block velocity remains 0.8224 and equation error worsens from 0.7731 to
1.1538. The 240-step tracer worst error remains 2.1956. Neither new recipe
qualifies; CPU latency remains unmeasured for their new weights.

A [generative curl velocity readout](temporal-curl-plan.md) is the next matched
representation trial. Its three learned velocity channels become edge
potentials, reconstructed through a fixed linear staggered curl layer inside
the model. It does not project an independently predicted velocity or solve
pressure numerically. Existing input divergence is retained, and float32
interchange has roundoff. Pressure/tracer outputs and every intermediate state
remain required. It has the same 163,068 trainable parameters, identical initial
tensors and initial physical outputs, the same increment-MSE objective, data,
sampling and two-block curriculum. Fixed readout arithmetic is additional work.
Independent impulse/divergence/wall/small-grid rank checks and actual two-block
CPU/CUDA qualifications pass with zero updates. Its managed fit, exact CPU
restore and complete development evaluation now finish. The control audit
passes. Divergence decreases strongly, but worst small-grid single-block
velocity error worsens to1.2467 and240-step tracer error to2.3915. This is a
negative accuracy result; new-weight CPU latency remains unmeasured. See
[the completed comparison](temporal-curl-results.md).

The next preparation step ports the successful frozen learned pressure expert
into a batched, differentiable component. The frozen coefficients and learned
geometry-head responses have been exported without reference data or optimizer
updates. [CPU/device parent parity and input-gradient qualification](temporal-pressure-core-port-results.md)
now pass on all four grids and both spatial densities, with128 native requests
and32 gradient cases per device. The maximum parent pressure difference is
1.24e-8. Two failed attempts remain retained; tolerances were not loosened.
This is not a temporal model or a new accuracy/latency claim. Both requested sharing architectures and complete
4/6/8ms block budgets remain in scope; see [the next implementation plan](temporal-pressure-expert-reuse-plan.md).

The [original plan](plan.md) and [registration](registration.json) remain the
scope of the active experiment. The confirmed multi-step timing target is
**less than 2 ms per generated physical step**, including encoding, prescribed
forcing construction, all output heads and physical field reconstruction.
Report full block latency too: K=2 below 4 ms, K=3 below 6 ms, K=4 below 8 ms.
The registered accuracy criterion requires equation, pressure, velocity and
tracer errors each at or below 4e-3. Application drift limits remain diagnostic.

## Pressure: a replicated improvement, with application limits

All four 1,200-update C107 continuations, both conditional architecture fits,
their exact CPU restores, historical evaluations and five-repeat timings are
complete. The best continuation reached 9.8744e-4 worst equation error, missing
7e-4. Channel redistribution was less accurate. A single correction stage was
faster but substantially less accurate. Negative outcomes remain in the report.

Residual analysis found that approximately 97% of the hardest historical
request's final squared residual lay in the lowest four cosine modes per axis.
An offline analytical control confirmed that those modes offered useful
headroom. That numerical control is excluded from the candidate frontier.

Three actual managed CUDA fits calibrated C107's existing 1,198-parameter global
head against offline modal teacher labels while freezing its 92,560 fine-network
parameters. No physical validation fields were used as teacher inputs or labels.
All three fits completed 10,000 head updates and independent, exact CPU restores.
Their historical worst errors were **4.6812e-4, 4.4639e-4 and 4.7026e-4**.
These are continuation/calibration seeds from a fixed C107, not independent
pretraining runs.

The plain calibrated heads regressed at 128³. The registered routed assembly
therefore retains both learned heads and the shared original fine CNN: use the
calibrated head only when every axis has at most 16 cells, and original C107's
head otherwise. Selection uses grid counts alone. This is entirely neural
pressure inference, with no numerical inverse or corrective projection. It has
94,956 parameters, a 1.28% increase. Its assembly is explicitly local, with zero
new optimizer updates; the parent components have managed training and CPU
restore provenance. Each assembled seed passed 196 fresh CPU checks proving
bitwise equivalence to the selected parent prediction.

The three weights, eager implementation and primary seed selection were
[frozen before capture](pressure-finalist-selection.json). All 16 independent
Track A scenes then produced 64 native reference pressure requests each.
On the **768 small-grid requests**, all three seeds pass 7e-4, with worst errors
**6.7324e-4, 6.6567e-4 and 6.7036e-4**. C107 reaches 1.4713e-3 and passes 764
of 768. On the separate **256 requests at 32³**, the routed models retain C107's
4.0459e-4 worst error. These independent pressure scenes do not consume the
temporal reserved bank. Five balanced CPU repeats measured the frozen primary
at **4.542 ms** median on the fresh small-grid mix, versus **4.494 ms** for C107.
The accuracy target passes; the strict 4 ms comparison does not. The separate
32³ medians are 76.43 ms for the primary and 74.83 ms for C107. These populations
differ from the historical chart and are reported separately.

Every model still rejects on the first step of all 16 fresh native trajectories
under the unchanged, stricter application acceptance gates. There is no fallback
or accepted-trajectory speed claim. Prepared-request accuracy success does not
establish application readiness or low drift over accepted simulation steps.

Frozen fine-network CPU traces passed output, gauge, label-independence and
cache invalidation qualification. Historical timing gains varied across seeds.
The reserved selection keeps eager execution; compiled timings remain a separate
comparison, including setup and trace preparation costs.

## Temporal: sharing is fast, but the first models are inaccurate

Native restart qualification reproduces eight subsequent reference steps
bitwise. All 16 training and 8 development scenes have dense 128-step captures;
development references extend to 240 steps through qualified restarts. The
8 reserved temporal scenes remain unopened until a finalist qualifies.

A pressure-gauge mismatch in the original temporal training feedback was fixed
before the corrected primary screen. The old completed B1 fit is diagnostic
only; its original queue stopped before B2 admission and reaped its children.
Corrected feedback matches deployed encoding exactly and preserves gradients.

All corrected K=1,2,3,4 fits, exact CPU restores, single-block evaluations and
own-history rollouts through 240 steps are complete. All fail the accuracy
criterion substantially. Worst single-block equation errors are approximately
1.03, 1.15, 1.51 and 1.94 respectively. A fixed-window diagnosis also found
approximately 8.1% median warm-start pressure error on sampled K1 training
windows. Cold starts and autonomous rollout errors are much larger. These results
indicate a fitting problem in this tested family, not merely an unseen-scene gap.

Five balanced CPU repeats include encoding and all physical outputs. Small-grid
K=2/3/4 median full block latencies are **1.722/1.831/2.148 ms**, or
**0.861/0.610/0.537 ms per generated step**. These are heterogeneous small-grid
medians, not 16³-only or production-grid claims. At 32³, per-step medians are
17.319/12.139/10.436 ms. Fast inaccurate predictions do not constitute a useful
speedup.

Sequential K1 controls, 72 actual native control episodes, and unexpected-forcing
events are complete. Native PCG and spectral references advance 240 steps;
C107 rejects at step one. Event tests invalidate unconsumed future outputs at
step 13 and count discarded work. Temporal state uploads are explicitly
structural research transfers, not native physical acceptance.

A [controlled loss follow-up](temporal-loss-followup-registration.json) tests
one K1 fit with relative squared errors instead of logarithmic compression.
It retains the exact architecture, seeded initialization, sampling stream,
training scenes, normalization, 10,000 updates and two-block curriculum. CPU and
CUDA full backward qualification passed before fitting. Its fit started after
clean pressure timing finished and completed with an exact CPU restore. All 24
comparison checks verified identical initial weights, sampled work, model and
inputs. The change did not solve accuracy: worst single-block equation error is
1.06879, compared with 1.03072 for the original loss. Warm-start training pressure
median improved modestly, but unseen-scene pressure and velocity medians worsened.
See [the complete comparison](temporal-loss-followup-results.md). No additional
temporal seeds are triggered by either failed screen. Stronger shared spatial
and global capacity remains the next fitting hypothesis.

## Evidence and remaining work

[Results](results.md), [interactive table](results.html),
[requirement audit](requirements-audit.md), and [next decisions](next-decisions.md)
separate measured outcomes from outstanding targets. The original whole-plan
goal remains active because temporal accuracy and reserved finalist evaluation
are outstanding. No application model has been promoted.

Machine-readable receipts and controller states live under
`runs/full-ai-next-priorities-v1/`. Current follow-up controllers are
`pressure-reserved-batch-01` and `temporal-mse-batch-01`; inspect their durable
states before considering any submission. Both follow-up batches and the
`temporal-mse-report-batch-01` diagnostic batch are complete. Completed primary results are in
`temporal-training-batch-02`, `architecture-training-batch-01`,
`temporal-development-evaluation-batch-01`, `post-training-evaluation-batch-02`
and `head-replication-batch-01`. Failed attempts and their retained logs remain
available; none are silently replaced.

## Shared-global capacity comparison and clarified block budgets

The [registered shared-global comparison](temporal-global-comparison.md) has
completed CPU/CUDA qualification, its bounded K1 managed fit, exact CPU restore
and development evaluation. Its two
learned global mixing blocks feed all five fields, with approximately 171,000
parameters. Data, original objective, sampling and own-history curriculum remain
fixed. Nonzero qualification probes verify active deterministic gradients
through every parameter; those probe weights are never training initialization.

The user's [additional generation options](multi-step-generation-options.md)
explicitly allow spending most of a block budget on an accurate endpoint plus a
small learned intermediate decoder, or reusing features from the first prediction
for later steps. There is no separate 2 ms cap on an internal step or stage.
Both options retain all physical outputs and the total 4/6/8 ms budgets.

A separate pressure execution experiment is testing a compiled complete
small-grid prediction graph at frozen successor weights. It retains larger-grid
eager behavior, learned gains and the original contraction gates. Qualification
must cover outputs, physical scalars, zero residuals, cache invalidation and
label independence before any comparative CPU timing. These already consumed
cases are execution checks, not a new independent accuracy population.

## Latest independent confirmation and resulting priorities

All scheduled fits, restores, evaluations and diagnostic processes in this
iteration are complete. Complete-graph pressure execution first passed bitwise
equivalence on all 1,152 consumed requests and measured 3.71 ms on the initial
fresh small-grid population. It was then frozen for a second independently
reserved 16-scene bank.

That new confirmation measures **3.908 ms** median and **7.9859e-4** worst
small-grid equation error. It passes 766/768 requests at 7e-4; two 15×6×8 startup
requests fail. The speed target passes on median, but the combined target is
not independently established. All outputs still match eager execution bit for
bit, all 256 separate 32³ requests pass 7e-4, and native candidates still reject
at step one. Loading, compilation and [short-sequence costs](pressure-confirmation-v1/cold-costs.json)
remain separate from warm inference.

The shared-global K1 model completed 10,000 actual managed updates, an exact CPU
restore, all-window development evaluation and 240-step own-history tests. Its
comparison audit verifies the original sampled work, inputs, objective and
training loop. Some velocity and rollout errors improve, while startup pressure
remains approximately 100% wrong. A pressure-history diagnostic and native
reference controls now identify excessive reliance on copied pressure as a
specific next learning issue. See [the updated decisions](next-decisions.md).

Additional pressure controls separate model hypotheses from numerical ceilings.
Exact early four-mode gains do not solve the confirmation failures. A final
learned four-mode block improves pressure-field error substantially but still
misses the equation target once. An offline eight-mode final block passes that
bank; its inverse eigenvalues are diagnostic only. A learned replacement needs
actual training, CPU restore, complete timing and another independent test.
Neither analytical controls nor future-RHS component oracles appear on the
candidate frontiers.

## Completed learned pressure confirmation and coupled full-state fit

The later independent pressure confirmation passes both prepared-request targets:
the frozen `a-tail8` primary measures 2.133 ms median CPU inference and 4.6963e-4
worst equation residual on 768 new small-grid requests. Both continuation seeds
pass. Larger-grid cost, cold setup and native rejection remain separate in
[the evidence](pressure-confirmation-v2/results.md).

The coupled full-state K2 fit completes 10,000 managed CUDA updates and an exact
zero-update CPU restore. All five restored artifacts match byte for byte, and
all eight development scenes return every horizon through 240 own-history steps.
The finite batch completes all four commands and reaps every child.

Worst small-grid single-block velocity improves from 0.1789 to 0.005251;
P/equation/tracer remain 0.1041/0.4291/0.01013. Every rollout stays finite, but
own240 accuracy still fails. The 32³ scenes pass every error criterion through
12 steps and fail later. No new CPU speed, recipe promotion, replica or reserved
capture follows from this failed accuracy screen.

The 43 comparison checks verify custody, unchanged inherited coefficients,
data, sample stream, objective, feedback and counted target work. Architectures,
widths (16 versus 12) and internal arithmetic differ. The initial reporting guard
incorrectly required equal widths; its failed attempt and source are retained.
The corrected guard checks the new width against its unchanged registration.
No training or evaluation was repeated.

[Complete coupled results and drift curves](temporal-coupled-results.md) ·
[Current next priorities](next-decisions.md) ·
[Original immutable scope](plan.md)

## Matched continuation, actual CPU cost and transport-representation controls

Both registered 4,000-update continuations complete managed training, exact CPU
restoration, all eight development scenes and 240-step own-history evaluation.
All 25 paired controls pass. Auxiliary supervision improves worst single-block
velocity only from 0.005229 to 0.005072 and does not qualify pressure, equation,
tracer or long-rollout accuracy. [The complete results](temporal-provisional-results.md)
retain both successes and regressions.

The parent/control/auxiliary checkpoints take 5.75/5.59/5.59 ms median complete
two-state CPU blocks on the small-grid mix, missing the 4 ms budget. The
five-repeat serial campaign retains 1,920 requests, per-grid/tail costs and
worker preparation/RSS. An initial constructor-logging protocol failure produced
zero timing repeats and reaped its worker. The corrected wrapper changes
logging only. [CPU costs](temporal-provisional-cost-results.md) are separate
from accuracy and application throughput.

Offline native-displacement readouts match 256 tracer primitive outputs exactly
at float32 precision. A 128-step scalar replay with reference velocities reaches
0.001207 worst small-grid tracer error; it is not a full-state or 240-step pass.
Both controls are excluded from candidate frontiers. The CPU profile identifies
tracer decoding and field/forcing work as substantial costs; the preliminary
denormal variant is not adopted. [Current priorities](next-decisions.md) now
target a learned transport representation and CPU overhead. The original goal
and eight unopened reserved temporal scenes remain unchanged.

## Face/cell readout qualification and timing clarification

The full transport label/readout qualification completes on CPU and CUDA:
80 registered cases per device, exact native label/readout agreement and
repeatable field/displacement gradients. These are zero-update offline checks;
no learned displacement predictor, new inference timing or full-state pass is
established. [Evidence](displacement-transport-results.md) is linked in the
generated report and excluded from candidate frontiers.

The combined budget remains below 4/6/8 ms for 2/3/4 generated states, without
an individual-stage speed cap. Endpoint-conditioned intermediate decoding and
retained features are both valid architectures under this definition. Their
initial fitted comparisons pass speed and fail accuracy; the later coupled
models fail both. The pressure-only successor already passes its separate
warm small-grid speed and accuracy benchmark. See
[the updated architecture status](multi-step-generation-options.md).

## Learned point coefficients and actual displacement model

The 972-coefficient point layer completes 1,000 ordinary managed CUDA updates
from zero and an exact CPU restore. All five artifacts match byte for byte.
Published float32 coefficients pass 24 native checks across four geometries,
both densities and zero/small/unit random fields; worst relative error is
2.51e-8. The first native evaluator reached a reporting-path failure; its source,
protocol and weights were unchanged when rerun with an absolute input path.
Both batch attempts and all child reaping evidence remain preserved.

The new full-state model, actual training objective and trainer are implemented.
It shares one encoder, combines learned point/local projections, predicts
subcell displacements and reconstructs all velocity/tracer fields through fixed
convex readouts. Native-qualified training labels supervise the small curvature
signal after prediction. No training label function enters inference.
CPU/CUDA actual-constructor checks for K1/2/3/4 are underway; full-state fitting
is not yet admitted and no full-state accuracy or latency is established.

[Completed component evidence](point-sampling-results.md) ·
[Implemented architecture and next gates](temporal-displacement-plan.md)

Both actual-constructor qualifications now pass: 64 cases on CPU and 64 on
CUDA, covering K1/2/3/4, every geometry/density and cold/warm two-block feedback.
The implemented model has 100,456 parameters, with 6,843 trainable. Registered
qualification verifies shared computation, repeatable active gradients,
frozen coefficients, staging and inference with training labels disabled.
[Full qualification evidence](temporal-displacement-qualification-results.md)
does not establish a fitted forecast or CPU speed.

The frozen K2 Project is captured and its finite managed batch is launched.
Training Run `52851a02f7cf4162b983a5be029f71a0` is admitted and observed running.
The batch owns 10,000 updates, exact CPU restoration and complete development
plus own240 evaluation. It does not retry failed or unknown mutations. New
model accuracy, CPU latency, finalist selection and reserved capture remain
unestablished.

## Displacement result and CPU-cost pipeline registered

The same managed Run is observed progressing through the two-block phase.
Its training sources, Project material and original plan/registration remain
unchanged. A dependent finite batch now waits for complete fitting, exact CPU
restoration, every-output/own240 evaluation and child reaping. It then runs
the fixed train/development fitting diagnosis, custody/accuracy audit and
five balanced serial CPU repeats against unchanged coupled and auxiliary
checkpoints. No additional training is submitted by postprocessing.

CPU timing uses the same 64 fixed development windows per worker, with full
physical encoding/forcing/reconstruction and separate tensor scopes. The new
model's complete budget remains 4 ms for K2. Larger-grid, cold setup and RSS
costs remain separate, and fixed-window block cost is not sustained application
time. The generated report can join actual restored-weight accuracy and costs
when available; incomplete-interval errors remain unavailable and are excluded
from scientific frontiers while retaining raw records.

The reused evaluation core contains a legacy static loader stamp. This is
recorded before evaluation and preserved as legacy metadata. The actual new
loader is bound independently through the frozen wrapper, source hashes,
model identity and exact restored weights; no evaluation math or population
changes. [Timing registration](temporal-displacement-timing-registration-01.json)
and [postprocessing registration](temporal-displacement-postprocessing-registration-01.json)
fix the upcoming checks before their results are observed.

## Completed displacement results and identified remaining component

The K2 fit completes all 10,000 updates, exact CPU restoration, all eight
development scenes and complete every-output/own240 evaluation. All comparison
controls pass. Small-grid single-block worst equation/V/T are now
0.01642/0.00014197/0.00001950; own240 equation/V/T are
0.01992/0.007352/0.005568. The five-repeat CPU campaign measures 6.350 ms per
complete block, or 3.175 ms per generated step. Both target gates still fail.
All timing workers and parent batches finish and reap. Reserved scenes remain
unopened, and no new finalist is selected.

New train-only controls reproduce normal metrics bitwise and leave candidate
coefficients unchanged. Native displacement alone barely changes fluid errors.
Native viscosity reduces warm worst equation error from 0.01098 to 0.000659;
on the cpm16 odd-grid 128-step training replay it reduces equation error from
0.01166 to 0.000566 and velocity from 0.003994 to 0.00003465. These numerical
controls remain offline and do not establish a new full-state/own240 pass.

The public viscosity primitive and exact dyadic label fixtures pass 56 cases.
Its first helper build failed before any numerical query because an axis
constant is crate-private. The retained v2 source uses the identical public
axis enumeration; primitive mathematics and captured Rheon sources are
unchanged. Both build attempts and terminal child evidence are retained.

The actual CPU stage profile preserves predictions and weights. Its inclusive
small-grid block medians include 1.59 ms in the fixed field readout, 0.75 ms
in displacement projection and 0.94 ms in preparation/reconstruction. These
overlapping instrumented costs are excluded from frontiers. Next work is a
managed calibration of neural viscosity and a separately qualified fused
readout implementation, followed by unchanged full-state and CPU gates.

[Accuracy and custody](temporal-displacement-results.md) ·
[Complete CPU costs](temporal-displacement-cost-results.md) ·
[Component controls and profile](displacement-component-results.md)


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


## Corrected-family training and bounded float32 execution comparison

All64actual constructor/gradient cases now pass on CPU and CUDA for K1/2/3/4.
The four Projects and finite fit recipes are captured with viscosity frozen
from initialization. The first managed K2 Run is5183fc284792458a902ec6c8ce123d4c;
K3,K4and the matched-physical-target K1control follow serially. Fit completion,
exact CPU restore and every-output/own240 results remain the authoritative gates.

The dependent postprocessing campaign is registered and waiting for that same
controller. It audits all four fits and actual36000-state work, qualifies every
physical byte for fast execution, and runs55balanced CPU timing responses
(3520blocks), including sequential2/3/4-call B1controls. It submits no new fits.

A separate zero-update float32 learned-map execution now passes every field,
horizon and own240 criterion on all eight development scenes. Worst small-grid
combined error is0.00339527, against0.00339409for the float64 parent. This changes
rounding and uses a new full evaluation; it does not inherit exact output bytes.
No inverse, numerical pressure solve or parameter learning is introduced.

Its1280-block CPU comparison is registered and waiting for the entire original
family/CPU campaign to finish and reap. Do not start seed-replication numerical
work while either CPU comparison is active. Select recipes after both actual
cost reports are available, then qualify replications and reserve final weights
and execution before capturing the eight untouched reserved scenes.

[Corrected-family work and gates](fixed-viscosity-family-plan.md) ·
[Actual float32-map accuracy](float32-learned-map-results.md) ·
[Registered family postprocessing](fixed-viscosity-postprocessing-registration-01.json) ·
[Registered float32 costs](float32-map-cost-registration-01.json)


A publication-order defect was identified before the float32 CPU campaign runs:
the original reporter expects its parent controller to be complete while running
inside that controller. A separately registered dependent publisher now waits
for the original owner to reap, checks that both numerical commands succeeded,
and accepts only that exact retained read-only publication exception. It then
publishes the existing timing records without repeating training or timing.
The original registered sources and command list remain preserved.
See [the bounded publication repair](float32-map-publication-repair-registration-01.json).


## Corrected K2 managed fit and complete accuracy pass

K2 Run `5183fc284792458a902ec6c8ce123d4c` completes all10,000updates. CPU restore
`35c5d3f6de734f268644ae39b8373b33` makes zero updates and reproduces all five published
artifacts byte for byte. The actual training work is18,000block forwards and
36,000physical state targets; setup takes74.53seconds and the fitting loop
1,021.48seconds. The managed viscosity coefficients remain frozen from the start.

The complete development evaluation passes every equation/pressure/velocity/tracer
criterion, every intermediate output and all240own-history steps on all eight
scenes. Worst combined error is0.0033966464. New-weight CPU latency is not yet
measured; the previous composition's timing is not inherited as a new result.
K3,K4and the one-step control continue through the existing finite controller.
The registered comparison and float32 timing/publication dependencies remain
owned; do not duplicate fits or overlap new numerical work with CPU campaigns.

[Full K2 raw summary](../../../runs/full-ai-next-priorities-v1/fixed-viscosity-evaluation-01/b2-fixed-viscosity/summary.json) ·
[Exact managed fit and CPU restore](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-pair-01/receipt.json)


## K3 accuracy pass and completed application preparation

The independently trained corrected K3model completes its managed fit, exact
CPU restore and all eight development scenes. Every output/field and own240
criterion passes; worst combined error is0.00353177. K4is training in the same
existing sequence, with the one-step control following. New-weight CPU costs
remain unmeasured until the registered serial comparison.

Bounded native transport/restart and12neural/native mechanics cases now pass.
The latter use K2checkpoint diagnostic horizon views and do not substitute for
independently trained horizon results. All eight fixed48-step development event
references are captured; states/RHS through13remain bitwise unchanged and the
forcing change affects velocity14. No reserved scene is opened. The forecast
application loop takes only causal current state and active forcing, excludes
offline reference metrics from timing and forbids native numerical advance.

[Application preparation evidence](application-preparation-results.md) ·
[Remaining application measurements](application-evaluation-plan.md)


## All four primary accuracies pass; application/control interfaces qualify

The corrected K1fit and exact CPU restore complete all20,000updates and the
matched36,000physical targets. Its complete eight-scene, every-output/own240
worst error is0.00372123. The K2/K3/K4models also pass, at0.00339665,
0.00353177and0.00336772respectively. The full four-fit work/source/custody audit
passes. New-weight CPU speed remains pending the registered serial comparison.
Fast execution already matches2,888K1,3,856K2and4,824K3physical outputs byte
for byte; K4is the final exact-execution qualification before timing.

Actual K2/K3/K4deployment/evaluator interfaces pass12limited48-step cases
across float64/float32-map policies and steady/event scenarios. Six native
controls qualify with unchanged gates: PCG/spectral complete, and C107rejects
its first step without fallback. These checks do not establish whole-bank
precision accuracy or measured application speed. The application controller
and report implementation require concrete frozen model/policy evidence before
use. No new registry or model selection is inferred by the executable.

[Complete four-fit accuracy and controls](fixed-viscosity-family-results.md) ·
[Application interface qualifications](application-interface-results.md)


## All four exact execution proofs pass; isolated CPU comparison active

Fast execution now preserves all17,360physical outputs byte for byte across
all four actual trained models:2,888forK1,3,856forK2,4,824forK3and5,792forK4.
This includes every reference-history horizon and all own240steps on all eight
development scenes. All neural tensors remain unchanged. The existing finite
controller has entered its five-balanced-repeat CPU campaign, including the
sequential2/3/4-call one-step controls. Do not overlap new numerical work with
this campaign or the queued float32 comparison. No new-weight timing result
is claimed until the complete responses and worker reaping are verified.

Application generation, native numerical/C107 controls, deferred metrics,
qualified-model admission, the seven-method serial controller and its reporter
are implemented. Their interface qualifications have passed; the full application
campaign still requires concrete frozen checkpoint/precision choices after
seed confirmation. Generic float32 deployment admission requires its own full
actual eight-scene evaluation and cannot inherit the old composition probe or
the limited48-step interface results.


## Family CPU campaigns complete; corrected K2 precision evaluation active

All55family timing responses/3,520timed blocks and all20old-composition
precision responses/1,280timed blocks complete, with every numerical worker
reaped successfully. K3/K4pass combined small-grid median speed and every
original field/horizon/own240 accuracy criterion. K2passes accuracy but its
4.1240 ms median exceeds the4 ms block budget. The earlier-composition
float32 policy measures3.7069 ms with actual accuracy retained. Its documented
read-only publication-order failure is recovered without numerical replay.

`fixed-viscosity-k2-precision-evaluation-batch-01` now owns an actual new-weight
float32 eight-scene/every-output/own240 evaluation. A new paired timing source
and reporter are prepared; their registration must bind actual completed
accuracy receipts before timing begins. No additional fits or reserved captures
start during these numerical comparisons. The original plan and completed
source/evidence remain unchanged. See [the current decision](next-decisions.md).


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


## Added-seed fits admitted; dependent confirmation owners registered

All96actual CPU and96actual CUDA constructor/gradient cases pass with zero
updates. The finite six-fit controller now owns the added temporal-cell seeds
forK1/K2/K4. The first managed training Run is
`ae4f4d36db0b4b40ad31698109220822` (K1seed2,20,000updates). Its live progress
is provisional; only drained artifacts and the exact typed CPU restore establish
completion. Original component pretraining remains shared and frozen.

A dependent postprocessing controller waits for this same existing owner,
without resubmission. It will bind nine actual primary/replica deployments,
prove float64 physical-byte equivalence or recompute float32 accuracy for each
new seed, then measure five balanced serial CPU repeats and matched sequential
B1controls. All seeds, including unfavorable outcomes, remain in the results.
K2float32 versusB1float64 compares complete deployment choices, including both
precision and computation sharing; it does not isolate the encoder contribution.
K4andB1both use float64.

A separately registered final confirmation owner waits for all of those workers
to finish and reap. It will freeze all four primary horizons and six added
seeds before opening any reserved data, capture all eight originally reserved
240-step scenes, evaluate every output/own240 for all ten checkpoints, measure
fresh reserved CPU costs, and finally run560steady/interaction application
measurements over four actual primary horizons and three native controls.
Native numerical gates remain unchanged, model-only budgets remain4/6/8ms,
and application acceptance limits remain diagnostic. No native numerical advance
or corrective projection enters a forecast candidate. Captures, evaluations and
CPU timing remain serial. Final target attainment is unproven until those
actual results and the original29requirements are audited.

The new orchestration modules parse and import under the qualified numerical
runtime. Original plan/scope, qualified inference code and managed training
source closures remain unchanged. This prepares and owns the remaining work;
it does not claim seed confirmation, reserved success or application speed.

[Replica postprocessing](fixed-viscosity-replication-postprocessing-registration-01.json) ·
[Final confirmation registration](temporal-final-confirmation-registration-01.json) ·
[Finite final sequence](temporal-final-confirmation-batch-specification-01.json)


## First additional seed: exact restore and all-eight-scene accuracy pass

K1seed2managed CUDA Run `ae4f4d36db0b4b40ad31698109220822` completes/drains
with20,000updates and36,000physical targets. All six published artifact hashes
are verified. Independent CPU restore `25be3d3d19f54d1b901d0e8ca28a054c`
completes with all five declared artifact comparisons byte-identical.

All eight development scenes pass every original equation/pressure/velocity/
tracer criterion and every intermediate/own240output. An additional raw-record
check independently recomputes all criterion decisions, summary extrema and
output coverage for2,888physical outputs. Worst combined error is0.0033227321.
This is control-seed accuracy, not a selected-policy CPU or reserved result.
See the [raw audit](../../../runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s2-raw-accuracy-audit-01.json).

The same finite controller moves toK2seed2training Run
`38ac29a9de52464393a02213bd6fdd75`. No Run is restarted. The other five pairs,
selected execution/precision proofs, CPU confirmation, reserved scenes and
application measurements remain owned by the existing dependent controllers.
Original sources, plan, thresholds and combined budgets remain unchanged.


## Completion audit and guarded final publication

A separate current-state completion audit now checks all29original requirements.
It verifies source/weight bindings, actual managed artifact parity, the original
reserved scene split, raw per-output criteria and drift crossings, fresh CPU
budgets and every application episode's causal/process/discard accounting.
Missing results stay pending; completing an experiment cannot substitute for
passing the target. No goal status is updated by the auditor.

The report publisher now requires a current passing scientific audit for
`--final`; its guard is verified to reject the current incomplete state before
writing any report. Normal publication remains interim. The first audit has
14passed scopes and15pending, with no failed completed gates. Reserved and
application evidence is still absent because the existing controllers are
waiting for the finite managed fit sequence. No extra numerical job or fit is
started by this audit. The currentK2seed2Run remains the existing
`38ac29a9de52464393a02213bd6fdd75`.

[Current-state completion audit](completion-audit.md) ·
[Auditor source](../../../experiments/full-ai-next-priorities-v1/audit_completion.py)


## Second completed pair: K2seed2 passes every development criterion

K2seed2CUDA Run `38ac29a9de52464393a02213bd6fdd75` completes/drains all10,000
updates,18,000block forwards and36,000physical targets. All six published
artifact hashes are verified. CPU restore `d2b17d21dead470cb8264d9da98cae42`
passes all five byte comparisons and its fresh CPU probe, with zero updates.

All eight development scenes, every intermediate output and own240criteria
pass. Worst combined equation/field error is0.0035599317. The independent
raw-record check verifies3,856physical outputs, each original near-zero/error
criterion and every summary count/extremum. See the
[raw K2audit](../../../runs/full-ai-next-priorities-v1/b2-fixed-viscosity-s2-raw-accuracy-audit-01.json).
This is the original restored float64 prediction policy; the selected float32
execution still requires its separately queued complete evaluation and timing.

Two of the six added-seed pairs now have exact restores and complete accuracy
passes. The same finite owner moves toK4seed2Run
`276729eada3b43caa4a376208b1de1c8`, without resubmission. Four pairs remain.
No reserved scene is opened and no selected-policy speed or whole-goal success
is inferred from these new accuracy results. Original sources, thresholds,
training budgets and pre-reserved primary choice remain unchanged.


## Supported boundary change: fresh prediction beyond the stop guard

The originalB4scope requires unannounced boundary changes to invalidate unused
outputs and trigger a new prediction. The existing unsupported-boundary guard
stops correctly, but that alone does not demonstrate supported continuation.
A separate registered functional comparison now covers a physical closed-box
rescale at fixed logical cell counts, switching8↔16cells per metre after step13.
This changes physical boundary positions and spatial sampling density; fluid
mass density, logical domain resolution and the closed boundary type stay fixed.
Concentration/velocity arrays are copied without resampling. The externally
imposed volume/mass mapping is explicit in the offline physical diagnostics.

Mechanics smoke tests pass allK1/2/3/4with a stub predictor: every stale state
is discarded, the next model call uses the new geometry at13, all48states are
consumed and all native children reap. A separate native-reference smoke test
checks actual pressure-request spacing metadata in both8→16and16→8directions,
with unchanged numerical gates and no numerical step during restore. Neither
smoke establishes trained-model accuracy or CPU speed.

`application-domain-change-batch-01` now waits for the existing final application
owner to complete/reap. It then captures eight consumed-development reference
episodes and tests all four actual frozen primary models:32episodes and1,536
physical output diagnostics. Native pressure/RHS and errors use the geometry
active at each timestep; every call's spacing and every stale-state discard are
checked. Actual fields and geometry, rather than a future reference endpoint,
enter each fresh prediction. No new fit or numerical candidate correction is
introduced. Accuracy remains an explicit diagnostic, with no arbitrary-boundary
or new CPU-performance claim. Unsupported boundary types retain the separate
stop guard. The completion auditor now requires this actual-model continuation
in addition to the original560steady/forcing application episodes.

The original training, timing and final-confirmation registrations remain
unchanged. No reserved scene is opened by this preparation. K4seed2continues in
the same owned managed Run; there is no resubmission.

[Supported-boundary registration](application-domain-change-registration-01.json) ·
[Finite dependent comparison](application-domain-change-batch-specification-01.json) ·
[Current completion audit](completion-audit.md)


## All second-seed fits pass; third-seed control begins

K4seed2CUDA Run `276729eada3b43caa4a376208b1de1c8` completes/drains5,000updates,
9,000block forwards and36,000physical targets. All six published artifact
hashes verify. CPU restore `184fc0d32eef4a689c4562af4106c159` passes all five
actual byte comparisons and the independent CPU probe with zero updates.
All eight development scenes, intermediate outputs and own240criteria pass.
The raw audit independently verifies5,792physical outputs and every criterion,
count and summary extremum: worst combined error0.0033281110.

All three ordinal2models now have exact managed restores and full accuracy
passes: B1worst0.0033227321, K2worst0.0035599317andK4worst0.0033281110.
These are actual original-policy accuracies, with new-weight execution/precision
qualifications and CPU measurements still separately queued. No seed is selected
as a replacement primary. See the
[raw K4audit](../../../runs/full-ai-next-priorities-v1/b4-fixed-viscosity-s2-raw-accuracy-audit-01.json).

The same finite controller advances toB1seed3Run
`f2a41e2042624421bfbde13f02a1de0c`. Three of six added-seed pairs remain.
The supported physical-box boundary comparison is registered and its dependent
owner is live; both native spacing directions and all four discard/regeneration
mechanics smoke cases pass. Its actual32trained-model episodes still wait for
the existing complete application campaign. The original source registrations,
thresholds, primary and reserved scenes remain unchanged. The completion audit
also distinguishes an interim report awaiting live-owner progress from altered
completed scientific evidence; immutable result checks remain strict.


## Third-seed control completes; late equation criterion fails and is retained

B1seed3CUDA Run `f2a41e2042624421bfbde13f02a1de0c` completes/drains20,000updates
and36,000physical targets. All six training artifact hashes verify. Independent
CPU restore `e0552a87be8042c5a6e40c39fc6e7950` completes/drains with all five
actual artifact byte comparisons, typed origin/closure and CPU probe verified.

The complete eight-scene evaluation remains finite through240steps, but the
original accuracy criterion fails: worst equation error0.0043489455. Six raw
own-history records fail on `temporal-g2-cpm16-r2`, a15×6×8grid at16cells/metre,
at steps207–212. The other seven scenes, all single blocks and own12/48intervals
pass; pressure/velocity/tracer field criteria pass throughout. Global field
maxima are pressure0.00093235, velocity0.00039357andtracer0.00135467.
The independent raw audit verifies all2,888outputs, counts/extrema and both
passing and failing decisions against the unchanged limits. See the
[retained negative raw audit](../../../runs/full-ai-next-priorities-v1/b1-fixed-viscosity-s3-raw-accuracy-audit-01.json).

Four of six added-seed pairs now have exact restores and complete evaluations:
three pass accuracy and this control seed fails. The original primary is not
replaced by a better seed. This is variation of the temporal-cell initialization
and sampled windows with common frozen component pretraining; it does not
establish a failure of the still-qualified original bundled primaries. Both
unfavorable control accuracy and its future measured cost remain in the reports.

The same finite owner startsK2seed3Run
`bff7ba1f5e484b569a39c1deee04ad5a`; K4seed3follows. Their exact restores,
all-field accuracy, selected execution/precision proofs, repeated CPU costs,
reserved scenes and application/boundary episodes remain pending. No new fit,
threshold change, retry or primary reselection is introduced in response to
this negative result. The whole-plan goal remains active.

## Conditional target update registered; current confirmation continues

The user's [ordered follow-on targets](deferred-target-program-01.json) are now
recorded separately from the original immutable plan. After current success is
verified, all K=1,2,3,4 models target maximum 1.2 ms per generated step and worst
error 1e-5. Then add 50/100 cells per metre with random aspect ratios below and
above those density dimensions; retain 1e-5 and allow latency growth at most
proportional to voxel count. Finally minimize drift over 600 steps at both
60 Hz and 30 Hz. The reference-grid clarification is pending. No future-phase
fit or capture starts while current work remains unfinished.

Current K2seed3 advances within the same owned pair to independent CPU restore
`f9e4a99c97784ce9a107bbd42f09da9f`. This is live, unverified custody; terminal
drain, artifact hashes, exact comparisons and all-field evaluation are still
required. The completion audit remains 14 passed, 15 pending, zero failed.

K2seed3 now completes both managed Runs with all terminal hashes, typed closure,
CPU probe and five byte comparisons verified. Its eight-scene original-policy
evaluation passes all 3,856 physical outputs and own240 checks; worst equation
error is 0.0039455145, close to 0.004. Independent raw decisions, interval
extrema and diagnostic crossings match. Four of the five completed additional
seeds pass accuracy; the negative B1 control remains retained. Final K4seed3
Run `bee6f205d1c247e5b01fc833fd1981d3` is live in the same serial queue.
Selected K2 float32-map evaluations and CPU timing remain separate pending work.

## Final publication owner waits behind current confirmations

`final-publication-batch-01` is live with a finite, source-bound
[publication registration](final-publication-registration-01.json). It observes
the four existing training, execution/timing, reserved/application and supported
boundary owners through completion and child reaping; it never replays their
fits or numerical measurements. It then refreshes durable state and publishes
actual interim results, audits all original requirements, and permits a final
report only if the scientific gates pass. A final independent audit must prove
all requirements after publication. Any target/evidence failure retains its
actual interim report and stops publication; it cannot update the thread goal.

The prior turn made concrete progress: fifth-seed typed restore and 3,856 raw
accuracy outputs verified, its negative/positive evidence published, and the
user's conditional future targets saved without changing current criteria.
The current final K4seed3 managed Run and all four existing controller handles
are freshly confirmed live. The goal remains active, with no blocking condition.

## Six additional seeds complete; actual execution qualification begins

Final K4seed3 CUDA Run `bee6f205d1c247e5b01fc833fd1981d3` and independent CPU
restore `d36eeed79ee54dbfb22db793aa6a0f54` both complete/drain with all terminal
hashes, actual parity, typed provenance and CPU probe verified. Every fit has
36,000 physical targets; this K4 fit uses 5,000 updates and 9,000 block forwards.
Its eight-scene evaluation passes all 5,792 physical outputs/own240 criteria:
worst equation error 0.0030208889. The independent raw audit also verifies field
criteria, interval summaries and diagnostic threshold crossings.

All six added seeds and all 140 complete-family report evidence bindings now
verify. Five pass accuracy; the B1seed3 late equation failure remains unchanged.
Training/admission owners complete and reap, and the existing postprocessor
advances to new-weight exact execution and selected F32-policy qualifications.
No CPU timing or reserved/application result is inherited. Completion is now
19 passed / 10 pending / zero failed, with the goal active.

The postprocessor completes two of six actual new-checkpoint qualifications.
B1seed2 fast-float64 matches every byte of all 2,888 physical outputs on eight
scenes. Actual K2seed2 fast-float32-map passes all 3,856 outputs/own240 checks;
the independent raw audit confirms worst equation error 0.0035592525, matching
managed weights and policy, unchanged field/near-zero decisions, all interval
extrema and diagnostic crossings. K4seed2 exact execution is now active. Four
checks and the complete serial CPU campaign remain; no replica timing or new
reserved/application accuracy claim is inherited.

Third execution/policy check completes: K4seed2 fast-float64 matches every byte
of all 5,792 physical outputs on eight scenes. Its checkpoint, source/protocol
hashes, all horizons/own240 and teacher isolation independently verify. The
existing live controller advances to B1seed3 exact execution, preserving that
control's original unfavorable accuracy. Three checks remain before CPU timing.

Fourth proof completes: B1seed3 fast-float64 is exact for all 2,888 outputs;
its original six step207–212 equation failures remain retained. Fifth check
completes: actual K2seed3 fast-float32-map passes all eight scenes and 3,856
outputs/own240, independently audited against unchanged limits. Worst equation
error is 0.0039463236, close to 0.004; actual weights, policy and evidence hashes
verify. Final K4seed3 exact execution is active. One check remains before the
already-owned nine-model CPU campaign; reserved/application results are pending.

Sixth qualification completes: K4seed3 is byte-exact for all 5,792 physical
outputs across eight scenes/own240; checkpoint/source bindings verify. All six
execution/policy commands complete and reap. Four new-weight F64 proofs and
both actual K2 F32 evaluations are complete, preserving the negative control.
The existing postprocessor starts its nine-model, five-repeat serial CPU
campaign (planned 120 responses / 7,680 blocks). Numerical audits/plotting stay
idle during timing. An independent raw-timing auditor is prepared for use after
the numerical workers finish; no new performance claim is made yet.

## CPU campaign complete and independently audited; reserved evaluation starts

All 120 responses / 7,680 blocks complete, with all nine CPU workers successful
and reaped. The independent raw-timing auditor recomputes all 48 summary rows,
window coverage, actual CPU context, repeat medians/tails and per-step divisors;
all match. Every one of the 154 cost-report evidence bindings verifies.
All three selected K2/K4 seeds pass the unchanged development accuracy and
combined small-grid median budgets. K2 is 1.789396–1.833132ms/state with worst
error 0.0039463236; K4 is 1.724521–1.746983ms/state with worst 0.0033677226.
These remain warm medians; larger-grid/tails, setup and application are separate.

The existing final owner freezes all ten models with original primaries retained,
then captures all eight original reserved scenes / 1,928 states. Receipt and
frozen-selection identities verify; every native capture child exits gracefully
and reaps. The ten-model held-out all-output/own240 evaluation is now live.
The tighter target program remains deferred. Current completion stays active at
19 passed, 10 pending and zero failed; held-out and application proof is required.

The temporal figure legends visibly clipped after adding the seed comparison.
Separate renderer v2 fixes shared legend layout; all non-plot AST nodes and
all 336 scientific points/full accuracy arrays remain unchanged. Original frozen
sources are preserved. A new live presentation owner waits behind the original
publication, snapshots its artifacts, and applies the visually qualified layout
with a fresh complete audit. Goal completion must also wait for this owner.

First three reserved primaries complete. Independent raw audits verify all
12,536 physical outputs: original K2 F32 passes with worst equation0.0031881576;
K4 F64 passes with worst0.0028193108. Original B1 F64 fails equation-only at
eight own-history steps221–228 on the15×6×8,cpm16 scene; worst0.0041023157.
All its field criteria pass and all episodes remain finite/full240. Negative
control retained, no reselection. K3, additional seeds, fresh reserved timing
and full application/boundary confirmation remain pending; goal stays active.

All three ordinal2 models pass complete eight-scene reserved accuracy/own240.
Independent audits verify another 12,536 outputs: B1 worst equation0.0030934196,
K2 F32 0.0037932540 and K4 F64 0.0028546537. All field/near-zero criteria,
interval extrema, drift crossings and exact frozen weights/policies/bank bindings
verify. Six of ten models / 25,072 physical outputs are complete and audited:
five passes plus the original negative B1 control. Ordinal3 B1 is now active;
ordinal3 K2/K4 and original K3 follow. No reselection, retuning or new fit.

Reserved evaluations seven/eight complete and independently audited. B1seed3
passes all 2,888 outputs, worst equation0.0038248141, while its earlier
development failure remains retained. K2seed3 F32 passes all 3,856 outputs,
worst equation0.0035699368, with exact pre-capture weights/policy verified.
All three K2 seeds pass development and reserved accuracy. Eight models /
31,816 outputs are complete and raw-audited: seven passes plus the original
negative B1 control. Final K4seed3 is live; original K3 follows. No new fit or
selection. Fresh reserved timing and application/boundary proof remain required.

All ten reserved evaluations complete. K4seed3's independent5,792-output audit
confirms an equation-only failure, worst0.0040399529, at steps223–227 on the
15×6×8,cpm16 scene; all field criteria and finite/full240 coverage pass.
Original K3 passes all criteria, worst equation0.0030412214. Eight models pass;
original B1 and K4seed3 fail. All original requested K2/K3/K4 primaries pass
all8scenes/everyhorizon/own240 without reselection. The registered report verifies
all1,928native archives. Nine models /37,608outputs have independent raw audits;
K3's4,824records await the final independent audit after CPU work.

The evaluation owner completes/reaps; the existing controller starts fresh
reserved CPU timing for all ten models, including failures. Planned totals are
135responses /8,640blocks, five balanced serial repeats on core23. Numerical
audits and plotting remain idle. Fresh speed, application and boundary
continuation are still required; goal active, tighter targets deferred.

Fresh reserved CPU campaign completes135responses /8,640blocks with all ten
workers successful/reaped. Published original-primary targets all pass: K2
3.539019ms/block /1.769510ms/state /0.0031881576worst; K3 5.526574 /1.842191 /
0.0030412214; K4 7.097961 /1.774490 /0.0028193108. These remain warm small-grid
medians;32cubed/tails, setup and application costs are separate. Added-seed
failures remain retained, with no primary reselection.

The same final owner starts the seven-method steady/forcing application
campaign. No additional numerical audit or plotting runs during application
timing. Independent reserved timing statistics and raw K3 verification remain
pending after CPU work, together with full application/boundary and final
publication audits. Reported original model metrics pass; goal remains active.


## Original goal complete; tighter follow-on activated

The original plan is complete and the thread goal is marked complete after an
independent review of all29original requirement texts, every final evidence and
publication hash, raw timing statistics,42432reserved physical outputs,
560timed application episodes and32supported-boundary episodes. The final
figures are visually inspected and readable. The path-bookkeeping failure is
preserved: version2reuses its one validated case and runs only31remaining
cases. No training, reference capture or CPU timing is replayed. The repaired
publisher snapshots the prior interim report and publishes the complete proof
using the stronger audit and readable renderer. All unfavorable model outcomes
and the native C107 rejection remain reported.

[Independent completion review](goal-completion-review-01.json) ·
[Final requirement audit](completion-audit.md) · [Final results](results.html).

The user-authorized follow-on is now active in
[the tighter-target activation](../full-ai-tight-targets-v1/activation-01.json):
full-state K1/2/3/4at maximum1.2ms/generatedstep and worst1e-5, provisionally
using32cubedas reference from the user's example and retaining smaller-grid
regressions. These new targets are not yet achieved. Higher-density50/100data
and600-step drift optimization remain behind their original ordered gates.
