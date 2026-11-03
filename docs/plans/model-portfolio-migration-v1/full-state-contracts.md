# Reviewed full-state execution contracts

These are reviewed family-level contracts, not complete coverage of the 100
expanded variants. The pressure-only benchmark is a separate task. All families
below return pressure, three staggered velocity fields and tracer; pressure is
float64 and the transported fields are float32. Preserve exact density, domain
dimensions, physical output interval, K and actual internal work.

| Family and source | Actual input/output contract | Applicable first comparison | Important distinction |
| --- | --- | --- | --- |
| Expanded shared-pressure generator, `expanded_cuda_generator_v1.py` | Prepared geometry; complete resident CUDA fields; scene; start; K=1–4; explicit prescribed controls; N=1 or 8. Output interval is fixed at 1/60 s. | Existing expanded native blocks and original N1/N8 reference schedules. Compare the complete native transfer entry separately from resident execution. | K outputs require K times N learned advances. A low-level CUDA method is not a complete CPU-input/CPU-output timing boundary. |
| Cell-coordinate graph/native entry, `cell_coordinate_pressure_graph_v1.py` | Inherited complete native contract, K=1–4 and N1/N8. Build 21 changes coordinates on N8; N1 keeps the old branch. | Frozen 1,018-output selection already replayed exactly. The g025 quoted anchor is N1. | Same trained coefficients do not imply equal arithmetic. The quoted K4 case still performs four advances. |
| Temporal eager/graph, `temporal_frozen_backbone_v2.py` and `temporal_frozen_graph_v1.py` | Explicit output rate 30/60/120; K=1–4; fine-grid start and positive group widths partitioning each whole output interval; complete current state and known controls. | The admitted two-master temporal protocol, keeping fine-reference, half-pass and single-pass schedules distinct. | K outputs span K/output_rate seconds. Internal advances are K times the number of groups, not K divided by a frame rate. Source support for a reference cadence is distinct from output cadence and from an actually qualified case population. |
| Three-/four-stage student, `macrostep_student_v1.py` | Fixed 1/60 s outputs, K=1–4, explicit teacher N1/N8 regime; managed readout component plus frozen backbone. Native entry includes upload/download/synchronization. | Existing 1,018-output student selection, scoring every output and unchanged-input/common-prefix checks. | On teacher N1 it keeps one stage per output. On N8 it uses three or four stages per output. “Student3/4” names internal stages, not a three-/four-frame endpoint interpolator. |
| Internal residual S1/S2/S3/S4, eager/graph/fused execution | Four 10,876-parameter correction fits; eight reviewed execution contracts. Fixed 1/60 s physical outputs, K=1–4, complete known controls/current fields. N8 uses group widths 2/2/3/1 with three residual-network calls per output. N1 retains its original single advance and bypasses the residual. | Each recorded 2,120-output consumed-case population; keep the four fit identities and execution implementations distinct. | S1–S4 are fit labels, not the number of internal stages. K outputs still span K/60 seconds and require 4K advances on the reduced N8 branch. |
| Paired interpolator, `paired_interpolator_graph_v1.py` | Source-bound N8 spatial donor interpolation using compensated float32 pairs and float64 coordinates; original N1 branch retained. | Native complete-output tests of the original source-bound arithmetic. | “Paired” describes spatial interpolation precision. It does not predict intermediate physical frames from an endpoint. |
| Remaining execution entries | Exact signatures, component parameters and physical schedule still require individual binding. | Keep their recorded evidence routes; admit each after source/parameter review. | No benchmark-complete or equivalence disposition follows merely from a shared parent or chart label. |

The expanded and student sources explicitly iterate through physical output
leads. These implementations do not implement the proposed accurate-endpoint
plus cheap intermediate-frame strategy. Their reduced-stage experiments reduce
work inside each physical advance. Native views must show physical K, stages per
output and total actual advances in separate fields.

The reviewed expanded lineage retains three explicit parameter sources:

- Parent transport/viscosity/flux material:
  `runs/full-ai-next-priorities-v1/b1-fixed-viscosity-pair-01`, checkpoint
  `fd93d97befafe3d336ef807160034b5841b0a7fef5ee3ed402c7915c191306cb`.
- Sampler material: `runs/full-ai-tight-targets-v1/sampler-head-pair-01`, checkpoint
  `b0fd9dfa81b6b85d2ae4542a8e0b9261001493b254c16ca67129a1d3eb81602f`.
- Shared spectral pressure: `runs/full-ai-data-diversity-v1/shared-spectral-pair-01`,
  checkpoint `1570880b28ed771baf1577a189ddf2951262391a98795550cd4b2e407a7bdc87`.
  Its retained identity describes four float64 coefficients fitted over 148
  training scenes. It is distinct from C107's pressure CNN.

Deployment also records quantized coefficients and build-specific arithmetic.
Retain those identities rather than labeling every execution by parent weight
hash alone. The expanded constructor initially restores a mixed-modal pressure
dependency and then deletes that model before installing the shared response.
Construction dependencies therefore need preservation even when their parameters
do not participate in the final arithmetic.

These contracts require complete known controls and reject SGS in the reviewed
paths. They are not evidence of arbitrary-boundary, nonuniform-cell or SGS
support. A consumed one-block test is not a 600-output own-history test. Pulse
timing, integration schedule and physical timestamps remain required case facts.
New Rheon cannot supply missing old-reference cases under an old identity.

[Source and parameter bindings](reports/full-state-contract-review-01.json).

The [residual binding review](reports/full-state-residual-bindings-01.json)
verifies the four original training Run identities, training/restore receipts,
weight and normalization hashes, eight implementation choices and their actual
row-level schedules. Ten diagnostic chart aliases share these accuracy sources;
that does not prove every timed call has the same execution boundary. S1 and S2
record 1,566 optimizer updates each, S3 records seven full-batch LBFGS updates,
and S4 records 24 with a direct normalized-delta objective. Their reported loss
values cannot be ranked across different objectives as a common accuracy score.

The [availability audit](reports/full-state-evidence-availability-03.json)
locates all historical evidence routes for the 100 expanded declarations and
verifies 2,632 distinct lab dependency claims and all 143 recorded protocol
references exactly. The initial audit's seven unresolved protocol references
resolve to four original files in a nested registration directory; the updated
tool and its nested-directory fixture cover that corpus. The one external
[cuFFT dependency](reports/full-state-external-cufft-01.json) also matches its
recorded hash in a separate explicit observation. Availability is not complete
execution binding or a fresh numerical benchmark.

Ten original restored [weight artifacts](reports/full-state-weight-availability-01.json)
match the claims published in model identities. Those declarations directly
cover 94 variants. The [control binding review](reports/full-state-control-bindings-01.json)
binds the other five learned controls through their original protocols and
factories; the remaining native CPU entry is explicitly a numerical reference.
These checks retain active versus construction dependencies and deployment
arithmetic as separate facts; they do not turn shared weights into equivalent
execution or establish a complete portfolio model count.

All eight reviewed residual executions now have fresh GPU accuracy replays.
Every one of their 16,960 output rows, field hashes, metrics and source/model
identities matches its original capture exactly. Each variant passes all 920
original N1 outputs. Its reduced four-group branch fails 1,147 or 1,151 of 1,200
outputs against the recorded 1e-5 diagnostics. These are consumed-case findings;
no new timing, independent confirmation or 600-output drift claim follows.
[Exact family replay](reports/full-state-residual-accuracy-reproduction-01.json).


Three [core-component source reviews](reports/core-component-method-reviews-01.json)
now explain method, purpose, training scope, observed outcome and limitations.
The shape-shared spectral pressure response has4learned float64 coefficients
trained on148scenes; its fixed cosine transforms/features are separate from its
fitted response. The local sampler has48float64 parameters trained on16scenes.
The fixed-viscosity parent has100456total/6600trainable parameters, with its
243-coefficient viscosity convolution frozen, and16training parents. These are
component-specific training populations and parameter counts, not one jointly
trained full-generator count. Captured component objective values are not field
errors or rollout drift.

The three component reviews are now adopted in the native project. Their
original receipt/configuration suffixes remain exact through the subsequent
32-review checkpoint; original scientific index membership and weights remain.
See the [current native custody audit](reports/native-thirtytwo-method-review-adoption-01.json)
and [current state](work-state-69.json). Clean retraining, all-model timing and
complete final lifecycle/organization acceptance remain separate requirements.


The common-boundary fresh GPU timing campaign is complete for four frozen
backend executions. All 24,000 measured calls / 60,000 timed physical outputs,
and 96,000 outputs including first calls/warmups, preserve their original hashes.
The two quoted cases reproduce 9.250025585850337e-17 and 9.15848991199606e-16
pressure error. Pooled median GPU latency is 0.232735ms/output for the cell
coordinate anchor and 0.411446ms/output for the 120-output/s temporal graph
anchor. Both grids contain15,360cells, with different aspect ratios, density and
integration work. These are complete NumPy-native input/output wall intervals,
not isolated kernels, CPU timings or new own-history results. See
[raw reduction](reports/frozen-full-state-GPU-timing-01.json) and
[scope/interpretation](regression-findings.md).

The prepared native block views keep fine/half/single schedules as separate
subjects sharing physical cases. Worst accuracy includes allKleads; timing
includes the wholeKblock and divides only byKphysical outputs. Reference updates
belong to cases, and actual student advances belong to observations. The new
native timing scopes are separate from the previously imported per-lead
accuracy-only packets, whose unavailable timing cells remain truthful. Native
publication, actual150-action chart workflow and after-custody acceptance now
pass. There are19native sources. See
[native custody](reports/native-frozen-full-state-timing-custody-01.json) and
[rendered comparisons](reports/frozen-full-state-timing-native-UI-01.json).
Both temporal masters are labeled train in the original captures and case rows;
the packet context's claim of a development master is a wording error, with an
explicit authored correction now adopted and read in the native overview. The scientific rows remain exact.
The separate600-output reference audit finds840missing required saved indices
per master; sparse20-second captures do not establish a complete drift test.


## Original joint temporal generators

The original protocol2K1/2/3/4generators are different from the sequential
expanded/frozen families above. They encode the starting five-field state once
through the width4compact spatial core. Joint local heads and a learned63-mode
pressure branch receive Kknown force increments and emit all Kfuture states.
Their counts are13,391/18,132/22,903/27,704parameters. A pass spansK/60seconds,
without Ksequential physical advances or an algorithmic pressure solve inside
the network. Later two-block training calls the network twice, using physical
reconstruction of the first endpoint as own history; that curriculum is not
an extra inference block.

Fitting uses16train-only scenes at8/16cells per metre, including32cubed and
rectangular9x11x11/15x6x8domains,129states per scene at1/60s. Both cadence and
known-force integration are fixed to1/60s; no separate dtchannel is present.
All80decoded F32field identities and original physical capture receipts are
verified, as are exact initial/final tensors, sample order and18,000optimizer
network calls per fit. Training pressure conversion has a recorded worst
relative L2of6.225179570786273e-8. This is a label-quantization observation,
not a fresh prediction-error result or a universal error floor. The model
calculates pressure inF32and upcasts/gauges during unpacking; F64output dtype
does not imply F64internal prediction. These methods cannot inherit the
near-exactF64component anchors or their timings.

A fresh complete-generator benchmark is now complete for these four original
joint models, superseding the pending status at review. It includes native
CPUinput fields, all packing/known-force work, transfers, Kcomplete owned CPU
output states and synchronization, reports block time and time/K, and scores every
field/lead against original native references. K1/2/3/4physical advancement and
training/current input quantization remain explicit. Original training/restore
recipes include CPUneural probes and must not run unchanged under current GPU-only
inference policy. [Exact source/array/work review](reports/captured-joint-temporal-generator-method-reviews-01.json).


## Endpoint and retained-feature prototypes

Six original K2/3/4fits implement the proposed compute-sharing alternatives.
The endpoint branch encodes current velocity/tracer and known forcing once,
predicts the final five-field state, then conditions one joint intermediate
1x1decoder on that prediction. The reuse branch retains the same spatial/global
features and generates later states through smaller latent updates conditioned
on the preceding prediction and known force. Neither receives a future label
at inference. Both exclude previous pressure from the encoder, generate fresh
pressure, and reference velocity/tracer increments to the original block state.
Two learned cosine-mode global mixers are neural channel mixing, not analytical
inverse eigenvalues or a numerical pressure solve.

Counts for endpointK2/3/4are163,068/163,220/163,372; reuse counts are163,140/
163,176/163,212. Same-K paired encoder/forcing/global initial tensors match
exactly; full initialization equality acrossKis not asserted. Six source/
recipe/input/output closures, all initial/final weights and private sample
orders verify. Each records18,000encoder calls during10,000updates; endpoint
fits generate18,000endpoints and18,000×(K−1)intermediate states, while reuse fits
record18,000×(K−1)latent updates. Those are fitting-work counts, not inference
latencies. Original16scene/F32/fixed1/60s training and own-history curriculum
match the joint-generator corpus. Equal-horizon losses do not guarantee an
especially accurate endpoint. Preserve these prototypes for fresh full-field,
complete-block GPUbenchmarks rather than discarding them from a name or target.
[Exact sharing-family review](reports/captured-sharing-temporal-generator-method-reviews-01.json).

## Joint-model GPU comparison on 64 common cases

All four original models completed two fresh CUDA process blocks each, with
20 timed samples per case, 5,120 timed block calls and 12,800 physical outputs.
The boundary includes native CPU inputs through all K owned CPU outputs.
All repeated output hashes agree, including across the two process phases.
No competing GPU compute was observed at worker boundaries; continuous
hardware exclusivity is not proven. Timing below is the median of case medians.

| Model | K | GPU ms/block | GPU ms/output | Worst pressure relative L2 | Worst velocity relative L2 | Worst tracer relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint b1 | 1 | 3.168 | 3.168 | 1.018 | 0.741 | 0.00859 |
| Joint b2 | 2 | 2.991 | 1.496 | 1.039 | 0.834 | 0.0173 |
| Joint b3 | 3 | 3.025 | 1.008 | 1.329 | 1.000 | 0.0260 |
| Joint b4 | 4 | 3.160 | 0.790 | 1.492 | 1.000 | 0.0248 |

These aggregate timings mix four shapes and two grid densities. On just the
16 cases at 32×32×32, GPU ms/output is 3.575/1.745/1.141/0.889 for K1/2/3/4.
The full report separates density, dimensions, their combination, and start.
The reference is the retained native temporal integrator using external full
spectral pressure on its actual requests; it is not a fresh native-PCG timing
or new-Rheon comparison.

The largest pressure errors occur at start0. Independently unpacking the
original saved CPU probe arrays reproduces first-case GPU field-error metrics
within the declared tolerance; no CPU neural forward was executed. This supports
that the large initial-case errors are retained model behavior, rather than a
new adapter-only discrepancy. That crosscheck covers one case per model, not
CPU/GPU equivalence over all 64 cases. Errors greatly exceed the recorded F32
label conversion error; precision alone is not an established cause.

All 16 scenes were consumed in fitting. These are diagnostics, not independent
generalization or 600-output drift qualification. The quoted F64 anchors still
need this exact same-case full-state comparison, as do the six sharing models.
No native publication of these new measurements has occurred yet.
[Full reduction](reports/joint-generator-GPU-benchmark-01.json),
[saved-probe crosscheck](reports/joint-generator-retained-probe-check-01.json).

## Fixed-cadence scaling and matched supervision follow-ups

Two K2 fits preserve the corresponding endpoint/reuse architecture and all
seeded initial tensors while changing fixed input/output units. Prescribed
forces are divided by training RMS0.021354100957838906; raw velocity/tracer
increments are multiplied by horizon-specific training increment RMS before
addition to the original block state. Physical field normalization and fresh
pressure output stay unchanged. Parameters remain163,068(endpoint) and
163,140(reuse). Independent NumPy decoding verifies all80archived training
fields and reproduces all16parent and aggregate horizon1–4increment statistics
plus the prescribed-force scale. Labels supply aggregate training statistics,
not future query inputs. The fixed JSON statistics are required constructor
assets outside the NPZ tensor inventory; weights alone cannot reproduce these
models. [Scaling evidence](reports/captured-step-scaled-temporal-generator-method-reviews-01.json).

A further matched K2endpoint pair keeps the same163,068-parameter network,
initial weights, data, order, budget and reconstruction. Both replace the
earlier logarithmic objective with MSE. The control uses relative full-state
velocity/tracer squared errors; the other divides mean squared full prediction
errors by fixed typical-increment RMS squared. Pressure/equation and physical
terms match between those two arms. The numerator still includes accumulated
full-state error, including the second own-history block. Comparing the pair
isolates denominators; comparing either with its earlier logarithmic ancestor
changes the robust objective too. [Matched supervision evidence](reports/captured-increment-supervision-generator-method-reviews-01.json).

All four still use the original16consumed scenes,2064states,8/16cells per metre,
four grid shapes and fixed1/60s outputs. K2advances2/60s. They do not establish
variable timestep or longer-duration training. All10,000updates/18,000encoder
calls, recipe/source/input/output identities and complete initial/final tensors
verify. Fresh common GPU measurements, own-history drift, clean GPU-only
retraining and new-Rheon data reproduction remain pending. These four source
reviews are not yet published into the native guide.

## Complete generator source coverage

All35recorded generator fits are now source-reviewed. This is method/provenance
coverage, not completion of their common benchmarks or reproduction. The20new
reviews preserve complete initial/final inventories, original inputs/outputs and
actual sample/work budgets. [Coverage](reports/review-coverage-83.json).

Four early controls separate a16-update K4cost pilot, raw-tensor second-block
feedback, corrected deployed feedback, MSE losses and larger global capacity.
The pilot is not a10,000-update competitor. The larger model changes capacity
and architecture together. Comparing its loss scalar with a different objective
does not establish an accuracy win. [Early controls](reports/captured-early-temporal-control-method-reviews-01.json).

The coupled K2parent and two provisional continuations share neural flux and
tied momentum coefficients, with a learned pressure map at every physical output.
The two4,000-update continuations start from the exact parent checkpoint and
fresh Adam, differing only in auxiliary weight. Both compute auxiliary labels,
including the zero-weight control; they do not incur different inference work.
No historical Trial ancestry is inferred from the weight proof.
[Coupled and provisional fits](reports/captured-coupled-temporal-generator-method-reviews-01.json).

Nine frozen-viscosity fits each predict36,000physical training states while
varying K, optimizer-update count and encoder work. Same-seed initial tensors
match across available K; model and sampling seeds both change between repeats.
All have100,456total/6,600trainable/93,856frozen coefficients. Learned displacement
is reconstructed with a fixed convex carrier readout; the numerical displacement
teacher is used only after prediction during training. Pressure/transport readouts
still repeat per output. [Nine fits](reports/captured-fixed-viscosity-generator-method-reviews-01.json).

The pressure-expert endpoint, combined lead/gauge/convex-tracer decoder and
trainable-viscosity displacement parent expose different precision/readout work.
One expert per endpoint block differs from Kexpert calls in recurrent designs.
Accurate pressure for a predicted request cannot certify the request itself.
[Expert/readout fits](reports/captured-expert-readout-generator-method-reviews-01.json).
The curl variant adds no learned coefficients; ideal divergence-free increments
preserve existing divergence and can incur F32rounding residuals. Its independent
NumPy identities do not substitute for actual full-model GPU tests.
[Curl fit](reports/captured-curl-temporal-generator-method-review-01.json).

Sixteen original inherited-pressure constructors perform neural work on CPU
while generating nonpersistent unit-response matrices, before .to(device).
They must not run unchanged under current GPU-only inference policy. A setup
variant needs device guards before construction, exact learned coefficients,
reference/probe checks and separate setup/memory reporting. Existing accepted
GPU composition backends have their own different constructors and evidence.
[Readiness and required qualification](reports/temporal-GPU-construction-readiness-01.json).

The 40-note native publication of 37 reviews is complete. Exact custody checks
preserve original note histories and all non-note scientific records. The guide
has 116 summaries and 696 literal paragraphs; actual UI verification passes
617 actions and 46 complete HTTP 200 reads. Twelve representative screenshots
were viewed out of 40 captured. [Custody](reports/native-116-method-review-adoption-01.json),
[UI acceptance](reports/native-116-review-UI-01.json).

## Guided temporal correction components

The guided shared and independent-pressure fits train 1,724 and 1,732 F32 residual
coefficients around a frozen learned guide. Those counts exclude the inherited
generator and its setup/cost. One shared encoder conditions endpoint/intermediate
readouts on current and guide fields, physical horizon/spacing, ordered prescribed
controls and event offsets. Guide evaluation is outside autograd. The independent
arm adds two four-coefficient pressure-RHS heads and uses two correction-pressure
responses per output; the shared arm ties pressure to the velocity correction.

Both consume the same 24 independent training scenes, 2,125 retained native states
and 240 scene/start windows. Every full score covers 2,880 blocks and 7,200 outputs:
30/60/120 outputs/s crossed with K=1/2/3/4. The fitting bank includes 8/10/16/25/50/100
cells per metre, 16 shapes through 64³ and rectangular 128-cell axes, four force
profiles and three motions. Targets advance by whole output intervals on the
240-step/s reference lattice; K=4 at 30 outputs/s spans 4/30 seconds.

Four complete-gradient iterations evaluate eight parameter trials and accept
four mean-loss-decreasing candidates. These are quadratic ray trials, not eight
Adam updates. Selection ranks the worst error across all fields before average
loss. The pressure arm selects probe-3 (worst pressure 0.442258, equation 0.572743)
while mean-best candidate-4 has worst pressure 0.157262 and equation 0.572892.
The selected probe was evaluated but not accepted as the continuation state.
Neither passes the original target. All numbers are retained training-population
scores with frozen denominator floors, not newly measured pressure-only accuracy.

All captured source members, original initial/final checkpoint tensors, restore
artifact bytes and the shared 2.81 GB archive hash verify in place. Native field
framing and temporal target indices verify; field payloads were not decoded.
Three focused positive/negative tests cover recorded-score selection and balanced
aggregation. No Torch/model import, neural forward, simulation or project mutation
occurs. Historical CPU restore checks typed weights and copies GPU probes without
CPU neural inference. Clean recipe reconstruction/replay and new-Rheon generation
remain unqualified. [Source review](reports/captured-guided-temporal-component-method-reviews-01.json).
