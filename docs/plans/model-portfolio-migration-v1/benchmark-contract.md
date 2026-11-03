# Common benchmark contract

Status: Draft. Owner: the lab benchmark runner. The exact all-model case manifest
and implementation bindings are not frozen yet. The existing anchor replay
protocols remain authoritative for their already executed, narrower populations.

The matrix must distinguish a training attempt, a retained parameter state and
an execution implementation. Every inventory item receives a benchmark,
an evidenced equivalence/alias link, an explicit unsupported-case result or a
visible unresolved availability gap. A missing checkpoint is not a passing case.
Intermediate and initial checkpoints remain accounted rather than silently
reducing the portfolio to selected winners.

## Comparison groups

| Task | Required model output | Primary measurements | Required contract check |
| --- | --- | --- | --- |
| Pressure proposal/component | Pressure for the exact supplied native pressure request | Same-request mean-free pressure error, equation residual, complete pressure-call latency | Boundary/operator, units, gauge, input features, shape and any spacing/timestep conditioning match |
| Pressure-request predictor | Predicted pressure RHS/request | Request error; pressure/equation error through its separately identified pressure component | Request prediction and downstream composition are separate records |
| Transport, viscosity or readout component | Its declared carriers or correction | Component-target error and work; complete-generator results when composed | Preserve the actual fitted component and its parent generator |
| Full-state generator/composition | Every pressure, staggered velocity and tracer output at each of K output intervals | Field error, equation residual, complete block latency, latency/K and own-history drift | K advances K complete physical output intervals; each intermediate output is scored |
| Execution variant | The same declared task through a different implementation | Identical task/case metrics plus actual precision, internal work and timing boundary | Equal weights do not establish output equivalence; measure it or retain differences |

Control references such as the `controls` collection's C106/C107 entries are
aliases of real models. They do not create extra training fits. A composition
with zero new full-state optimizer updates still retains the training cost and
provenance of its components.

## Existing populations to bind

| Population | Retained source | Valid evidence and limitations |
| --- | --- | --- |
| Original 128 pressure requests | Captured validation packets and per-model replay entries used by the cohort diagnostics | Consumed pressure diagnostics; preserve exact pressure inputs and operator. The metric is relative equation residual, with pressure-field error added separately when reference pressure is available. |
| Expanded native full-state blocks | `cell-coordinate-blocks-protocol-01.json` and its `macrostep-student-block-cases-01.json` selection | 1,018 emitted states, K=1–4, original N1/N8 integration schedules and explicit train/development membership. A fresh exact reproduction is retained. This does not cover every later model or all possible timesteps. |
| Temporal pilot | `temporal-frozen-backbone-qualification-protocol-03.json` and the two audited long-reference masters | 540 emitted states across the original three schedules, 30/60/120 outputs per second, K=1–4 and three starts. These are two consumed training masters; preserve failures in reduced-pass schedules. |
| Larger/rectangular pressure inputs | Existing captured packet views and corresponding source/split manifests | Freeze complete case membership and training exposure. Endpoint fields alone cannot replace a captured pressure request for a model needing additional pre-projection features. |
| Own-history trajectories | Existing old-reference rollout banks and audited long masters | Compare matching physical timestamps without teacher resets. Report supported duration; do not claim a 600-output test from a shorter recording. |
| Reserved data | Existing reserved manifests, after inventory and protocol freeze and the applicable owner disclosure | Keep exposure explicit. Do not use a final evaluation to tune weights or repair a candidate while retaining an independent-test label. |

The original joint-generator population is separately frozen at64cases:
16consumed fitting scenes × starts0/24/64/120, at1/60s outputs with leads1–4.
The320original native files retain F64pressure/F32carriers and label-only RHS.
Four original K1–4joint fits have completed the full GPU boundary, two order/
process phases and20samples/case; the six K2–4endpoint/reuse fits are prepared
on the exact same cases/boundary. Both report full block and block/K, every
field/lead, and separate density/dimensions/combined groups. This population
has no independent holdout or600-output drift claim. Its retained native
reference uses external full-spectral pressure on actual integrator requests;
its capture binary is ecf8252c88c248603325bf36be19c4aae0633b62a71de60740ab9610117c2283,
separate from the pressure-request bank below.
[Joint reduction](reports/joint-generator-GPU-benchmark-01.json),
[sharing protocol preparation](reports/sharing-generator-GPU-preparation-02.json).

Paths in the first three rows refer to existing plans under
`docs/plans/full-ai-data-diversity-v1/`, except the pressure replay entries.
The [worklist](named-model-worklist.md) and ignored `model-worklist-05.json`
link pressure entries to their retained projects and fit receipts.

The first pressure population is now executed for all 127 available named
checkpoints. [Common-score evidence](reports/pressure-common-score-01.json)
independently verifies every output and scores all models with the same frozen
finite-volume operator. All recorded pressure metrics match exactly, and the
common equation metric equals each captured-operator metric on all 16,256
requests. Two unchanged controls verify the model-independent packet reader.
This does not complete other pressure populations, intermediate checkpoints,
timing, compositions or own-history testing. C01/C02 remain unavailable.

The pressure-request lane now has a [frozen input contract](reports/frozen-pressure-requests-01.json)
with all128existing native case IDs, exact input/reference hashes, original
capture metadata/build receipt and all127named source/weight bindings reverified.
Its reference is the retained capture binary ec4c6ae2320afdeb90b13e97d7fbb44d79f7f996a427d395ab1197669cbad1b0;
no new reference was generated. This bank has64cases at8cells/metre and64at16,
all with1/60s input timestep. Grid dimensions, grid density and fluid density are
separate fields. Native frame indices do not imply recovered physical timestamps.
The frozen task produces one pressure proposal; its call time cannot stand for
a complete simulation advance or be divided by physical K outputs. Existing
labels remain consumed diagnostics. This freezes the pressure lane's existing
population only; timing adapters, other pressure populations, full-state/temporal
manifests and the complete portfolio matrix remain pending.

[Reviewed full-state contracts](full-state-contracts.md) distinguish the initial
fixed-cadence, explicit-time and reduced-stage families. Their signatures are not
interchangeable, and their stages are not physical K outputs.

For every model, derive applicability from its captured implementation. A fixed
60-output-per-second generator is not silently modified to accept 30 or 120.
Likewise, original N1/N8 metadata is not interchangeable with the temporal
pilot's grouping of a 240-update-per-second reference. Cases with incompatible
inputs, boundaries or integration schedules receive explicit reason codes.

Generate any missing reference inputs using the frozen old solver and record a
new old-reference capture identity. New Rheon cannot fill gaps in an old target
by reusing its scene name. Adoption of the new solver follows complete old-model
benchmarking and history preservation.

## Metrics and aggregation

Preserve historical metric definitions and add separately identified current
metrics. For each field retain absolute error, denominator, normalization floor,
whether the floor was active and the corresponding relative error. Pressure is
mean-free where the reference contract has a gauge freedom. Equation residual
and pressure-field error have distinct identities.

The existing full-state measurement uses mean-free pressure RMS difference
divided by the larger of reference RMS and a frozen training-derived floor.
Its equation metric is residual L2 divided by `max(RHS L2, 1e-30)`. The cohort
diagnostic also reports the equation L2 ratio. A maximum across cases of an L2
ratio is not a maximum voxel error; retain the spatial norm in the metric name.
Record absolute RMS and maximum absolute field errors for the new common suite.
Define any added relative spatial-maximum metric before computing it.

Never drop zero-reference, nonfinite, rejected or failed cases from a summary.
Their statuses and denominators remain inspectable. Thresholds are diagnostics
unless an application limit has separately been established. An authored Tuldok
target is not proof of scientific acceptance.

Aggregation keys include reference epoch/scheme, task, exact population and split,
grid dimensions, cells/metre, output interval, K, internal schedule, precision,
device and timing scope. Keep density and grid dimensions separate. Retain every
case/output row and the identifier/start/lead of each worst value. Compare a
single quoted case with that same case, then show broader population summaries.

## Timing and CPU reproduction

The primary GPU measurement begins with the declared complete CPU input tensors
and ends after independently owned CPU outputs are available and the GPU is
synchronized. Include input validation and packing, transfers, known-control
construction, all actual model advances/projections, output allocation and
reconstruction. Pressure-only input assembly and full-state preparation have
different contracts and must be labeled accordingly.

Keep constructor/setup, compilation, geometry preparation, first call and warmups
separate. Retain repeated full block wall timings and divide by K physical
outputs. GPU event or resident-input timings are secondary, explicitly labeled
scopes. Record actual internal advances and pressure calls; never divide by
internal substeps. Check measured outputs against the accuracy output identities.

Use balanced paired order where implementations share a comparison. Record host
affinity, thread settings, GPU/driver/runtime, dtype/TF32/autocast and resource
observations. Do not accept timings collected under unrelated GPU contention as
uncontended measurements. Accuracy-only replays may proceed separately.

Retain CPU optimization sources, pinned reproduction commands and historical
thread counts. Keep those results separate from GPU timing. A CPU-only legacy
implementation requires an explicit execution disposition; its historical CPU
time cannot fill a fresh GPU timing slot. Do not alter the model architecture
merely to obtain a GPU number.

Rheon reference workers together use no more than half the physical CPU cores.
Reference timings measure the corresponding pressure request or complete state
advance, with process startup separated. A pressure model is not assigned a
speedup against a differently scoped full simulation step. No training or
reference capture overlaps accepted model timing.

## Required result fields and completion

Each work item binds model/source/material references, all required parameter
artifacts, adapter version and case manifest. Its results include original input
and output identities, all output times, metrics and denominator policy, actual
internal work, numerical failures, ownership checks, setup/warm samples, hardware
and precision, and exact reproduction commands. Missing data or unsupported
execution remains visible and prevents a broader completion claim.

Publish native Tuldok scientific records and views from these retained results.
The migration worklist is temporary input to that owner, not a second permanent
model registry. Fresh benchmark Trials are genuine new executions; imported
training history keeps its original timestamps and namespace. Record inferred
or authored retrospective explanations separately from measured facts.


## Original joint-generator common case lane

The first four reviewed protocol2K1/2/3/4joint generators have one qualified
case selection:16original train scenes at8/16cells per metre, four starts
(0,24,64,120),1/60s output spacing and all four available target leads. All320
unique current/reference files match original receipts; the independent sealed
finite-volume target equation error is at most1.153583589958727e-13. References
come from the original native temporal probe with external full-spectral pressure
on its actual requests, binaryecf8252c88c248603325bf36be19c4aae0633b62a71de60740ab9610117c2283.
This is distinct from the128pressure-request capture binary and is not a new
native-PCG or new-Rheon solver timing. RHS is a target label only.

A preregistered eight-process campaign reverses model/case order in phase2,
retains20timed samples per model/case, and measures5120joint calls/12800physical
outputs. The interval includes native CPU5field validation/views, sourcepacking/
known forcing, H2D, joint forward, reconstruction and every Kowned CPU5field
output/D2H/synchronization. Setup, first calls,warmups,finite/hash/ownership
oracles and scoring are separate. Every field/lead is scored; blocktime and
block/Kare both retained. A global module pre-hook forbids CPUneural forwards.
Three analytic NumPy metric fixtures pass, including gauge invariance and an
explicit undefined zero-reference boundary. No CUDAqualification is yet claimed.

The original dispatch check refuses while OverwatchPID6138is active onGPU;
no benchmark directory/worker is created. Do not stop unrelated processes.
Invoke run_joint_generator_gpu_benchmark.py through the registered managed
runtime with common.environment(runtime,'cuda') once resources are quiet.
Using its 'cpu'environment deliberately hidesCUDAand is unsuitable for child
GPUworkers. Follow the original campaign directory/session after dispatch;
retain failed/contended evidence and do not replay over its outputs. Independently
reduce all eight results before any timing/accuracy qualification or native import.
All cases are consumed training scenes; held-out and600-output drift remain
separate requirements. [Case qualification](reports/joint-generator-case-preflight-01.json),
[readiness](reports/joint-generator-benchmark-readiness-01.json).
