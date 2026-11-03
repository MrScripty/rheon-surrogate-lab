# Inspectable model statistics and comparison contract

Status: Adopted user requirements; coverage audit in progress. Prediction deletion
is paused until the applicable requirements below have an explicit disposition.
This contract supplements [benchmark-contract.md](benchmark-contract.md).

The retained scientific result is a small, inspectable description of the task
and its measured behavior. It is not an archive of input or prediction fields.
Keep case/output scalar rows as well as summaries: aggregates alone cannot answer
new combinations of filters, support paired comparisons, or expose a bad
intermediate output hidden by an accurate endpoint. Retain source/reference/model
identities and reproduction commands; preserve original training and reference
assets in place. No additional raw-field copies are required.

## What each model does

Each result binds the fit, selected parameter state, execution variant and source.
Describe input quantities, units, layout, conditioning, supplied context and
output quantities explicitly. Distinguish a pressure request, predicted pressure
RHS, local transport/viscosity operation, stage correction/readout and complete
state generator. Record required geometry, grid spacing, timestep, prescribed
forcing/events and any supplied solver/stage/guide outputs. A model receiving
supplied intermediate state is not a standalone full-state generator.

Evaluate the output against its matching reference at the requested physical
time. Separately record scalar changes from the initial state, and improvement
against the appropriate unchanged-input, zero-correction or parent-model baseline
when meaningful. A small output-minus-input difference does not itself establish
accuracy. Keep baseline identities and paired differences; do not manufacture
a persistence baseline for a task where it has no defined physical meaning.

## Case and temporal dimensions

Retain scene/parent, split and training exposure; reference solver/build, physics
epoch and integration scheme; counts X/Y/Z, voxel count, aspect ratio, cell
spacing/cells per metre and physical extents; fluid mass density separately.
Record force profile, magnitude/range, location/motion, pulse/event timing and
configured pulse/motion/variance frequencies when known. Missing data stays
unavailable. Output cadence is not a forcing frequency. Irregular pulses do not
have a single measured periodic frequency merely because a configuration has
a nominal frequency field.

For temporal generators retain these separately:

| Field | Meaning |
| --- | --- |
| Output interval, seconds | Physical time between requested outputs |
| K | Number of complete physical outputs emitted by one call |
| Block horizon, seconds | Total physical time advanced, normally K times output interval |
| Lead and physical lead horizon | Which output is scored and time since input state |
| Endpoint/intermediate role | Final output versus each intervening output |
| Internal schedule | Substep widths/durations, passes/iterations, refinement and learned calls actually used |
| Computation strategy | Sequential, shared-work, endpoint plus reconstruction, or another source-verified method |

Do not infer internal passes from K, fit names, or a source method-call count
that observes only part of the computation. Preserve observed counters with
their scope. Stage index is not physical output lead. For K=1 the only output
is the endpoint; it is not evidence of an interpolation strategy.

Every lead retains its own pressure, velocity and tracer errors and sample count.
Summarize by K × output interval × lead × grid × density as well as endpoint
versus intermediate. Keep the worst lead and its case identity. Compare the same
physical target across K where the captured references allow it. This is essential
for methods that spend more work on the endpoint and approximate intervening
states: endpoint accuracy cannot stand in for all emitted states.

For example, a four-output endpoint-plus-interpolation model must expose four
separate accuracy rows: intermediate leads 1, 2 and 3, and endpoint lead 4. Each
row includes best, worst, mean, median, p95, sample count and worst-case identity
for every applicable output quantity. Retain the underlying per-case scalars
so those rows can also be filtered jointly by cadence, grid and density. Label
how each lead was generated when verified from the source; do not infer that
the endpoint receives a full solve just because it is the last output.

Also expose the worst intermediate error and worst error across all physical
outputs, with the responsible lead and case. An endpoint-only score or a pooled
mean must not qualify a model against an all-output worst-error target. Preserve
unsupported, failed and nonfinite output counts by lead rather than silently
summarizing only successful endpoints. Internal-stage targets remain separate
from these physical output rows.

## Scalar measurements

| Measurement family | Retain when applicable |
| --- | --- |
| Pressure | Gauge-aware relative L2, absolute RMS/MAE/max, signed bias; reference scale and floor activation |
| Velocity | Combined and X/Y/Z errors, absolute RMS/MAE/max, reference scales; magnitude/direction diagnostics with a declared near-zero policy |
| Tracer | Relative and absolute errors, amount/integral error, extrema, negativity and overshoot |
| Physical consistency | Pressure-equation residual, divergence, boundary violations; kinetic-energy and vorticity summaries where relevant to the task |
| State and advancement | Input/reference/prediction scalar ranges and RMS, output-minus-input and reference-minus-input norms, baseline improvement |
| Stability | Per-output scalar drift curves under own history, elapsed physical time, failure/threshold crossing time, reset policy |
| Performance | Complete block wall time, block/K for physical outputs, setup/compile/cold/warm costs, memory, device/runtime/precision/thread settings and contention scope |
| Coverage and failures | Attempted/succeeded/unsupported/nonfinite counts, denominator-floor counts, scene counts and sample weighting |

These are task-dependent requirements, not a claim that every historical result
already contains them. A pressure-only result has no velocity/tracer accuracy.
Local synthetic operator tests do not establish trajectory stability. Additional
spectral-band or regional error summaries may be added before field disposal
when a specific research question requires them; arbitrary future spatial plots
will require a reproducible replay rather than retained field archives.

For measured error/cost populations keep min/best, max/worst, mean, median,
standard deviation, p90/p95/p99, sample count and worst/best case identities.
Preserve per-case scalars so percentiles, ECDFs and new stratifications remain
computable later. Signed/state quantities use min/max rather than automatic
best/worst labels. Declare weighting: per voxel, output, block or scene.
Repeated timing samples and outputs from one scene are not independent scenes.
Do not silently omit failures or use zero for unavailable measurements.

### Timing multi-output methods

The primary timing is the measured full block and block/K, counting only physical
outputs. Preserve separately measured endpoint solve, intermediate reconstruction,
shared preparation and transfer costs when instrumentation supports them. Mark
such measurements as instrumented stage costs and retain instrumentation scope.
Do not assign an equal fraction of a jointly measured call to each lead and call
it a measured lead latency. Shared computation may have no unique per-lead cost.
CPU and GPU figures, resident and complete-transfer boundaries, and component
versus full-generator costs remain distinct.

In that four-output example, report the measured block latency and block/4
alongside the four accuracy rows. If endpoint and interpolation costs have not
been measured independently, mark those stage timings unavailable. Accuracy is
always evaluated separately for each emitted state even when its computation
shares work with the endpoint or other outputs.

## Plotting and disposal gate

Support within-task accuracy/latency frontiers, paired per-case improvements,
lead-error curves, endpoint/intermediate comparisons, K/schedule tradeoffs,
grid/density/voxel scaling, forcing/event breakdowns, error distributions and
own-history drift. Match task, reference, case population, metric definition,
physical target and timing boundary before ranking points. Use explicit filters
or separate panels for fundamentally different workloads. A full generator's
pressure error may be inspected alongside a pressure component's error only with
their different input contexts and end-to-end costs made explicit.

Before deleting any further derivative prediction capture:

1. Inventory its outputs, case/lead membership and downstream consumers.
2. Preserve all applicable scalar metrics above, metadata and existing baselines;
   mark unsupported/not applicable/missing measurements explicitly.
3. Verify every lead and accepted case joins its source, including denominators,
   flags, failures and timing qualification; verify summary/re-read agreement.
4. Resolve required missing metrics by scoring existing captures in memory or
   by a planned statistics-only replay. Do not treat planned work as collected.
5. Verify training/reference/weight/setup assets are protected and write a
   disposition receipt before retiring only audited derivative fields.

Eight earlier derivative captures were already retired after the narrower
documented scalar/spatial audit. Their accepted scalar evidence remains, but
their original retirement does not establish completion of this expanded
contract. Subsequent statistics-only replays now supply expanded diagnostics
for all205selected fits on their admitted populations; see
[current coverage](reports/common-fit-benchmark-coverage-16.json). Source/interface,
internal-work, consumer and portable-source audits, other parameter states and
own-history drift remain separate requirements. Further deletion remains paused;
the portfolio is not benchmark-complete.
