# Accuracy replay findings

## Inherited generators: broader accuracy and the shared pressure limit

All 16 inherited-pressure fits now pass CUDA construction and complete-field
accuracy checks on the same 64 old-reference cases. The run verifies 35,920
constructor unit-query batches, 2,048 complete predictions and 4,608 physical
outputs including repeat calls. Original final learned tensors remain exact;
GPU-built F64 maps/bases are labeled as an execution/precision variant. Both
rectangular map routes and cube CNN fallback routes execute. All first-case
saved-probe metric checks pass without a new CPU neural forward.

These are untimed consumed-scene diagnostics. They establish accuracy coverage
for another 16 fits, bringing common-case generator accuracy coverage to 26 of
35. Nine earlier/global/scaled/MSE/curl fits still need the common comparison.
No inference-speed or independent-holdout claim follows from this run.

| Captured family | Fits | K | Worst pressure relative L2 | Worst velocity relative L2 | Worst tracer relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Endpoint pressure expert | 1 | 2 | 0.23915 | 0.40102 | 0.001761 |
| Endpoint decoder | 1 | 2 | 0.40553 | 0.17616 | 0.005141 |
| Coupled | 1 | 2 | 0.08195 | 0.002440 | 0.001758 |
| Provisional control | 1 | 2 | 0.06189 | 0.002310 | 0.001408 |
| Provisional auxiliary | 1 | 2 | 0.06287 | 0.001783 | 0.001493 |
| Displacement | 1 | 2 | 0.006457 | 0.000292 | 7.50e-6 |
| Fixed viscosity, base/seed repeats | 3 | 1 | about 5.24e-4 | 2.91e-4 | 3.06–3.45e-6 |
| Fixed viscosity, base/seed repeats | 3 | 2 | about 5.24e-4 | 2.91e-4 | 6.46–7.66e-6 |
| Fixed viscosity | 1 | 3 | 5.24e-4 | 2.91e-4 | 1.77e-5 |
| Fixed viscosity, base/seed repeats | 3 | 4 | about 5.24e-4 | 2.91e-4 | 1.60–2.65e-5 |

All ten fixed-viscosity fits share their worst pressure case:
`temporal-g3-cpm16-r1`, start 24, lead 1, **32×32×32 at 16 cells/m**. This uses
the pressure component's CNN fallback, rather than the complete learned maps
used for its smaller registered geometries. Different K/seeds do not remove
that first-output limitation.

A separate diagnostic reproduces all 64 original K1 generator output hashes,
then evaluates the same frozen pressure component on the exact future native
RHS. That oracle-fed pressure-only call is deliberately noncausal; it is not an
improved generator. It separates the total pressure error into request
propagation through the component, pressure-component error on the true request,
and output quantization. At the shared worst case:

| Error term, normalized by native mean-free pressure norm | Relative L2 |
| --- | ---: |
| Original complete-generator pressure error | 5.237459e-4 |
| Pressure component on the exact native request | 5.236959e-4 |
| Effect of the generator's request difference | 6.587273e-8 |
| Output quantization/reconstruction difference | 3.956858e-8 |

The component remains the dominant limitation at this case even with the true
request. This localizes the bottleneck; it does not distinguish learned
approximation from precision effects inside that component. Other cases also
have meaningful request error: its largest propagated pressure effect is
7.06e-5, and the complete-map route's largest pressure-component error is
1.07e-4. Improving only the 32-cubed fallback therefore does not prove the
whole model will reach 1e-5. Three literal vector/gauge/undefined-denominator
tests pass for the diagnostic decomposition.
[Full 16-fit coverage](reports/inherited-generator-CUDA-qualification-01.json),
[controlled error diagnosis](reports/fixed-viscosity-pressure-error-diagnosis-01.json).

## Timing instrumentation needs a common boundary

The older joint/sharing worker retains its CUDA forward-validation hook inside
the timed interval. The pressure and F64 workers remove that hook for timing.
Actual original captures confirm that the joint/sharing guard observes all
896 first/warmup/timed network calls per process, rather than only 256 first/
warmup calls. This leaves their saved numerical accuracy intact, but the ten
generator times below describe instrumented calls. The callback overhead and
its effect on production rankings have not been measured.

Retain the original packets and add a native scope disclosure. Before a
cross-family speed conclusion, use one hooks-outside-timing protocol plus a
paired on/off calibration after heavy native work finishes. This observation
does not establish that instrumentation explains the entire speed difference.
[Source and original-capture evidence](reports/generator-timing-hook-scope-01.json).

## Ten generators on the same retained physical cases

The six endpoint/reuse models now complete the same 64-case comparison as the
four early joint generators. Every case retains the same current native fields,
known controls, reference solver, grid density, dimensions and 1/60-second output
duration. These are 16 consumed fitting scenes at four starts, not an independent
holdout or a 600-step rollout. All new neural forwards are CUDA. Timing includes
native CPU input through every owned CPU output and synchronization.

| Model | K | Median block ms | Median ms/output | Worst pressure relative L2 | Worst velocity relative L2 | Worst tracer relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint b1 | 1 | 3.168 | 3.168 | 1.018 | 0.7410 | 0.00859 |
| Joint b2 | 2 | 2.991 | 1.496 | 1.039 | 0.8337 | 0.01726 |
| Endpoint b2 | 2 | 4.801 | 2.400 | 0.8717 | 1.0000 | 0.01676 |
| Reuse b2 | 2 | 4.941 | 2.471 | 0.7405 | 0.9824 | 0.01650 |
| Joint b3 | 3 | 3.025 | 1.008 | 1.329 | 1.0001 | 0.02600 |
| Endpoint b3 | 3 | 5.127 | 1.709 | 0.8719 | 0.9998 | 0.02601 |
| Reuse b3 | 3 | 5.246 | 1.749 | 0.7362 | 0.9963 | 0.02478 |
| Joint b4 | 4 | 3.160 | 0.790 | 1.492 | 0.9998 | 0.02481 |
| Endpoint b4 | 4 | 5.342 | 1.336 | 0.8032 | 0.9997 | 0.02758 |
| Reuse b4 | 4 | 5.635 | 1.409 | 0.7604 | 0.9969 | 0.03291 |

These instrumented timings are medians of the 64 case medians, each from twenty samples across
two process/order phases. Error columns take the worst over all cases and every
physical lead; their extrema can come from different cases. Both full block
time and amortized time are shown. GPU: RTX 5090 Laptop. Before/after worker
checks observe no competing compute; continuous hardware exclusivity is not
proven, and the campaigns are not a controlled thermal experiment.

Endpoint/reuse have lower worst pressure error than the early joint models at
matching K, with higher measured time. They do not establish improved accuracy
for every field: the velocity tails remain close to unit relative error. Several
sharing models' worst pressure cases occur at start 120, so the start-zero
problem seen in joint models does not explain their entire error distribution.
Every sharing model reproduces its original saved CPU probe's first-case field
metrics within the declared tolerance, without running a new CPU forward.
This supports the retained model behavior on that case; it does not identify
the training cause or prove all-case CPU/GPU equivalence. The near-exact F64
compositions still need this exact population before any claim of a matched
regression from those anchors.

All 20 original process blocks and every lead's metrics are now packaged in
four native K-specific tables. Independent verification recomputes all 6,400
displayed block scalars from raw phases, checks 400 native density/domain/start
queries and rejects all six incompatible cross-K table pairs. Six focused
fixtures cover missing intermediate leads, late-lead worst error, undefined
relative error, an incorrect time divisor and changed grid density. Native
ingestion and chart/UI verification remain pending.
[Sharing results](reports/sharing-generator-GPU-benchmark-01.json),
[joint results](reports/joint-generator-GPU-benchmark-01.json),
[native packet qualification](reports/common-generator-GPU-measurements-qualification-01.json).

## Fitting population and measured recurrence can differ

The source audit now distinguishes C62/C64–C67/C69's physical calibration
population from their actual optimizer inputs. All use the original 16 physical
trajectories / 32 frames to calibrate scales, but their optimizers consume only
manufactured spatial fundamental fields. They are capacity probes, not additional
multi-scene training runs. Each begins from the exact C53 checkpoint.

| Fit | Actual manufactured cases | Optimizer updates | Example presentations | Student neural forwards |
| --- | ---: | ---: | ---: | ---: |
| C62, one pattern/grid | 1 | 2,000 | 2,000 | 8,000 |
| C64, eight grids × three axes | 24 | 4,800 | 4,800 | 19,200 |
| C65, one grid × three axes | 3 | 600 | 600 | 2,400 |
| C66, eight grids × one axis | 8 | 1,600 | 1,600 | 6,400 |
| C67, same cases/later rate drops | 8 | 1,600 | 1,600 | 6,400 |
| C69, whole-cycle mean before Adam | 8 | 200 | 1,600 | 6,400 |

C66/C67/C69 have the same private case-presentation order and 200 exposures
per case. C69 averages eight losses/gradients before one clipped Adam update;
its smaller update count is not eight times less neural work. C67 changes the
learning-rate schedule. Original configurations and recorded histories remain
unchanged. Independent order/exposure/work checks and four focused tests verify
the interpretation; they do not rerun neural fitting.

C63's "frequency curriculum" instead retains the physical/five-stream fitting
setup and changes only supplemental spatial patterns. It assigns 12/12/8 patterns
to fundamentals/other-low/high spatial modes at four amplitudes on eight grids;
the fundamental patterns repeat. Its actual 1,500 category draws are 579/542/379.
This is a static spatial distribution, not a staged temporal-frequency or
timestep curriculum.

These seven fits train four pressure corrections, but their default inference
and common GPU benchmark execute two. For example, C62's final four-correction
equation diagnostic is 1.00858e-8 on its one fitted manufactured pattern. Its
common two-correction worst pressure error is 0.0193265 on 128 consumed physical
requests. Both the population and executed work differ, so those figures cannot
be read as a matched accuracy regression. Separate four-correction common
measurements remain required (I44). None of these pressure corrections advances
another physical simulation frame.
[Exact reviews](reports/captured-pressure-capacity-method-reviews-01.json).

## Original direct joint generators: separate common-case comparison

The four early F32joint K1–4generators now have fresh complete-boundary GPU
measurements on the same64retained native temporal cases:16consumed training
scenes × four starts. Median GPU ms/physical output is3.168/1.496/1.008/0.790,
with full blocks3.168/2.991/3.025/3.160ms. At32cubed alone it is
3.575/1.745/1.141/0.889ms per output. These show compute amortization, not
successful accuracy: worst mean-free pressure relative L2is1.018/1.039/1.329/
1.492 and worst velocity relative L2is0.741/0.834/1.000/1.000. Largest pressure
errors occur at start0. Original saved CPU probe arrays reproduce the first-case
GPU field metrics within declared tolerance, without new CPU inference. This
supports retained model behavior on that case; it does not certify all-case
CPU/GPU equivalence or identify the training failure’s cause.

The population/reference differs from the quoted F64anchor cases below. Do not
compare the mixed-shape medians directly with a single15360-cell/cpm100anchor,
or promote consumed diagnostics to an independent holdout. The F64backends still
require this exact same-case comparison; the six endpoint/reuse models are now
measured above.
[Full reduction and every field/lead](reports/joint-generator-GPU-benchmark-01.json),
[saved-probe check](reports/joint-generator-retained-probe-check-01.json).

## Fresh complete-generator GPU timings

The common complete-generator benchmark now reproduces the two quoted accuracy
anchors and their approximate GPU speeds. All 24,000 timed complete calls and
60,000 timed physical outputs are retained as scalar/hash evidence. Including
first calls and warmups, 96,000 returned physical states matched the original
field bytes. An independent reducer reopened all 102 current-state archives,
bound every original metric/field hash, and pooled 20 samples per exact case.
There were no new fits, reference simulations or CPU neural forwards.

The primary interval starts with owned CPU NumPy pressure/vx/vy/vz/tracer fields
and includes layout checks, views, packing, H2D, known controls, all actual
learned advances, reconstruction, allocation, D2H, synchronization and NumPy
output views. Setup, first calls, warmups and correctness oracles are separate.
RTX 5090 Laptop GPU, Torch 2.14.0+cu132, one host thread on logical CPU 4. GPU
resource samples show no other compute PID; this is not uninterrupted hardware
exclusivity or a hard real-time bound.

Both quoted anchors use 15,360-cell rectangular domains. The cell case uses
16x24x40 at 100 cells/m, start 192, 60 outputs/s and one learned advance/output.
The temporal case uses 16x40x24 at 50 cells/m, start 0, 120 outputs/s and two
learned advances/output. K4 therefore advances four versus eight internal
stages. The physical case, reference policy, output cadence and grid dimensions
must stay visible when reading a speed/accuracy frontier.

| Execution, exact quoted case | Median ms/output | P95 ms/output | Maximum observed ms/output | Worst pressure error over K | Worst equation error over K |
| --- | ---: | ---: | ---: | ---: | ---: |
| cell_bound | 0.239540 | 0.263700 | 0.436692 | 9.250026e-17 | 2.415125e-15 |
| cell_coordinate | 0.232735 | 0.263167 | 0.315852 | 9.250026e-17 | 2.415125e-15 |
| temporal_eager | 0.787353 | 0.849881 | 0.980105 | 9.158490e-16 | 1.459761e-14 |
| temporal_graph | 0.411446 | 0.422756 | 0.423318 | 9.158490e-16 | 1.459761e-14 |

The cell-coordinate/bound-sampler comparison has a median paired latency ratio
of 0.9991 across all 384 matched blocks; 201 favor cell coordinates. Every tested
output is byte-identical between them. This supplies no consistent speed gain
from the coordinate change. Temporal graph execution is faster than eager on
all 216 matched blocks, with identical outputs and a median paired ratio of
0.5973. This is an execution optimization; neither comparison adds trained
parameters or independent scenes.

The cheaper temporal controls reproduce their accuracy losses. Below, each
number is the median of three start-specific case medians for the existing
50-cells/m training master, K4. Pressure error takes the worst of all three
starts and all K output leads. All rows use the same 240-updates/s reference.

| Output cadence | Captured graph schedule | Median case ms/output | Worst pressure error |
| --- | --- | ---: | ---: |
| 30 | fine_reference | 1.374100 | 3.108227e-15 |
| 30 | half_passes | 0.798796 | 2.013389e-02 |
| 30 | single_pass | 0.369096 | 1.282026e-01 |
| 60 | fine_reference | 0.731811 | 3.108227e-15 |
| 60 | half_passes | 0.443409 | 2.013389e-02 |
| 60 | single_pass | 0.298270 | 6.122897e-02 |
| 120 | fine_reference | 0.405094 | 3.108227e-15 |
| 120 | half_passes | 0.258261 | 2.013389e-02 |
| 120 | single_pass | 0.261518 | 2.013389e-02 |

At 120 outputs/s, half-pass and single-pass schedules are the same one-width-2
advance per output; their timing difference is sample variation. These are
untrained coarsening controls, not successful trained macrostep predictors.
The existing fine-reference models retain their accuracy. The reduced-work
errors are real on these consumed cases, rather than a lost weight or dtype
restoration problem. This does not establish generalization, pulse coverage,
600-output drift or the research target of under 0.2 ms per physical output.
Both original temporal masters and their native case rows are labeled train.
The packet context's sentence claiming one training and one development master
is incorrect; an explicit correction is now adopted in the portfolio
overview, with its actual UI read accepted. Preserve the original evidence, and do not treat either temporal
master as held-out validation. Equation error here uses the fine-reference
saved RHS, not the coarse candidate's own independently assembled RHS; coarse
integration differences do not alone establish pressure-solve failure.

Two native packets are qualified with 9,600 scalar observations and 96 actual
native-table filter/maxima checks. Integration schedules are distinct temporal
subjects on 72 shared physical cases; actual student advances are observation
metadata. Cell and temporal reference scopes remain incompatible. The importer
runs through the ordinary native source/measurement operations. Both imports
acknowledge HTTP200, bringing the native measurement source count to19. The
after-custody audit rehashes all53,915canonical bodies, reconstructs the exact
previous header and verifies all8,715old payload identities plus the three new
evidence blobs. The actual browser passes150actions and17complete HTTP200reads,
including seven filtered comparisons and both original-source inspections.
All nine screenshots were visually inspected. These are UI/API observations,
not additional model inference samples.
See [independent raw reduction](reports/frozen-full-state-GPU-timing-01.json),
[native packet qualification](reports/frozen-full-state-timing-native-packets-02.json),
[native custody](reports/native-frozen-full-state-timing-custody-01.json),
[actual native UI](reports/frozen-full-state-timing-native-UI-01.json),
and runs/model-portfolio-migration-v1/native-frozen-full-state-timing-import-01.

The own-history reference availability audit freshly verifies all153saved
states in each temporal master. Each capture spans4,800fine updates/20seconds,
but saved states cover only61of901unique indices required for600outputs at
30and60outputs/s. Another840indices per master are missing. These sparse
windows cannot support a complete600-output drift claim. See
[reference availability](reports/own-history-600-reference-availability-01.json).

The new [matched correction-work comparison](pressure-correction-work-findings.md)
separates training changes from deployment work for C37/C44/C53 and nine detached
fine-tunes. C37's saved default uses one correction, while its fine-tunes use two.
Their gains persist at matched two/four/eight corrections, but their worst
pressure error is worse at one correction. All C44 and C53 fine-tunes regress
against their own parents on both worst metrics at every tested matched budget.
All 7,680 retained fields passed independent numerical rescoring. The native
C37 cards now include the matched-work qualification, preserving their original receipts.

Fresh C106/C107 GPU timings are now imported and rendered in a separate native
scope. Median-of-case-medians is 4.91430875 / 5.003978 ms per complete pressure
request; chart maxima are 5.626163 / 5.762195 ms, the worst case medians. All
10,240 measured calls retained exact original outputs. This measures the
captured eager pressure policies with CPU input/output ownership; it does not
replace the historical cell/temporal full-generator timings or establish a
speed regression between those differently scoped implementations. See
[timing](reports/pressure-GPU-timing-pilot-01.json),
[native custody](reports/native-GPU-pressure-timing-custody-01.json), and
[rendered comparison](reports/native-GPU-pressure-timing-UI-01.json).

The earlier accuracy-only replay reproduced both cited implementations exactly. Across
1,558 full-state outputs, every saved field hash, accuracy row and metric matched
the original result. These are consumed-case regression checks, not independent
validation or new latency measurements.

| Exact quoted case | Fresh worst mean-free pressure error | Interpretation |
| --- | --- | --- |
| Cell-coordinate g025, start 192, K4 | 9.250025585850337e-17 | The N8-labeled implementation executes its unchanged N1 branch here: four learned advances for four outputs. |
| Temporal continuous-varying cpm50, start 0, K4, 120 outputs/s | 9.15848991199606e-16 | Fine-reference grouping uses eight learned advances for four outputs. |

The wider cell-coordinate replay scored 1,018 outputs with no diagnostic failures.
The temporal replay scored 540 outputs. Its fine-reference schedule passed all
180 outputs; the half-pass and single-pass schedules each failed 176 of 180.
Their worst pressure errors were respectively 0.02013 and 0.12820. Reducing work
changes accuracy; the fine-reference result cannot describe those cheaper modes.
[Exact replay comparison](reports/anchor-accuracy-reproduction-01.json).

C106 and C107 also completed fresh CUDA pressure-only evaluation on the same
128 original validation requests, with unchanged parameters and architecture.
All paired input and reference hashes matched. The recorded neural forwards
were on CUDA; none were on CPU.

| Model | Worst relative equation residual | Worst mean-free pressure relative L2 error |
| --- | --- | --- |
| C106 | 0.006787566655874528 | 0.015504814574048292 |
| C107 | 0.0010294370663109186 | 0.008153290052382488 |

C107 remains better than C106 on both metrics for these requests. The old chart's
C107 error near 1.03e-3 measures equation residual, not pressure-field error.
Neither pressure-only request timing nor its error should be directly ranked
against a differently scoped full-state case. Historical CPU and current CUDA
numerics differ slightly; cross-device bitwise equivalence is not claimed.
[Paired CUDA comparison](reports/pressure-GPU-comparison-01.json).

A further 21-checkpoint CUDA batch has now completed. Together, 23 pressure
checkpoints answered 2,944 requests on the same 128 original cases. Every case's
input/reference hashes pair exactly. The captured portable equation-operator
function definitions also match structurally. Source bindings, architecture and
parameters stayed unchanged; no CPU neural forward was observed.

This broader comparison reveals another ranking distinction. C53's worst
pressure-field relative L2 error is 0.00602154, below C107's 0.00815329, although
C107 has lower pressure error on 122 of 128 individual requests and lower equation
residual on 125 of 128. The maxima occur on different cases. C107's pressure
maximum is the 5 by 6 by 15 case at 8 cells/metre, frame 63. Preserve that tail
case alongside the broader improvement; neither a typical-case improvement nor
an equation-residual improvement guarantees a better worst pressure error.
[All 23 pressure results](pressure-GPU-accuracy.md), including separate density
and grid summaries in the retained comparison.

The evidence so far supports a comparison-scope problem and real reduced-pass
accuracy losses, rather than a loss of the two frozen anchors' accuracy. It does
not resolve every later model's performance. Historical 0.223700 ms and
0.426637 ms timings remain historical until an uncontended GPU rerun. The full
portfolio benchmark, native project migration and clean reproduction remain open.

The named pressure evaluation has now expanded to all 127 available checkpoints
(C01/C02 remain missing): 16,256 requests on the same 128-case population.
Every saved field was independently rehashed and rescored. Pressure metrics
match the worker results exactly; scoring every model with one frozen operator
also gives exactly the same equation metrics as its captured operator on these
outputs. C53/C107 controls retain all earlier output hashes and accuracy rows.

On this consumed pressure population, C128 has the lowest worst pressure-field
relative L2 error, 7.93679e-5. C99 has the lowest worst equation residual,
1.47498e-4. These are different methods and aggregates; fresh latency and
full-state/own-history results are required before choosing a model. The raw
local-smoother components remain included as diagnostics; their one-call results
do not measure the complete numerical hierarchies for which they were intended.
[All named pressure results](pressure-accuracy.md).

The C128/C129/C130 source reviews clarify the conditional improvement: each
freezes the same 94,956-parameter routed parent and trains a 33,982-parameter
eight-mode pressure tail. The actual head-only recipe consumes offline labels
for 1,526 training and 202 development geometries, rather than the inherited
scene-feedback configuration. Of the 128 common pressure requests, 118 have
teacher-training geometries and ten teacher-development geometries; that is not
an independent physical-scene split. The tail is bypassed above 16 cells on any
axis. Later seeds have lower final teacher-development loss but slightly worse
worst pressure-field error here. Modal fitting loss, pressure/equation errors,
larger-grid support and complete-generator timing therefore stay separate.
[Source-bound review preparation](reports/eight-method-review-preparation-qualification-01.json).

## Reduced residual schedules reproduce their failures

Fresh serial GPU replays of eight reviewed residual executions, representing
four trained correction fits, reproduce all 16,960 historical rows exactly.
Output hashes, metrics, model identities and frozen source bindings all match.
The same original population contains 53 scenes, ten grid shapes and densities
8, 10, 16, 25, 50 and 100; native row-level density/grid separation is retained.
This is consumed-case reproduction, not independent validation.

Every variant passes all 920 outputs on the original N1 branch. On cases with
an eight-advance reference, the candidate uses four groups (2/2/3/1), with three
residual-network calls per physical output. S2 fails 1,147 of those 1,200 output
rows; the other seven executions fail 1,151. Worst pressure errors across the
four fits remain near 0.00144, and worst equation errors near 0.0152. Velocity
and tracer also have real failures. These per-field metrics and their worst
cases remain separate in the report. The diagnostic threshold is 1e-5; this
does not assign application-specific drift limits.

The evidence rules out changed restoration/evaluation as the cause of these
models' recorded accuracy losses on these requests. Source review shows that
they compress internal evolution, while still computing every physical output
in sequence. They do not implement endpoint prediction plus inexpensive
intermediate frames. The paired-interpolator prototype likewise changes spatial
interpolation precision, rather than generating temporal intermediate frames.
New timings remain unqualified because the GPU is shared with desktop/game
activity. [Exact residual replay](reports/full-state-residual-accuracy-reproduction-01.json).


The C125/C126/C127 source reviews identify the earlier calibration step: all
three begin from the exact same complete C107 weights and calibrate1,198
existing coarse-head parameters while freezing92,560 fine parameters. Their
worst common pressure errors are0.0024197173/0.0023671402/0.0024062009, versus
C107’s0.00815329. Offline teacher geometry covers axes5..128, but the retained
physical diagnostic covers small grids; it cannot prove larger-domain pressure
accuracy. Keeping the same inference graph gives no demonstrated speedup from
calibration alone. Different private sampling seeds do not imply different
loaded initial weights or sequential continuation.
[Eleven-card source preparation](reports/eleven-method-review-preparation-qualification-01.json).


C119–C124 distinguish further C107 continuation and architecture experiments.
The first four share complete initial weights and population; reducing the head
rate and reweighting difficult frames produce small, metric-dependent changes.
The C121 tail name means hard-case sampling, not an added pressure-tail network.
C123 redistributes channels and has137,227parameters; C124 adds511global modes,
126,542parameters and one pressure pass. Both worsen worst pressure error on the
common128requests, to0.01583/0.04810, versus0.00752–0.00797for the four continuations.
The one-pass fit repeats each original draw twice and matches2,400CNN forwards;
that count does not equate FLOPs, memory traffic or measured latency. Its saved
pressure pass is distinct from physical-time multistep prediction.
[Source-reviewed preparation](reports/seventeen-method-review-preparation-qualification-01.json).

The four reduced-integration correction fits now have source-reviewed explanations.
All use10,876trainable float32 parameters. The nominal148scene objective contains
29parameter-dependent N8 scenes and119fixed N1 controls. Active training covers
783local samples over nine starts and three correction groups, eight shapes
through64³ and8/10/16/25/50/100cells per metre. The larger128³ shapes occur in the
complete control population; that does not establish learned128³ correction
coverage. The encoder explicitly includes spacing, duration, group width and
ordered known forces.

S1's two AdamW epochs select epoch2 at physical objective1.9069764312e-5.
Reweighting losses in S2 still selects epoch0: the final tensor inventory is
identical to initialization despite1,566executed updates. S3 starts from the
exact selected S1 state and uses full-objective LBFGS, reaching1.9016923678e-5
at closure18. S4 starts from selected S3 and improves its direct-delta objective
from0.7665071604to0.7451582696, while the separately recorded physical objective
is6.3435674502e-5. These objective definitions are different; a lower direct
correction loss is not evidence of a better physical result.

All original source/static-input/checkpoint tensors and warm-start bindings were
checked without model execution or rereading the large training archives.
The four cards' original receipt/configuration suffixes remain exact in the
prepared revisions. Exact complete-generator replay still shows the N8 failures;
these reviewed optimizers do not establish endpoint-plus-intermediate multistep
prediction. [Four-fit review](reports/captured-internal-residual-method-reviews-01.json),
[incremental preparation](reports/four-residual-method-review-preparation-qualification-01.json).

## Global temporal readout versus reduced integration

The three/four-internal-stage readout fits learn only9/12global double
coefficients, grouped by pressure/shared velocity/tracer, on a frozen backbone.
The two-start training view includes148scenes but29provide gradients;119are fixed
N1controls. LBFGS makes160parameter-changing calls with413/398closures. Final
training objectives9.2549343084e-7 /3.3163564068e-7are means of squared normalized
metric residuals, not worst pressure errors.

Original deployed GPU evaluation has1,018output rows per fit. All920N1bypass rows
pass, while all98N8reduced-integration rows fail the diagnostic1e-5gate. Worst
N8pressure errors are0.00594525(three stages) and0.00357059(four); equation residuals
are0.0186561and0.0111379. Lower training loss and many passing bypass rows do not
establish accurate reduced integration. Koutputs still advance sequentially;
these models do not make cheap in-between physical frames from one distant
endpoint. The full-field QR training objective also precedes deployed float32
carrier casts, so final field scoring remains necessary.

These are source/artifact checks and retained historical outcomes, not new model
replays or qualified latency. See
[the exact two-model review](reports/captured-macrostep-readout-method-reviews-01.json)
and [six-card incremental preparation](reports/six-incremental-method-review-preparation-qualification-01.json).

### C37 attached/detached fits used eight optimizer patterns

Source review of C75/C76/C77 finds that their retained physical datasets are
calibration/diagnostic inputs. The actual fitting-bank updater trains on eight
single-mode analytical patterns,200exposures each, at8or16cells/m. It explicitly
rejects companion losses. All three start from the same370,016-parameter C37
checkpoint and run1,600updates. U uses four corrections and6,400counted network
forwards; A/D add one auxiliary correction per update and count8,000forwards.
A keeps the carry gradient; D detaches it. Four correction passes solve one
pressure equation; they do not advance four physical simulation frames.

On the already retained common128pressure requests, worst pressure errors are
U0.01129882497652308, A0.011092770966602578 and D0.011180070449168685 versus
parent C37's0.05382213638741031. U has the lowest worst equation residual among
these three, so the attached arm does not dominate both error metrics. These
are pressure-only accuracy results with unqualified timing, not trajectory drift
or clean retraining. Exact32source bindings, original inputs/receipts/history,
parameter inventory, matched sampling/depth schedules and original checkpoint
bytes are checked by review_detached_C37_fits.py. See
[three source reviews](reports/captured-detached-C37-method-reviews-01.json).

The narrow optimizer stream matters when interpreting generalization: counting
all retained physical datasets as optimizer scene variety would be misleading.
These three authored reviews are not yet prepared or adopted into native notes.

### Native full-state views preserve matched comparisons and unavailable timing

All24real comparison API reads agree exactly with direct original-packet reductions,
including binary64 maxima/rows and status denominators. The actual GUI passes118
interaction assertions across seven comparison views and original-source inspection.
The same1,018outputs give cell-coordinate worst pressure1.336598269583449e-15
versus approximately6.9e-4for the reduced-pass residual variants. Wider2,120-case
maxima are not the same comparison population. Temporal180-case worst pressure is
fine-reference3.108226952420847e-15, half-pass0.02013388607301235 and single-pass
0.12820259877350038. This supports real loss from reduced integration work in
these retained experiments, alongside differences in benchmark populations.

Fresh qualified GPU latency is explicitly unavailable; the native plots show no
measured time and retain every unavailable observation. Browser/APIwall times
remain slow and are not inference timings. See
[API qualification](reports/native-full-state-comparison-API-01.json),
[actual UI](reports/native-full-state-comparison-UI-01.json), and
[navigation](native-comparison-navigation.md). These do not close all-model
benchmarking, genuine Experiment progression, clean reproduction or final cold
portability acceptance.

### C106/C107 now have matched complete GPU pressure-call observations

The four-block ABBApilot retains10,240complete calls on the same128old native
pressure requests, each with exact original output hashes. Inputs begin as
owned CPU NumPy fields/scalars; timing includes validation, packing, H2D,
captured CUDA pressure graph, allocation/reconstruction/D2H and synchronization.
Pooled40sample case medians give C1064.914ms and C1075.004ms median-of-case-medians;
worst case medians are5.626ms and5.762ms. Per-case timings vary; C107is faster on
53/128cases, paired median ratio1.0168. Block medians and all raw samples remain.

C107's retained worst equation error is0.00102944versus C106's0.00678757, and worst
mean-free pressure error0.00815329versus0.01550481. Thus this comparison supports
its accuracy improvement without establishing a speed improvement. These are
pressure-only GPU request times, not full-state/K-output step times or comparisons
with the historical0.22msanchor. No new weights/reference simulations were made.
[Raw reductions and settings](reports/pressure-GPU-timing-pilot-01.json) own the
scoped evidence; sampled resource observations are not continuous exclusivity.


## The original joint temporal nets use a different numerical method

The four original protocol2joint-head full-state nets (K1/2/3/4) genuinely share
one spatial encoder over multiple future fields. They use F32parameters,
F32internal pressure prediction and F32pressure fitting labels. The saved data
conversion records a worst6.225179570786273e-8relative pressure discrepancy from
the original native fields; its lossless transport now independently decodes to
all80original F32field hashes. Upcasting their pressure output toF64does not
recover F64internal arithmetic. This is another reason not to transfer the
nearly exact F64component-anchor score to these nets. It does not allocate all
observed accuracy/speed differences or establish a fresh model error floor.
The same-case, complete-generator GPU benchmark is still needed.
[Joint-generator source and data review](reports/captured-joint-temporal-generator-method-reviews-01.json).


## Matched full-generator comparison: the regression persists

All 35 original generator fits now complete the same 64 native cases in two fresh
process/order phases with forward-validation hooks excluded from primary clocks.
The 44,800 primary calls return 103,680 physical outputs; 98,560 complete calls include
first/warmup/calibration work. All 26 previously evaluated fits retain exact prior
field hashes. Paired hook-on/off medians span 0.0593285–0.499676 ms/block, so the
historical instrumentation mismatch is real but does not change numerical
accuracy. CPU neural forwards are zero.

The unchanged original cell-coordinate F64 native entry also completes all 64 cases
at each K, in two fresh phases:512 cases, 5,120 primary calls and 12,800 physical outputs.
All repeats/phases and cross-K sequential prefixes match exactly; parameters remain
unchanged. This uses one original N1 history advance per 1/60-second physical output.
The N8-only coordinate change is unused. It predicts every native field; no
future pressure/RHS/state enters preparation or inference.

| Physical outputs K | Median case GPU ms/output | 32 cubed median case GPU ms/output | Worst pressure relative L2 |
| --- | --- | --- | --- |
| 1 | 0.258388 | 0.585517 | 1.55386e-16 |
| 2 | 0.179265 | 0.474112 | 2.79755e-16 |
| 3 | 0.141102 | 0.431277 | 2.79755e-16 |
| 4 | 0.128931 | 0.400990 | 2.79755e-16 |

Times above are medians of per-case medians,each based on20 complete CPU-input/
GPU-prediction/owned-CPU-output samples. They are not worst-case latency promises.
All F64 variants have worst combined velocity relative L2 of 8.85113e-16 and zero
tracer difference on this population. Worst native-reference equation relative
L2 is 1.17674e-13; equation residual and pressure error are distinct metrics.

Every one of the 35 fits is less accurate in worst pressure error and 7.645–31.085
times slower by the matched-K median-time ratio. The best pressure tails among those fits
remain around 5.24e-4. Thus the older near-exact behavior also holds on this
common population; the reported performance gap cannot be explained by unmatched
benchmark populations or forward-hook overhead alone. Different learned
components,precision and deployment implementations matter. These data do not
isolate the speed contribution of each difference. The existing controlled
pressure-component diagnostic specifically localizes the fixed-viscosity32 cubed
worst case to its pressure CNNfallback.

This comparison is between distinct retained designs and deployment
implementations. It does not show that one checkpoint became worse when rerun.
The F64 composition remains available, and all 26 previously measured fit
outputs stay exact under the corrected timing boundary. A chronological
training-regression claim would require a matched lineage comparison.

K4 amortizes host/transfer/graph-call overhead, but still performsK sequential
learned advances; it is not a learned joint temporal shortcut. The original
temporal entry requires 240/480 fine-clock force schedules missing from these
legacy scenes. Its matched comparison is unsupported here; its original cadence
measurements remain separately retained. No fine histories were fabricated.

Both families use the 16 consumed fitting scenes at four starts, 1/60-second outputs,
grid densities 8/16 and dimensions15x6x8,9x11x11,16cubed,32 cubed. There is no new
independent-holdout or600-step own-history drift qualification. GPU before/after
observations show no competing CUDA compute PID; host asset processing is present
and continuous exclusivity is not established. New-Rheon physics remain separate.

The first F64launch stopped before any prediction because its tooling closure
omitted a metadata helper. Its original failure is retained. The corrected
protocol binds all task Python imports, changes no scientific worker/weights/kernel,
and all 8 fresh workers pass actual imported-source enforcement.

[35fit reduction](reports/complete-generator-GPU-benchmark-01.json),
[matched F64reduction](reports/same-case-F64-GPU-benchmark-02.json),
[exact comparison](reports/matched-generator-F64-comparison-01.json). Four native
K-specific tables are prepared and independently qualified:39 configurations,
24,960 scalars,all 78 original raw blocks,600 density/grid queries and6 cross-K rejections.
Their native import/custody and actual UI are accepted. See
[native import](reports/native-matched-generator-GPU-import-custody-01.json) and
[actual matched UI](reports/native-matched-portfolio-UI-01.json).


## Original causal request heads: numerical error survives matched GPU reproduction

Both original F32 future-q heads now complete the same 2,048 transitions from
16 retained training-parent scenes in four fresh GPU workers. Inputs, labels,
normalization, published weights and captured architectures retain their original
identities. The boundary returns a normalized future pressure-equation request,
not pressure or a complete simulation step.

| Original fit | Median GPU ms/request | Worst q relative L2 | Worst q relative L2 on main-loss sampled cases | Worst original pressure-filtered relative L2 |
| --- | --- | --- | --- | --- |
| a9e199535346c3d9, causal | 1.31822 | 7.24269 | 6.42109 | 5.19988 |
| ccb969376d283836, factored | 1.09882 | 0.250929 | 0.213316 | 0.103942 |

Timing includes owned normalized CPU inputs, packing, H2D, the unchanged GPU
head and owned CPU q output/D2H/synchronization. Force construction and original
normalization are preparation. The original fixed F32 pressure-filtered training
objective is evaluated separately on saved outputs, outside inference timing;
it is not native-F64 pressure accuracy or a candidate pressure generator.

Hash-verified replay of both original 10,000-sample main-loss sequences identifies
1,834 sampled transitions and 214 not sampled, all from the same parents. Of the
latter, 112 are beyond the original sampler's current-state 0..120 support.
Neither group is an independent holdout. Errors are substantial even within the
actually sampled group. Factored fitting also had its own earlier linear
calibration; the membership split describes main q-loss fitting only.

On the 12 original retained CPU probe cases per head, maximum GPU-to-saved-CPU
relative differences are 1.75022e-7 and 2.08333e-8. No new CPU model is run. These
small differences make backend rounding an unlikely sole explanation for the
much larger request errors on those checked cases. They do not establish
whole-population CPU/GPU equivalence or identify the architectural cause. The
source's recorded F32 q-label quantization error, 3.73526e-8, also must not be
confused with near-exact F64 pressure qualification.

The factored head has lower observed q and pressure-filtered error and a smaller
overall median time on this matched population. Grid-specific costs still
matter: its 32-cubed median is 2.06495 GPU ms versus 1.44952 for the causal head.
This is evidence for retaining both the method and its limitations, not an
application accuracy pass or a claim that every grid became faster.

[Independent request results](reports/causal-request-component-GPU-benchmark-01.json),
[original-objective diagnostic](reports/causal-q-original-objective-GPU-01.json),
[saved CPU probes](reports/causal-q-retained-CPU-probe-comparison-01.json),
[195-fit coverage](reports/common-fit-benchmark-coverage-09.json).


## Original temporal readout training behavior reproduces exactly

The original nine- and twelve-coefficient F64 readouts reproduce both saved
CUDA training-residual probes bit for bit, with unchanged published coefficients.
Their original GPU objective values also match exactly:9.254934308418676e-7
for three internal stages and3.3163564067785536e-7for four. Four guarded direct
CUDA .residuals calls cover two repeats of each original model. These direct
methods do not pass through nn.Module forward hooks; operand/parameter/output
devices are checked explicitly. No new CPU model inference runs.

The training view contains148parents at two windows each. Only29N8parents
(58windows) have coefficient-dependent losses. The other119N1parents
(238windows) contribute fixed losses and bypass the readout in field deployment.
Their near-zero errors must not be counted as learned temporal correction
successes. The same-parent windows are not an independent holdout.

In this original supervised QR view,the active-subset worst pressure residual
is0.00446737for three stages and0.00264337for four. Velocity instead rises from
0.000439876to0.000481982. These are original normalized affine training-loss
residuals before final carrier casts,not independently measured deployed-field
accuracy or latency. Mean loss alone hides both the fixed branch and metric
tradeoffs. The58active full-field windows for each model remain required;they
span six grid resolutions(8,10,16,25,50,100cells/metre)and eight grid shapes,
including64cubed. Stage-generation cost and bare readout cost stay separate.
The published raw coefficients and the existing student's2^-40deployment
rounding are distinct variants.

[Actual original GPU probe replay](reports/macro-readout-original-GPU-probe-replay-01.json),
[admitted field comparison](../../../runs/model-portfolio-migration-v1/macro-readout-component-contract-01.json).
Default benchmark coverage remains195/205;this diagnostic does not substitute
for either full-field component benchmark.


## Original temporal readouts: full physical fields now independently scored

The earlier QR-only limitation above is resolved for all58active fitting windows
perhead. Two fresh CUDA processes reproduce causal stage features and final
fields exactly;all232recomputed original full-grid QR factors also match bit for
bit. NumPy scoring of retained fields agrees with the original Torch diagnostic
within2.68e-15. Raw coefficients and original frozen backbone are unchanged.

| Internal stages | Worst pressure | Worst velocity | Worst tracer | Worst native-reference equation |
| --- | --- | --- | --- | --- |
| 3 | 0.00446737 | 0.000439876 | 0.00194492 | 0.0186561 |
| 4 | 0.00264337 | 0.000481981 | 0.00119068 | 0.0111379 |

Four stages improve pressure/tracer/equation tails but worsen the velocity tail.
These errors already occur on consumed fitting scenes and reproduce the original
loss view;the migration/CUDA restoration did not introduce them. The equation
metric uses native future RHS,so it does not isolate the pressure inverse on its
own coarse request. Each call advances one1/60sphysical interval;internal stages
are not simultaneous future outputs.119N1parents bypass the head and are excluded
from its learned success count. This remains neither independent holdout nor
own-history drift. No latency was collected or accepted under shared GPU use.

[Independent fields,grouped scores and worst-case identities](reports/macro-readout-field-GPU-accuracy-01.json),
[197accuracy/195timed coverage](reports/common-fit-benchmark-coverage-10.json).


## Original internal corrections: checkpoint selection and objective tradeoffs

All four original checkpoints now have full 783-request component accuracy
results in two fresh CUDA processes. Their saved original GPU probes match
exactly; every full-population field repeats exactly. The original 9.77 GB
training corpus was read in place. NumPy and captured Torch physical scoring
agree within 1.10e-16; 12 retained worst cases also pass a separate no-Torch
reduction. Only 10.10 MB of prediction arrays were added.

| Fit | Worst pressure | Worst velocity | Worst tracer | Worst equation |
| --- | --- | --- | --- | --- |
| S1, first AdamW fit | 0.0481556 | 0.000597382 | 0.000928138 | 0.0764394 |
| S2, loss-conditioned | 0.0482020 | 0.000344192 | 0.000928271 | 0.0764840 |
| S3, full-objective warm start | 0.0482061 | 0.000572213 | 0.000928150 | 0.0764543 |
| S4, direct correction labels | 0.400596 | 0.000540093 | 0.000924786 | 0.199671 |

S2 retained epoch0, with all output heads exactly zero. Every one of its 783
outputs equals the supplied baseline bytes. A trained fit identity therefore
does not imply its selected checkpoint learned a useful correction.

S4 worsens pressure on 601 requests. On its worst case (g020,25cells/metre,
32cubed,start32,group1), pressure error rises from 0.000240287 at the supplied
baseline to 0.400596 after correction. Absolute pressure RMS error is 0.228784
against target RMS0.571108; no normalization floor is active. Velocity error on
that same request remains 1.07426e-7. The captured response derives pressure
from the correction's flux and then couples it back into velocity. This supports
the interpretation that small carrier errors can have a much larger pressure
impact; direct carrier-label loss does not ensure pressure accuracy. The exact
causal contribution of each correction channel has not been isolated here.

These are original supplied internal-stage requests, including fine-history
context; they are neither full-generator rollouts nor comparable populations to
the earlier 58-window temporal readout results. No timing or holdout claim. The
normalization-reader failure was in the new harness, preserved and corrected
before accepting these results; scientific settings and raw weights are unchanged.

[Full grouped results and paired baseline counts](reports/internal-residual-GPU-accuracy-01.json),
[reader reconciliation](reports/internal-residual-normalization-reader-reconciliation-01.json),
[201 accuracy /195 timed coverage](reports/common-fit-benchmark-coverage-11.json).


## Guided fits: original checkpoint selection explains another pressure regression

Both original guided compositions now reproduce their full training objectives
and worst errors in fresh CUDA processes across2880blocks/7200outputs perfit.
All repeated fields and scores match exactly. The shared source manifest's v2
family label is historical; the actual captured trainer executes v3 with guide-RMS
scaling. A migration or wrong-parameter explanation does not fit this evidence.

| Original selected composition | Worst pressure | Worst equation | Worst velocity | Worst tracer |
| --- | --- | --- | --- | --- |
| Shared guided residual | 0.157269 | 0.572893 | 0.112095 | 0.165023 |
| Independent pressure head | 0.442258 | 0.572743 | 0.112078 | 0.207382 |

The pressure-head publication selects probe-3 by the maximum across all four
metrics. Equation error dominates that maximum, so its slight improvement allows
pressure/tracer to worsen. The original retained best-mean candidate-4 instead
records worst pressure0.157262 andmeanobjective0.000385712,versus0.442258/
0.000957134for published best-worst. This candidate-4comparison is historical
scalar evidence,not new inference;its weights remain a priority within524state
coverage. Do not call every publication's default checkpoint universally best.

The pressure-head worst occurs at K4lead1,30outputs/s on an irregular pulse
scene at100cells/metre,24x32x48,start4768. For the identical first physical target,
pressure error rises across K1/2/3/4:0.00404172/0.116526/0.254646/0.442258.
Reference RMS1.20549,no floor. Thus evaluating endpoints alone would miss a
material interpolation/conditioning failure. The shared residual on this same
target instead ranges0.000490129to0.00235690. Every comparison uses the same
native target,not different physical step durations. Detailed endpoint/interior
and cross-Kscalar comparisons are retained.

[Full guided evidence](reports/guided-GPU-accuracy-01.json),
[actual source/population](reports/guided-source-and-population-reconciliation-01.json).

The user explicitly requests scalar statistics instead of stored predictions.
Eight recent fits have complete available scalar distributions;additional spatial
diagnostics have narrower explicit sample coverage. Their178.35MBderivative
field captures are intentionally retired after extraction;raw scalar metrics,
identities/hashes,source/model/reference data remain. Historical references to
those capture paths describe evidence at collection time;the retirement receipt
is current storage authority. No current retention claim requires those arrays.
[Scalar distributions](reports/recent-component-scalar-statistics-01.json),
[spatial coverage](reports/recent-component-spatial-statistics-01.json),
[retirement](reports/recent-prediction-retirement-01.json).
