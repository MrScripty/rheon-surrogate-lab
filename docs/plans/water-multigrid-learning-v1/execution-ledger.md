# Execution ledger

2026-10-15: inspected clean HEAD 5bf5c2d and accepted previous tranche. Admitted
the recorded follow-ups on experiment/water-multigrid-learning-v1. Historical
results remain immutable. Two changes are controlled separately: per-evaluation
CPU stopping for both existing checkpoints, and multi-grid analytical training.
The broader cohort and architecture changes remain conditional on measured fit.

Implemented optional per-network-evaluation stopping in the existing corrector,
trajectory adapter and native receipt consumer. Captured recurrence owner is
reused without changing model configuration. Seven focused tests pass, including
an actual four-evaluation recurrence versus four separate evaluations: identical
pressure bits, four observed forwards in each, original metadata unchanged.
Existing two-cell RHS/divergence and budget checks remain passing. This is an
inference-policy measurement change, separate from training-population changes.

Added closed schema-v4 feedback with explicit grid counts, one cubic spacing per
grid, balanced private-RNG sampling and an exact 768 MiB cache projection.
Legacy analytical bank dispatch retains tensors/order/evidence. Three new
contract/eigenvalue tests, five historical analytical tests and six residual
contract tests pass. Scope explicitly includes the new test_multigrid.py and
training configuration files. The 10-update smoke covers every grid twice.

The complete authored bank qualifies on CPU: 640 entries across five grids,
worst analytical-eigenvalue versus implemented-operator relative mismatch
6.90286e-14. The original captured bank source independently reproduces every
legacy tensor, sample order and evidence byte; model RNG stays unchanged.
Resident tensor budget and the exact per-grid exposures are retained in
reports/bank-qualification.json. No optimizer or CUDA context in this check.

Controller environment correction: the Pumas interpreter lacks psycopg; the
Tuldok interpreter lacks NumPy for the observer. Both failed before Run
submission. The repository .venv has the established controller dependencies;
managed training still selects the qualified Pumas CUDA runtime. Both startup
logs remain under runs/. No extra model attempts are counted for these errors.

Added reusable configured CPU study driver. Every checkpoint episode uses a
fresh Python process, exact captured sources and a distinct output/log. Fixed
worker/core partitions prevent contention between candidate jobs. Concurrent
checks are numerical/throughput evidence; repeated timings must run separately.

Matched fine-stopping CPU study completed all seven native controls and fourteen
candidate episodes: 112/112 AI steps qualify, every native/controller child is
reaped. Latest weights need 22.875 evaluations on 40-cubed versus 29.75 for older
weights, under the same per-evaluation stopping. These overlapping timings carry
no latency claim. Older checkpoint development recheck passes 128/128 with mean
13.4375 evaluations, versus its prior block-of-four mean 14.96875. Retain fresh
latest-checkpoint development verification separately before final attribution.
The two existing native process failure/deadline tests also pass.

Fresh latest-checkpoint development replay again qualifies 128/128. With stopping
fixed per evaluation, the earlier/current means are 13.4375/12.6015625, a 6.22%
reduction attributable to changed training/checkpoint on this population. The
previous 15.81% comparison also included stopping granularity and is not a
training-only claim. Larger-grid qualification is short and development-only.

Tuldok enforces immutable manifest snapshots: reusing the previous project ID
for new content returned HTTP409. Preserved that snapshot and registered the new
multi-grid project a5e968a9ceb740c3b775ed14103bc519 in the same catalog. This is a
project snapshot, not a separate dataset catalog. Local CPU profile startup also
caught a stdlib module-name collision; renamed profile.py to cpu_profile.py and
retained its failed startup log. Neither issue is a model numerical result.

Smoke completed/drained training Run 6327bdba13f3465ab82474241f23d52f and fresh
CPU restore d488ff6ca3a84764ad9b33f9909d62fc. Ten CUDA updates, two analytical
exposures per grid, 488,651,776 resident tensor bytes, peak CUDA allocation
531,972,096 bytes; all weight/prediction bits restore exactly and artifacts
reopen/redownload identically. Full 5,000-update one-pass multi-grid sequence
was then submitted through the same workflow (source capture fb0eff6).

CPU profiles cover 16/24/32/40-cubed with 1/4/8 Torch threads, identical mode
mixtures and stopping. All sampled results satisfy the equation gate. Profiled
convolution and tensor-fill cost dominate; four threads help this mechanism,
but eight regress relative to four at 32-cubed. These microprofiles do not
prove whole-CFD latency or fastest production settings. Added optional 1/4/8
threads to the existing rollout owner and disjoint core groups to the study
controller for a matched native/AI follow-up. Defaults remain one thread.
Two process ownership tests still pass. Scientific source and deployment stay
separate; this extension is execution policy, not a model change.

Long-window native controls and current checkpoint complete 512 steps on all
three small-grid scenes, with 1,491/1,491 learned steps qualified after warm-up.
Forcing is explicitly quarter strength to keep the native controls physically
valid at the longer fixed timestep; this does not extend the original full-force
claim. Readiness of the broader cohort remains conditional on new fit results.

The optional four-thread trajectory fixture qualifies native and learned tracks
with one AI step after 15 native steps, explicit CPUs1–4 and observed four-thread
budgets. All children reaped. Retained separate-seed 120-mixture CPU baseline
uses up to all 23 independent worker cores while CUDA fitting proceeds; low modes
and arbitrary-frequency mixtures are distinguished rather than averaged away.

Full one-evaluation fit completed 5,000 CUDA updates at 44.3782 updates/s.
The retained post-update fitted equation ratio is .0082280, still above 1e-6.
This satisfies the recorded condition for the prepared two-evaluation training
trial: compare whether recurrence during training improves error reduction.
Admit its unchanged 5,000-update config, same seeds, real/analytical sample
exposure and learning rates; it uses twice the neural evaluations/voxel work.
The first trial's full CPU restore/validation qualification remains pending;
no learned acceptance is inferred from this training diagnostic. Its optimizer
has ended before the second job is submitted through Tuldok's serial owner.

Prepared the conditional cohort feedback contract separately from the running
captured trials. Schema v5 names train-correction-cell-rms: all cached real
training cell-states determine analytical amplitude, with common rho/dt checked.
It does not select an arbitrary first parent or use validation statistics.
Historical v3/v4 one-frame paths remain unchanged. Cohort preprocessing still
fits only real training data; that population change is explicit, not a
single-factor learning claim. Added independent weighted-RMS, authority and
physics-mismatch tests. Admission of a cohort Run remains conditional on the
current trials' actual iterative fit/development/grid evidence; no Run starts
merely because these configurations exist.

One-pass multi-grid development completes 128/128 physical passes, averaging
13.46875 evaluations (worst 23), versus current reference 12.6015625 (worst 26).
This is a mean-work regression on small saved states despite a better tail;
retain it rather than presenting every new training run as an improvement.
Larger grids and two-evaluation weights still need their full comparison.
Two-evaluation optimizer completed 5,000 updates at 23.5147 updates/s; fitted
post-update ratio .00309170 is diagnostic, not final CPU physical acceptance.

Prepared cohort bank independently qualifies all 640 entries, worst relative
operator mismatch6.44257e-14, global real-train correction RMS2.853573464078259 Pa.
Legacy bank remains bit-identical and model RNG unchanged. No cohort Run has
been admitted yet. Corrected the mode-probe description to 60 distinct mixtures
at two signed amplitudes (120 cases); generated inputs are unchanged.

Broader-cohort admission review: one-pass multi-grid weights pass known fit1/1,
development128/128 and all seven supported short grid scenes under the original
32-evaluation cap and unchanged physical gates. They improve40-cubed work from
22.875 to 13.625 evaluations but regress16-cubed from 10 to 12.75. Preserve the
current reference and report this Pareto tradeoff; this is not default promotion.
Those physical fitting/generalization checks satisfy the evidence prerequisite
for a bounded cohort qualification. Admit only the prepared10-update cohort
smoke plus CPU restore now. Full cohort fitting still follows smoke and the
one/two-evaluation comparison. This clarifies 'no material regression' as no
lost physical acceptance within the named compute cap, while retaining per-grid
work regressions explicitly rather than hiding them in an aggregate.

First full trial record sealed after all three managed Runs reopened/drained,
whole artifacts redownloaded identically, raw native checks, adaptive fit128-state
checks, 120 analytical signed cases, seven grid settings and three 512-step
trajectories. All20 native trajectory children are reaped/exit0. Raw output
remains0/128; iterative policy retains full physical acceptance. Three control
parities (initial parameters, normalization, real sampling) are exact. Forward
voxel-evaluation proxy139,776,000 is 3.4125 times the prior single-grid fit despite
equal forward/example counts; do not call the training compute equal.

Second fit's CPU development evidence passes 128/128 at mean 12.4140625 and
maximum 19 evaluations, versus reference 12.6015625/max26 and multi-grid one-pass
13.46875/max23. Independently seeded low-frequency40-cubed mixtures improve from
23 to 12.5 evaluations, while broad-frequency mixtures regress; report both.
Seven physical grid scenes and longer trajectories remain pending at this point.

Second trial completes all managed and CPU stages: development128/128, seven
physical grid settings and 1,491/1,491 longer-window AI steps qualify, with 20 native
trajectory children reaped/exit0. Forty-cubed means11.75 (fixed spacing) and
12.125 (fixed volume), versus reference 22.875; small16-cubed needs11 versus10.
Select two-evaluation training for the cohort experiment, retaining this tradeoff.
Raw two-evaluation outputs remain physically unqualified; final inference iterates.

Cohort qualification Run d25f7512a7294af9bfb0d7db4fc89a1f and CPU restore
cd403fc646f14bb39cb8ba6fa5cd51b0 complete/drain, reopen and redownload identically.
Ten CUDA updates, positive global real-train amplitude, all five bank grids and
exact CPU weight/prediction parity. Admit the prepared5,000-update two-evaluation
cohort fit at source capture c4e4a50:16 real simulations/32 states, uniform parent
then frame sampling, same analytical grid schedule and train-only global units.
Evaluate all 32 real train states separately from 128 development states. The
mode-probe fixture remains the original common physical inputs for comparison.

CPU preparation during cohort CUDA fitting: eight channels-last3D mechanism
probes cover16/24/32/40-cubed at 1/4 threads. All parameter values remain bit-exact
and sampled equation gates pass. Timings are mixed across sizes/threads and the
GPU fit overlaps, so no uniform gain or whole-track speed claim is supported.
Retain the profiles and optional diagnostic flag, without adding a rollout or
checkpoint-adapter layout migration. Isolated final timings retain the qualified
contiguous execution path and compare1/4 threads. No model/training change here.

Timing-readiness review found the CPU study published aggregate observations
only after a whole fixed-core partition completed; a serial study could hide
intermediate results until every episode ended. Replaced that scheduling join
with one in-flight job per core group and parent-owned streaming consumption.
Every completed episode now updates retained reports and live activity. This
also extends exclusive core leases to native controls with uneven durations.
Independent event-driven scheduling/failure tests cover progress, exclusion and
failure cleanup; existing native child lifecycle ownership remains unchanged.

Cohort completes all managed stages and CPU checks: fit32/32 at mean 9.125/max11,
development128/128 at mean 8.8828125/max12, seven grid settings and all 1,491 longer
AI steps. The early32-state and final32-state CPU predictions reproduce identical
numeric inventories. Relative to the preceding latest checkpoint, development
mean work falls29.51%, maximum work53.85%, and 40-cubed work56.28% (22.875 to 10).
Retain this as the development reference, with no application deployment implied.
Raw managed outputs remain unqualified; success uses explicit adaptive iteration.

All13 managed stages complete/drain and repeat-download identically: three 5,000-
update fits plus two 10-update smokes total 15,020 CUDA updates. All task training,
CPU qualification controllers and native children drained before the final timing
sequence began. The serial one/four-thread studies each use five physical scenes,
four restored checkpoints and three alternating repetitions. Per-episode reports
and live activity stream as completed; no concurrent numerical work is launched.

Prepared a final equal-budget CPU diagnostic to distinguish improved equation
accuracy from merely stopping at the same acceptance target. It compares the
preceding latest and cohort checkpoints after a fixed eight-evaluation policy
on the same128 development states, using independent native gates. It starts
only after both isolated timing studies drain. The reusable evaluator now accepts
an explicit1–32 network budget and fixed/adaptive policy; defaults preserve the
qualified32-cap adaptive path. No training, labels during inference or relaxed
physical targets are introduced. Fixed-policy receipts name a declared budget;
adaptive receipts retain observed per-case counts.

Both isolated timing studies complete with all 960 AI steps qualified and all 130
native trajectory children reaped/exit0. Four threads improve cohort latency on
all five measured scenes. At40-cubed, prior/new AI times are 1442.14/701.78 ms with
four threads, paired native 419.11 ms; one-thread prior/new3221.29/1470.35 ms.
Native still wins and scaling is not monotonic. Source/per-repetition timings,
resource budgets, whole-track scope and distinct equation targets are retained.

The subsequent fixed-eight CPU comparison uses identical128-case identities:
median relative residual 6.92990e-6 to 1.21265e-6 (5.71x improvement), worst
1.67301e-3 to 4.53286e-5 (36.91x); physical passes 11 to 46. Both policies still
fail some cases at eight evaluations. Adaptive32-cap acceptance remains128/128.
The experiment therefore separates equal-work accuracy from stopping policy.

Pressure-drift synthesis checks12 common-scene trajectories and retains60 actual
samples plus worst/final values. Drift is mixed at the same equation target;
cohort cube maximum 3.95e-6 exceeds the reference 5.70e-7, while rectangular and
compact maxima improve. These field differences are not equation residuals.
All finite-window physical checks remain passed. New datasets/weights, altered
physics targets, application deployment and final-test claims are absent.

Final project1fd1f315ecf8431abc10d16f9de546ce ('Rheon Surrogate Lab · multi-grid
results') is registered/reopened in the same cataloga7da4a8de53541a2a0e2aad08005bb71.
Its exact manifest hash and 64 evidence references are retained. Eight old source/
workflow shortcuts were retired from the index to respect the UI limit; their
files and all historical scientific results remain intact. The renewed observer
PID1482974 was argv-verified, interrupted, waited and reaped; final root activity
is completed 13/13. The original observer1444821 was likewise drained on renewal.

Lifecycle audit opens221 native trajectory receipts (all exit0/reaped) and 19
native diagnostic populations/1362 cases (all owned groups drained). New local
retention375,005,265 bytes stays below2 GiB plus the declared reserve. Source tests,
independent label/recurrence/scheduling fixtures, real CPU/CUDA integration,
whole-artifact custody, explicit timing conditions and fixed-budget checks are
retained. Reviewed the full linear branch range from 5bf5c2d; the profiler-name
mistake has an explicit corrective rename and no history rewrite. No merges,
publication, Rheon/Tuldok source edits, model weights or datasets enter Git.
Acceptance C1-C5 is satisfied for this finite development experiment. The
next-steps document prepares subsequent work and admits no additional jobs.
