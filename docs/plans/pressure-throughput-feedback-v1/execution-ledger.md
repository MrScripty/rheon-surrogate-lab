# Execution ledger

## Admission and scheduler discovery

The user authorized concurrent CPU/CUDA execution followed by the proposed
pressure-feedback sequence. Created separate Tuldok and lab branches from the
completed migration heads. Protected the original checkouts and existing PR89.
CPU preparation used four processes and regression had an explicit two-worker
cap; Tuldok's catalog coordinator executed one attempt synchronously.

Built `test_run_concurrency.py` through the existing real PostgreSQL/ExternalCase
and actual neutral subprocess custodians. Before the fix, the first attempt was
running/go_intent and the second stayed queued/prepared after5 seconds; the
test failed in9.659s. An initial isolated cluster used SQL_ASCII, rejected by
Tuldok's UTF8 contract before the test; created a separate UTF8 test database in
the same new private Unix-socket cluster. No original catalog was used.

The first receipt harness incorrectly resolved the virtual-environment Python
symlink into system Python, losing psycopg. Preserved scheduler-tests-01.json;
corrected interpreter selection to retain its absolute venv path. Tests-02 and
04 each retain an unavailable legacy TinyCNN-runtime test, failing at preflight
before scheduling. No runtime package was installed or substituted. All12 other
built-in lifecycle tests and all11 external lifecycle tests passed in those runs.

Added the standard-library executor behind the existing Runs owner, per-device
admission budgets, CPU affinity inherited from each executor thread, current
host/cgroup memory observation and optional captured recipe resource requests.
No numerical worker, model weight, physical threshold or database schema changed.
Unknown custody fences dispatch; executor futures retain ownership through drain.

Tests-03:5 capacity,3 concurrency,7 progress tests pass,46.689s total. GPU device
classification in the concurrency test is explicitly simulated; CPU custodians,
job contexts, cancellation and PostgreSQL are real. Tests-05:6 capacity,3
concurrency,9 recipe,3 bound-Torch and4 captured-CPU contract tests pass. Its CPU
concurrency test also observes disjoint actual custodian affinity. Final
source-frozen test selection is ongoing; original receipts are retained unchanged.

## Complete CPU workload benchmark

Used the current migrated fixed control package and native binary through the
existing rollout owner,16 consumed development scenes,15 native warm-up steps
and8 AI steps per scene. All worker processes use one CPU thread/core and fresh
model startup with complete installed-runtime verification. No training or unused
validation scene is involved. Raw outputs are under
`runs/pressure-throughput-feedback-v1/cpu-benchmark` in the artifact root.

Two workers:110.374s, conservative summed process-tree RSS peak1,786,155,008B.
Eight workers:71.309s,4,762,517,504B. Initial comparison failed because the
signature included each attempt's intentionally fresh request SHA. Inspection of
all frame differences found only reference/AI seconds and diagnostic request SHA;
all physical values and network-call sequences were exact. Retained the failed
raw receipt and a separate corrected comparison (`cpu-throughput-02-vs08.json`).
The corrected projection excludes precisely those metadata fields; a focused
regression proves physical pressure differences still fail the comparison.

Sixteen workers:53.366s,7,544,217,600B summed RSS;49,210,724,352B minimum
available RAM. All16 numerical frame projections and all128 AI call-count entries
match the two-worker baseline exactly. All workers/native children reaped.
One repetition per configuration so far; these are complete short-workload
throughput observations, not full74 regression timing or isolated inference
latency. Startup hashing slows individually with concurrency, but aggregate
completion improves. Reverse-order repetitions are in progress.

## Scheduler acceptance and repeated throughput

Two repetitions in order 2,8,16,8,2,16 gave median complete times
112.2521252905s /67.3790247255s /54.3725686680s. Sixteen independent
one-thread workers are 2.064499214x faster than two on this shortened workload.
All physical frame values and call sequences match exactly. See
reports/cpu-throughput-summary.json for every raw receipt hash, resource scope
and timing. This does not establish full74 speed or isolated inference latency.

Independent standards/spec reviews exposed a real startup fence race. Retained
the failing actual-child regression (scheduler-fence-red.json), then fixed both
startup checks and coordination of drain-failure fence writes with dispatch.
All13 builtin lifecycle tests pass (scheduler-tests-10.json); actual external
lifecycle11 and concurrent cancellation/shutdown3 also pass (tests-09). The
unavailable legacy TinyCNN numerical runtime is explicitly excluded, not passed.
Both reviews close with zero remaining findings. Tuldok commits are
5e18837e1b580b5b831118adb0b0dc8b1dbba4e7,
7fbc82d9abe80182cfb3186a6691649298e98eb7 and
a9e99d28fb88320323dbcbfb63ae8224b89043ff.

Actual-device qualification03 completed in270.023226304s with1.255602121s of
measured CPU-computation/CUDA-optimizer interval overlap, disjoint assigned
cores, one float32 CUDA update and exact fresh-CPU checkpoint restore. Peak CUDA
allocation767402496B. All three Runs completed/drained; executor joined. Attempts
01(relative path) and02(obsolete Project method) failed before any optimizer
update and remain retained. This is qualification, not a scientific fit or
model accuracy result. See reports/actual-device-overlap-03.json.

Low-band feedback source passes five meaningful CPU tests in0.646s (2.070s
process total), including an analytic odd-grid cosine, bounded weak-mode loss,
zero-weight loss/gradient parity, first/final derivatives and strict policy
validation. No extra network passes. See reports/band-feedback-tests-01.json.
CUDA qualification and full matched fits are pending.

## Live scheduler and frozen training comparison

Deployed exact Tuldok a9e99d28fb88320323dbcbfb63ae8224b89043ff as current-04.
Verified original PID/start identity, zero active Runs, graceful SIGINT drainage,
HTTP health and all six retained Run states unchanged. Switch10.464s. Same
original catalog, owner session and Project roots. Operator profile:6CPU slots,
4GiB CPU minimum,8GiB CUDA minimum,8GiB host reserve.

The current training owner now freezes v10 configurations and current typed
Projects without invoking closed migration gates. Reuses exact train-only16
parents/32states and qualified direction bank; one already-consumed validation
probe is a diagnostic only. Control weight0 and challenger weight0.1 use max DCT
mode3 per axis, DC excluded, signal floor0.1, first/final existing predictions.
Both5000update fits retain identical seeds, sampling and learning-rate changes
at3000/4500. Only the band-feedback weight differs scientifically. No new data,
no optimizer resume, no model promotion, no untouched validation access. A
one-update challenger qualification precedes both full fits.

Version10 permits an explicit30–1800s trainer deadline; all historical policies
keep their prior600s cap. This provides a bounded allowance for additional
spectral reductions, not an increased update budget. All40 affected trainer
tests pass in1.107s. Current Tuldok SDK owns checkpoint and field evidence;
finite final-update clocks replace the one-update migration-only glue. Each
API mutation retains dispatch intent before sending and never auto-replays.

Assembly01 failed before Project publication or execution because a relative runtime path was not normalized before making its rooted reference. Preserved partial material; fixed path normalization and use fresh attempt02. Zero optimizer updates.

Assembly02 reached current extension validation and rejected contribution IDs because they require namespace:role rather than namespace.role. No catalog write or optimizer update. Corrected identity syntax before a fresh03 assembly; provider and recipe schemas remain current Tuldok owned.

Prepared the reusable full-development episode owner using the current rollout, existing independent neighbor-stencil oracle and existing physical/pressure gate verifier. Full episodes retain sparse spatial-band/gradient diagnostics and paired trajectory metrics. Worker count, reserved/excluded cores and deadline are explicit configuration; shortened benchmark defaults remain unchanged. One comparison-signature regression passes. The first new-weight fixture compares exact initial inputs while deliberately allowing predictions to differ.

Full control assembly01 rejected the restore recipe deadline before any execution: current ordinary CPU recipes cap600s while bound training recipes permit2400s. Corrected only restore custody to600s; full CUDA training remains explicitly1800s. Partial material retained, no Run admitted.

Assembly02 also rejected the CUDA deadline: inspection showed the2400s exception
is exclusively the captured-CPU large-output profile, not bound CUDA recipes.
Earlier expectation of a longer bound CUDA deadline was incorrect. Preserve the
existing600s contract for both fits; remove the unused v10 deadline extension.
The prior control needed417s of optimizer time, leaving measured room for the
bounded projection cost. If this limit proves insufficient, retain the failure
and extend execution policy deliberately in a separate change. No full fit yet.

Low-band CUDA qualification completed one optimizer update and fresh CPU restore
in the original live catalog. Runs20c3de2d9be14873a5fa0fc9ae391323 /
2d79680874d343c7978c6249667b2df9 completed/drained. Initial readback attempted
to use an owner token through agent-only bootstrap; no numerical failure.
Corrected with an owner-operation adapter to the same existing artifact chunk
verifier and performed read-only collection of the acknowledged terminal Runs.
No training or admission was repeated. Full artifact parity passes. Complete
command interval including inspected readback recovery337.833s; numerical
timing remains separately retained in receipts.

Regression fixture01 failed before simulation because the retained build's
dependency selection bound the pre-commit base, while Rheon subsequently committed
the identical source. Fixture02 independently measured every complete compiled
input and license byte against the build receipt, with current commit recorded;
passed7.527s,760549376B RSS. Historical build receipt remains unchanged.
This verifies input bytes without asserting that advanced Git metadata matches
the original base. Full74 case failures will be retained independently; memory,
deadline or custody failures still stop the pool.

## Concurrent complete development baseline

All74 current26b cases passed,8523 AI steps and233 sparse independent checks.
Sixteen one-thread workers completed in737.499358296s, including startup, native
controls and independent oracles. Peak conservative summed RSS11227897856B;
minimum available RAM46977703936B. Every call sequence and worst paired pressure
metric is exact to the independently reproduced historical74case result. All
worker/native children reaped. CUDA control admission/startup and optimizer work
overlapped this CPU phase. Retained migration workers previously took1708.119
and2462.151s for two fixed shards; this historical maximum versus737.499s is
3.3385x faster, but is observational, not the matched repeated short-workload
benchmark. New episodes additionally retain gradient and low-mode diagnostics.

Both independent lab reviews found the same infrastructure-failure classification
issue; Standards also found partial readback could not resume after one completed
download. Waited for the running74pool to drain before changing its source.
Only an explicit drained native numerical rejection may now be retained; memory,
timeout, source, protocol, orphan and all unclassified failures stop the pool.
Readback reuses only regular nonsymlink files with exact published size/hash and
refuses to overwrite mismatches. Tests cover interruption after first download,
corruption, and scientific versus infrastructure rejection. Four targeted tests
pass. No original baseline failed or changed; all74passed before this correction.

Replaced only the verified old observation publisher PID536521/start6106702 with
a source-only current-experiment publisher, preserving previous files in the
artifact root. Existing main Tuldok Project and already-open Rheon preview paths
stay usable. Pressure frames are actual completed sparse captures, cubic and
hash-bound; no velocity/tracer field is fabricated. No catalog/Run authority is
assigned to this observer. The isolated scheduler-test PostgreSQL fixture was
stopped after completed qualifications; original live database untouched.

Review of the failure-classification patch caught an accidental extra positional
argument in the oracle module loader, introduced after the full74baseline had
finished. Corrected before running another case; a new real first-frame fixture
passed8.216s,760303616B RSS, including sparse independent oracle validation.
No candidate result was produced using the invalid loader. Both original
review findings and this introduced source error are corrected.

Prepared a finite233state spatial frequency/amplitude audit to use independent
CPU capacity while CUDA continues. The audit bins mode amplitudes and errors
rather than retaining a large state-by-mode tensor. Separable two-input DCT
contractions retain the previous OOM fix. A known cosine has0.5Pa² signal energy,
0.005Pa² error energy and exactly10percent relative error; analytic test passes.
This is descriptive analysis of consumed autonomous states, not a causal noise
floor experiment or a temporal-frequency intervention.

## Evidence-driven CUDA deadline replan

The full control reported updates through4500, with losses matching archived
control values (including exact2700loss0.007556987421906473), then its neutral
custodian returned reason=wall_timeout, drained=true/echild=true, leader=-15.
No completed checkpoint/accuracy result is claimed. Retain original failed
Run f8e6bd15f5ba4e6bbd5571071ecb2798 and explicitly stopped its Study via a new
journaled mutation; no existing mutation is replayed.

The measured600s execution bound is insufficient for the current complete
worker. Replan one infrastructure slice: shared managed-CUDA maximum1800s in
recipe admission and neutral custody; keep ordinary CPU600s and captured-CPU
large2400s. Both fresh matched fits retain5000updates/data/seeds/LR schedule.
No resume, no accuracy gate change, one CUDA slot. Isolated recipe9, bound-Torch4
and custody10 tests pass, with both1801CUDA and601CPU rejected. Trainer40tests
pass; v10 alone admits an explicit bounded1800s resource request. Captures use
the current custody validator and material-derived extension namespaces, preserving
previous immutable provider/Study identities rather than replacing them.

The reviewed CUDA-only deadline change1688f7d9fc6f941d8a23d8b077df6af4213f683a
is deployed as current-05 after zero active Runs;15.765s switch, all nine Run
states preserved. The next control attempt is a new Study/Run with fresh
initialization; previous failure is retained. Source-only observation selects
the latest explicitly materialized attempt, not an assumed success.

The16-worker spatial audit completed233consumed states in2.786784348s, no
simulation or optimization. Weak low modes (coefficient/RMS1e-6–1e-4) have
relative RMS error3.31833e-4 versus3.49097e-6 for coefficients/RMS>=1e-2.
Their absolute error energies are6.9499e-14Pa² versus6.57076e-9Pa²: high relative
weak-mode error does not dominate total pressure error. The aggregate is
descriptive over autonomous states, not evidence of a causal noise floor.
Raw finite per-state bins remain in the artifact root; small retained summary
is reports/current-233-spatial-amplitude.json.

Prepared both fresh matched Studies through the live catalog. Control Run
9c9777fcb4b541e98dc03e66c1df4f73 and challenger4f5bfea59ebd474d934ca967ff01927c
use separate captured Models/Trials; one CUDA slot orders execution. This allows
control CPU restore to overlap the following GPU fit. No second GPU execution
is permitted by the active capacity policy. Current observer follows the latest
explicit attempt; completed/failed prior attempts stay retained.

CPU preparation profiled complete installed-runtime validation without importing
Torch or changing any runtime file. cProfile24.665s included13.631s cumulative
manifest validation, millions of Path operations and only1.762s direct hashing.
A separate unpromoted prototype caches canonical path components while retaining
all content hashes, topology, links and owner checks. Verified26818containment
pairs and both full inventories; manifests byte-identical. One illustrative
reference/prototype pair took9.875s/6.747s. Order/cache/load are uncontrolled;
no isolated benchmark or deployed speedup is claimed. This is prepared subsequent
infrastructure work, not a change to either running numerical experiment.

A completed new control may reuse this turn's74case campaign only after exact
weights, executable/numerical/runtime bytes and physical/stopping contract match,
plus a new current-package first-frame all-array check. Any numerical difference
requires a fresh74campaign. Evidence reuse will be labelled explicitly, not
counted as additional simulations or an accuracy gain.

## Completed matched control and reuse proof

Control Run9c9777fcb4b541e98dc03e66c1df4f73 completed5000updates; fresh CPU
restore5c5b54957b1544ba87240263aba9a8e6 completed with exact artifact parity.
Complete Study command1015.366106756s; optimizer362.149376154s, trainer
365.771807598s, CUDA peak770565632B. Weights exactly reproduce26b6623...
and normalizationf73b6cf... . These are observed timings, not an isolated
training speed comparison.

Packaged the restored result and ran a new current-package first-frame fixture:
8.200s,760426496B RSS. All numerical/executable/runtime package files, physical
contracts and stopping policy match the just-qualified control, and every fresh
first-frame input/output array is byte-exact. Only id/version/provenance metadata
differ. Reuse the already executed74cases/233oracles, explicitly labelled with
reports/control-equivalence.json; do not count another74executions. The complete
control fixed-call manufactured amplitude diagnostic finished3.976s;370016
parameters,1485452B weights. Challenger still requires fresh qualification.

CPU qualification budget is explicit1500s per native child and1800s per complete
worker for the candidate, allowing up to32calls over long512step scenes. This
does not relax any physical/model criterion; a time failure remains an execution
failure. Ordinary managed CPU restore still has its separate600s limit.

## Completed CUDA challenger and initial CPU qualification

Challenger4f5bfea59ebd474d934ca967ff01927c completed5000updates, then CPU
restore210a52fd86524ebf9411fed3aae2dd4b completed with exact artifacts.
Weights2a6be6b42a4a3b0baf63e0b58a878e3138cc4af490c3fdb53b0a602ecd6b9788;
optimizer397.049598455s, trainer400.858344044s, CUDA peak837691904B. Complete
Study command1161.684300094s includes waiting behind control. Both actual
initial checkpoints, inputs, normalization, runtime and sample-order hashes match;
only feedback weight differs scientifically. No native/model physical gate changed.

The new package first-frame fixture passed in7.652s,760406016B RSS,6network
calls and3.1803889258e-7 paired trajectory pressure error after15native warm-up
steps. Full74qualification follows this gate. Fixed-call manufactured analysis
finished3.920s. At8calls on33x11x9, pure-low1Pa coefficient error fell from
3.85696e-5Pa to6.91840e-7Pa (55.75x); high-only leakage fell8.83150e-6Pa to
4.46795e-6Pa (1.98x); mixed low0.001Pa/high1Pa low coefficient error fell
8.87867e-6Pa to4.46870e-6Pa. These are matched stationary diagnostics, not yet
autonomous trajectory acceptance or a universal spectral improvement claim.


## Live observation recovery during M4

An episode receipt may be observed between file creation and completed JSON.
The read-only observer now counts that source as unreadable and retries; preview
refresh errors publish unavailable without inferring scientific results. One
meaningful partial-to-complete receipt regression passes in the existing lab
venv. System Python lacks NumPy and is not the observer environment. Independent
standards/spec reviews report no findings. Restarted only the exact owned observer
PID/start identity; original catalog, Project and Rheon paths retained. Public
receipt: reports/observer-refresh.json. Scientific worker source closure unchanged.

All84matched manufactured rows remain external artifacts. Retained complete
cross-sample summary reports/spatial-comparison.json: among21samples, low-mode
absolute error improved/worsened15/6 at1call,10/11 at4,18/3 at8 and14/7 at12.
The favorable33x11x9example is not a universal improvement claim. Full autonomous
population gates, rather than selected manufactured examples, decide eligibility.

Serial timing configuration selects consumed scenes8,10,12,14:16³,24³,32³,40³
cells in a fixed1m cube, each at its corresponding16/24/32/40cells per metre.
This measures resolution refinement, not growing volume at fixed spacing. Every
cell remains cubic. Eight AI steps after15native warm-up steps; complete paired
step/adapter/protocol cost, with startup separate. Broad work must drain first.


## Full M4 development result

Completed74candidate trajectories,8523AI steps,233independent sparse checks in
723.304438136s with16one-thread workers. All physical gates passed; all parent
workers/native children drained. Peak conservative summed RSS11783892992B,
minimum host available47515836416B. The control737.499s campaign is reused only
through the exact numerical/package and fresh-frame equivalence proof, not
counted again. Uncontrolled wall difference is not isolated inference speedup.

Each population independently compares against reference-current, late-pressure
and harder-data control. No model is eligible for the unused eight. Original58
calls7.7351->7.7245 mean and12->10max versus control; worst trajectorypressure
4.68189e-6->4.48189e-6 improves, but remains above prior accurate3.14479e-6.
Late-amended8worst1.262996e-6->1.536552e-6 regresses; minimum8-failed8max11->12
regresses although worstpressure2.385126e-6->1.573261e-6 improves. Preserve the
challenger as a rejected experiment; no thresholds relaxed, deployment promotion
or unused scene consumption. See reports/regression-summary.json,csv and
qualification-decision.json. Independent standards/spec read-only review applies.

All233candidate oracle defects<=1.646e-13; every pressure bound covers its
independent oracle. Median same-RHS meanfree pressure error2.36459e-7->1.48805e-7,
median gradient1.51305e-7->1.16380e-7, but maxima regress. These compare each
model's autonomous trajectory states; they are not matched-RHS causal effects.
The separately retained manufactured fixed-input diagnostics provide that control.


## Serial timing and terminal closure

After full candidate pool drainage, ran one core22, one episode at a time:
control rep1, lowband rep1, lowband rep2, control rep2. Each command used
`taskset -c 22 python3 -B experiments/pressure-throughput-feedback-v1/benchmark.py
--config <artifact-root>/<model>-timing-config.json --workers 1
--output <artifact-root>/<model>-timing-rep<N>`, with the exact rep1 receipt
passed as `--baseline` for rep2. Retained configs select consumed scenes8/10/12/14;
15native warm-up+8AI steps, fixed1m³ volume,16/24/32/40cells per metre. Benchmark
selects the Pumas interpreter/site-packages from the captured runtime manifest,
sets OMP/OPENBLAS/MKL/RAYON threads1 and hides CUDA for inference. Source remains
unchanged across repetitions; second repetitions prove exact physical outputs and
call sequences. No sparse capture/oracle runs in timing. Full paired-step costs
include iterative adapter, IPC and logging; startup is separate.

Complete measured batch times73.038174058,73.593580000,72.585434462,
73.632448845s. All child processes drained. Serial report binds each complete
campaign/config/source hash and every raw frame/callback hash; includes memory,
startup, step means/medians, calls, affinity and complete child times. AI remains
slower than native full steps across this tested range; ratios are not monotonic
with resolution. Full table and timing limits in results.md. Independent reviewers
recomputed all16rows and table aggregates without simulations; no remaining
findings. Corrected a malformed documentation SHA before final commit.

Read-only final original-catalog lookup confirms both CUDA fits and both CPU
restores completed/drained. Final qualification observation published to existing
Project/Rheon paths, candidate rejected/no holdout used; exact owned observer
stopped after publication. Original applications stay open with recorded pressure
preview, honestly marked completed. All scientific/serial jobs complete; isolated
fixture database stopped. No model weights, simulation arrays, credentials or
runtime packages added to Git. No promotion, push, merge or additional fit.

Next proposed science: smaller feedback weights0.01/0.03 with matched controls,
then independent late-correction-state or mixed-amplitude training ablations only
if necessary. Prepared runtime path prototype remains unpromoted infrastructure.
Keep scheduling/portability changes separate from scientific objectives and retain
all failures and unchanged population criteria in subsequent work.
