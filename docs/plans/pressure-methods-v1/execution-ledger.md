# Execution ledger

## Scope and preserved controls

The user authorized all eight cohorts in [request.txt](request.txt), including
follow-up tests for promising results and recorded outcomes. The completed campaign
is [campaign.json](campaign.json), with [final results](results.md). Historical
preparation receipts below retain their original pending states. Earlier nine-fit results remain retained; no
old fit, experiment directory, chart or checkpoint has been overwritten.

## Frozen-model correction screen

Completed all three checkpoint workers: C53, C44 and C37. Seven numerical
qualification tests passed. All pure-neural residuals exactly reproduce their
prior measurements on every one of the 128 consumed requests at 1/2/4/8 passes.
The comparison includes 240 manufactured modes and 32 separately specified mixed
cases. Work was serial, CPU-only, one thread, and every worker was reaped.
[Results](cohort-02-results.md), [raw gates/hashes](cohort-02-results.json),
[execution](cohort-02-execution.json).

Numerical correction improves some error/call-budget tradeoffs but does not make
the tested four-pass hybrids satisfy all 128 requests. The full numerical
spectral control passes all requests at substantially lower prepared-input cost.
That control triggers native follow-up; it does not qualify arbitrary geometry
or prove a learned architecture improvement.

## Native producer and larger-domain data

The versioned producer extends the qualified local capture source with a
262,144-cell bound, two/four substeps, and matching physical snapshot times.
The underlying advection, forcing, pressure, viscosity and scientific gates are
preserved. Build, tests and Clippy passed against the selected current Rheon.
Generated source, exact edits, dependency hashes and binary identity are retained
under `runs/pressure-methods-v1/native-01`.

The first capture wrapper attempt stopped at an import-path error before making
an output directory or starting a native producer. Its stopped controller and
logs are retained in `native-pilots-attempt-01.json` and `runs/`. The corrected
wrapper then completed all nine fixtures and nine full pilot trajectories; all
native children were reaped. [Execution](native-pilots-execution.json).

Qualified training-only pilots cover 32³/64³ at 8/16 cells per metre, a 16³
matched-time comparison at dt/dt/2/dt/4, and 16³ at 4/32 cells per metre.
[Pilot results](native-pilots-results.md), [evidence](native-pilots-results.json).

The larger-domain view contains all 16 original small-grid trajectories and all
four new larger-grid trajectories: 40 snapshots, both original spacings.
Two shards of about 19.5 MB each passed lossless roundtrip checks.
[View identity](domain-training-view.json). This is qualified data, not a trained
or evaluated larger-domain model. Versioned training consumption, shape-budget
qualification, matched fits and independent per-tier evaluation remain pending.

The temporal pilot shows smaller pairwise differences for velocity, tracer and
pressure after halving dt. This is timestep sensitivity evidence from one force
profile, not exact-solution accuracy. It warrants further teacher-data coverage;
fine-step pressure arrays must not substitute for unrelated coarse-step labels.

## Numerical and periodic trajectory follow-ups

The full numerical solver passed its first actual post-warmup native request in
`runs/pressure-methods-v1/followup-fixture-full-01`, with zero network calls.
The follow-up matrix completed under [followup-plan.json](followup-plan.json):
three shuffled serial repetitions of the full numerical, eight-pass C53 and
native algorithm controls; periodic numerical solves every 4/16/64 requests; and
development checks of the four-pass before/after hybrids.
[Terminal worker receipts](followup-execution.json).

All 14 trajectory workers completed and drained. Native algorithm, full numerical
and eight-pass C53 controls each completed three 512-step repetitions. Median
instrumented diagnostic-loop times were 18.119s, 16.798s and 43.171s respectively.
This includes both reference/candidate tracks, reference diagnostics and IPC;
these are not single-track production timings. Periodic numerical correction at
4/16/64-request intervals completed all 497 post-warmup steps, with worst pressure
trajectory errors below 7e-7 on this consumed development scene. Both four-pass
hybrids rejected their first pressure request, so neither is qualified despite
its offline improvement. [Follow-up results](followup-results.md),
[raw comparisons](followup-results.json).

The existing diagnostic wire exposes pressure trajectory error. Velocity/tracer
drift instrumentation remains to be added before claiming completion of the
drift requirement. No model was promoted and no unused whole-scene holdout was
accessed in these stages.

## Larger-grid numerical model qualification

The versioned reader loaded all 40 mixed-domain snapshots and checked all target
and incoming-residual closures. A captured model with only its cell ceiling
extended to 262,144 reproduced all 512 prior small-grid case/pass residuals
exactly. Two forwards and backward on a real 64³ case produced finite loss and
gradients, with unchanged parameters, zero optimizer steps and no CUDA. Peak
worker RSS was 1,208,262,656 bytes. This qualifies numerical shape handling and
differentiability, not trained larger-grid accuracy or managed training admission.
[Qualification](domain-model-qualification.json).

The first model qualifier encountered a generic module-name collision before
field qualification. Its failure is retained in
[attempt 01](domain-model-qualification-attempt-01.json); an explicit reader-module
load resolved it. The successful qualifier used the same captured model sources
and data, with no repeated training.

## Remaining campaign work

Complete the trained domain, teacher, numerical-coarse/fine, sparse-anchor, revised learned
coarse/fine and spacing-conditioning comparisons, with appropriate controls,
qualified restores, small-grid regressions, cost and drift measurements. Follow
promising signals with additional tests. All eight requirements stay active;
completed pilot data or one successful control does not close the campaign.

## Renewed access and matched training campaign

The user explicitly instructed the agent to renew the expired owner session.
The documented local CLI login and HTTP session exchange renewed the same owner
and catalog; the token remains in the private 0600 session file. No credentials
were printed. The control CUDA fit was not repeated: a separate recovery journal
verified the original 401 denial and absence of a restore graph, then completed
its fresh CPU restore. All five retained artifacts match exactly. See
[control custody](domain-control-training-custody.json).

The expanded-domain fit also completed 1,600 CUDA updates and a matching fresh
CPU restore. Its first controller stopped before creating a Study because the
Project preview exhausted its cumulative hash budget. The replacement controller
explicitly hashes every declared input locally, preserves the actual API preview
statuses, and retains the server's complete recipe material/preflight/admission
checks. The original interrupted command and its sole Project-open journal are
preserved; no training was duplicated. Independent 140-request evaluations of
the matched domain pair are running. Their timing includes final residual
evaluation and cannot be inserted into the original chart without matching its
timing scope.

All 118 broader native trajectories completed and drained: 48 matched teacher
trajectories, 64 spacing trajectories, and six new development-validation domain
trajectories. Teacher triplets use one, two or four native substeps at matched
physical times; the model timestep stays 1/60 s. Fine pressure and RHS fields are
converted to the unchanged model equation, independently reconstructed from the
pre-projection velocities, and paired with a common coarse-history initial
guess. This tests pressure requests from finer teacher trajectories; it does
not claim to reproduce a fine-integrated coarse simulation transition exactly.
All three teacher views passed physical closure and the bounded training reader.

The broader spacing view contains all 16 original grid/force profiles at each
of 4/8/16/32 cells per metre. Matched conditioning fits use the same two-channel
architecture, identical initialization and data: the second feature is either
zero or q/train_q_RMS times log2(h/(1/8 m)). The original C53 tensor bytes are
preserved and the new first-convolution column starts at zero.

The numerical-coarse and sparse-anchor implementations passed six independent
mathematical tests. Numerical coarse correction uses conservative transpose
restriction and the actual Galerkin fine operator. Sparse anchors use a full
global forward DCT, all-mode division and selected inverse reconstruction; all
that cost counts. Neural infill is pinned to preserve the computed anchors.
Both projects passed actual objective, recurrence, label-isolation, inventory,
and largest-grid backward qualification without optimizer updates.

The revised learned branch replaces the earlier unconstrained coarse mixing
with 63 positive learned low-mode gains and an actual fine-residual contraction
gate. It uses no analytical inverse eigenvalues. The original fine-network
tensors are preserved, and qualification verifies the added parameters and
differentiable objective. The gate covers the coarse proposal; deployment and
long-trajectory accuracy still require evaluation.

The remaining qualified fits are owned by two finite serial controllers under
`runs/pressure-methods-v1/teacher-spacing-sequence-01` and
`runs/pressure-methods-v1/hybrid-spacing-sequence-01`. Each stops on a failure
and preserves its mutation journal; there are no automatic fit retries.

Velocity/tracer observations are being built in an isolated Rheon copy. The
selected Rheon and all earlier native capture bindings are unchanged. A read-only
method measures committed independent tracks. The versioned diagnostic extends
the step cap to 3,600 and the declared paired-history budget to 384 MiB, preserving
the physical acceptance gates. Build/test qualification and observed native
parity are required before drawing drift conclusions.

## Independent drift observations and promising follow-ups

The first isolated observation build compiled, but library-test discovery stopped
because required example files were absent from the copied crate. It was not
used for scientific measurements. A fresh complete copy under
`drift-observation-02` passed 13 diagnostic Rust tests and Clippy. Its native
physical fields match the original probe exactly on 16 frames. The selected
Rheon source remains unchanged. [Qualification](drift-observation-qualification.json).

The drift matrix runs two disjoint single-CPU workers, with three development
seeds and declared 1% relative L2 thresholds for pressure, velocity and tracer.
Accepted frames contain host-observed committed velocities and tracer values;
rejected pressure requests do not advance the simulated history. The first
512-step matrix and selected 3,600-step follow-ups are retained under
`drift-matrix-01`; the execution report is `drift-execution.json`.

The expanded-data model completed 113 post-warmup predictions on a new 32³ native
trajectory. The matched small-only model rejected its first pressure request.
The expanded model's maximum pressure/velocity/tracer drift in that test was
about 1.50e-7 / 1.18e-7 / 3.41e-8. Its small/rectangular regressions still prevent
a general qualification. [Domain results](domain-training-results.md).

The trained numerical-coarse model qualifies 103/128 standard requests at four
passes and all 128 at five passes (worst residual 8.38e-7). At eight passes its
worst residual is 9.07e-10. Intermediate-pass and native follow-ups are therefore
required; added numerical cost is included in measured inference.

Frozen C53 with one low-mode correction before six neural passes qualifies all
128 standard requests (worst residual 8.90e-7). It also completed 497 native
post-warmup steps, with pressure/velocity/tracer drift below 1e-6. A separate
3,600-step follow-up uses a declared 900-second native deadline based on its
observed 512-step cost. Its plan is `six-pass-long-followup-plan.json`. Seven-pass
measurements additionally qualify before/after C44; the original extra-evaluation
plan remains unchanged and these separate follow-ups are retained explicitly.

Twelve further validation trajectories cover three new force/grid profiles at
each of 4/8/16/32 cells per metre. All captures completed, passed independent
physical gates, and drained. Their data never enter any optimizer. All restored
models are being compared on this same independent spacing population.

The revised learned branch improves large-cube accuracy but leaves two standard
requests above tolerance at eight passes. Its added logits moved only -0.00431
to +0.00658 at the inherited learning rate. A bounded follow-up keeps all fine
weight rates fixed and gives only the 63 added logits a 1000x multiplier. The
actual parameter groups are disjoint/complete, and CPU qualification verifies a
finite nonzero coarse-head gradient with no optimizer step.
[Qualification](coarse-head-rate-qualification-02.json),
[follow-up plan](coarse-head-rate-followup-plan.json). One earlier preparation
was rejected for an overlong extension namespace before API mutation. A first
optimizer-group wrapper qualification stopped on an input-slot lookup error; its
completed base-objective receipt is retained, and the corrected full qualification
uses a new output. No fit was repeated for either preparation issue.

Exploratory timings so far overlap other campaign work and differ between CPU
cores. They are suitable for including numerical overhead, but a quiet matched
repeat is still required before claiming a small speed improvement.

## Final bounded-campaign completion

All eight cohorts are complete and tested. The final artifact audit verified
12 managed fits (19,200 CUDA optimizer updates), exact five-artifact CPU restore
parity for each fit, all 140-request primary evaluations, 55 drained native
episodes, independent spacing comparisons, and quiet single-CPU timing with
three repetitions per request. The additional frozen-policy native workers
finished before quiet timing began. The live campaign activity is terminal.

The trained numerical-coarse six-pass policy has worst standard residual
8.57e-8 at 19.627 ms, compared with 1.45e-7 at 22.160 ms for frozen C53 eight
passes in the same quiet prepared-input scope. The five-pass coarse policy
qualifies all 128 requests at 16.268 ms and completes its native follow-up.
Frozen C53 with low-mode pre-correction and six passes qualifies all requests
at 17.098 ms and completes 3,585 post-warmup steps in a 60-second simulation;
maximum pressure/velocity/tracer relative L2 drift stays below 1e-6.

The full spectral numerical control remains the best measured pressure solver
on the current sealed uniform benchmark (about 0.360 ms and residual 5.53e-15).
Native times are diagnostic dual-track costs; no production application speed
claim or model promotion is made. The head-rate follow-up improves the revised
learned branch but does not beat the best numerical hybrids. The final
[results](results.md) and [machine-readable evidence](results.json) supersede
historical preparation-stage pending labels.

The final read-only [live catalog audit](live-run-completion-audit.json) confirms
that all 24 CUDA-training/CPU-restore Runs are completed and drained. Quiet
pressure-policy timings use warm geometry caches; one-time setup and application
startup are excluded, with first-invocation measurements retained separately.
