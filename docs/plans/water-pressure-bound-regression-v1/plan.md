# Broader pressure-aware stopping regression

Status: Accepted. Measurements and records complete; no replacement.
Admission: start, user-authorized continuation of the next
pressure-aware regression and conditional independent validation sequence.
Canonical plan: docs/plans/water-pressure-bound-regression-v1/plan.md.
Owner/integrator: this task. Private branch experiment/water-pressure-bound-regression-v1
continues experiment/water-stopping-floor-v1 at56f07ace48624e7d2e00e0b2ab21a02779bd8340.
Retain for review/continuation, synchronize with new commits, retire after accepted
integration or explicit retirement. No push, rewrite, deployment or weights change.

Objective: determine whether the already-qualified sealed-grid pressure-error
bound generalizes beyond six consumed scenes, before spending another independent
validation set. Record methods, accuracy, actual work, latency, failures and timing.

Decisions:
- Freeze the late checkpoint and v2 pressure bound at1e-6 mean-free relative error,
  preserving original equation/divergence gates, cubic cells, history warm-up,
  fixed sealed-water physics, float32 network arithmetic and32-call maximum.
  No tuning, new model inputs, optimizer, training examples, fallback or refinement.
- Run all 58 original standard regression scenes plus both subsequent consumed
  eight-scene sets (late amended validation and the failed minimum8 validation).
  The original setup amendment remains historical; these populations are consumed.
  Keep elongated exploratory cases outside this standard replacement gate.
- Reuse authoritative completed reference trajectories after exact per-case scene,
  binary, checkpoint, normalization, single-call policy, stopping profile, receipt,
  frame/callback completeness and child-reaping checks. Perform a fresh one-frame
  native/reference/candidate fixture and compare reference physical frames with
  the archived prefix. Archived latency is not a current comparative benchmark.
- Run new native controls before candidate episodes through the existing bounded
  CPU phase/episode owners, two independent one-thread workers pinned to distinct
  cores. Preserve120s native deadlines,32-call cap,>=8GiB available memory,
  <=512MiB additional oracle address space and<=1GiB raw output. Sparse bounded
  same-RHS oracles execute only after the native child is reaped.
- Qualification gates: every scene completes and passes physical and declared
  pressure-bound stopping checks. For each of three populations separately
  (original58, late amended8, failed-minimum8 recent8), worst paired pressure
  deviation and mean/maximum network calls must not regress against its retained
  reference. This prevents hiding a recent population's regression in a larger
  aggregate. Report all constituent subgroups and case-level tradeoffs as well.
- Record serial uninstrumented reversed-order timing repetitions on three
  predeclared consumed representative scenes, with live reference and candidate.
  Exclude one explicit AI timing warm-up per episode; ordinary desktop services
  remain. Concurrent correctness timings support no isolated latency claim.
- Author and commit eight new independent scenes before final evaluation. Fresh
  AI validation is conditional on all three consumed population gates. Freeze the
  sole candidate before qualification, qualify all native fixtures/full controls
  before any fresh AI episode, and apply unchanged replacement gates. No
  final-informed refit/reselection, new threshold or claimed deployment activation.
- Rejection completes the scientific measurement and preserves the reference.
  Keep independent validation unconsumed if development fails. No blind CUDA
  refit while the inference-policy question remains unresolved.
- Retain JSON/CSV source/config/checkpoint/receipt bindings, per-frame metrics,
  sparse pressure/gradient/band checks, failures/retries, elapsed time and lifecycle
  receipts. Use the existing Tuldok catalog and live project activity; preserve old
  snapshots and records. Do not commit weights, arrays, raw simulation datasets,
  credentials or the four unrelated cohort/spectrogram files.

Write set: experiments/water-pressure-bound-regression-v1/**;
docs/plans/water-pressure-bound-regression-v1/**; tuldok-project.json; README.md.
Read-only reuse: stopping-floor run/episode/summary owners, learned_iteration,
native rollout and historical reference records. A necessary owner repair needs
an explicit write-set extension and evidence before continuing.

Milestones:
- M1 Accepted: reproducible population/reference binding and first real fixture.
  Gate: all 74 declared references verified, one real prefix parity observation,
  pressure-bound coverage, CPU-only inference, observed children reaped.
- M2 Accepted: all 74 candidate trajectories, per-population comparison and current
  serial timing. Gate: complete retained measurement, including negative results;
  replacement eligibility is a separate all-population decision.
- M3 Deferred (consumed maximum-work gate failed; fresh scenes unconsumed):
  independent fresh validation only when M2 replacement gates pass.
  Otherwise defer it explicitly, retain unused fixtures and record rejection.

Acceptance satisfied for the measurement task. Current phase: terminal; next
slice: none in this plan. All 74 scenes and 233 bound checks qualify; worst pressure
deviation improves in each population, but recent maximum calls 17 vs 11 reject
replacement. Fresh validation remains Deferred and its eight scenes unconsumed.
Serial timings complete. Saved-input replay indicates most extra calls are needed
for pressure accuracy; future matched CUDA training for fewer passes is proposed,
not executed. No deployment or final-informed refit.
Replan on invalid reference/setup, resource/lifecycle failure, bound violation,
changed numerical semantics or evidence invalidating conditional continuation.
Standards snapshot: snapshot:v1:8003a2b9-6b39-48ec-820f-fe66f59fc90a;
16 routed policies, zero unresolved facts. Records: execution-ledger.md, issues.md,
reports/. Measurement completion does not imply replacement qualification.

[Complete results](reports/summary.json) · [Population CSV](reports/population-comparison.csv) · [Timing CSV](reports/timing-comparison.csv) · [Hard-case replay](reports/hard-case-frontier-summary.json) · [Methods](execution-ledger.md).
