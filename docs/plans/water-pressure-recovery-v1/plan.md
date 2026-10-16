# Recover pressure accuracy at lower correction counts

Status: Accepted; acceptance satisfied. Admission: user's continuation of the five-step
sequence. Owner: this task on experiment/water-pressure-recovery-v1, base
9fec0f2ae6d4de6963c056cc0a3003b35fbe94a4. Target is experiment/water-early-pressure-v1, carrying the pressure-bound lineage;
integrate serially with new commits. Retain for local experiment review and later
training; retire after accepted integration or explicit retirement. No rewrite,
push, deployment, or branch deletion in this task.

Objective: recover the prior late model's pressure accuracy while retaining the
harder-data control's reduced correction work. Measurement acceptance does not
require a successful replacement. Compare same-input convergence separately from
paired autonomous trajectories. Keep float32 CUDA neural training and CPU inference,
float64 physical operators, cubic cells, sealed full water, forces, 15-step warm-up,
32-call maximum, no native pressure refinement/fallback, existing residual and
divergence limits. Preserve all earlier results and unrelated user files.

Decisions frozen before measurement:
- Replay every retained sparse capture from the three control scenes with largest
  trajectory pressure error in their respective consumed populations, plus scene73
  (deduplicate). Compare old late and harder-data control on exactly the same inputs,
  calls0..32, original pressure target1e-6 and stricter3e-7/1e-7. Include absolute
  and relative spectral signal/error by low spatial modes and amplitude bins.
  No optimizer or new trajectories for this diagnostic; one real fixture before
  expanding any new trajectory protocol.
- Evaluate harder-data control at both stricter targets on all74 consumed scenes.
  Keep native controls, fixed scene inputs and all physics gates. This is an
  inference-policy experiment, not a model change. No loosening or posthoc sweep.
- Two5000-update CUDA fits, first-pressure weights0.1 and0.25, original seed and
  identical192-pair train-only bank/sampling/optimizer. Reuse the archived weight0
  control after verifying its exact config, source, initial weights and sampler;
  do not spend another identical fit. No new training examples or architecture.
  Existing qualified v9 trainer; smoke uses actual CUDA and fresh CPU restore.
- Evaluate both fits at unchanged1e-6 target on all74 consumed scenes. Require no
  physical failure, no worse group worst pressure, mean calls or max calls against
  BOTH retained reference and old late model. Report original58/amended8/recent8
  separately. Rank eligible combinations by global maxcalls, worstpressure, meancalls.
- Only if eligible, freeze one configuration then repeat training at a second
  predeclared seed20261016 (for an inference-only winner repeat its underlying
  weight0 training at that seed). Qualify on74 again. Only then freeze the winning
  original checkpoint/policy and evaluate the eight already reserved untouched
  scenes. No fresh-informed fit, target change or reselection. No candidate means
  replication/fresh validation explicitly Deferred.
- Current serial CPU timing: old late versus best eligible candidate; if none,
  diagnostic-only lowest-work candidate. Same three cases and reversed-order two
  repetitions as prior timing. No GPU or parallel inference during latency runs.
- One GPU optimizer; at most two one-thread CPU workers, pinned to measured P cores;
  8GiB available-memory reserve and1GiB per-phase raw budget. Record clocks,
  command failures, hashes, per-step scalars, peak RSS/CUDA memory and child reaping.
  Existing Tuldok catalog; new project snapshot, older snapshots preserved. Never
  commit weights, raw arrays, credentials or the user's graph files.

Write set: experiments/water-pressure-recovery-v1/**;
docs/plans/water-pressure-recovery-v1/**; tuldok-project.json; README.md;
experiments/water-pressure-accuracy-v1/diagnose.py (optional diagnostic callback,
default unchanged). Existing canonical trainer, managed lifecycle, stopping,
native rollout, oracle and CPU phase owners remain authoritative. No runtime
portability or model architecture changes. Composed-design review not applicable:
config-selected existing owners plus a bounded measurement coordinator.

Milestones:
- M1 Accepted: freeze protocol, qualify callback signal/error statistics, CUDA smoke
  and CPU restore; saved-input replay. Evidence: focused parity/statistics tests,
  independent neighbor oracle and exact captured source/model bindings.
- M2 Accepted: stricter-policy and matched-training consumed comparisons with timing,
  memory and complete failures. Gate all declared measurements retained, not improvement.
- M3 Accepted (seed replication and fresh validation Deferred): current serial timings, conditional seed replication/fresh evaluation,
  sealed records and Tuldok readback. Gate consumed selection respected, all task
  children reaped and Runs drained; conditional deferral explicit.

Current phase: terminal. Next slice: none. Seed replication and fresh validation
are Deferred because the consumed comparative gates failed. No model replacement
or deployment change.

All named source/environment/measurement/retention claims are satisfied;
comparative improvement is reported separately. Evidence:
[source tests](reports/source-tests.json), [smoke](reports/smoke-result.json),
[matched admission](reports/training/admission.json),
[policy comparison](reports/policy/regression-summary.json),
[training comparison](reports/training/regression-summary.json),
[serial timing](reports/training/serial-timing.json), [seal](reports/summary.json),
[Tuldok readback](reports/project-readback.json), and
[host process check](reports/host-process-check.json).

Replan on invalid lineage, changed
physics/numerics, custody failure, resource breach or lifecycle failure. Claims:
source/spectral semantics (automated CPU), CUDA training/CPU restore (real managed
environment), consumed comparison (real CPU), retention (artifacts/UI readback).
Standards snapshot:v1:f5ce584b-dc8e-4bf8-b2aa-4d860e5cd109,16 policies,0 unresolved.
