# Frozen-model stopping floor and pressure accuracy

Status: Active. Admission: start, user-authorized continuation of the proposed
CPU stopping diagnostic, conditional CUDA experiment, and fresh validation.
Canonical path: docs/plans/water-stopping-floor-v1/plan.md.
Owner and integrator: this task. Private branch experiment/water-stopping-floor-v1
continues experiment/water-late-residual-v1 at3cc74c9. Retain the branch for review,
synchronize through new commits, retire after integration or explicit retirement.
No push, history rewrite, deployment change or replacement of historical reports.

Objective: determine whether early equation-based stopping explains the late
checkpoint's better equal-call but worse autonomous pressure accuracy; select
subsequent work from observed evidence rather than launch another blind fit.

Decisions:
- Freeze both retained reference and extra-late checkpoints, normalization and
  physical setup. CPU inference only; no optimizer in the diagnostic. Keep cubic
  cells, sealed water, native history warm-up, all physical gates and32-call cap.
- Add an explicit optional minimum-network-evaluations argument to the existing
  learned-iteration owner. Default zero preserves current stopping and zero-work
  behavior. Nonzero requires adaptive stopping, a reachable whole-proposal floor,
  and a sufficient unchanged budget. Continue beyond the floor until checks pass.
  Keep persisted v1 stopping profiles unchanged; do not redefine their semantics.
- Compare reference/current, reference/minimum8, late/current, late/minimum8 on
  consumed whole scenes. Native controls must qualify first. Qualify one native
  fixture and all four learned episodes before expanding the population.
- Use two independent one-thread CPU workers for correctness, each pinned to a
  separate permitted core, retaining120s native deadlines and process receipts.
  No isolated latency claims from concurrent or instrumented observations.
- Capture only predeclared sparse actual input/prediction states, then independently
  solve those same RHS for pressure, gradient and spatial-band diagnostics. Keep
  same-RHS error distinct from full-trajectory pressure differences versus native.
  Use bounded separable cosine transforms, <=512MiB added address space per oracle,
  <=1GiB raw experiment output and>=8GiB available before parallel work.
- Run repeated serial uninstrumented episodes on a consumed representative scene
  for timing after correctness work finishes. Alternate policy order; exclude one
  AI timing warm-up. Record actual seconds as well as network calls.
- If the floor recovers trajectory pressure accuracy, test a deployable stopping
  criterion using residual information, not oracle labels. If it does not, bound
  the failure before selecting on-policy training; do not refit on final data.
  A new CUDA fit is conditional on an identified, train-only data/objective change.
- Before any fresh validation, freeze the selected policy and commit new scene
  configurations; qualify native setup before AI evaluations. Fresh scenes may
  validate a policy only after consumed-scene acceptance. Do not consume the
  independent-seed training confirmation population for inference development.
- Retain configurations, source/checkpoint hashes, commands, timings, per-frame
  metrics, failures and child reaping; publish progress in the existing live
  Tuldok project/catalog. No weights, raw arrays, credentials or unrelated files
  in Git. Preserve all previous experiment documents and results.

Write set: experiments/water-stopping-floor-v1/**;
docs/plans/water-stopping-floor-v1/**;
experiments/water-stability-learning-v1/{learned_iteration.py,test_learned_iteration.py};
experiments/water-residual-learning-v1/rollout.py;
experiments/water-multigrid-learning-v1/study.py;
tuldok-project.json; README.md. Ignored runs/water-stopping-floor-v1/** and live
activity are task-owned. Existing lifecycle, checkpoint and numerical oracle
owners are reused; no new runtime framework or solver implementation.

Milestones:
- M1 Accepted: optional floor, focused boundary/parity tests and one real fixture.
  Gate: same default decisions; correct floor/count/cap behavior; unchanged physical
  qualification; CPU-only restore; all children reaped.
- M2 Accepted: consumed autonomous comparison plus independent sparse-state errors
  and serial timings. Gate: complete declared population, including failures,
  reproducible source/checkpoint bindings and machine-readable analysis.
- M3 Active: evidence-selected follow-up and, when eligible, fresh validation.
  Gate: freeze decisions before final evaluation; qualify setups first; no
  accuracy regression and record actual work/time. A negative scientific result
  completes measurement but does not authorize replacement of the reference.

Acceptance partial. Current phase M3; next slice: qualify the residual-to-pressure stopping bound.
Replan on resource/lifecycle failure, numerical mismatch, invalid setup, changed
experimental question, or evidence ruling out the planned conditional follow-up.
Standards snapshot: snapshot:v1:12194777-eab9-47d9-b486-7ef0779f2ae7;
21 routed policies, zero unresolved facts. Findings: issues.md; methods/results:
execution-ledger.md and reports/.
