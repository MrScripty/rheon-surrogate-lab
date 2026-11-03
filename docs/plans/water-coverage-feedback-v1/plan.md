# Separate coverage and intermediate-feedback experiments

Status: Accepted. Admission: start, user-authorized continuation of the prepared
next steps. Owner/serial integrator: this task. Canonical plan:
docs/plans/water-coverage-feedback-v1/plan.md.

Objective: isolate mixed-odd operator coverage from intermediate residual
feedback, with controlled CUDA fits and native CPU qualification. Retain all
methods, timings, failures and checkpoint decisions for later review.

Branch experiment/water-coverage-feedback-v1 starts at
edb9992fe5c8df05a7486d6ceb47e95cbfe6b8e6, targeting the retained experiment lineage.
It remains private for review/continuation, synchronized by new commits; retire
following accepted integration or explicit retirement. No push/history rewrite.

Binding decisions:
- Retain the two-correction/5,000-update checkpoint as control. Two new fits use
  the same seeds, 5,000 updates, schedule, architecture, float32 parameters,
  16 real parents/32 states, train-only scales and two shared corrections.
- Coverage-only: original eight analytical grids plus the three independently
  qualified mixed-odd grids; canonical objective. Balanced sampling across 11
  changes old-grid exposure from 625 to 454/455 draws; disclose this data/sampling
  effect. Keep all old tensors and original configurations unchanged.
- Feedback-only: original eight grids and draw order; add weight 0.1 times mean
  relative residual squared at nonfinal training corrections to the existing
  final loss, on both real and analytical training samples. No later-than-two
  residual sampling, coverage extension or inference policy change in this fit.
- A versioned v6 experiment contract explicitly selects a 16-grid capacity and
  the optional named intermediate objective. Old schemas retain their 8-grid
  restriction, tensors, RNG, loss, predictions and gradients. All banks retain
  the 768 MiB tensor projection and existing cubic spacing/cell limits.
- Existing managed workflow owns CUDA training and Run custody; one optimizer,
  600-second/32MiB Run limits. CPU qualification/preparation can overlap on
  disjoint cores, using bounded single-thread worker pools. No blind retry.
- Reuse exact native binaries and existing catalog/project observation. Match
  reference checkpoint identity, scenes and binary before starting. Keep water,
  sealed/free-slip boundaries, scalar cubic h, 15 history steps, 32-call cap and
  physical/equation gates. No deployment or GPU inference qualification change.
- Include 42 development trajectories: standard 8, prior 7, long 3, consumed 16 and
  the last eight final simulations. Elongated 4 remain exploratory. Eligibility
  requires all fitting/familiar states and 42 trajectories to qualify; rank
  maximum calls, worst pressure relative L2, then weighted mean calls. Familiar
  mean and analytical probes remain separately retained diagnostics.
- Reserve eight fresh whole final simulations before fitting. Freeze the best
  eligible new challenger (best diagnostic if neither qualifies) and overall
  development choice before final. Compare control/challenger once. Promotion
  requires development choice plus final qualification and no regression in
  maximum/mean calls or worst pressure error. No final-informed fit/reselection.
- CPU preparation may run eight fixed learned corrections from the retained
  checkpoint on training snapshots only, retaining late residual/teacher-label
  defect diagnostics. This does not modify either fit or consume final data.
- New local retention 2 GiB plus 1 GiB reserve. No weights/raw simulation arrays in
  Git. Preserve all historical results and unrelated cohort-progress files.

Composed-design review: not-applicable. Network architecture and runtime
composition are unchanged. Existing policy, analytical, physical-residual and
training owners keep their responsibilities; training_loss encapsulates the
new objective and optional internal intermediate trace. Numerical and lifecycle
execution still delegates to existing managed/native owners.

Write set: experiments/water-conditioning-v1/training/{policy.py,analytical.py,
model.py,physical_residual.py,train.py,test_multigrid.py,test_residual.py};
experiments/water-multigrid-learning-v1/qualify_bank.py;
experiments/water-coverage-feedback-v1/**;
docs/plans/water-coverage-feedback-v1/**; README.md; tuldok-project.json.
Ignored outputs: runs/water-coverage-feedback-v1/** and live activity.
Clean capture may create /tmp/rheon-coverage-feedback-capture; remove only this
task's clean worktree after recording exact head, retained-ref reachability and
retained disposition, verifying the protected commit remains reachable.

Milestones/acceptance (automated):
- M1 versioned trainer capabilities/configs and focused parity/operator tests.
  C1 focused/contract, not-applicable environment: legacy bounds/tensors/order/
  prediction/loss/gradient parity, intermediate feedback meaning/gradient,
  capacity rejection, CPU bank label qualification and source capture.
  State Accepted; claim satisfied; reports/configuration.json, unit-tests.json,
  reference-bindings.json and capture.json.
- M2 two managed CUDA fits with fresh CPU restore and native development.
  C2 system, required-real RTX5090/Tuldok/nativeCPU: terminal custody, identical
  redownloads, restored weights, real numerical results and work. State Accepted;
  claim satisfied; reports/trial-*.json, comparison.json and final-selection.json.
- M3 frozen final qualification, result/timing/UI/lifecycle records.
  C3 system/artifact, required-real nativeCPU: final after selection, all native
  controls checked, attempted failures retained, owned jobs reaped, sources and
  evidence discoverable in existing project. State Accepted; claim satisfied;
  reports/summary.json, timings.json,
  decision.json, project-registration.json and worktree-lifecycle.json.

Current acceptance: satisfied. Current phase: terminal.
No active slice remains. Prepared follow-ups in next-steps.md require a new
bounded admission; the two fits and frozen final evaluation are complete.
Blockers: none. Replan on native control failure, unknown submission outcome,
resource overrun, lost legacy parity or changed physics/custody contracts.
Improvement is not required for acceptance. See execution-ledger.md, issues.md,
reports/. Standards snapshot: snapshot:v1:70977344-d8d9-4f7c-b10b-fb9d9ce11726.
