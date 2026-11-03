# Two-correction longer training control

Status: Accepted. Admission: start, authorized by the user's continuation of the
prepared next steps. Owner and serial integrator: this task. Canonical plan:
docs/plans/water-two-pass-long-v1/plan.md.

Objective: determine whether 10,000 updates with two training corrections retain
robustness while reducing learned CPU inference work and pressure error. Retain
method, elapsed times, negative results and operational outcomes for review.

Branch experiment/water-two-pass-long-v1 starts at
f03a5162220f06c9193fa05860fbd9bb01df3f34, targeting the retained experiment lineage.
This private experiment branch remains available for review/continuation, with
new commits only; retire after accepted integration or explicit retirement.

Binding decisions:
- One fresh CUDA fit, original initialization/sampling seeds, 10,000 updates,
  two shared corrections, learning-rate reductions at 6,000 and 9,000. Existing
  370,016-parameter float32 network, 16 real parents/32 states, eight-grid bank,
  train-only scales, canonical objective and normalization remain fixed.
- Compare against retained two-correction/5,000 control and historical
  one-correction/10,000 development result. Equal updates and equal forward
  calls are different comparisons. Do not claim isolated latency from call count.
- Existing managed workflow owns submission, approval, cancellation and Run
  custody. One optimizer, 600-second/32 MiB Run limits. Reuse existing catalog
  and live project; never blindly retry an unknown submission.
- CPU restoration and qualification, up to 23 independent single-thread jobs
  where enough cases exist. Use bounded pools rather than oversubscribing a
  single small convolution. CPU bank preparation may overlap CUDA on other
  cores. Concurrent durations are throughput observations, not latency evidence.
- Preserve cubic scalar spacing, water density/viscosity, sealed/free-slip
  boundary contract, 15-step history, cap 32 and unchanged physical/equation gates.
  Reuse the prior model contract and native binaries; no deployment changes.
- Prior final 16 are consumed development, including 16x23x25. Include all 34
  development trajectories (fresh 8, prior 7, long 3, consumed 16) in eligibility.
  Rank eligible models by maximum actual calls, worst pressure relative L2,
  then step-weighted mean calls; familiar mean remains diagnostic. Elongated 4
  remain exploratory because the control already fails the thinnest scene.
- Declare eight new whole final simulations before fitting. Freeze checkpoint
  and development choice before final. Always report challenger versus control;
  an ineligible challenger is diagnostic. Promote only if development chooses
  it and final qualification, maximum calls, mean calls and worst pressure error
  do not regress. No final-informed refit or alternate-checkpoint selection.
- Prepare a separate mixed-odd three-grid auxiliary bank on CPU, with independent
  operator-label checks and legacy parity. It is not included in this fit.
  Combining banks or changing sampling/objective remains a subsequent slice.
- New ignored retention budget 1 GiB plus 1 GiB reserve. No model weights or
  simulation arrays in Git. Preserve all historical evidence and unrelated files.

Composed-design review: not-applicable. Existing numerical, managed-workflow,
process-lifecycle and tensor-contract owners are unchanged. A bounded experiment
entrypoint delegates and records one new training/evaluation decision.

Write set: experiments/water-two-pass-long-v1/**;
docs/plans/water-two-pass-long-v1/**; README.md; tuldok-project.json.
Ignored outputs: runs/water-two-pass-long-v1/** and live activity.
Clean source capture may use /tmp/rheon-two-pass-long-capture at a committed HEAD;
remove only this task-created clean worktree after recording exact head,
retained-ref reachability and retained commit disposition.

Milestones and claims (all automated):
- M1 configuration, capture and active catalog registration. C1 focused/artifact,
  environment not-applicable: exact controlled differences, policy validation,
  final identities and capture parity. State Accepted; claim satisfied; reports/configuration.json, capture.json and project-registration.json.
- M2 managed CUDA fit, CPU restore and development qualification. C2 system,
  required-real RTX 5090/Tuldok/native CPU: terminal managed custody, artifact
  restoration parity, fixed/adaptive work and physical results retained.
  State Accepted; claim satisfied; reports/trial-two-pass-10000.json and comparison.json.
- M3 frozen final evaluation and retained review evidence. C3 system/artifact,
  required-real native CPU: native controls, frozen selection, timings, failure
  disposition, UI references and owned child reaping. State Accepted; claim satisfied; reports/decision.json, timings.json, summary.json, verification.json and project-registration.json.

Current acceptance: satisfied. Current phase: complete.
Exactly one next slice: none within this completed tranche. Prepared follow-ups
are in next-steps.md. All owned jobs and observers are stopped.
Blockers: none. Replan on native-control failure, unknown submission outcome,
exceeded resource budget or changed physics/custody contracts. Improvement is
not an acceptance condition. See execution-ledger.md, issues.md and reports/.
Standards authority: snapshot:v1:245ee439-9f5d-4d0e-b2d6-24011e4cb484.
