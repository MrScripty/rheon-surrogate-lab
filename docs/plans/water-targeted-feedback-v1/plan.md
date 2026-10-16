# Targeted coverage with intermediate feedback

Status: Active. Admission: start, user-authorized next sequence. Canonical path:
docs/plans/water-targeted-feedback-v1/plan.md. Owner/integrator: this task.
Objective: diagnose the hard-shape slowdown and test targeted coverage without
diluting the original grid stream; retain methods, times, failures and decisions.

Branch experiment/water-targeted-feedback-v1 starts at
77ad2e23ac5e0b6b6ad803c5d6b6c4eed30113fe, targeting the retained experiment lineage.
It remains private for review/continuation; synchronize by new commits, retire
after accepted integration or explicit retirement. No rewrite or push.

Binding decisions:
- Keep qualified control 17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8.
  No physics, architecture, precision, inference policy or application changes.
- First reproduce the slowdown on a single actual post-warm-up input. Compare
  checkpoints on identical fields, with deterministic call counts and retained
  trace/source hashes. Investigate a bounded set of axis/mode responses; stop
  when evidence selects targeted coverage versus a different mechanism.
- Paired recipe selected from diagnosis: two 5,000-update
  CUDA fits, intermediate objective, same original seeds/real states/scales,
  original eight-grid order/exposure preserved and explicit extra feedback.
  Supplemental control reuses the primary bank; target adds six cyclic grids
  around 16×23×25 and 10×7×25, each at 16 cells/metre. Both add one weight-0.1
  sample per update (5,000 extra examples); disclose differing cell work.
  Replicate a promising improvement across independent seeds before definitive
  promotion; a single-seed winner remains provisional.
- Version any sampler extension explicitly; old schemas retain numerical and
  RNG/evidence parity. Existing train/analytical/policy owners remain canonical.
- Existing Tuldok workflow/catalog owns CUDA, fresh CPU restore and validation.
  One optimizer, 600-second/32 MiB Run caps; bounded independent CPU pools may
  prepare labels or real CFD scenes on separate cores. No unmanaged training.
- CPU preparation qualifies late labels independently of captured teacher error
  and prepares varied larger mixed-odd water parents for a subsequent fit.
  Neither preparation is silently added to these two fits or final data.
- Include all 50 consumed development trajectories plus fit/familiar checks.
  Elongated four are exploratory. Freeze selection before eight new whole final
  simulations, reserved/committed before fitting. Rank max calls, worst pressure
  relative L2, mean calls; promotion also requires no final regression in these.
- Keep cubic h, sealed water, existing gates, 15 native warm-up steps and 32-call
  cap. No native cleanup credited as AI. Record actual work and elapsed time.
- Retention: 2 GiB plus 1 GiB reserve; no weights/raw datasets/secrets in Git.
  Preserve historical results and unrelated cohort-progress files.

Composed-design review: not-applicable. Network/runtime composition is unchanged;
this extends training data selection inside its existing owner and delegates
managed/native execution, label oracles and observation to existing owners.

Write set: experiments/water-conditioning-v1/training/{policy.py,analytical.py,
train.py,model.py,test_multigrid.py,test_residual.py,test_supplemental.py};
experiments/water-multigrid-learning-v1/{qualify_bank.py,record.py};
experiments/water-targeted-feedback-v1/**;
docs/plans/water-targeted-feedback-v1/**; README.md; tuldok-project.json.
Ignored outputs: runs/water-targeted-feedback-v1/** and live activity.
Clean capture may use /tmp/rheon-targeted-feedback-capture. Before removing only
that clean task worktree, record its exact head, retained-ref reachability,
retained disposition and protected-OID check; verify path/registration removal.

Milestones and objective evidence:
- M1 deterministic real-input differential repro and bounded mode diagnosis.
  Focused/system CPU evidence: actual model code, same inputs, red-capable
  call-count signal, minimized solve, hypothesis predictions/observations.
  Accepted; claim satisfied by reports/diagnosis-*.json, probes.json and
  diagnosis-findings.md.
- M2 versioned supplemental sampler and controlled configs, preserving legacy
  tensors/order/loss/RNG; memory/operator validation and clean source capture.
  Accepted; claim satisfied by reports/unit-tests.json, configuration.json,
  reference-bindings.json and capture.json.
- M3 two managed CUDA fits, CPU qualification, frozen final comparison and
  independent CPU preparation. Required-real RTX5090/Tuldok/nativeCPU evidence:
  numerical outcomes, terminal/reaped custody, actual timing/source identity,
  existing project visibility and complete retained record. Active; pending.

Current acceptance: partial. Current phase: M3.
Exactly one next slice: complete paired managed fits and CPU qualification,
with independent CPU label/CFD preparation; freeze selection before final.
Blockers: none. Replan if controls fail, diagnosis contradicts the selected
mechanism, resources/custody become unavailable, or legacy parity/physics changes.
Improvement itself is not required for experiment acceptance. See execution-ledger.md,
issues.md and reports/. Standards snapshot:
snapshot:v1:9e601b82-37ef-414f-bff6-479137cd4df2.
