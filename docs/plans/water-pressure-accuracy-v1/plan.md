# Pressure accuracy versus correction work

Status: Accepted. Admission: start, authorized by the user's request to continue
the recorded next steps. Canonical path: docs/plans/water-pressure-accuracy-v1/plan.md.
Owner/integrator: this task. This is a new bounded follow-up; the targeted study
remains accepted and unchanged.

Objective: distinguish same-input pressure-solve error from autonomous state
differences, measure whether additional learned corrections remove the accuracy
tradeoff, and retain a decision for the subsequent late-residual training test.
This tranche is CPU inference/diagnosis; no optimizer or deployed model change.

Branch experiment/water-pressure-accuracy-v1 starts at
c8857ae243e53580e5c11affa7308d390cf6efa8, targeting the retained experiment
lineage. The task owns this private continuation branch, retained for review and
subsequent experiments; synchronize by new commits, retire after integration or
explicit retirement. No rewrite, push, or task worktree is required.

Binding decisions:
- Preserve both checkpoint receipts, original normalization, input channel,
  physical operator, cubic h, float32 neural arithmetic, existing physical
  accumulation and native paired engine. Keep the retained learned control.
- Use only the three already consumed pressure witnesses, their original
  scene seeds/forces, 15 native warm-up steps, and native binary. First qualify
  a one-AI-frame fixture. Capture first/previous peak/last input on independent
  control and targeted trajectories, then replay both checkpoints on each exact
  field. Samples remain ignored; retain their identities and small receipts.
- Qualify the existing independent cosine pseudoinverse using independently
  enumerated neighbor action; it is an analysis oracle, never native cleanup or
  learned-inference credit. Measure gauge-anchored and mean-free pressure error,
  pressure-gradient error, equation error and low-frequency error energy.
- Compare actual calls 1..32 and existing adaptive stopping against separately
  named stricter 1e-7 and 1e-8 equation stopping policies, with unchanged absolute
  and divergence limits. These are diagnostic inference policies, not a relaxed
  acceptance target, new network precision, or a checkpoint improvement.
- Reuse process/termination/reaping owners. Bound each native trajectory to
  120 s, pool to available CPUs, one thread per solve. Qualify before broadening;
  record observed CPU utilization/affinity, not assumed speedup from core count.
- No promotion or fresh holdout consumption in this diagnostic study. Do not run
  the conditional seed confirmation or change model/training/data recipes here.
  A controlled late-label fit gets its own versioned admission after this result.
- Retain failures and elapsed boundaries; do not overwrite an attempt. Limit
  local outputs to 1 GiB with 1 GiB free reserve. No weights, raw arrays, secrets
  or local auth adapters in Git. Preserve unrelated cohort-progress files.
- Keep the existing Tuldok catalog and historical project snapshots; register
  the new manifest as its own project snapshot, as required by the observed
  immutable-ID contract. Activity and unpinned evidence remain live. No UI changes.

Write set: experiments/water-pressure-accuracy-v1/**;
docs/plans/water-pressure-accuracy-v1/**; README.md; tuldok-project.json.
The evidence-list limit requires docs/plans/experiment-history.md to retain all
64 historical references through one index; original documents stay unchanged.
Ignored outputs: runs/water-pressure-accuracy-v1/** and runs/live-project/activity.json.
Existing numerical, native process, checkpoint and timed-command owners remain
canonical. Composed-design review: not-applicable; this diagnostic composes those
unchanged owners without adding a runtime or deployment abstraction.

Acceptance (automated, required-real local CPU/native engine):
1. Reproduce the same-input pressure-error tradeoff on a qualified one-frame
   fixture; retain the red-capable comparison command and result. Independent
   neighbor action qualifies the spectral oracle; negative control demonstrates
   the diagnostic detects a wrong pressure field.
2. Capture/replay the consumed witnesses and measure all per-call frontiers;
   all children finish/reap, source/checkpoint/sample identities and timings are
   retained. No CUDA context or optimizer is created.
3. Run baseline and stricter-policy complete native paired trajectories for both
   checkpoints. Preserve physical failures, actual calls, pressure witnesses and
   elapsed cost, and keep isolated-solve claims distinct from trajectory claims.
4. Retain methods, results, decision, limitations and the next controlled training
   protocol; publish the experiment references/activity to the same Tuldok
   catalog. Stage/review only the declared scope and commit the coherent result.

Current acceptance: satisfied. Claims 1–3 are supported by summary.json, the
replays/trajectory tables, reconciliations, timings and lifecycle tests. Claim 4
is supported by project-readback.json, verification.json and the reviewed scoped
change. Evidence is under reports/; detailed decisions are in results.md and the
execution-ledger.md. Issues are dispositioned in issues.md; no ADR is changed.
Milestones: M1 fixture/repro accepted (diagnose.py, repro.py, fixture receipts);
M2 same-input frontier and whole-trajectory comparison accepted (run.py, reports);
M3 durable synthesis and publication accepted (docs, manifest, retained receipts).
Current phase: terminal. Next slice: none. The next controlled CUDA training
comparison is prepared in next-training-protocol.md, not implemented or running;
it requires its own versioned admission. No model or inference policy promotion.
Blockers: none. Stop or replan on incompatible oracle error, identity mismatch,
unreaped child, resource breach, or diagnosis requiring a model contract change.
Scientific improvement is an outcome to measure, not an acceptance prerequisite.
Standards snapshot: snapshot:v1:3b77bf95-ce70-4f8a-8f9e-338a878d3e12,
23 selected policies, zero unresolved routing facts. Serial root work only.
