# Findings and dispositions

- R1, limitation: current reference requires iterative inference; raw output is
  unqualified. Owner: learning experiment. Retain failures and fixed-budget errors.
- R2, limitation: prior pressure-field drift is mixed despite passing equation
  gates. Owner: qualification. Measure fresh/long trajectories; do not claim
  universal stability or accuracy from per-step gates.
- R3, deferred: application adapter remains separate from this one-channel lab
  model. Owner: Rheon integration; revisit only for explicit deployment work.

- R4, deferred: retained native scene contract caps trajectories at 512 steps.
  Owner: native capture/integration. A longer window needs a separate resource/
  lifecycle qualification; this experiment stays within the actual limit.

- R5, measured learning limitation: reference passes 6/8 new shapes;17×13×9
+  and 33×17×9 fail unchanged equation gates after 32 evaluations while native
+  controls qualify. Owner: learning feedback. Compare all new trials before
+  deciding on shape coverage; no threshold relaxation or native cleanup.
+  Evidence: reports/reference-fresh-development.json and ignored full traces.

- R6, preserved unrelated work: docs/plans/cohort-progress.html appeared untracked
+  during execution. Owner: external to this task. Leave untouched/uncommitted;
+  source preparation uses an isolated clean committed checkout. No blocking overlap.
+
- R7, fix-now by feedback experiment: seed3 reaches a genuine step 32 hard stop
+  on33×17×9 after native warm-up despite passing128 familiar states. Owner:
+  learning/qualification. Admit the separately qualified shape bank; prohibit
+  promotion from teacher-state metrics alone. Preserve failed trajectory.
+- R8, fix-now: trajectory report sampling omitted the terminal frame for early
+  hard stops and described all inputs as quarter-strength. Owner: existing
+  trajectory_summary.py; include actual last step/completeness and declared
+  scene forcing. Verify against the real step 32 failure and complete reference.
+
- R9, operational recovery: one-pass CUDA succeeds, but CPU approval exceeds
+  the 180-second controller request limit under overlapping managed preparation.
+  Owner: local execution controller. Resume CPU-only stages with a600-second
+  bounded request; retain failure and normal recovered custody. Root-cause
+  profiling of Tuldok preparation latency is deferred to its owner, triggered
+  by repeated timeouts or an end-to-end throughput task. No Tuldok source edits.
+
## Final dispositions
+
+R1–R3 remain explicit scientific/deployment limitations; next-steps.md owns their
+triggers. R4 remains deferred to the native resource-contract owner. R5/R7 are
+addressed for the tested coverage by the selected bank, with all failed weights
+and trajectories retained; broader guarantees remain unqualified. R6 preserves
+both unrelated progress files. R8 is verified against the actual step 32 failure
+and full windows. R9 is recovered through normal CPU-only execution/custody;
+Tuldok preparation profiling remains separately deferred. No required current
+claim is blocked or left pending.
+