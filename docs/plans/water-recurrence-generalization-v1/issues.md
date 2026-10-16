# Findings and dispositions

- R1, limitation: current reference requires iterative inference; raw output is
  unqualified. Owner: learning experiment. Retain failures and fixed-budget errors.
- R2, limitation: prior pressure-field drift is mixed despite passing equation
  gates. Owner: qualification. Measure fresh/long trajectories; do not claim
  universal stability or accuracy from per-step gates.
- R3, deferred: application adapter remains separate from this one-channel lab
  model. Owner: Rheon integration; revisit only for explicit deployment work.

- R4, deferred: retained native scene contract caps trajectories at512 steps.
  Owner: native capture/integration. A longer window needs a separate resource/
  lifecycle qualification; this experiment stays within the actual limit.

- R5, measured learning limitation: reference passes6/8 new shapes;17×13×9
+  and33×17×9 fail unchanged equation gates after32 evaluations while native
+  controls qualify. Owner: learning feedback. Compare all new trials before
+  deciding on shape coverage; no threshold relaxation or native cleanup.
+  Evidence: reports/reference-fresh-development.json and ignored full traces.

- R6, preserved unrelated work: docs/plans/cohort-progress.html appeared untracked
+  during execution. Owner: external to this task. Leave untouched/uncommitted;
+  source preparation uses an isolated clean committed checkout. No blocking overlap.
+
- R7, fix-now by feedback experiment: seed3 reaches a genuine step32 hard stop
+  on33×17×9 after native warm-up despite passing128 familiar states. Owner:
+  learning/qualification. Admit the separately qualified shape bank; prohibit
+  promotion from teacher-state metrics alone. Preserve failed trajectory.
+- R8, fix-now: trajectory report sampling omitted the terminal frame for early
+  hard stops and described all inputs as quarter-strength. Owner: existing
+  trajectory_summary.py; include actual last step/completeness and declared
+  scene forcing. Verify against the real step32 failure and complete reference.
+
- R9, operational recovery: one-pass CUDA succeeds, but CPU approval exceeds
+  the180-second controller request limit under overlapping managed preparation.
+  Owner: local execution controller. Resume CPU-only stages with a600-second
+  bounded request; retain failure and normal recovered custody. Root-cause
+  profiling of Tuldok preparation latency is deferred to its owner, triggered
+  by repeated timeouts or an end-to-end throughput task. No Tuldok source edits.
+