# Recurrence and generalization experiments

Status: Accepted. Admission: start, authorized by the user to continue the retained
next steps and record methods/results. Owner and serial integrator: this task.
Canonical plan: docs/plans/water-recurrence-generalization-v1/plan.md.

Objective: measure repeatability of the current cohort result, compare one/two/
conditional four evaluations during training, and test fresh physical setups.
Training uses CUDA; qualification uses CPU. Improvement is an experimental
outcome, not an acceptance requirement. Failed hypotheses remain retained.

Branch experiment/water-recurrence-generalization-v1 starts at
c079347a3d69eba44022a7f003b9f741fb7f1b8c and targets the retained
experiment/water-multigrid-learning-v1 lineage. This private experiment branch
is retained for review/continuation, synchronized by new commits, and retired
after accepted integration or explicit retirement. No history rewriting,
remote publication, model deployment or Rheon/Tuldok source changes.

Binding decisions:
- Preserve historical results, training data and configuration. Use the same
  existing Tuldok catalog and normal managed train/restore/validation stages.
- Reuse the 370,016-parameter one-channel network, fixed train-only normalization,
  16 training simulations/32 states, analytical bank, 5,000-update schedule and
  float32 neural values. Vary declared recurrence or model/sampling seeds in the controlled trials.
  The separately admitted shape-feedback trial adds 3 declared grids, preserving
  all real data, scales, initial seed, recurrence and objective.
- One optimizer at a time; each Run retains the existing 600-second and 32 MiB
  limits. New local retention: 2 GiB plus 1 GiB reserve. No datasets/weights in Git.
- Preserve cubic cells, water density/viscosity, sealed walls, native history,
  independent physical gates and model 1e-6/native 1e-8 equation profiles.
- Report raw and fixed-eight results separately from adaptive 32-cap results.
  Record actual evaluations/cell exposures and elapsed compute; updates are not
  equal compute across recurrence depths. No native refinement counts as AI success.
- Fresh development has 8 elongated/odd grids. Final 16 whole simulations have
  separate predeclared setups; execute only after freezing candidate selection.
  Once inspected they cannot be reused as untouched final evidence in later work.
- CPU correctness can overlap CUDA training on distinct cores, up to 23 independent
  single-thread workers. No isolated latency claims under overlapping work.
- Admit the CPU-qualified eight-grid feedback trial after the seed repeats reveal
  persistent shape failures and a genuine hard stop. This is a separate coverage
  comparison with changed grid exposure/work, not another controlled seed repeat.
- Require all familiar, fresh and prior-grid development gates before promotion eligibility;
  freeze final candidate selection before executing either final holdout.
  An unqualified diagnostic candidate cannot become a qualified replacement.
- Extra objective/architecture changes follow this experiment's results as a
  separate experiment. CPU profiling/deployment remain independently owned.

Composed-design review: not-applicable; no new runtime composition or contract
replacement. Existing trainer, process lifecycle, native oracle and custody own
their boundaries. This slice supplies configuration and retained observations.

Write set: experiments/water-recurrence-generalization-v1/**;
docs/plans/water-recurrence-generalization-v1/**; README.md; tuldok-project.json.
Additional reporting write set: experiments/water-multigrid-learning-v1/{record,trajectory_summary}.py,
for explicit recovered custody and complete/partial trajectory evidence.
Ignored material: runs/water-recurrence-generalization-v1/** and live activity.
Existing experiment sources remain unchanged unless a recorded finding justifies
an adjacent write-set extension.

Acceptance claims (automated, required-real unless noted):
- C1 contract/artifact: explicit configurations preserve controlled quantities;
  different seeds and fresh setup identities are recorded. Satisfied; reports/configuration.json, prepared-shape-bank.json and checkpoint-interface.json.
- C2 system: every admitted CUDA fit completes or has an explicit failed outcome;
  successful Run artifacts reopen/redownload and fresh CPU restore matches. Satisfied; five sealed trial records and the explicit CPU-only recovery.
- C3 system: matched adaptive/fixed-eight development evaluation and native-
  controlled fresh-grid/long-window trajectories retain errors, work and failures.
  The selected reference/candidate face the frozen final population. Satisfied; reports/summary.json and final-selection.json.
- C4 artifact/system: results, methods, next decisions, lifecycle and existing-
  catalog Tuldok project references are retained; all own jobs drain. Satisfied; reports/project-registration.json and lifecycle-retention.json.

Current acceptance: C1–C4 satisfied. Current phase: complete.
Exactly one next slice: none within this completed tranche.
Subsequent unexecuted work is recorded in next-steps.md. All task jobs and
observers are stopped; the existing Tuldok server remains untouched.
Blockers: none. No architecture, objective replacement or deployment is admitted.
Branch consumers are experiment review and subsequent learning/integration work.

Standards snapshot: snapshot:v1:84ef7125-33ed-4377-b8f7-5bed6fecf00c.
See execution-ledger.md, issues.md and reports/.
