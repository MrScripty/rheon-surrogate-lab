# Shape repeatability and correction efficiency

Status: Accepted. Admission: start, user-authorized continuation of the prepared
next steps. Owner and serial integrator: this task. Canonical plan:
docs/plans/water-shape-repeatability-v1/plan.md.

Objective: test repeatability of the eight-grid recipe and compare one-correction
training at equal updates and approximately equal forward evaluation work.
Retain accuracy, instability, timings and operational failures as results.

Branch experiment/water-shape-repeatability-v1 starts at 776e8237ed672763a7457f4ad61c14250610ba09,
targeting the retained recurrence experiment lineage. Private experiment branch
retained for review/continuation, synchronized through new commits and retired
after accepted integration or explicit retirement. No history rewrite or push.

Binding decisions:
- Reuse the current 370,016-parameter float32 network, train-only scales,
  16 real training simulations/32 states, eight-grid analytical bank and objective.
- Run two independent seed pairs at two training corrections/5,000 updates;
  one same-seed correction/5,000 updates; one same-seed correction/10,000 updates
  with learning-rate changes at 60% and 90%. The longer run approximately matches
  two-correction forward calls, not optimizer/backward work or wall time.
- CUDA optimization only, one optimizer at a time. Each managed Run keeps the
  existing 600-second/32 MiB limits. Retain all attempts; no automatic replay.
- CPU qualification pools serialize, use up to 23 independent single-thread
  workers and may overlap CUDA. Their timings describe throughput, not isolated
  inference latency. Keep the user's existing Tuldok catalog and live project.
- Preserve cubic spacing, water/sealed-wall physics, 15 native history steps,
  adaptive cap 32, raw/fixed 8 distinctions and unchanged equation/physical gates.
- Reuse prior development/long/grid cases as regressions. Declare 16 fresh final
  whole simulations before fitting. Freeze development selection before final
  evaluation; prior final16 are consumed, not new final evidence. If the reference
  wins development, compare the best eligible new challenger as well; if none
  qualify, retain the reference and label the best new challenger diagnostic.
  Final results cannot trigger reselection or refitting.
- CPU preparation may create audited32/64-training-parent views from the existing
  qualified source packet for subsequent experiments. Do not substitute these
  into any current fit or consume reserved/final data. A separately named
  64-step elongated-domain diagnostic (65/95/127 cells along one axis) may
  test the retained reference within existing native limits. Its native controls
  must pass before judging AI; this exploratory population is not a selection
  gate or final test. The proposed11-grid preparation is rejected by the enforced eight-grid
  limit. Prepare only a separate three-grid auxiliary bank; any future combined
  bank requires an explicit trainer-contract change and controlled regression.
- No model deployment, application source edits, capacity/objective changes or
  enlarged native grid/history contract. Profile/integrate in a separate slice.
- Local new retention budget 2 GiB plus 1 GiB reserve. No weights/data in Git.

Composed-design review: not-applicable. Existing trainers, managed workflow,
native oracle and lifecycle owners remain authoritative. The sequence delegates
to them and adds configuration, timed command records and bounded observation.

Write set: experiments/water-shape-repeatability-v1/**;
docs/plans/water-shape-repeatability-v1/**; README.md; tuldok-project.json.
Ignored outputs: runs/water-shape-repeatability-v1/** and live activity.
Preserve unrelated cohort-progress HTML/refresh script and historical records.

Milestones:
- M1 configure/capture/qualify inputs and publish the existing-catalog project.
  Gate C1; state Accepted.
- M2 execute four fits with CPU restore/development/regression qualification.
  Gate C2/C3; state Accepted.
- M3 freeze selection, evaluate new finals and retain results/timings/lifecycle.
  Gate C4; state Accepted.

Acceptance claims, automated:
- C1 focused/artifact, not-applicable environment: policy accepts configurations,
  controlled quantities and fresh final identities verified. Satisfied; reports/configuration.json and reports/capture.json.
- C2 system, required-real CUDA/Tuldok: fits have terminal outcomes, artifacts
  reopen/redownload, CPU restore identity matches. Satisfied; four sealed trial records and reports/checkpoint-interface.json.
- C3 system, required-real native CPU: adaptive/fixed 8, native-controlled fresh,
  prior-grid and 512-step results retain work/error/failure. Satisfied; sealed trial records and reports/comparison.json.
- C4 system/artifact, required-real: frozen selection precedes final population;
  commands/methods/elapsed times/results/UI references retained; own jobs drain.
  Satisfied; reports/summary.json, timings.json, qualification-decision.json,
  lifecycle-retention.json and project-registration.json. Improvement itself
  is not an acceptance condition.

Current acceptance: satisfied. Current phase: complete.
Exactly one next slice: none within this completed tranche.
Subsequent unexecuted work is in next-steps.md. All owned jobs/observers are stopped.
Blockers: none. Replan on native-control failure, exceeded budgets, unknown
submission outcome or changed custody/physics contracts. See execution-ledger.md,
issues.md and reports/. Standards snapshot:
snapshot:v1:f153468b-a309-4dca-8cac-3b949a613959.
