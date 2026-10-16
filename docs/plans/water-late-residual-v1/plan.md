# Matched early and late residual feedback

Status: Active. Admission: start, user-authorized continuation of the prepared
pressure-accuracy protocol. Canonical path: docs/plans/water-late-residual-v1/plan.md.
Owner/integrator: this task. Base a1b84ff593034e747417e063cb261e12d9c429aa,
private branch experiment/water-late-residual-v1, targeting the retained experiment
lineage. Retain for review/continuation, synchronize by new commits, retire after
accepted integration or explicit retirement. No push/history rewrite.

Objective: test whether learning the residual directions left after eight calls
improves pressure accuracy at comparable work. Preserve the existing learned
control and every historical result. No promotion from training loss alone.

Decisions:
- Two matched 5,000-update CUDA fits, extra early versus extra late directions.
  Same original initialization, seeds, real/primary/supplemental exposure,
  two-correction intermediate objective, weight 0.1 and one extra draw/update.
  CPU inference retains original adaptive 1e-6/physical gates and 32-call cap.
- User reaffirmed direct low-frequency improvement first. Frequency-loss
  curricula stay deferred. Retain configuration/source/checkpoint identities,
  exposure, accuracy by band, timings, failures/retries and selection decisions
  as durable small receipts for later review, alongside the live Tuldok project.
- Freeze extra pairs from targeted checkpoint 5bbdc315… using its actual eight
  single-call corrections, independently solved compatible early/late residuals.
  Use original 16 training parents/32 states plus eight fixed pattern directions
  from each of the existing 14 analytical train grids (positive amplitude slot).
  Keep manufactured fields distinct from real CFD. No development/final labels.
- Scale both pair members separately to the original train-correction RMS,
  scaling RHS/target together. Qualify homogeneity at 0.3 and 2.0, operator/gauge
  and float32 representability. Explicitly exclude both members of a pair if its
  late/initial norm is <=1e-12 (numerical-noise guard), require every group active.
  One private balanced group sampler, identical indices/shapes/cell work for both.
- Add closed v8 direction-feedback input contract in the canonical trainer.
  Capture manifest plus <=32 MiB numeric bank as typed project inputs for
  train/restore. Existing v1–v7 paths retain arithmetic/order/evidence parity;
  evaluation CPU recipes keep their original inputs. No inference bank access.
- Preserve cubic scalar h, sealed full water, fixed rho/mu/dt/origin/boundaries,
  float32 neural weights/gradients/Adam and existing physical arithmetic.
  No architecture, inference policy, new fluid data or deployment change.
- Existing Tuldok catalog owns one CUDA optimizer at a time, fresh CPU restore,
  validation and reserved inference. Existing workflow/native/custody owners
  remain canonical. CPU pools prepare labels and qualification independently;
  use observed workload concurrency, retain timeouts and reaping outcomes.
- Reserve and commit eight new whole final scenes before fitting. Include all
  58 consumed standard development trajectories; elongated four remain
  exploratory. Rank eligible models by max calls, worst paired pressure error,
  then mean calls, including retained control. Freeze new challenger and overall
  choice before final; no final-informed refit or reselection. Promotion also
  requires no final regression in qualification, max/mean calls or worst pressure.
- Include unchanged targeted checkpoint as fixed-call/same-input reference.
  Compare pressure, gradient, low-mode error at 4/8/12 calls on common fields,
  fitting/familiar physical checks and all full trajectories. Fixed-call failures
  remain reported, not hidden by adaptive successes. Independent-seed confirmation
  remains conditional on a complete improvement and outside this first pair.
- CPU input preparation <=768 MiB combined resident tensors. Training per-Run
  bounds stay 600 seconds/32 MiB outputs. Local retention <=2 GiB plus1 GiB free.
  No weights/raw arrays/auth credentials in Git. Preserve unrelated cohort files.
- OOM recovery: explicit separable cosine transforms replace the unoptimized
  four-operand contraction that requested32.8 GB on one40-cubed case. This shared
  numerical-oracle memory fix is committed separately from training changes.
  Preparation uses4 workers, bounded submission, parent additional virtual
  memory <=1 GiB, worker <=512 MiB and >=8 GiB available before startup. Generate
  one selected analytical grid at a time. Preserve the interrupted24-worker
  attempt as host-OOM/reboot, not a normally completed/reaped scientific run.
- User hypothesis check: calculate temporal sinusoid sampling bounds for dt,
  distinguish observation duration from sample spacing and spatial wavelength.
  Reuse consumed saved states for a fixed-input dt-metadata check and three
  stationary fundamental spatial modes. CPU only, <=512 MiB address-space
  growth, no new trajectories, training examples, temporal history or refits.
  Record limits: this does not qualify actual changed-dt dynamics, and wavelength
  alone gives no temporal frequency without a motion/dispersion relationship.
- Follow-up hypothesis diagnostic: split the same consumed saved residuals into
  the existing first-three-cosine-modes-per-axis band (excluding constant gauge)
  and its complement. Measure intrinsic low-input solve error, other-band leakage
  and their superposition at4/8 calls, bounded CPU only. This is input isolation,
  not a curriculum fit, temporal spectrum or a frequency-independent guarantee.
  A high/middle-first then full-spectrum curriculum remains a proposed separate
  ablation against full-spectrum and band-balanced training at matched budgets.
  Keep all-band evaluation/physical gates throughout; do not alter current fits.

Write set: experiments/water-conditioning-v1/training/{policy.py,model.py,
train.py,project.py,pipeline.py,direction_feedback.py,test_direction_feedback.py};
experiments/water-multigrid-learning-v1/{record.py,qualify_bank.py};
experiments/water-late-residual-v1/**; docs/plans/water-late-residual-v1/**;
README.md; tuldok-project.json. Ignored runs/water-late-residual-v1/** and activity.
OOM fix also owns experiments/water-residual-learning-v1/audit.py and the existing
mode-analysis caller in experiments/water-pressure-accuracy-v1/diagnose.py; old
reports remain immutable. Reachable four-operand production contractions: these
two owners, now replaced; a tiny dense contraction remains only in the parity test.
Clean source capture may use /tmp/rheon-late-residual-capture; retain its exact
commit via branch reachability and remove only that clean owned worktree after
capture, recording protected-OID and registration/path postconditions.
Composed-design review: not-applicable. This adds one bounded extra-bank consumer
to existing project-input/training owners, preserving numerical/runtime composition.

Milestones and acceptance, automated:
- M1 CPU bank qualification: independent neighbor-action defects, gauges,
  homogeneous prediction, train-only lineage, paired sampling/cell work and
  memory bounds; one example before broad generation. Accepted:144 paired cases,
  no exclusions, independent defects <=4.08e-13,12.158s bounded preparation.
- M2 focused contract/gradient/parity evidence and clean committed-source capture;
  managed one-update CUDA fit and CPU restore qualify the real new input boundary
  before full fits. Required-real local RTX5090/Tuldok. Pending.
- M3 two complete managed CUDA fits, fitting/familiar/fixed-call CPU comparisons,
  58 development trajectories, frozen final comparison, complete source/custody,
  process/timing/failure records and same-catalog project visibility. Pending.

Current acceptance: pending. Current phase: managed source/fixture preparation.
Next slice: committed clean source capture and one-update CUDA/CPU restore fixture.
Blockers: none. Replan on invalid labels, incomplete lineage, parity violation,
resource breach, unavailable CUDA/custody or unreaped work. Numerical improvement
is measured, not a completion prerequisite. Decisions/issues/receipts live in
execution-ledger.md, issues.md and reports/. Standards snapshot
snapshot:v1:be8a88a8-99b2-4355-a04c-f1306390fe13;23 policies, no unresolved facts.
