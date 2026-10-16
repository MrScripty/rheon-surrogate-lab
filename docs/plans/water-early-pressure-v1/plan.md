# Earlier accurate pressure corrections

Status: Active. Admission: start, authorized by the user's continuation of the
matched CUDA / harder training-side examples / CPU regression sequence.
Canonical plan: docs/plans/water-early-pressure-v1/plan.md.
Owner: this task, branch experiment/water-early-pressure-v1, based on bdf2238.
Retain branch for review; no push, rewrite, deployment or deletion authorized here.

Objective: learn accurate pressure corrections earlier, reducing peak inference
work without surrendering the pressure-aware model's accuracy. Also test the
user's amplitude/noise-floor hypothesis, distinguishing temporal and spatial
frequency. Scientific measurement acceptance does not require improvement.

Decisions:
- Preserve sealed full water, fixed physics, cubic scalar spacing, previous
  pressure/history warm-up, float32 CUDA neural training, CPU inference, current
  equation/divergence and 1e-6 mean-free pressure-bound gates, maximum32 calls.
- Prepare a paired early/late bank from the previously qualified frozen 5bbdc315 checkpoint after8
  calls on original train-only states and manufactured train-only fields. Add
  six new long-axis grids, not inspected evaluation RHSs or reserved scenes.
  Independent exact pressure labels and existing bounded-memory preparation.
- Two 5000-update CUDA fits from the original seeded initialization: identical
  new bank (late stage), real/primary/supplemental exposure, sampler, two forward
  corrections, optimizer and learning-rate schedule. Control retains old loss;
  challenger adds weight1 full pressure/gradient/equation feedback on first
  correction. Existing first-equation contraction and final objective remain.
  This isolates feedback at fixed example/cell/network-call budget. Separately
  report changes from the archived older-bank late model; those are not a pure
  loss ablation. No optimizer concurrency. Per Run600s /32MiB outputs.
- Version9 adds only training feedback. Weight0 preserves v8 arithmetic; no
  additional network forward pass or inference inputs. Canonical trainer and
  captured source witnesses own this contract. Fresh CPU restores required.
- Run one actual CUDA smoke/restore before full fits. While GPU trains, CPU
  diagnostic and preparation run independently with bounded worker/thread counts.
- Supplement diagnostics with4/8/12-call replays on the15 existing consumed
  common RHSs through the established fixed_compare owner; reuse verified archived
  baselines. Diagnostic only, no training or selection-gate change.
- Measure reference signal and prediction error temporal PSD on retained teacher
  states with the same detrending/Hann windows; record absolute and signal-relative
  power and weak-signal cases. Controlled spatial pure/mixed amplitude sweeps
  distinguish scaling, cross-band leakage and float32 rounding. These are
  explanatory diagnostics, not causal proof about autonomous trajectories.
- Freeze both resulting checkpoints and run all74 consumed scenes through the
  existing pressure-bound CPU owner (two independent one-thread workers), first
  native controls then candidates. Report each of the original58 / amended8 /
  recent8 separately. Compare both against archived late-pressure and retained
  reference: no physical failure, worse worst pressure, mean or maximum calls in
  any group. Rank eligible challengers by maxcalls, worstpressure, meancalls.
- Only if these gates pass, freeze a sole challenger and run eight untouched
  scenes already reserved in water-pressure-bound-regression-v1/configs/fresh-scenes.json,
  native fixtures/control first. No final-informed refit or reselection. Fresh
  scenes remain unused on rejection. Repeat representative serial timing after
  concurrent work; ordinary desktop apps remain.
- Preserve every config/source/checkpoint/bank binding, managed Run, CPU receipt,
  lifecycle failure, timing, memory, scalar result and selection decision. Same
  Tuldok catalog, new immutable project snapshot; older snapshots retained.
  No weights/raw arrays/credentials or unrelated spectrogram/cohort files in Git.

Write set: experiments/water-early-pressure-v1/**;
docs/plans/water-early-pressure-v1/**; README.md; tuldok-project.json;
experiments/water-conditioning-v1/training/{policy.py,model.py,train.py,project.py,
test_first_pressure.py}; experiments/water-late-residual-v1/prepare.py (additive
config-selected grids/report destination, old defaults preserved).
No runtime portability or native/model architecture changes.

Composed-design review: not-applicable; this extends an existing captured trainer
loss and reuses existing managed lifecycle, preparation and episode owners.
No new subsystem or overlapping lifecycle authority. Two reviewable outcomes:
source-qualified training contract, then measured comparison with retained results.

Milestones:
- M1 Accepted: qualify bank and training objective; gate independent labels,
  v8/weight0 loss+gradient parity, additive first-loss proof, real CUDA smoke and
  CPU checkpoint restore. Write set above; replan on lineage/resource/contract failure.
- M2 Active: matched full fits, explanatory CPU amplitude diagnostics and all74
  trajectories. Gate complete records including failures, unchanged physical
  thresholds, custody drained and children reaped. Improvement is reported separately.
- M3 Planned: current serial timing and conditional fresh validation; gate frozen
  eligible selection or explicit fresh deferral; records sealed and reviewed.

Acceptance: partial. Claims M1 source/numerics (automated CPU plus real CUDA/
CPU restore), M2 full comparison (environment-qualified GPU/CPU), M3 retention and
conditional validation (artifact/real CPU); M1 satisfied (reports/smoke-result.json and source-tests.json); M2/M3 pending.
Current phase: matched CUDA fitting. Exactly one next slice: complete both frozen
5000-update fits, CPU amplitude diagnostics and74-scene regression. Blockers: none.
Replan on changed numerical semantics, invalid lineage, resource breach, unavailable
custody, unreaped workers or evidence invalidating the admitted composition.
Standards: snapshot:v1:10196c62-2edb-4466-a3c7-c3448f7bc2bc,16 policies,0 unresolved.
[Ledger](execution-ledger.md) · [Issues](issues.md) · [Reports](reports/).
