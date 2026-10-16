# Execution ledger

## Admission and controlled design

The last plan is terminal. User continuation admits this new bounded tranche:
coverage-only11-grid and feedback-only8-grid fits, both5,000 updates with the
original initialization/sampling seeds. Neither changes neural architecture,
physics, inference policy or real CFD population. The v6 contract is explicit;
legacy schemas keep their existing contract and numerical behavior.

Intermediate feedback penalizes the residual at the first correction, rather
than only the final two-correction output. It does not yet introduce later
residual distributions. The coverage fit changes balanced grid exposure; this
is disclosed, not treated as equal old-grid exposure. All final identities are
committed before fitting, and the most recent final cases become development.

## Implementation qualification

The focused 25-test suite passes. Additional residual tests verify exact legacy
analytical criterion gradients after introducing training_loss. Full CPU bank
qualification checks 1,408 coverage labels and 1,024 feedback labels, worst relative
operator error 6.44257e-14. The v6 eight-grid control preserves old tensor/order
bytes; the coverage bank preserves all1,024 old tensors. Projected bank residency
is 642,528,256 versus 529,469,440 bytes. No optimizer update/CUDA initialization
occurred during these checks. Seven reused reference bindings match checkpoint,
scenes and binary, and a deliberately wrong checkpoint is rejected before work.


## Managed execution and parallel preparation

Training source was captured from commit
6b9ef674b25a738f87bb24e4f04b4d9f85e8c549 into two byte-identical 37-file
projects. Assembly took 2.509 s and 2.365 s. The clean temporary worktree was
removed after its exact head was verified reachable from the experiment branch;
reports/worktree-lifecycle.json retains that disposition.

The existing Tuldok catalog/project was registered and read back. The scientific
controller began at 2026-10-16T06:12:31.583669Z. Coverage training completed 5,000
CUDA updates in 209.590 s, producing checkpoint
e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f.
A fresh CPU Run restored identical weights with zero updates and no CUDA
initialization. All three managed stages completed/drained in 850.058 s. The
coverage CPU qualification overlaps preparation/execution of the feedback fit;
there is only one CUDA optimizer. No training retry has occurred.

Independent CPU preparation applied eight fixed learned corrections to the
retained control's 32 training states (16 parents), with zero updates/CUDA.
Elapsed time was 8.401 s; median residual became 2.69826e-5 of its initial norm.
The largest captured teacher defect was 1.6597% of a late residual. Future late
labels must account for that finite teacher defect before admitting a fit.
This preparation neither changes the current fits nor consumes final cases.

Early coverage diagnostics: 32/32 fitting and 128/128 familiar adaptive states
qualify. Familiar mean/max calls are 13.3984/19 versus the retained control's
9.41406/12. Fixed eight calls qualify 0/128, with worst relative residual
3.95082e-4. Native fallback diagnostics are retained separately and never count
as learned success. Autonomous/final judgments remain pending at this point.


Feedback-only training completed 5,000 updates, checkpoint
74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631.
Its historical optimizer interval is 224.214 s; the train receipt's broader
elapsed interval is 229.573 s. These boundaries are retained separately rather
than treating Run preparation or the full receipt as optimizer-only time.
The final post-update sampled residual is 0.00152274 versus coverage's 0.0105054;
these are training diagnostics, not evidence of qualification or promotion.
CPU restore and full validation are pending at this ledger entry.


The completed 18 standard coverage trajectories (8 fresh, 7 prior grids, 3 long)
all qualify. A read-only comparison of matched native/callback traces records
median per-scene successive residual ratios 0.586626 for coverage versus 0.303818
for the control; neither has a step with residual growth in this population.
These observed ratios explain higher work, not a global contraction proof.
Source trace hashes and exact command timing are retained in
reports/coverage-standard-contraction.json and the local command receipt.
Feedback CPU restore completed/drained with identical weights, zero updates and
no CUDA initialization; its managed validation remains active.


## Development completion and frozen selection

Both fits pass all 32 fitting/128 familiar adaptive states and all 42 required
whole development simulations. All six managed stages completed/drained with
fresh CPU restore, re-opened Runs and verified duplicate downloads. Optimizer
intervals are disjoint and total 433.804 s. Managed sequence plus overlapping
standard CPU qualification takes 2026.160 s. CPU qualifications take 276.261 s
and 214.208 s; their sum is not elapsed workflow makespan.

Frozen selection at 2026-10-16T06:54:36.249756Z retains the control as development
choice. Rank [maximum calls, worst pressure relative L2, weighted mean calls]:
control [13, 6.13957e-6, 9.67760]; coverage [21, 1.05214e-5, 15.20848];
feedback [28, 5.82528e-5, 9.61993]. Coverage is the new challenger under the
predeclared worst-case ranking. Feedback's familiar 30% mean-work improvement
does not overcome its worst-case regression. Final evaluation compares only
control and coverage; there is no final-informed checkpoint reselection.
The eight new whole final simulations started after this selection froze.


## Final evaluation, audit and closure

The eight final simulations complete at 2026-10-16T06:57:52.095452Z. Control and
coverage both qualify 8/8 scenes and 904/904 AI steps, with no cap hits/hard stops.
Mean/max calls are control 10.1980/12 and coverage 17.1250/21. Worst pressure
relative L2 is 1.92709e-6 versus 5.65607e-7. Coverage improves worst final pressure
difference by 3.41× but needs 67.9% more calls; it is not promoted. Feedback was
not evaluated on this final population. The final cases are now consumed.

Scientific controller elapsed is 2720.512 s; capture-to-final is 2748.715 s.
The final study takes 195.778 s. The timing index retains 56 fresh delegated
commands with terminal return/reaping plus six managed Run queue/execution
intervals. Independent source-bound correction traces retain both successes
and failed native step IDs. No scientific process failed, no training retry or
postprocessing recovery was required, and no acceptance threshold changed.
Scientific failed qualification is distinct from execution/lifecycle success.

Final indexing verifies six unique completed/drained Runs, 208 native trajectory
children exited zero/reaped, 10 drained diagnostic populations/856 cases, all
1,808 final callback/frame pairs and qualified steps, frozen configuration and
selection before final, no surviving owned controllers, and 223,521,880 local
bytes within the 2 GiB budget. Both CPU checkpoints restore identical weights
with zero optimizer updates; actual CUDA intervals do not overlap.

The project was read back from the same catalog with unchanged manifest ID/hash
and 64 evidence references. Initial registration is separately retained. The
capture worktree remains removed with its exact source head protected by the
experiment branch. Historical results and the two unrelated cohort-progress
files are untouched. Source commit 6b9ef674b25a738f87bb24e4f04b4d9f85e8c549 owns
training/config changes; this evidence closure adds preparation/index source,
methods, numerical results, timings, lifecycle receipts and prepared follow-ups.
The branch remains retained for review/continuation under plan.md's contract.
No model weights/raw datasets, application deployment or history rewrite.


Closure review covers the complete branch range from
edb9992fe5c8df05a7486d6ceb47e95cbfe6b8e6 through the training source commit and
this evidence closure. The source/evidence split is intentional; no fixup,
regression/revert sequence or unexpected merge topology is present. The 58-file
staged closure is restricted to the declared experiment/README paths, contains
text receipts/source only, and passes sensitive-pattern and whitespace checks.
All 38 JSON files parse with finite values, 12 current documentation links
resolve, and executed preparation/controller/index source and timing hashes
match. Existing 25-test evidence plus the eight-test gradient follow-up, full
operator/parity bank checks and actual managed/native system evidence satisfy
the selected checks; documentation-only edits do not require training replay.
