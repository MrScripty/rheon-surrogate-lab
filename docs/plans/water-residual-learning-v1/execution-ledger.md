# Execution ledger

2026-10-15: User authorized the prepared sequence and durable method/result
retention. Clean base 18159af; new experiment branch created. Existing managed
trainer and source capture path inspected. Previous no-GroupNorm fitted residual
0.0003982221868309573, validation median 0.0828159014298824; 0/128 qualified.
This is a fitting improvement, not improved held-out accuracy. No new runs yet.

Initial focused test: 4/5 passed; superposition's per-cell tolerance rejected
5/315 near-cancelling float32 values (largest absolute discrepancy 2.15e-7).
Use a correction-vector norm-relative 3e-5 comparison for this algebraic
arithmetic check; unchanged 1e-6 physical acceptance is evaluated independently.
The operator, preprocessing and prepared gradient checks already passed.

Focused checks: all five residual tests and seven historical conditioning tests
pass. First registration returned Tuldok HTTP 409 because an immutable project
ID already referred to the prior snapshot. Assign a new snapshot ID, retaining
the old project and the same catalog/root. No Run was submitted by that failed
registration. The initial observer used system Python without NumPy; restart
with the existing laboratory virtual environment. No source/model change.

Source 725b100 captured physical-residual smoke and full nonlinear/linear trials.
CUDA smoke Run 455586b9f3c24a06ba0b50df1d932c12 completed two actual updates;
float32 params/gradients/Adam moments on RTX 5090 Laptop GPU, deterministic,
AMP/TF32 off. Fresh CPU restore remains pending. Distinct project snapshot
54e60e2d5f8a4267a002d8d11e5e2666 registered in the original catalog.

CPU diagnostics commit fc2b160: independent spectral regression passes; old
no-GroupNorm validation audit completed on 128 states, 23 one-thread workers
(22 observed cores), 7.59s, no CUDA. Summary retained in reports. Full analytic
control is not AI success; eight raw learned proposals do not converge well.

Physical-residual smoke accepted: train 455586b9f3c24a06ba0b50df1d932c12 and
CPU restore 87786368c8794a8eb827a7f35a9db458 completed/drained, reopened and
whole artifacts redownloaded with parity. CPU has no optimizer or CUDA context,
exact fresh prediction inventory. The full nonlinear 10,000-update input ablation
is now executing through the same catalog. Exactly next scientific slice remains
its native evaluation, followed by the matched linear architecture comparison.

Prepared the later physical-feedback comparison while managed execution proceeds:
version-2 residual configuration selects canonical or correction-gradient-divergence
objective explicitly. The changed objective uses 0.25 relative correction MSE,
0.25 relative correction-gradient MSE, 0.40 relative equation MSE and 0.10
normalized maximum equation error squared. dt/h^3 maps equation error to
predicted divergence; normalization cancels that per-case scalar. This is
training feedback, not a substitute for corrected-f32 native physical gates.
Canonical loss/preparation remains unchanged for the earlier input/linearity
controls. All six residual tests pass, including zero loss at a manufactured
reference and an independent finite-difference directional gradient. The later
objective trial has not been assembled or run yet.

Trajectory preparation: the old application bridge explicitly admits only seven
channels. Preserve that independently deployed contract. A separate lab probe
includes the existing Rheon paired Engine/Scene source without editing its
checkout and uses the qualified prepared Rheon library. A CPU parent serves the
new captured checkpoint directly. Release build and four actual existing engine
tests pass; Clippy passes after correcting a local wrapper argument placement
(the first malformed invocation failed, no source issue). The native/native
64-step control has 15 native warm-up +49 actual projected steps, 49/49 qualified,
64/64 reference frames qualified, no hard stop, child exited/reaped. No model
inference is claimed for that algorithmic control. Model trajectories follow
after fresh CPU restores. This adds a bounded lab verification seam, not a
production deployment migration or changes to the Rheon team's source.

Initial one-channel nonlinear CPU evidence: fitted native residual
0.0005193531068407955 (0/1 physical qualification), validation diagnostic median
0.04441808198, max0.29100174022. This improves held-out residual relative to the
matched historical one-frame no-GroupNorm model, but not prior larger-cohort
best, and not the acceptance gate. Full native validation remains pending.

Add a 5,000-update single-correction control matched to the recurrent run's
seed, input, objective and 3,000/4,500-step schedule. The 10,000-update linear
trial alone cannot isolate recurrence because its budget/schedule differ. The
new control uses the numerically qualified prepared path; recurrence necessarily
recomputes intermediate residuals. Report compute/sample exposure separately.

Independent nonlinear trajectory: new cube force/seed, 15 native warm-up steps,
26 actual CPU AI predictions before the unchanged hard stop at step41. All
41 reference steps qualify, 0/26 AI steps qualify; worst residual3.07294 and
actual divergence92.39115. Child exits/reaps. This reveals accumulated deployment
error despite lower teacher-state validation median; it is not a cold-start
failure or accepted simulation. Source/build and native-control receipts retained.

Fresh CPU linear evidence: fitted residual0.0004192441490, validation diagnostic
median0.04773792218. Superposition discrepancy falls to2.23e-7 (f32 arithmetic),
but eight raw applications diverge to median222.126. Thus structural linearity
is established without a convergent approximate inverse. Do not promote the
linear model based on the algebraic test or assume recurrence is safe. Its
separately trained four-correction alternative remains an experiment. Native
validation and independent trajectories will retain the actual physical failures.

GPU training complete for recurrent and physical feedback. The four-step model's
last sampled CUDA residual is0.0003498759 (5,000 updates,24.18 updates/s); physical
feedback's last sample is0.0009405187 (10,000 updates,79.06 updates/s). These are
training diagnostics awaiting independent CPU/native checks. Loss values across
different objectives are not directly comparable. Recurrent proposals use four
network applications and extra operator work; eight audited wrapper proposals
therefore mean32 network applications versus8 for single-step models. The
matched5,000-update control is finishing. Keep accuracy and compute cost separate.

Recurrent CPU diagnostic median improves to0.01433483391 (worst0.2218051), but
eight wrapper applications diverge to median930.444. Independent trajectories
hard-stop after5/4/6 AI steps (cube/elongated/compact), all after15 native steps,
with0 qualifying AI proposals. A bounded CPU manufactured-mode diagnostic probes
24 known compatible Neumann pressure fields across the three grids, at the fitted
correction RMS2.1793Pa.21/24 first proposals reduce equation error; worst increases
it7.25x and eight wrappers reach1.65e10. These are operator diagnostics, not
simulations or cold-start qualification. They identify unstable spatial modes
that single-frame supervised fitting leaves unconstrained. Preserve the negative
rollout evidence alongside improved saved-state error; no deployment or cohort
expansion follows. Full native validation remains pending.

All planned GPU optimization finished:40,002 updates across five fits and the
two-update smoke. The paired5,000-update control has CPU validation diagnostic
median0.05493786090 versus recurrent0.01433483391, about73.9% lower for recurrent
with four network applications per proposal. Physical-feedback median0.04358148
is modestly lower than the10,000-update linear control, but its fitted error is
worse and repeated solves remain unstable. Final two native validation Runs are
still being drained/evaluated. The next experiments document is prepared from
the measured instability; no additional GPU experiments were started.

Closure: all 17 managed stages completed/drained, reopened and re-downloaded with
whole-artifact parity; 40,002 actual CUDA updates and fresh CPU-only restores.
All five candidates received 1 fitted  + 128 validation native checks (645 total)
and three independent post-warm-up trajectories each. Each 0/128 physical gates;
all 15 trajectories hit hard stops, while native/native controls qualify 49/49
post-warm-up steps in each scene. 20 focused checks pass. Methods, results, CPU
mode diagnosis, future experiments, JSON/CSV indexes and per-trial source/config/
checkpoint custody are retained. No new reserved test, no deployed model change,
no new Rheon/Tuldok source edits. Larger-cohort training is deferred because the
agreed known-case gate remains unmet; 16 independent simulations / 32 native states
are prepared for later admission. Source/record branch is retained for continued
experiments and reviewed adapter integration, with no worktree created.
