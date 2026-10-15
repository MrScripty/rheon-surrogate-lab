# Residual learning results

Completed: 17 normal Tuldok stages, all completed/drained, reopened and whole
artifacts redownloaded with parity; 40,002 CUDA optimizer updates. Five controlled
fits plus a two-update CUDA/CPU smoke. No model meets physical qualification.
No new final test or deployed checkpoint was used. Historical results remain intact.

## Accuracy comparison

Entries are relative equation residuals; the unchanged target is 1e-6. These
are one-frame capacity experiments evaluated on the same 64 independent native
validation parents at two saved states each. Validation is development evidence.

| Trial | CUDA updates | Network applications per proposal | Fitted residual | Validation median | Validation worst | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained seven-channel no-GroupNorm baseline | 10,000 | 1 | 0.000398222 | 0.082815901 | 0.293090166 | 0/128 |
| [One physical residual channel, SiLU](reports/residual-nonlinear.json) | 10,000 | 1 | 0.000519353 | 0.044418082 | 0.291001740 | 0/128 |
| [One physical residual channel, linear](reports/residual-linear.json) | 10,000 | 1 | 0.000419244 | 0.047737922 | 0.289982696 | 0/128 |
| [Matched 5,000-update linear control](reports/residual-linear-5000.json) | 5,000 | 1 | 0.000427563 | 0.054937861 | 0.291362902 | 0/128 |
| [Four shared linear corrections](reports/residual-recurrent.json) | 5,000 | 4 | 0.000349860 | 0.014334834 | 0.221805073 | 0/128 |
| [Correction/gradient/divergence objective](reports/residual-physical-loss.json) | 10,000 | 1 | 0.000940524 | 0.043581483 | 0.270796572 | 0/128 |

The recurrent median is 73.9% lower than its matched single-correction
control. Its optimizer interval is 206.77 seconds versus 51.17 seconds, with four
network applications and residual recomputation per proposal. Initial parameter
bytes, train inputs, normalization, sampling and learning-rate schedule match;
see [parity receipt](reports/recurrent-control-parity.json). This is improved
saved-state accuracy at greater computation, not demonstrated simulation success.
The older 1,024-parent candidate had median 0.0385793 and worst 0.207926; its
population/budget differs, and its lower worst error should not be hidden.

## Independent simulations and stability

All five candidates were evaluated in three new paired native/AI development
trajectories: cube [16,16,16], elongated [24,16,12] and compact [12,8,10], with
varied force centers, strength, swirl, fluctuation and frequency. Cells remain
cubic, with one spacing and proportions determined by per-axis counts. Every
trajectory starts with 15 native steps; first actual AI inference is step 16.
The native/native controls qualify all 64 reference frames and 49 subsequent
projected steps in each scenario. These are full new simulations, distinct from
recorded-state teacher evaluations. Reusable scene JSON and receipts are retained.

| Candidate | Actual AI steps before stops, cube / elongated / compact | Qualifying AI steps | Hard stops |
| --- | --- | ---: | ---: |
| One physical residual channel, SiLU | 26 / 23 / 12 | 0/61 | 3/3 |
| One physical residual channel, linear | 7 / 6 / 7 | 0/20 | 3/3 |
| Matched 5,000-update linear control | 7 / 6 / 7 | 0/20 | 3/3 |
| Four shared linear corrections | 5 / 4 / 6 | 0/15 | 3/3 |
| Correction/gradient/divergence objective | 8 / 6 / 7 | 0/21 | 3/3 |

The lower recurrent validation median does not carry over to stable simulation.
All native references observed before the learned stops qualify. Raw learned
pressure failures remain visible; no native fallback is counted as AI success.
CPU native warm-start checks also require fallback on all 128 validation states
for every candidate. There is no qualified full-simulation speedup.

Structural linearity reduces median superposition discrepancy to approximately
2e-7 for the single-step linear models, but does not ensure a stable inverse.
Eight wrapper proposals amplify median equation residual to 201/222 for the
single-step controls and 930 for recurrent (32 internal network applications).
The nonlinear model reaches 0.01134 after eight proposals, still unqualified.

[Manufactured Neumann mode probes](reports/recurrent-mode-probes.json) test 24
compatible modes on the three grids, at the fitted correction's 2.1793 Pa RMS.
The recurrent model reduces first-proposal error on 21/24; all three finest
checkerboard modes grow (up to 7.25x), and repeated proposals reach worst 1.65e10.
This identifies an unstable spatial component, without claiming a global proof
or calling these algebraic probes simulations/cold-start qualifications.

## Method and scientific controls

Train on the same native 16-cubed frame 15, with actual previous solved pressure;
seed 20261015 and sampling seed 2026102501. CUDA float32 model parameters, gradients
and Adam moments, deterministic eager execution, AMP/TF32 off. Physical operator,
labels and pressure reconstruction retain their existing f64 arithmetic. Fresh
CPU restores construct no optimizer or CUDA context and reproduce the exact
prediction inventory. All physical equation/divergence/flux/Courant gates remain
unchanged. Twenty focused checks pass (16 Python, four existing native engine
tests), including independent stencil/spectral and real process cleanup checks.

The new input is q=(rho/h)*(b-A p_previous), divided by train-only q RMS, one
channel on the original grid. For the admitted uniform-density cubic sealed
operator, A=(h/rho)L. The output uses the historical train pressure RMS, cell-zero
gauge and supplied base pressure. Coordinates/density/timestep are not network
maps; density/spacing act through physical assembly and scaling. The versioned
contract is restricted to the existing full-fluid sealed Neumann setting.
Masks, free surfaces and different material operators are not established.

First compare representation against the retained seven-channel no-normalization
baseline, then remove SiLU and offsets for the linear experiment. Unrolled
training uses shared weights and recomputes actual residuals after each proposal.
Canonical input/linearity/recurrent comparisons retain the same pressure/equation
loss. The separate physical-feedback objective weights correction-relative MSE
0.25, correction-gradient MSE 0.25, relative equation MSE 0.40 and maximum normalized
equation error squared 0.10. Loss values between objectives are not comparable.
Exact schedules, configurations, source/input hashes and traces are per-trial.

CPU ideal float32 correction controls pass the equation-only gate on all 128
validation states for each model; earlier fitted controls also pass native
physical checks. The target is representable for these controls. Analytic
low/full spectral corrections remain explicitly algorithmic controls; the full
f64 inverse removes the equation error and is not counted as learned accuracy.
The measured learning gap and unstable modes remain after precision controls.

## Retention and disposition

[JSON index](reports/summary.json) and [CSV comparison](reports/comparison.csv)
index immutable per-trial method/result records, Run/catalog IDs, source commits,
config/input/preprocessing hashes, sample order, checkpoint hashes, traces, UTC
training intervals, CPU/native outcomes and trajectory receipts. Large weights
and simulation arrays remain in the original Tuldok catalog and verified ignored
local downloads; none are added to Git. All native children are drained/reaped;
the lab trajectory parent retains failure/deadline cleanup evidence. See
[commands/environment](reports/commands.md) and [checks](reports/tests.json).
All earlier v1 records remain unchanged; later v2 adds explicit runtime/time fields.

The prepared next cohort has 16 genuine independent simulations and 32 actual native
snapshots with preceding pressure and varied physical setups. It has not trained:
all five fitted candidates still miss 1e-6 and actual simulations are unstable.
The larger-cohort/several-seed stage therefore remains conditional, as agreed.
No unused final-test population was consumed. Current comparison UI deployment
remains unchanged; it requires an explicit new adapter contract before admitting
one-channel checkpoints. See [findings](issues.md).

The next priority is constraining the measured high-frequency amplification and
closing the known-frame fitting gap before expanding the corpus. The
[evidence-based next experiments](next-steps.md) are prepared, not executed.
