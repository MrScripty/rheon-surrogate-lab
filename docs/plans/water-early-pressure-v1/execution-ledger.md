# Methods and execution ledger

2026-10-16: admitted user-authorized continuation at bdf2238. Previous task's
74-scene pressure-bound run improved pressure accuracy but recent maxcalls17vs11
rejected replacement. Three saved-input replays showed at most one avoidable
certificate call; target faster learned convergence, preserving all thresholds.

New branch experiment/water-early-pressure-v1. Four unrelated untracked user
spectrogram/cohort files remain read-only. User's spectrogram builder inspected:
it plots temporal error PSD, normalized by full-record reference mean-square,
not per-frequency signal power. Thus it cannot yet answer signal-amplitude
conditioning. New diagnostics will explicitly distinguish these quantities.

Preparation replan: current late teacher failed the existing float32 homogeneity
check on train parent0004/frame63 (1.01689e-5 vs1e-5 at gain0.3). No bank
published or fitting started; retain attempt and threshold. Reuse the prior
qualified targeted teacher5bbdc315 instead. This also keeps the original144
directions identical to historical preparation, isolating added long-axis data.
Original failed total elapsed was not durably timed; explicitly unavailable.
All subsequent expensive commands use the existing retained command timer.

Qualified bank:192 pairs,0 exclusions,24,476,086 bytes per stage. CPU preparation
15.012s inside owner (outer timing retained), four bounded one-thread workers.
Original train-only teacher source preserved; no validation RHSs used.
V9 source tests verify closed weights, v8 loss/gradient parity at weight0,
unchanged inference and full first-pressure loss with no extra forward pass.

Baseline amplitude diagnostics completed:two one-thread CPU processes on cores4/5,
~85.1s each.16 consumed teacher trajectories x49 states x1/8 calls.
Temporal periodic Hann32/hop8, lowest positive bin1.875Hz; cannot resolve slower
cycles from these short windows. At the lowest bin, late-model median relative
RMS error after8calls is3.10e-7/4.99e-7/2.71e-7 across weak/middle/strong signal
tertiles: no monotonic weak-signal degradation there. These are pooled, correlated
observations, not a controlled temporal noise experiment.

Controlled spatial sweeps: globally scaling a pure low mode preserves relative
error approximately. Adding a1Pa high mode produces low-mode leakage that can
overwhelm a weak low mode. On9x11x33 late-model8calls: pure1Pa low coefficient
error -0.001838Pa; high-only low leakage -0.018253Pa. Mixed low0.001Pa error
-0.018256Pa, about18.26 times the low signal. This is learned cross-band error
on this manufactured case, not proof that float32 cannot represent the mode.
No training policy or data changed in response; predeclared ablation retained.

M1 accepted:real CUDA1-update smoke and optimizer-free CPU restore both completed
and drained, repeated artifact downloads verified. Run IDs and actual arithmetic
are retained in smoke-result.json. Source admitted at e43d4d9. A premature pair
invocation before workflow final receipt was refused by the guard, before any
new Run; full pair admitted only after receipt publication.

M2 diagnostic detail:reuse fixed_compare.py on15 already consumed common saved
RHSs at4/8/12calls for both new fits, with archived identical-input baseline
receipts. This directly measures earlier-correction accuracy without autonomous
trajectory confounding. No new samples enter training or change selection gates.
Control replay runs on CPU6 after control restore, during next managed preflight.

Control completed5000updates:416.9863s optimizer,11.9908updates/s; managed train
plus CPU restore/repeat download889.3034s, child reaped. Added data alone reduces
manufactured9x11x33 pure-low8-call coefficient error39.66x and high-only leakage
143.93x, but worsens15-state equal-call worst mean-free pressure error by
24.2%/28.4%/92.2% at4/8/12calls. Those distinct probes show a tradeoff, not
replacement qualification. CPU control amplitude69.89s; common-input replay2.98s.
New first-pressure feedback fit proceeding unchanged.

Input-rounding diagnostic:analytical q uses exact current train normalization,
one float32 input cast, then existing exact physical oracle (no neural pass).
All unquantized references agree with closed-form modes below1e-10. On9x11x33,
high-only input rounding induces5.16e-8Pa low coefficient error; mixed low0.001Pa
gives -2.37e-7Pa, whereas learned8-call high-only error is -0.01825Pa (oldlate)
or -0.0001268Pa (newcontrol). Thus this quantization contribution is much smaller
than observed model leakage here. At low1e-8Pa, input quantization can overwhelm
the low mode (4.16e-8Pa coefficient error):amplitude/dynamic range does matter.
This is not an iterative inference floor:physical residuals are recomputed in
float64 between float32 neural calls. Initial diagnostic attempt incorrectly
read scale metadata as a number; TypeError and0.9s attempt retained, corrected
to scale.value; qualified actual numeric work0.012s. No model/fit changes.

Both full fits and fresh CPU restores completed/drained. First-pressure optimizer
453.7712s,11.0188updates/s (8.82% slower than control); full managed969.4033s.
Both peak CUDA allocated770,565,632B and weights1,485,452B; initialization,
normalization, sampling/cell-work streams proven identical. Frozen weights are
in training-and-admission.json; no files switched into deployment. Real
native/control/first-pressure warm-up-plus-one fixture all qualified.

First-pressure common-input8-call worst mean-free relative error3.848e-7,
47.37% below control and32.43% below oldlate. This is not uniform:oldlate is
better at4 and12calls (first-pressure16.85% and9.69% worse respectively).
On manufactured9x11x33 after8calls, first-pressure pure-low error -0.0003712Pa
and high-only low leakage -0.0018948Pa; improved versus oldlate but worse than
newcontrol. No selection inferred from these tradeoffs; full74-scene suite runs.

Original58 population closed while remaining consumed scenes continued:all58
qualify per model. Control mean7.7351calls/max12/worstpairedpressure4.6819e-6;
first-pressure mean7.7404/max11/worst3.5943e-6. Both worsen priorlate worst
3.1448e-6 despite lower mean work. Therefore no candidate can satisfy the
frozen all-population replacement gate; independent8 stays reserved. No profile
tuning, refit or final-informed reselection. Conditional final executor was
prepared but not invoked or retained once this gate failed; existing native
phase owners remain canonical. Full remaining trajectories and timing continue.

M2 accepted:all74 native controls and148 model trajectories qualify,17,046 AI
steps and466 independent pressure-bound coverage checks, no failed episodes.
Development phase2270.99s (~37.85min). No replacement:control preserves all
retained-reference group gates and reduces recent max17→11 (overall17→12),
but worsens oldlate worst pressure in original58 and recent8; first-pressure
also regresses pressure and peak work on some groups. Full CSV/JSON results
retained. Freeze control for consumed timing only by predeclared max/pressure/
mean ranking; this is not a promotion or final selection. Fresh8 remains unused.

M3 accepted:all12 serial episodes qualified. Small-grid median7.64% lower, long
rectangular0.71% higher,40³1.51% lower (limited14steady samples/model, within
repetition variation). Timing307.85s. Scientific closure interval85.93min from
qualified prep through last measurement, including gaps; initial untimed failure
and later documentation excluded.237 native and261 recorded measurement command
children reaped;6 managed Runs completed/drained. Max individual RSS866,668,544B;
raw retention498,760,027B. Independent8 remains untouched; model weights and
physics/runtime sources unchanged since admission. summary.json seals source
bindings, all population gates, clocks, resources and lifecycle receipts.

Terminal review:executed admit/compare/seal source hashes match current files;
source bindings still match. Host task process scan empty. Existing Tuldok catalog
serves completed status and preserves previous project5204e37e…; no application
restart/catalog replacement. Explicit history range bdf2238 through02d4e35 plus
reviewed final staged result receipt, no history rewrite/merges. Four unrelated
user files remain untracked and untouched. Final commit retains full measured
tradeoffs, clocks, source/Run bindings, failures, and conditional validation
deferral; no binary weights, raw field arrays or credentials staged.
