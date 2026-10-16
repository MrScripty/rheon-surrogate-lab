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
