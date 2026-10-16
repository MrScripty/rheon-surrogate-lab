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
