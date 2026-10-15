# Execution ledger

2026-10-15: admitted the user's next sequence on a new experiment branch after
checking the clean repository and accepted previous tranche. Counted actual
retained checkpoint tensors: current 370,016 float32 parameters, 1,480,064 raw
bytes, 1,485,452-byte NPZ; original successful 32-cubed modal checkpoint 32,767
float64 learned gains plus a 1,024-element fixed basis, 272,149-byte PT file.
Earlier ten-channel neural inventory is 372,377 parameters. Repeated inference
shares one parameter set. File format/dtype and fixed-grid specialization differ.

The existing engine/model limit is 65,536 cells. Bound the cube scaling study at
40-cubed (64,000), preserving the contract rather than admitting 48/64-cubed.
Follow-ups depend on measured fitting and stability; no production speedup or
arbitrary-grid accuracy is presumed.

Captured both unchanged single-pass configurations at 8c6ddc7. First normal
managed full sequence admitted training Run 406b606cd7a8431db744e86b2631f28e.
Implemented optional bounded stopping using original reconstructed RHS and
predicted divergence, never labels. Five independent two-cell tests pass; two
real process cleanup tests pass using system Python (Pumas path spaces make its
shebang unsuitable for those disposable script fixtures). Historical fixed
correction path remains covered. New scaling driver compares serial same-core
whole-track timings, three repetitions and an excluded neural warm-up step.
Native/AI retain their distinct existing 1e-8/1e-6 equation profiles.

Adaptive CPU development check: all 128 states pass the unchanged float32
physical profile, averaging 14.96875 neural evaluations (8–28), 53.22% fewer
than fixed 32. Median relative residual 2.33592e-7, worst 9.95478e-7; none
meet the stricter original native 1e-8 profile at this stopping point. All
147 AI steps pass across three independent 64-step trajectories after native
warm-up. Cube/rectangular use 12 evaluations; compact averages 15.8367.
Distinct-core execution overlaps managed work, so no speedup claim from those
non-isolated timings. Three altered real receipts (relaxed target, false cap,
false actual work) reject before native execution. CPU source ad89d11.

Replan: combined single-pass Run 2500fc7690c14049876250d8c3f5a981 failed
and drained. Custody-verified progress reports 2,400 completed updates, loss
77,382.87, then gradient-norm non-finite rejection; no artifacts were published.
Exact terminal update is unavailable, so retain a lower bound rather than
counting 5,000. No CPU restore/inference is claimed for that failed fit. Keep
the original capture/Run and add one distinct lr1e4 trial: all learning rates
tenfold lower, other settings/exposure fixed. This is a controlled optimization
follow-up, not an automatic replay. First single-pass fit completed normally.
Adaptive extended windows qualify all 854/854 AI steps; source receipts are
authoritative (the session extended wrapper initially printed short-run summaries
after executing correct extended commands; corrected its display path, no rerun).

Completed the first single-pass managed sequence and full CPU qualification.
Known frame residual .00737933; development median .0164437, worst .106447;
0/128 physical passes. All three 64-step trajectories finish, with 0/147 AI
steps qualified. Independent mode mixtures all reduce first-pass error (72/72),
but worst after eight one-network proposals is 6.36510e-5, above the target.
Initial parameters, real-frame order, auxiliary bank/order and normalization
match the four-pass control exactly. No capacity/generalization success is
inferred from completed optimization. Original CPU controller stopped upon the
retained failed second fit; a separately owned follow-up controller waits only
for the new lr1e4 trial, then qualifies it before serial scaling.

Lower-rate combined single-pass fitting completed 5,000 CUDA updates at
47.3483 updates/s and all managed CPU restore/validation artifacts reopened
and redownloaded identically. Native CPU fit residual .00781890, development
median .0258949, worst .107453, physical passes 0/128. Three short trajectories
complete with 0/147 qualified AI steps. Initialization, real/analytical order
and train-only normalization match the four-pass combined control; one pass
and lower learning rate are both changed, so no single-factor attribution.
Six managed stages completed and drained; one failed/drained training Run
is separately retained. Isolated serial scaling began only after all those
stages and CPU workers finished. New retention remains below 2 GiB.
