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
