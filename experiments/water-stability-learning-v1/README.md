# Stability experiment diagnostics

The [active plan](../../docs/plans/water-stability-learning-v1/plan.md) and its
[method commands](../../docs/plans/water-stability-learning-v1/reports/commands.md)
own the comparison and results. Managed training remains in the existing
water-conditioning trainer; native evaluation remains in the existing residual
diagnostic owners. No additional optimizer, catalog or deployment adapter.

`mixed_modes.py --project CAPTURE --artifacts CPU_RESTORE --packet FIT_VIEW
--output FRESH [--seed INTEGER]` tests 72 signed mixtures of analytical Neumann
modes on three development grids. Its seed differs from training. All cases
assert the exact analytical solution against the existing finite-volume operator,
then compare one/eight learned proposals and representable float32 output.
The program uses the retained CPU checkpoint loader, verifies source/checkpoint
identity, records numerical results and source hashes, and distributes independent
work over available cores with one Torch thread per process. It constructs no
optimizer or CUDA context. A sampled pass does not establish global contraction;
these artificial operator inputs are not additional fluid simulations.

Failed numerical or process checks propagate; the output directory must be fresh.
Keep per-trial summaries in the plan's reports and numeric artifacts in ignored
runs/water-stability-learning-v1 under existing Tuldok custody/retention rules.
