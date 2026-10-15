# Execution ledger

2026-10-15: inspected clean HEAD 5bf5c2d and accepted previous tranche. Admitted
the recorded follow-ups on experiment/water-multigrid-learning-v1. Historical
results remain immutable. Two changes are controlled separately: per-evaluation
CPU stopping for both existing checkpoints, and multi-grid analytical training.
The broader cohort and architecture changes remain conditional on measured fit.

Implemented optional per-network-evaluation stopping in the existing corrector,
trajectory adapter and native receipt consumer. Captured recurrence owner is
reused without changing model configuration. Seven focused tests pass, including
an actual four-evaluation recurrence versus four separate evaluations: identical
pressure bits, four observed forwards in each, original metadata unchanged.
Existing two-cell RHS/divergence and budget checks remain passing. This is an
inference-policy measurement change, separate from training-population changes.
