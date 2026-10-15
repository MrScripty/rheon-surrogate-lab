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

Added closed schema-v4 feedback with explicit grid counts, one cubic spacing per
grid, balanced private-RNG sampling and an exact 768 MiB cache projection.
Legacy analytical bank dispatch retains tensors/order/evidence. Three new
contract/eigenvalue tests, five historical analytical tests and six residual
contract tests pass. Scope explicitly includes the new test_multigrid.py and
training configuration files. The 10-update smoke covers every grid twice.

The complete authored bank qualifies on CPU: 640 entries across five grids,
worst analytical-eigenvalue versus implemented-operator relative mismatch
6.90286e-14. The original captured bank source independently reproduces every
legacy tensor, sample order and evidence byte; model RNG stays unchanged.
Resident tensor budget and the exact per-grid exposures are retained in
reports/bank-qualification.json. No optimizer or CUDA context in this check.

Controller environment correction: the Pumas interpreter lacks psycopg; the
Tuldok interpreter lacks NumPy for the observer. Both failed before Run
submission. The repository .venv has the established controller dependencies;
managed training still selects the qualified Pumas CUDA runtime. Both startup
logs remain under runs/. No extra model attempts are counted for these errors.

Added reusable configured CPU study driver. Every checkpoint episode uses a
fresh Python process, exact captured sources and a distinct output/log. Fixed
worker/core partitions prevent contention between candidate jobs. Concurrent
checks are numerical/throughput evidence; repeated timings must run separately.
