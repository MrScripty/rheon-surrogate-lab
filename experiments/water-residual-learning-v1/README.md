# Residual learning diagnostics and retention

The [active plan](../../docs/plans/water-residual-learning-v1/plan.md) owns the
sequence. Training/configuration and managed recipes remain in
`experiments/water-conditioning-v1/training` and `configs/residual-*.json`.

`audit.py --project PROJECT --artifacts CPU_RESTORE --packet VIEW --output FRESH`
restores the exact captured architecture and checkpoint on CPU, one thread per
worker and distinct available cores. Add `--training-only` for the fitted state.
It measures residual, pressure/gradient error, sign/scaling/superposition,
eight repeated proposals and separately labelled analytic low-frequency/full
cosine controls. These operate on recorded states, not autonomous simulations.
The spectral control is checked against an independently specified rectangular
Neumann cosine eigenfunction; it is not learned inference or native qualification.

`record.py --trial TRIAL` retains a small immutable report under the plan's
`reports/`: source/config/input lineage, Run/catalog IDs, CUDA and CPU evidence,
checkpoint hashes, training trace, timings and available native/audit results.
Call after the required evaluations complete. It refuses changed evidence under
an existing record ID. Do not put weights/datasets in Git. Keep their managed
artifacts and verified local downloads; the record identifies both custody and
local paths. Failed Runs remain in the catalog with their local failure receipts.

Use the installed qualified Torch interpreter for audits and tests. `record.py`
needs only standard Python. Run spectral regression with `python -m unittest
discover -s experiments/water-residual-learning-v1 -p test_audit.py`.
