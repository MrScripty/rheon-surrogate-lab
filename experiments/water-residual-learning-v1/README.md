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


`build_rollout.py --prepared PREPARED_RHEON --rheon-source RHEON_CHECKOUT
--cargo CARGO --output FRESH` builds/tests/lints the separate
`probes/residual-rollout-v1` against the qualified prepared native library and
selected existing paired-engine/scene source. It retains source/dependency/build
commands and binary hashes; it does not edit the Rheon checkout or application.
`rollout.py --binary PROBE --scene SCENE_JSON --output FRESH --project PROJECT
--artifacts CPU_RESTORE` serves that captured checkpoint to independent native/AI
tracks after the scene's native warm-up. `--reference-control` substitutes a
native solve for the prediction callback and needs no checkpoint. The parent
owns the native process group, a 120-second deadline, termination/reaping and
failure receipts. Existing hard physical stops remain enforced; failed pressure
qualification stays visible and is not fallback success. These runs are new
simulation trajectories, unlike the recorded-state audits. The deployed UI
bridge's seven-channel contract stays unchanged. Run `test_rollout.py` for actual
failed-exit and timeout process cleanup evidence.


`summarize.py` builds `reports/summary.json` and `reports/comparison.csv` from the
retained per-trial records. It indexes their hashes and Run custody, training
UTC intervals, update rates, fitted/held-out errors, consistency/repetition
checks and actual post-warm-up trajectory outcomes. This derived comparison can
be regenerated; per-trial records are immutable. Older v1 reports remain valid;
v2 records add explicit training timestamps and runtime snapshot identity.


`modes.py` takes the same project/artifacts/packet/output arguments as the audit.
It probes 24 manufactured compatible Neumann modes on the three development
grids, matching pressure RMS to the fitted correction. It measures whether one
proposal reduces equation error and whether repeated proposals amplify it.
This isolates unconstrained spatial modes; it is not a simulation, training run,
cold-start model acceptance or physical qualification.
