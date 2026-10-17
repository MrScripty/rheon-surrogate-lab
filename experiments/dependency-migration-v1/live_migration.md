# Migration live observations

`live_migration.py` reads completed receipts from the explicitly selected current
cohort, plus bounded GPU qualification and named operations receipts. It uses the
existing Observer to publish ordinary schema-1 `activity.json` and `frame.json`.
These are external observations. They establish no Tuldok Run, execution custody,
migration acceptance or independent scientific acceptance.

Publish to `runs/dependency-migration-v1/live-migration`. The historical
`runs/live-project` files are migration source evidence and must remain unchanged.
The publisher rejects that destination. The current read-only viewer can select
the new `frame.json` and sibling `activity.json`; a new current Project can refer
to that activity through its ordinary evidence reference.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B \
  runs/dependency-migration-v1/lab/experiments/dependency-migration-v1/live_migration.py \
  --root "$PWD" --interval 2
```

`--once` publishes one observation and exits. `--cohort` selects a directory
inside `runs/dependency-migration-v1/numerical-qualification-v1`; the default is
`new-cohort-01`. `--live-root` must name the dedicated migration observation root.
One advisory lock prevents competing publishers. SIGINT/SIGTERM stops the loop
and publishes `idle`; it never signals, starts or changes a scientific worker.

Progress counts passed completed scene receipts out of the declared 74-scene
already-consumed cohort. Failed and unreadable counts remain visible. GPU
lifecycle and schema rehearsal statuses report their own receipts separately.
Even 74/74 is external progress; the publisher does not infer full migration
acceptance. Named operations status reads exclude private selections, catalogs,
credentials and source inventories. No database, HTTP service, GPU, solver,
trainer, model or producer module is opened.

The latest completed passing scene receipt selects its latest sparse capture by
generation. The publisher requires a completed reaped native process, matches
the NPZ SHA-256 in the scene receipt, checks the scene's actual XYZ counts, and
reads only that capture. Prediction is the captured full float64 pressure in Pa,
with x-fast ZYX storage. Actual captured h, density and dt define SI geometry and
physical time (`generation * dt`). The original receipt and capture identities,
dataset label, scene, seed and field name travel with the preview. This is a
field preview from a completed case, rather than a new accepted result.

Current sparse captures provide pressure, base, residual, h, rho and dt. They do
not provide tracer. The preview omits tracer and the viewer must state that it
was not supplied. It does not create zeros, a static sphere, synthetic fluid or
an invented force field. Current diagnostic scenes expose their exact declared forcing parameters under
`source.forcing_parameters`, with kind `rheon-diagnostic-scene-v1`, raw source
scene SHA-256, units and explicit nominal/base scope. The projected keys are seed,
center, acceleration, radius, swirl, fluctuation, frequency, counts, cells per metre,
steps, warm-up steps and SGS. Center is normalized, radius scales the minimum
physical domain extent, acceleration is m/s², frequency is Hz, and swirl and
fluctuation are dimensionless. The source scene hash is checked before and after
projection; a change prevents publication. The authoritative force kernel remains
`Rheon/src/diagnostic/engine.rs::force`; the publisher never evaluates it or
invents a force vector. Existing canonical regime/amplitude metadata is emitted
only if its exact named fields are present. A missing canonical regime does not
hide the current scene's recorded nominal parameters.

The loop holds at most one latest frame, releasing its previous in-memory
preview before opening another capture. It retains only small receipt metadata
and scalar summaries between polls. The limits are 128 scene receipts, 16 sparse
captures per selected case, 64 lifecycle receipts, 2 MiB per JSON read/publication,
4 MiB compressed NPZ, 8 MiB expanded archive and 65,536 cells. Headers are checked
before array loading to reject object arrays, oversized shapes, truncated payloads,
Fortran order and unexpected floating storage. Poll interval is 1–60 seconds,
default two. Deployment uses one numerical-library thread and low process
priority; no simulation or optimizer executes.

An invalid new frame leaves the previous atomic frame file untouched and adds
an explicit unavailable-preview message to activity. The retained source hash
continues to identify the older actual frame. A receipt refresh failure publishes
failed observation status; it does not rewrite failed previews as accepted
results. Successful field preview validation does not validate scientific
accuracy, migration success or any other case.

Run focused checks with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover \
  -s runs/dependency-migration-v1/lab/experiments/dependency-migration-v1 \
  -p test_live_migration.py -v
```

The fixtures verify exact pressure/geometry/time/provenance, truthful tracer and
forcing absence, changed hash/path/failed/unreaped case refusal, header allocation
bomb rejection before NumPy loading, one-frame reads and reuse, previous-frame
retention on failure, private-file exclusion, historical-root refusal, exact current-scene nominal forcing
metadata/units and unchanged source hashes, and rejection of a scene that changes
during projection. Their
small explicit arrays are contract fixtures; no simulation is executed.
