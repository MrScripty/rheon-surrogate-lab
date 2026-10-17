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

`--once` publishes one observation, prints one JSON summary and exits. A summary
transport failure exits nonzero without relabelling a valid durable observation.
Continuous watch writes no per-poll stdout: atomic activity/frame publications
provide its bounded current status and heartbeat, so a regular-file log cannot
grow indefinitely or turn a full log into a receipt-refresh failure.
`--cohort` selects a directory
inside `runs/dependency-migration-v1/numerical-qualification-v1`; the default is
`new-cohort-01`. `--live-root` must name the dedicated migration observation root.
One advisory lock prevents competing publishers. SIGINT/SIGTERM stops the loop
and publishes `idle`; it never signals, starts or changes a scientific worker.

Progress counts passed completed scene receipts out of the declared 74-scene
already-consumed cohort. Failed and unreadable counts remain visible. GPU
lifecycle and current catalog rehearsal statuses report their own receipts separately.
Even 74/74 is external progress; the publisher does not infer full migration
acceptance. Named operations status reads exclude private selections, catalogs,
credentials and source inventories. No database, HTTP service, GPU, solver,
trainer, model or producer module is opened.

Catalog progress comes from the exact public `ops/rehearsal-status.json`, with
its current stage, corrected `prepare04`, publication, cold-read, completed-current
roundtrip and browser fields projected separately. The first schema-only rehearsal
and superseded prepare03 passes never substitute for an absent or pending current
catalog field. A missing current file is `absent`; a malformed, oversized, symlinked
or wrong-format file is `unreadable`. Neither state retains a previous pass.

M3 observation selects only the named preflight, quiesce, fresh-bookend, schema-upgrade,
domain-prepare/publish/reconcile, verify-current and app-launch receipts, plus
`ops/m3-cutover.json`. Service-switched, M3-complete and ordinary-admission booleans
are displayed as recorded, without inferring one from another. The fresh bookend
uses `rheon-fresh-stopped-original-bookend-v1`; the other named steps use
`rheon-m3-orchestrator-v1`. Final completion is a separate `ops/m3-complete.json`
with schema `rheon-tuldok-migration-completion-v1`. Its explicit `status`,
`m3_service_switched`, `m3_complete` and `ordinary_admissions_open` fields are
projected independently. Missing fields are `not_recorded`; they are never inferred
from another flag or a successful smoke. The immutable consumed cutover authorization
continues to show its original `m3_complete:false` even if the separate final
completion later reports true. The smoke has
separate observations for `ops/typed-pressure-smoke-launcher.json`, the fourteen
exact numbered phase journals under `typed-pressure-live-01/mutations`, and
`typed-pressure-live-01/{receipt,failure}.json`. Only whitelisted bounded status
scalars and receipt hashes are retained; commands, arguments, results, private
credentials and arbitrary sibling files are excluded. The actual live receipt's
`machine_qualified_browser_pending` and `scientific_acceptance=not_asserted` remain
visible. Acknowledged mutation journals report dispatch progress, not Run drainage
or final acceptance. Status refreshes reuse the same captured pressure/frame source.

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
captures per selected case, 64 lifecycle receipts, 30 exact operational status
paths, 160 characters per projected status scalar, 2 MiB per JSON read/publication,
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
Status fixtures cover stale prior passes, absent/malformed/current catalog progress,
M3 and smoke phase separation, unchanged frame reuse, private/unknown-file exclusion,
and oversized/symlinked status refusal. Publisher restart remains a separate root
operation; these source changes do not restart the existing owned publisher.
