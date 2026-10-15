# Fresh native / AI water comparison

This integrates the frozen water checkpoint with Rheon's native desktop. It does
not train or select a new model. Training remains CUDA; comparison inference uses
CPU float32 weights, no optimizer, and the measured installed runtime.

The desktop runs 15 qualified native steps first. Its first AI call is step 16,
using the solved pressure from step 15 and the new pressure-equation residual.
The minimum is exported from the training reservation, not a replacement historical
hash. The old captured training snapshots have zero-based indices 15 and 63.
A cold-start request is `unsupported_context`; it must not count as a model failure.
A warm-start prediction can still fail the unchanged pressure/physics checks.

After the prefix, the two tracks use the same scene, geometry, force and timestep,
but evolve separate velocity, tracer and previous-pressure histories. AI pressure
failures stay visible as raw diagnostics; hard bounds stop the paired run. No
native fallback is substituted into the AI track after handoff.

## Export and verify a deployment

Use a fresh output directory. Paths below describe the completed local baseline;
all exporter inputs are arguments. Historical source, artifacts and deployments
remain immutable.

```bash
.venv/bin/python experiments/live-water-comparison-v1/deployment.py \
  --project runs/sealed-water-v1-3/project \
  --artifacts runs/sealed-water-v1-3/managed/stage-00/artifacts \
  --run runs/sealed-water-v1-3/managed/stage-00/run.json \
  --runtime runs/cuda-pressure-runtime-1/manifest.json \
  --output runs/live-ai-comparison/deployment-NEW
```

The exporter checks completed/drained training custody, whole artifact hashes,
source witness, final declared checkpoint and registered runtime identity. It
copies measured source, normalization, weights, runtime inventory and bridge into
one deployment. The bridge measures installed runtime files at startup, checks the
CPU device and float32 weights, and enforces shape, cubic spacing, water material,
gauge, finite fields, bounded records and the native-prefix policy.

```bash
.venv/bin/python experiments/live-water-comparison-v1/verify_deployment.py \
  --deployment runs/live-ai-comparison/deployment-NEW/deployment.json \
  --packet runs/sealed-water-v1-3/packets/train_validation \
  --inference runs/sealed-water-v1-3/managed/stage-01/artifacts \
  --output runs/live-ai-comparison/deployment-verification-NEW.json
```

Verification requires byte-exact prediction parity with the existing CPU witness,
exact zero-residual bypass, no CUDA initialization or optimizer updates, and a
cold-start rejection before inference. Boundary regressions:

```bash
.venv/bin/python -m unittest discover -s experiments/live-water-comparison-v1 -p 'test_*.py'
```

## Open fresh simulations

From the isolated Rheon checkout, build `rheon-ai-compare` with `--features desktop`.
The regular desktop also has **Compare Rheon / trained AI**. Pass a deployment,
not model hashes compiled into application code.

```bash
runs/live-integration/native-target/debug/rheon-ai-compare \
  --deployment "$PWD/runs/live-ai-comparison/deployment-2/deployment.json" \
  --project-dir "$PWD/runs/live-ai-comparison/project" --auto-run
```

The controls change seed, cell counts, cells per metre and localized force.
Cell size is a scalar; physical extents are counts multiplied by that size.
The synchronized panes show reference pressure and AI pressure or signed
`AI − Rheon` pressure in Pa. Blue is lower, white zero, red higher. Both panes share
camera, timestamp and Z cut. Click a voxel for exact paired values. Current-frame
full-volume scales are the default; whole-run and fixed scales are optional.
Shared native steps show **AI not evaluated** and never inflate model passes.

Use `--verify-run --steps 24` for a bounded non-GUI run: 15 native steps, 9 AI
predictions. `--steps` must leave an AI step after warm-up. A project-directory
lease prevents overlapping publications. Cancel/close drains the owned child and
readers; retained frames survive. Scene/model/frame/completion JSON is preserved
per run, while `latest.json` and `activity.json` update the existing Tuldok project.
These observations are not managed training Run custody.

## Tuldok collection export

Export a completed retained run through the existing scientific-field format:

```bash
.venv/bin/python experiments/live-water-comparison-v1/export_fields.py \
  --run runs/live-ai-comparison/project/RUN-DIRECTORY \
  --tuldok runs/live-integration/Tuldok \
  --output runs/live-ai-comparison/fields-NEW
```

This validates and retains all frames, reference pressure, raw AI pressure, signed
error, both passive tracers, scene/checkpoint provenance and per-step verdicts.
It creates `project.json` and `comparison.zip`. Open that project in the existing
Tuldok workspace and import its data reference into the normal collection.
Importing each future interactive restart is explicit; it is not automatic.
The first verified warm-start record is `899f4cc68ea64a91a4abc4ad9ac31b7e`.
It contains 65 frames, including the shared native prefix, and is a diagnostic draft.

See [measured acceptance](../../docs/plans/live-ai-pressure-comparison/reports/acceptance.json)
and [results](../../docs/plans/live-ai-pressure-comparison/results.md). The cold-start
records are preserved and excluded from model-accuracy claims. The new warm-start
run remains inaccurate; no threshold or trained weights were changed.
