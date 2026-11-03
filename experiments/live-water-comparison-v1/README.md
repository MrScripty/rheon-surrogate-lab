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

## Package and verify the current adapter

The active process reads only `rheon-pressure-adapter-v1`. The explicitly invoked
one-time converter archives the original v2 bundle and emits a fresh descriptor
with unchanged weights, normalization and numerical sources. It is separate from
normal model startup. The aa82 seven-channel model and 26b residual comparator
remain separate packages; neither is promoted by migration.

```bash
python3 -B experiments/live-water-comparison-v1/deployment.py aa82 \
  --bundle runs/live-ai-comparison/deployment-2 \
  --runtime runs/dependency-migration-v1/runtime/current-manifest-v2.json \
  --runtime-owner runs/dependency-migration-v1/Tuldok/torch_runtime.py \
  --output runs/dependency-migration-v1/packages/aa82-NEW \
  --archive runs/dependency-migration-v1/archive/models/aa82-NEW
```

The complete conversion contract and the residual control command are documented
in [model_adapter.md](../dependency-migration-v1/model_adapter.md). The package
uses the current measured managed Torch manifest, checks all inventoried files
before importing Torch, and returns full float64 pressure. Neural computation
remains CPU float32, with no optimizer or CUDA initialization. Prior pressure is
already included by the lab numerical owner and must not be added by the host.

The independent Q4 tool converts the exact historical capture to a current saved
request and retains the original numerical witnesses. Verify a current request
against the independently retained little-endian float64 pressure bytes:

```bash
python3 -B experiments/live-water-comparison-v1/verify_deployment.py \
  --deployment runs/dependency-migration-v1/packages/aa82-NEW/manifest.json \
  --request PATH/TO/current-request.json \
  --expected PATH/TO/expected-pressure.f64le \
  --output runs/dependency-migration-v1/receipts/aa82-saved-input-NEW.json
```

This command checks exactly one saved pressure result. Q4 separately owns exact
preprocessing, raw neural output, reconstruction, zero bypass and independent
operator checks. Run the boundary tests after the fixed saved-input gate passes:

```bash
python3 -B -m unittest discover -s experiments/dependency-migration-v1 -p test_model_adapter.py -v
```

## Open fresh simulations

From the isolated Rheon checkout, build `rheon-ai-compare` with `--features desktop`.
The regular desktop also has **Compare Rheon / trained AI**. Pass a deployment,
not model hashes compiled into application code.

```bash
runs/live-integration/native-target/debug/rheon-ai-compare \
  --deployment "$PWD/runs/dependency-migration-v1/packages/aa82-runtime-current/manifest.json" \
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
