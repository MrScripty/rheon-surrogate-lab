# Variable-shape pressure correction with cubic cells

This new model accepts positive X/Y/Z grid counts, including odd, rectangular
and singleton dimensions. The eager CPU profile admits at most 65,536 cells
per sample and batches of 1–4. It has no 16³/32³ shape whitelist. Capture profiles
retain their separately declared physical/count and payload budgets.

Each physical input has **one shared positive cell side length** on X/Y/Z.
`model.predict_pressure(network, sample, normalization)` rejects unequal spacing
before inference. The operator and training loss enforce the same condition.
The result is f64 pressure, one value per original cell, with cell-zero gauge;
an exactly zero residual returns the original pressure without invoking the net.

The non-affine GroupNorm blocks and 372,377 parameters are retained. Pooling uses
per-axis kernel/stride `min(2, count)`, ceiling and excluded padding; decoder
features return to each skip shape. This pools hidden features while preserving
the physical input/output grids. Physical cell centres in metres replace the
old unit-box coordinate features. The original 16³ hidden-feature forward and
seeded parameter initialization remain exactly equal in regression checks.

Training packets preserve separate arrays for each whole trajectory. A batch
can contain different shapes: each frame contributes its loss divided by the
batch size before one clipped Adam update. No physical field is stretched,
padded, resampled, converted to an object array or flattened into another shape.
Normalization reads training parents only; validation parents stay held out.

[The reusable configuration](configs/development.json) selects two densities,
8 and 16 cells/metre, four training parents and two validation parents, 24 frames
each, with cubic and odd/rectangular grids. Its **top-level training policy** is
the executed `cubic-groupnorm-v2` policy. Nested capture configurations retain
the frozen producer's unexecuted legacy training declaration solely to satisfy
its existing capture contract; they never launch the old model.

Generate and independently qualify a one-frame fixture before every full capture,
then build the per-grid residual packets:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/variable-pressure-v1/pipeline.py \
  --prepared-rheon /absolute/path/to/qualified/prepared-rheon \
  --output "$PWD/runs/new-variable-pressure-material"
```

`project.assemble(packet, runtime_manifest, tuldok_source, output)` freezes
committed sources and qualified packets into normal Tuldok train/restore recipes.
`workflow.execute(authenticated_loopback_client, project, output, observer)` uses
the user's existing catalog and worker, exact material preflights and owner grants,
typed artifact restore inputs and complete SHA-verified artifact downloads.
No optimizer runs during preparation. Source/data/runtime hashes are measured,
not substituted historical constants. Recipe source paths are relative to the
actual repo project root; interpreter and native dependency paths are host inputs.

`evaluate.run(packet, native_binary, training_artifacts, restored_artifacts,
output, observer)` tests actual restored predictions on each first/final parent
frame through Rheon's native solver. It retains the original physical accuracy
gates, checks the candidate's operator residual independently, and records raw
acceptance, correction/fallback work and baseline iterations for each grid.

Capture and managed Run progress appear in the existing Tuldok project; actual
reference fields and labelled pressure proposals appear in Rheon's external egui
viewer. These are teacher-state snapshots, not autonomous learned rollouts.
Shape support does not guarantee accurate learned pressure at every resolution.
The native diagnostic reports that accuracy separately. Historical models,
weights, reservations and reported results remain unchanged.
