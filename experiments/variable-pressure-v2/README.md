# Seven-channel cubic pressure model

This version removes all three voxel-size network channels. Actual input is
`[N,7,Z,Y,X]`: native pressure residual, previous pressure, physical cell-centre
X/Y/Z coordinates, density and time step. There is no replacement spacing
channel or hidden broadcast spacing conditioning. `cell_size_m` is one positive
float64 scalar per sample. Preprocessing uses it for physical coordinates;
the pressure operator uses it as three equal scalar edge lengths. These are
physical calculations, not additional network input maps.

The four-level non-affine GroupNorm architecture retains variable positive
shapes, odd dimensions and singleton axes within 65,536 cells. Its first
convolution accepts seven channels; the model has 371,729 parameters. Seeded
initialization retains the original seven surviving first-layer columns and
every other original initialized tensor. It does not import trained weights.
The previous ten-channel model, captured recipes, weights and results remain
unchanged in variable-pressure-v1. Its checkpoints cannot be restored into v2.

The versioned namespace deliberately snapshots the bounded managed execution
contract, avoiding changes to frozen historical readers and source bindings.
It retains the original pressure/residual loss, deterministic CPU optimizer,
train-only normalization, sampling, exact typed-artifact restore and native
quality/fallback gates. Normalization contains no hx/hy/hz scales. Packets store
one cell-size scalar per parent. Native protocols and the native probe remain
unchanged: three equal scalar lengths are supplied only at that existing seam.

Use a fresh output path and the qualified prepared Rheon checkout. To compare
this input change using the exact same development fields, reverify existing
whole-parent captures instead of generating new simulations:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/variable-pressure-v2/pipeline.py \
  --prepared-rheon /absolute/path/to/qualified/prepared-rheon \
  --cargo /absolute/path/to/rust-1.92.0/bin/cargo \
  --captures /absolute/path/to/density-8/capture /absolute/path/to/density-16/capture \
  --output "$PWD/runs/new-seven-channel-material"
```

The reusable configuration selects the ordered capture profiles. Reverification
checks each original physical manifest and all saved fields, then recomputes
native residuals against the independent MAC oracle. Without `--captures`, the
pipeline generates and qualifies one-frame fixtures before full histories,
using the existing cubic capture producer. No capture or prior packet is edited.

After committing source, `project.assemble(packet, runtime_manifest,
tuldok_source, output)` freezes qualified material into ordinary train/restore
recipes. `workflow.execute(client, project, output, observer)` uses the existing
catalog and captured CPU runtime. `evaluate.run(...)` diagnoses restored
first/final-frame predictions through the unchanged native gates. Progress and
actual diagnostic fields publish to the existing Tuldok/Rheon observers.

The development comparison reuses previously observed train/validation parents;
it is not a new held-out test or an accuracy guarantee. Training remains 200
updates with no GPU, checkpoint search or autonomous rollout. New measurements
are recorded in docs/plans/seven-channel-pressure-model/ after execution.
