# Variable domains with cubic physical cells

This is the current physical-grid capture setup. The default is **16 cells per
metre on every axis**, so each cell measures 0.0625 × 0.0625 × 0.0625 m.
The domain may be rectangular; its cells remain cubic.

| Domain in metres | Cell counts X × Y × Z | Cell size |
| --- | --- | --- |
| 1 × 1 × 1 | 16 × 16 × 16 | 0.0625 m |
| 1.5 × 1 × 0.75 | 24 × 16 × 12 | 0.0625 m |
| 2 × 2 × 2 | 32 × 32 × 32 | 0.0625 m |
| 0.5 × 0.5 × 0.5 | 8 × 8 × 8 | 0.0625 m |

[Configuration](configs/development.json) owns `cells_per_metre` and physical
lengths. Counts are derived as `length × cells_per_metre`; equivalently, each
length is `count / cells_per_metre`. A length requiring fractional cells fails
with an error. No rounding of the domain, stretching, padding or resampling is
performed. Native capture currently admits 2–128 cells per axis, at most 65,536
cells, with the configured wall and payload budgets.

Build against an independently qualified prepared Rheon source directory:

```bash
.venv/bin/python experiments/cubic-pressure-v1/pipeline.py describe
.venv/bin/python experiments/cubic-pressure-v1/pipeline.py build \
  --prepared-rheon /absolute/path/to/qualified/prepared-rheon \
  --output /absolute/path/to/new/build
.venv/bin/python experiments/cubic-pressure-v1/pipeline.py generate \
  --binary /absolute/path/to/new/build/rheon-cubic-pressure-capture \
  --output /absolute/path/to/new/capture --live-root "$PWD/runs/live-project"
```

Use `--config` with another complete configuration to reuse the setup. A binary
is bound to its selected configuration and measured sources; rebuild after
changing them. Every generation qualifies a one-frame fixture first, then the
selected whole development trajectories. The retained independent MAC oracle
checks RHS, pressure residual, velocity correction, divergence, walls, flux and
Courant limits using the derived counts and the shared physical spacing.

The live publication feeds the existing Tuldok project and Rheon's egui external
observer. Export a complete verified parent through the normal Tuldok field
bundle parser with `experiments/live-project-v1/bundle.py --capture-profile cubic`.

**Training limitation:** the unchanged GroupNorm model accepts only exact 16³
and 32³ grids. `pipeline.py model-preflight` rejects this profile's 24 × 16 × 12
grids before training. Rectangular-grid model support remains a separate model
change; this setup does not train a model or claim learned pressure success.
Historical fixed-count captures, model weights and results are preserved.
