# Varied sealed-water pressure experiment

[Measured execution, held-out accuracy and previous-run comparison](../../docs/plans/sealed-water-pressure-v1/results.md) record the completed baseline: 0/128 pressure proposals qualified.

This experiment trains the qualified seven-channel pressure-correction CNN on
256 distinct training simulations, with 64 validation and 64 reserved test
simulations. See [configuration](configs/water.json) and the
[execution plan](../../docs/plans/sealed-water-pressure-v1/plan.md).

Every domain is completely filled with water: density 1000 kg/m³, dynamic
viscosity 0.001 Pa·s (engineering approximations). Boundaries are stationary,
impermeable and free-slip. There is no mass emitter, outlet or free surface.
Rheon performs advection, localized acceleration, viscosity and reference pressure
projection. All 64 timesteps must pass the native reference gates. Two snapshots
at native indices 15 and 63 also pass independent full-grid MAC checks.

Each scene varies axis counts (5–16), physical cell spacing (8 or 16 cells/m),
force center, direction, strength, extent, rotation, temporal frequency, center
motion and spatial fluctuation. All voxels are cubic: length on each axis is
its cell count divided by cells/m. The compact Gaussian force tapers smoothly to
zero at four radii. This is controlled fluctuating forcing, with no claim that
these small grids resolve real-world turbulence. Tracer is passive visualization.
Different seeds alone are not counted as different physical setups.

The network architecture, float32 parameters and loss remain unchanged from
[cuda-pressure-v1](../cuda-pressure-v1/README.md). Inputs remain residual,
previous pressure, three physical coordinate channels, density and timestep;
scalar voxel spacing stays physical metadata. Normalization is fitted to training
parents only; the existing constant-unit density policy is preserved. This is an
identified baseline, with no normalization or architecture comparison mixed in.
Training samples whole scenes and their actual retained frames. Training and
validation have 512 and 128 snapshots; all 128 reserved test snapshots are used
only after the predeclared final checkpoint is frozen.

Training uses CUDA Adam for 2000 updates, batch 2, with learning rate 0.001 then
0.0001 from update 1000. Checkpoints are 0, 100, 500, 1000 and 2000. A fresh CPU
Run restores all checkpoint evidence and checks exact CPU prediction parity; a
third CPU Run evaluates the independent reserved population without optimization.
The current relative residual target remains 1e-6 and physical gates are retained.
Execution success does not establish pressure accuracy or autonomous stability.

## Reusing the workflow

Choose a fresh ignored run directory. Change configuration values within the
explicit qualified bounds; historical source or result hashes are not replaced.
The native build embeds the entire reproducible reservation and measured source
closure. Use the qualified prepared Rheon dependency and Rust/Cargo 1.92.0:

```bash
python experiments/sealed-water-v1/pipeline.py build \
  --config experiments/sealed-water-v1/configs/water.json \
  --prepared "$PREPARED_RHEON" --cargo "$CARGO_192" --output "$WATER_RUN/native-build"
python experiments/sealed-water-v1/pipeline.py generate \
  --config experiments/sealed-water-v1/configs/water.json \
  --binary "$WATER_RUN/native-build/rheon-sealed-water-capture" \
  --output "$WATER_RUN/fixture" --fixture
python experiments/sealed-water-v1/pipeline.py generate \
  --config experiments/sealed-water-v1/configs/water.json \
  --binary "$WATER_RUN/native-build/rheon-sealed-water-capture" \
  --output "$WATER_RUN/pilot" --qualified-fixture "$WATER_RUN/fixture" --pilot-parents 4
python experiments/sealed-water-v1/pipeline.py generate \
  --config experiments/sealed-water-v1/configs/water.json \
  --binary "$WATER_RUN/native-build/rheon-sealed-water-capture" \
  --output "$WATER_RUN/capture" --qualified-fixture "$WATER_RUN/fixture"
python experiments/sealed-water-v1/pipeline.py pack \
  --capture "$WATER_RUN/capture" --output "$WATER_RUN/packets"
```

Use a NumPy/Torch-capable Python. Derive a separate training JSON from the dataset
configuration's `training` object. Commit qualified source before assembly, then
run `pipeline.py assemble --packet .../packets/train_validation --test-packet
.../packets/reserved_test --runtime-manifest ... --tuldok ... --output ... --config
.../training.json` using the selected Pumas Torch Python. Assembly captures sources,
physical contracts, complete input hashes and optimizer-free grid qualification.

`workflow.execute(client, project, output, observer)` uses the existing normal
Tuldok authenticated HTTP client. The catalog owns measured runtime resolution,
material preflight, exact grants, all three Runs, typed checkpoint custody,
artifacts and teardown. It reopens the completed Runs and redownloads every output.
No separate catalog or local substitute optimizer is used. `pipeline.py evaluate`
then checks reserved CPU predictions with the independently qualified water
native pressure diagnostics binary. Build it with `pipeline.py build-evaluation
--prepared ... --cargo ... --output ...`; its pressure arithmetic and gates are
the frozen variable-pressure-v1 diagnostic, with a separate water-density admission
contract. The historical binary still admits only its original 1.25 kg/m³ material.

`pipeline.py export-fields` creates validated scientific-field bundles.
`collection.publish` imports them through normal Tuldok project/data endpoints,
one independently grouped parent record per simulation. Publish training and
validation before execution; publish reserved fields after the final candidate
and test are frozen. Records are draft synthetic references, without invented
human review or release approval. The existing project activity and Rheon egui
observer display real generation/training/test progress. Reserved arrays are not
previewed during development generation.

Bulk arrays, material, checkpoints and retained failures belong under ignored
`runs/`; durable completion evidence records their hashes and normal catalog Run
identities. Reproducibility is scoped to the measured platform/toolchain.

## Verification

Run `python -m unittest discover -s experiments/sealed-water-v1` and
`cargo fmt --manifest-path probes/sealed-water-v1/Cargo.toml --check`.
Both native builders run release build, tests and Clippy with warnings denied.
The water evaluator adds a regression admitting water and rejecting non-cubic
cells; its unchanged independent face-balance and fallback tests also pass.
The narrow-force regression exercises the real viscosity stage: infinite Gaussian
tails previously created subnormal values and were rejected; the finite smooth
support fixes the source without weakening native arithmetic or physical gates.
