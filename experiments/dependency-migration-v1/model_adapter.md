# Current pressure model adapter

`model_adapter.py` owns the lab preprocessing/reconstruction and current generic
process implementation. Rheon owns context construction, physical support checks,
model process lifetime and candidate acceptance. `package_model_adapter.py` is an
explicit offline converter; the running model never imports it or parses the old
v2 deployment, training recipe, Run, receipt or checkpoint-history formats.

The only process request is `rheon-pressure-context-request-v1`, at stage
`sealed-mac-pressure-context-v1`, carried by a `rheon-pressure-adapter-v1` package.
The host launches the measured interpreter with `-I -S -B`, the measured entrypoint
and `--deployment manifest.json`. Startup emits bounded loading notices followed
by `adapter_ready` binding the exact manifest SHA256. Replies are current
`candidate` messages with `full_pressure`, f64, x_fast ordering and cell-zero gauge.
The prior has already been included exactly once by the lab owner.

Four named fields are mapped by the manifest: `base` is `previous_pressure` in Pa,
`residual` is `initial_residual` in m3/s2, `rho` is global `density` in kg/m3, and
`dt` is global seconds. Geometry stays counts XYZ and one cubic h; neural fields
reshape to N,1,Z,Y,X without reordering values. The adapter rejects bad shapes,
units, nonfinite scalars/fields, unsupported physical contexts, missing or stale
prior lineage and unqualified diagnostic history before calling the model. Explicit interval endpoints follow Rheon's maximum one-ULP rule for indexed
clocks; prior endpoint alignment remains exact. The
host checks viscosity and the actual native-step count before dispatch because
those facts are host-owned and are not additional fields in the current wire.

The numerical mappings stay distinct:

| Package | Preprocessing | Correction policy |
| --- | --- | --- |
| Original aa82 | Original seven-channel residual, prior, XYZ centres, density, dt; unchanged frozen scales | One original network prediction; float32 correction converted/gauged and accumulated with prior in float64 |
| Frozen 26b control | One channel q=(rho/h)r; unchanged train-only RMS; float32 input/output | Original linear two-step architecture; explicit original iterative policy selects one network evaluation per proposal, maximum 32, minimum zero, unchanged v2 equation/divergence/pressure stopping |

An all-zero residual returns the supplied base exactly without a network call.
The current control assembly composes the exact captured portable model and
physical residual functions, in the original construction order. The six original
iteration functions are extracted verbatim; their dataset/CLI decoder is omitted.
No original recipe reader is needed to assemble the model. Weight tensor names,
shapes, dtype, byte lengths and numeric hashes are checked before strict restore.

The installed Torch manifest uses the current canonical
`managed_torch_python_v1` contract, including its mandatory measured `file_links`.
Fresh current-owner inventory adds link identities while preserving the exact
installed numerical files. The new manifest therefore has its actual new hash.
This does not assert that a historical Run executed under new material. Adapter
startup captures and calls the exact current `torch_runtime.py` owner to recheck
asset topology, link targets and every measured file before importing NumPy or Torch,
requires the explicitly hidden CUDA device, sets one Torch thread and verifies
CPU float32 parameters. Current material identity, model identity and historical
execution identity remain distinct.

## Explicit conversion

Use fresh package and archive directories. The archive retains exact original
bytes plus a measured inventory; conversion receipts map those originals to the
actual new descriptor identity. Existing packages are never overwritten.

```bash
python3 -B experiments/dependency-migration-v1/package_model_adapter.py control \
  --project /ABS/LAB/runs/water-early-pressure-v1/control/project \
  --artifacts /ABS/LAB/runs/water-early-pressure-v1/control/managed-full/stage-01/artifacts \
  --runtime /ABS/LAB/runs/dependency-migration-v1/runtime/current-manifest-v2.json \
  --runtime-owner /ABS/LAB/runs/dependency-migration-v1/Tuldok/torch_runtime.py \
  --iteration experiments/water-stability-learning-v1/learned_iteration.py \
  --output /ABS/LAB/runs/dependency-migration-v1/packages/control-NEW \
  --archive /ABS/LAB/runs/dependency-migration-v1/archive/models/control-NEW
```

For aa82, use subcommand `aa82`, `--bundle` pointing at the exact original
`runs/live-ai-comparison/deployment-2`, and the same explicit runtime/output/archive
arguments. The converter verifies the complete original bundle inventory. Control
conversion requires the frozen 26b/f73 numerical identities, the measured source
witness and original CPU restore custody. Both package unchanged normalization and
weight bytes. Archives and generated packages remain under ignored `runs`; no
weights or dataset are committed.

`model_adapter.load(path)` returns a state exposing `np`, `torch`, `model`,
`network`, `normalization`, `policy`, `manifest` and `iteration`.
`sample_from_request(value, state)` exposes the exact mapped tensors;
`predict(value, state, trace=None)` applies the explicit saved policy. A process
owns one package because measured numerical modules use flat import names. Compare
independent packages and historical references in separate processes.

The migration acceptance protocol Q4 owns the first exact saved-input gate. It
must pass before broader numerical tests or trajectories; explanatory numerical
differences do not pass. The 74-case acceptance remains separate and uses the
preselected control only. Package generation itself does not run inference,
qualify accuracy, alter model weights or enable desktop deployment. Source and
execution receipts are retained in `runs/dependency-migration-v1/receipts`.
