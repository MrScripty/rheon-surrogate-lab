# Lab-owned Tuldok pressure project

This project uses Tuldok's existing schema1 project references and
`external_recipe_v1` CPU recipes. The lab owns physical meaning, numerical
algorithms, experiment configuration, split policy, measurements and replay.
Tuldok owns generic source/runtime/input/output custody and Run lifecycle.
No project opening or assembly starts training or installs dependencies.

## Clean build and project assembly

Prerequisites: clean committed lab source, Git/Cargo/Rust1.92.0, CPython3.12,
a read-only compatible Tuldok checkout, and a reviewed captured CPU runtime
containing Torch2.8.0+cpu/NumPy2.3.5. Install the explicit CPU requirements into
a separate virtual environment; no recipe installs dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tuldok/requirements-cpu.lock
python3 tuldok/project.py build --receipt runs/tuldok-build.json
python3 tuldok/project.py native-policy --tuldok-root ../Tuldok --destination runs/tuldok-native-policy.json
python3 tuldok/project.py assemble --build-receipt runs/tuldok-build.json --runtime-snapshot-sha256 ACTUAL_CAPTURED_CPU_SNAPSHOT_SHA256 --destination runs/tuldok-project
```

Use the exact actual snapshot hash from Tuldok's explicit quiesced capture
workflow; assembly only checks its syntax. A runtime inventory hash alone is
not an installed/admitted snapshot. Unknown/missing runtimes remain unavailable
at generic preflight. The [captured-runtime contract](https://github.com/MrScripty/Tuldok/blob/d856a98/docs/contracts/captured-cpu-runtime.md)
owns capture; the application never installs the lab runtime. Initial contract
qualification used Tuldok main d856a98. Select a tested compatible revision
with the parent before integrating; no release tag is created by this task.

`build` refuses dirty lab or fetched Rheon source and records both exact Git
commits, source/manifest/lock hashes, measured native hash and tool versions.
The new Cargo manifest pins historical trajectory Rheon2209012b with parallel
kernels. Existing pressure34235d33 and sibling-path trajectory manifests are
preserved. Cargo Git fetch is an explicit build prerequisite, never a recipe
side effect. No cross-machine binary reproducibility is claimed.

## Generic workspace and local observation

Start Tuldok with `--project-root lab=/absolute/path/to/runs/tuldok-project`,
then Open project `project.json` in Workspaces. Data contains original checkpoint,
freeze, fixed config and measured build receipt; Pipeline offers Train/Compare;
Runs exposes the original historical comparison as external evidence.

For Train bind `config` and `build_receipt` to their exact project references.
For Compare additionally bind `checkpoint` and `freeze` to the original project
references for unchanged-model parity. Review the exact generic material,
explicitly approve it and submit with the existing Runs API/UI. To evaluate a
new training checkpoint, bind checkpoint/model.pt and freeze/freeze.json from
that completed training Run via the existing `run_artifact` input variant.
Do not edit bindings after approval or copy a trained model over the original.
The paired Tuldok owner qualifies managed execution, runtime, approvals and UI.
Direct local recipe evidence never claims those boundaries.

A direct observation uses the same assembled recipe/source bytes and generic
host-generated native policy, while recording its distinct execution scope:

```sh
.venv/bin/python tuldok/project.py run-local --project runs/tuldok-project --recipe compare --native-policy runs/tuldok-native-policy.json --output runs/pressure-compare --python "$PWD/.venv/bin/python"
.venv/bin/python tuldok/audit.py runs/pressure-compare --receipt runs/pressure-compare-audit.json --viewer-run pressure-parity
.venv/bin/python tuldok/project.py run-local --project runs/tuldok-project --recipe train --native-policy runs/tuldok-native-policy.json --output runs/pressure-train --python "$PWD/.venv/bin/python"
```

All output destinations are new. A partial output is never a verified success.
Configured generic wall/output bounds apply; local invocation only demonstrates
this trusted lab recipe, not the managed executor's descendant custody. No
filesystem/network isolation is promised by either path.

## Dataset, model and evaluation

`configs/pressure-v1.json` is a closed fixed protocol: sealed32³ unit box,
dt1/60s, density1.25, prescribed forcing/passive tracer. Train seeds100..111
with8frames/seed; validate500/501 over60frames; exposed tests900..903 over120.
Trajectories are indivisible; no validation/test arrays enter fitting. Train
reuses the unchanged original generator and ModalPressure.fit, produces actual
float64 RHS/pressure[96,32768] training.npz, model.pt and a new CPU freeze. It
verifies restore/refit equality and the actual unit-gain native physics rejection.
No optimizer epochs or synthetic loss history exist for closed-form modal fitting.

Compare uses unchanged trajectory_cpu.py with three rotated-order repeats,
equal8native/Torch threads, interop1, CUDA hidden and model tensors on CPU.
It checks4320timed frames, native residual≤1e-8, divergence≤1e-5, Courant≤1,
zero learned PCG iterations and full velocity/tracer relativeL2<1e-5 against
both Jacobi/SGS. Full six-field replay NPZs are retained for all methods/seeds.
A cold audit recomputes norms and reruns restored pressure predictions from the
retained RHS. Display voxels are a thresholded projection of full numeric fields.

Resident speed covers native startup/teardown, all simulation stages, inference,
pipes, gates and parsing; imports/model setup/warmup/capture/compression/rendering
are excluded. `*-process.json` records a separate fresh worker wall including
imports/setup/warmup/accuracy retention/compression. These scopes must not be
combined into a speedup. Approximately8x is the historical CPU result to compare,
not a requirement. This new CPU runtime differs from historical Torch2.14.
Agreement with Rheon's discretization is not general fluid or aircraft/submarine
validation; the operator also has an analytic spectral inverse not benchmarked
here. Later passenger jet and submerged forward-plus-ascent submarine projects
require validated solver/domain/data contracts before fitting.

## Existing lab viewer

The audit's `--viewer-run` installs only a new ignored replay directory matching
the unchanged existing viewer layout. Serve the lab root and open its reported
`/viewer/?run=pressure-parity&compute=cpu` URL:

```sh
python3 -m http.server 8098 --bind 127.0.0.1
```

Existing run paths and GPU historical displays remain preserved. The viewer reads the new replay's actual Torch2.8 CPU label; old replay files
retain the historical Torch2.14 default.

## Artifacts and checks

Source Git contains configuration, code, small recipes generated by assembly,
plans and compact receipts. Runtime, native binary, datasets, predictions and
assembled projects stay in ignored runs/target/. Model and freeze input references
remain byte-identical to the original. Generic output declarations name roles,
per-file/aggregate limits and SHA-verified download custody.

Run `TULDOK_CONTRACT_ROOT=/read-only/Tuldok .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v`
for focused real generic-reader compatibility and lab regressions, with the
original serial native probe built at its pinned sibling revision for old tests.
A zero runtime hash in a schema-only fixture never represents runtime admission.
