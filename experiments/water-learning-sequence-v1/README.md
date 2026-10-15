# Water learning sequence

The user approved a one-snapshot fit, longer training, and a controlled comparison
of 256 versus 1,024 independent training simulations. The canonical plan and
results live in `docs/plans/water-learning-sequence-v1`.

This namespace captures the established seven-channel float32 CUDA model and loss
without architecture changes. It adds explicit lossless views over qualified
water captures, bounded two-part numeric packets, CPU validation/test recipes and
native diagnostics supporting several capture roots. Historical source and
results remain untouched. Code copied from sealed-water-v1 retains its numerical
operations; source identities are frozen for each new managed trial.

Trials use 10,000 CUDA Adam updates, rate 1e-3 then 1e-4 at 1,000 and 1e-5 at
6,000. States 0, 2,000 and 10,000 are retained; progress is published every 100
updates. Inference and native qualification use CPU. Model acceptance remains the
existing 1e-6 float32 residual profile plus unchanged physical gates.

- `fit-one`: one original training simulation's native index-15 snapshot, batch 1.
- `long-256`: both saved states for all original 256 training simulations, batch 2.
- `scale-256` / `scale-1024`: matched one-state-per-parent views, alternating
  native indices 15 and 63, batch 2. Shared initialization, optimizer schedule,
  sampling policy, validation parents and training budget.

The data-size pair uses one state each to fit the existing Tuldok material bound.
It is a distinct paired experiment; comparing long-256 directly to scale-1024
would change both parent count and snapshot selection. All original captures
retain both saved states. The training views preserve every selected float64 bit,
full-grid shape, cubic spacing and native preceding-pressure context.

`prepare.py` constructs views from the existing original packet plus three fresh
256-training-parent native cohorts. Use the original sealed-water pipeline to
build/generate/pack each `configs/cohort-N.json`: fixture, four-scene pilot, then
full generation. Those physical/numeric capture contracts remain frozen. Cohort
3 reserves 64 fresh test parents. Their fields never enter normalization,
optimization or checkpoint selection.

```bash
.venv/bin/python experiments/water-learning-sequence-v1/prepare.py \
  --root runs/water-learning-sequence-v1 --include-large
.venv/bin/python -m unittest discover -s experiments/water-learning-sequence-v1 -p 'test_views.py'
```

Commit qualified source before assembly. Use the selected Pumas Torch Python:

```bash
"$TORCH_PYTHON" experiments/water-learning-sequence-v1/pipeline.py assemble \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --validation runs/water-learning-sequence-v1/views/validation \
  --reserved runs/water-learning-sequence-v1/views/reserved-test \
  --runtime-manifest runs/cuda-pressure-runtime-1/manifest.json \
  --tuldok runs/live-integration/Tuldok \
  --config experiments/water-learning-sequence-v1/configs/fit-one.json \
  --output runs/water-learning-sequence-v1/fit-one/project
```

`workflow.execute(client, project, fresh_output, observer)` admits train, CPU
restore and validation inference through the existing authenticated Tuldok client.
It does not run reserved inference by default. It reopens completed/drained Runs
and verifies every downloaded artifact. A two-update managed pipeline smoke
qualifies this new packet/recipe path before full trials.

Native fitted-case checks use the restore artifact population and `--training-only`;
validation checks use the validation view and inference artifact population:

```bash
.venv/bin/python experiments/water-learning-sequence-v1/pipeline.py evaluate \
  --packet runs/water-learning-sequence-v1/views/fit-one \
  --inference runs/water-learning-sequence-v1/fit-one/managed/stage-01/artifacts \
  --training-only --binary runs/sealed-water-v1-3/evaluation-build/rheon-sealed-water-evaluation \
  --output runs/water-learning-sequence-v1/fit-one/fitted-evaluation
```

The native evaluator and independent MAC oracle are the already measured water
versions. They measure actual equation residual, divergence and physical bounds;
loss reduction alone never counts as a solved pressure proposal.

After validation-only candidate selection, call `workflow.execute` with
`stage_names=['reserved_test']`, `previous={'train': reopened_training_run}` and a
fresh output directory for the selected trial only. The final candidate is
frozen before reserved inference. The current Tuldok catalog owns runtime,
material, grant, checkpoint custody, all CUDA/CPU Runs and artifacts. There is no
private catalog, local substitute optimizer, automatic CPU training fallback or
silent runtime resource extension. Scientific results and limitations are recorded
for every stage, including failures.

The independent observation path replays real native training scenes without
modifying the frozen captures or in-flight recipes. `preview.py build` qualifies
`probes/water-preview-v1` against the same prepared native source and reservation.
`preview.py replay` publishes a bounded timeline only after all original fields
at indices 15 and 63 reproduce byte for byte. For example:

```bash
.venv/bin/python experiments/water-learning-sequence-v1/preview.py replay \
  --binary runs/water-learning-sequence-v1/preview-native-build-2/rheon-water-preview \
  --capture runs/sealed-water-v1-3/capture --trajectory-id water-train-0000 \
  --steps 256 --stride 4 --output runs/water-learning-sequence-v1/my-preview
.venv/bin/python experiments/water-learning-sequence-v1/tracer_check.py \
  --session runs/water-learning-sequence-v1/my-preview/session.json \
  --output runs/water-learning-sequence-v1/my-preview/tracer-diagnostic.json
```

Use a fresh output path. The 256-step example runs 4.27 seconds of native
simulation and records 64 states. Steps beyond the original 64 are observation
continuation only. The native preview's independent test moves a compact tracer
exactly one cell in each axis under a known velocity. `tracer_check.py` measures
centroid travel and width in physical units; `--require-visible` returns 2 when
travel is under half a voxel and width changes under 10%. That is a display
diagnostic, not a physics or model-accuracy gate. Semi-Lagrangian concentration
transport is diffusive and does not conserve tracer mass exactly.

`preview_fields.py` exports the verified replay through Tuldok's existing
scientific-field bundle format, including pressure, tracer, cell-centered
velocity components, speed and exact force provenance. Import through the
normal project's data reference into the existing catalog. Verify the stored
bundle through `/api/workbench/asset/{record_id}`. Project manifests are immutable
catalog snapshots; registering changed references requires a fresh project ID,
while activity and unpinned source contents remain observable at their existing
paths. Do not mutate an old project snapshot or create a separate catalog.

Rheon's `rheon-training-viewer` accepts one to sixteen `--session` paths. It pins
one scene, uses its real timestamps and shares native translucent voxel rendering
with the ordinary desktop. The force inspector and water velocity arrows describe
the actual recorded scene. The separate AI-comparison application still owns
native-versus-model dynamics and qualification.
