# Global pressure CPU smoke through generic Tuldok

This is slice 2 of the accepted pressure-correction design. It uses complete,
verified development trajectories and trains one fixed eager global model for
exactly 200 Adam updates. Final reservations remain sealed. The model and numeric
weight reader belong to this lab; Tuldok owns source/input capture, material
grants, process custody, artifact publication, progress and downloads.

Read [the slice plan](../../../docs/plans/global-pressure-training-v1/plan.md)
and its reports for actual source, runtime, input, Run and artifact identities.
Historical modal gains and the failed small CNN remain separate evidence.
A successful train/restore workflow alone does not qualify an accelerated solver.

The actual fixed smoke completed 200 updates and exact fresh restore, but produced
zero learned correction on all 32 development witnesses. All short native warm
attempts failed and required full previous-pressure fallback. The observed model
has no demonstrated solver benefit; see the [negative diagnostic](../../../docs/plans/global-pressure-training-v1/reports/physics.md).
Do not run full/GPU training or expose final reservations from this outcome.

The isolated CPU environment uses the existing lab requirements lock. Data packing
requires the pinned native operator outside the captured recipe; training and
restore consume only numeric arrays/JSON, Torch, NumPy and the captured Tuldok
artifact-source validator. There is no mandatory Rheon runtime adapter in Tuldok.

```sh
# Qualify the new native operator, then pack only the accepted development capture.
python experiments/global-pressure-v1/training/evaluate_native.py build \
  --cargo /workspace/task-cargo/bin/cargo \
  --prepared-source /workspace/rheon-surrogate-lab/runs/global-pressure-v1-dependency/rheon \
  --output /tmp/global-pressure-training-reviewed-native
python experiments/global-pressure-v1/training/data.py pack \
  --capture /workspace/rheon-surrogate-lab/runs/global-pressure-v1-capture/committed-development-complete \
  --native-binary /tmp/global-pressure-training-reviewed-native/rheon-global-pressure-training \
  --output /workspace/rheon-surrogate-lab/runs/global-pressure-v1-training/reviewed-packet
```

Use the actual qualified CPU interpreter for packing and unit checks. Fresh names
are required; failed/partial destinations are not reused. Each flat input is
bounded before allocation and the independent capture verifier runs first. The
reader checks whole trajectories across both train shards and validation,
original per-row fields, numeric history fingerprints, and a caller-frozen native
residual origin digest. Shared initial fields are explicit diagnostics. Original
full-history physics is a packager attestation; the standalone reader does not
claim to rerun native physics.

Before assembly, commit the source and complete the model qualification report.
The exact actual runtime manifest, native build receipt, measured packet and
qualified source/report are required; no fake runtime hash is submitted.

```sh
python experiments/global-pressure-v1/training/project.py \
  --package /absolute/reviewed-packet \
  --runtime-manifest /absolute/catalog/runtime-snapshots/ACTUAL_SHA/manifest.json \
  --native-build-receipt /absolute/reviewed-native/build-receipt.json \
  --qualification-report /absolute/model-qualification.json \
  --tuldok-source /absolute/clean-pinned-Tuldok \
  --output /workspace/rheon-surrogate-lab/runs/global-pressure-v1-project/reviewed-project
```

The assembler only produces immutable project bytes. A catalog owner captures
and qualifies the actual CPU runtime using the existing Tuldok runtime CLI, starts
Tuldok with a configured project root, and supplies a private scoped AgentClient
configuration. Assembly, opening and preflight do not execute training.

Use the existing `project_cli.py --config PRIVATE_CONFIG` commands: `open`,
`inspect`, `recipe`, `prepare`, `preflight`, `submit`, `status --progress`, and
`download`. Train graph inputs are the five exact project data references.
Training has no editable step, architecture or tolerance parameters. Review the
returned exact material before the existing owner one-use grant; submission binds
its digest, grant, graph revision and stable request key. On uncertain admission,
reconcile that same body/key instead of creating a second training attempt.

Restore binds those same five data references and five actual same-Run
`run_artifact` inputs: trained weights, receipt, normalization, predictions and
splits. It checks the captured cold custody-owner source, typed artifact evidence,
raw trained bytes, model/input/config/runtime closure and bitwise prediction
parity; it constructs no optimizer. Downloading artifacts and relabelling them as
project data cannot satisfy trained artifact custody.

UI path: Workspaces → open the configured project → inspect its references →
current catalog → Pipeline saved graph → inspect preflight and grant exact
material → Runs progress/result → verified checkpoint download. Reopen the
catalog/service and verify the same completed Runs and exact artifact bytes.
A task-local catalog is actual owner-local execution, not hosted/cloud evidence.
Tuldok's same-UID cooperative custody is not hostile-code/network isolation;
reported process RSS is an observation, not aggregate memory containment.

Six flat outputs are declared: numeric `weights.npz`, persisted `splits.json`,
train-only `normalization.json`, `receipt.json`, bounded `predictions.npz` and
`summary.json`. Predictions are a fixed 32-case witness: frames 0 and 23 of each
of the 16 development histories. Their same-runtime pressure bytes must match
fresh restore exactly. Clocks are kept outside prediction identity.

The local native diagnostic uses those same 32 teacher-state cases, unchanged
full residual/divergence/boundary/flux/Courant gates, learned warm PCG ≤32 then
one fresh retry from previous pressure ≤3,000, zero-Jacobi and true full-budget
temporal-PCG baselines. The separate temporal 32-step/reset path is labelled a
bounded diagnostic. Include proposal rejection, any PCG, reset cost and model
inference/custody overhead. These snapshots qualify no autonomous rollout,
held-out geometry/regime, engineering flow or acceleration claim. Retain poor
physics/cost outcomes without additional steps, restarts or checkpoint selection.

Full/GPU training and final-test materialization require a later reviewed slice.
