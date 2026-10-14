# Tuldok execution contracts and implementation handoff

All commands naming `global_pressure.py`, its project assembler or new native
probe below are **proposed implementation interfaces**, not tools delivered here.
Tuldok `project_cli.py` commands are existing contracts at merged PR58.
No project opening, generation or training is executed by this design PR.

## CPU smoke resource admission

Resource caps are proposed experiment limits, to be measured before submission.
Exceeding a cap stops the preset and requires review; no implicit enlargement.

| Stage | Preset/cap | Publication |
| --- | --- | --- |
| Reference generate/verify | 16³; 12 train+4 val ×24 frames; native one thread; 600s per stage; RAM <=4 GiB | Separate numeric train shards, val shard and manifests; each named input/output <=32 MiB |
| Captured train | 200 updates, batch 2, Torch/BLAS one thread, CPU only; child wall <=600s; RAM <=4 GiB | Flat weights/splits/normalization/receipt/prediction witness/summary, aggregate <=32 MiB |
| Restore-dev | Zero updates, new process, same captured runtime; <=600s/RAM4 GiB | Exact trained weights plus prediction identity and native dev quality |
| Final evaluation | One suite and one method per Run, <=600s/RAM4 GiB; split fresh-process samples if necessary in frozen protocol | Compact per-frame metrics and at most 8 selected-frame witness fields per trajectory; aggregate <=48 MiB |
| Whole smoke | Proposed active computation cap 45 CPU-wall minutes on a four-CPU-equivalent host, serial runs; stop/report overrun | Stage receipts retain actual elapsed time; provisioning/build time separate |

A 16³ frame's raw canonical fields are 32,768-byte RHS, 32,768-byte pressure,
52,224-byte MAC velocity and 16,384-byte tracer: **134,144 bytes**. Previous pressure
can be referenced by frame lineage; no duplicate stored field required. Train
raw payload is 38,633,472 bytes for 288 frames; split into two <=32MiB shards.
Validation is 12,877,824 bytes; these are sizing arithmetic, not observed file sizes.
Include arrays/metadata overhead and additional fields in measured preflight.
No compressed-size assumptions. Model recipe needs RHS/pressure/previous-pressure
and geometry/splits; native witness inputs may be separate dev-evaluate stages.

Existing captured CPU runtime limits: named inputs <=40MiB each, aggregate inputs
<=256MiB and total capture <=320MiB; default output <=64MiB aggregate, <=32 flat
names and wall <=600s. Keep smoke within the default profile, using measured bounds.
At most32 inputs/parameters and64 source files. The smoke workload is56,448method-frames across both timing regimes (see acceptance.md).
Use train-only measured throughput with startup/custody/output headroom to admit
the complete45minute cap before freeze. If final method/suite exceeds600s,
stop/replan before the freeze rather than splitting a timed rollout after access.
Complete 96-frame trajectories cannot be cropped to fit a runtime cap.

The full raw training dataset cannot fit these captured input bounds. Do not
pretend many references evade the aggregate limit, add path arguments to uncaptured
data or train chunks with implicit mutable weights. Full training remains an
explicit separate local-GPU procedure. Requalify any compact full checkpoint on CPU
before admitting a bounded existing CPU restore/evaluation recipe.

## Proposed lab source and recipe surfaces

Keep implementation under a new experiment namespace with an isolated native
manifest, lock and build receipt. Reuse native public functions and current generic
contracts; no copy of a Tuldok executor/reader. Capture every imported local source,
exact binary/runtime identities and dependency notices at the distribution boundary.
A planned lab CLI is:

```sh
# Future interfaces: not executable from this design-only branch.
python experiments/global-pressure-v1/global_pressure.py generate --config smoke.json --output NEW_DATA
python experiments/global-pressure-v1/global_pressure.py verify --dataset MANIFEST --output NEW_VERIFICATION
python experiments/global-pressure-v1/global_pressure.py train --config smoke.json --inputs INPUT_MANIFEST --splits SPLITS --output EMPTY_RUN
python experiments/global-pressure-v1/global_pressure.py restore --checkpoint WEIGHTS --receipt TRAIN_RECEIPT --normalization NORMALIZATION --splits SPLITS --inputs DEV_INPUTS --witness PREDICTIONS --output EMPTY_RESTORE
python experiments/global-pressure-v1/global_pressure.py evaluate --freeze FREEZE --suite SUITE --method METHOD --output EMPTY_EVAL
```

The same entrypoints are declared as `generate`, `verify`, `train`, `restore-dev`
and `evaluate` recipe references using `external_recipe_v1`. Treat generic
parameter bounds as maximums; v1 scientific configs close the permitted values.
There is no remotely tunable geometry, checkpoint-selection or tolerance field.

Definition uses schema_version1, fixed `format`, lab-owned id/name/version,
`runtime:captured_cpu_python_v1`, actual `runtime_snapshot_sha256`, source
`path/sha256` inventory and entrypoint. Arguments are literal/input/parameter/
output_dir tokens, with `native_policy` only through admitted native declarations.
No shell command, runtime installer, network download, ambient sibling checkout,
environment override, compiler/JIT or GPU access inside the recipe.

Project schema1 contains immutable 32-hex id, name, data/recipes/evidence references
`id,label,path,sha256`, and declared requirements. Paths are relative to the server's
configured project root, not necessarily the manifest directory. Generate the
project only after actual source/input/runtime hashes exist; no fake digest or
placeholder project is submitted. New inputs/source changes create a new snapshot
identity as required by the project owner.

Train outputs, all required and flat:

| File | Existing generic role | Content and bound |
| --- | --- | --- |
| weights.npz | checkpoint | finite numeric trained arrays, <=8MiB |
| splits.json | training_splits | exact persisted development partition, <=256KiB |
| normalization.json | preprocessing | training-only scales/provenance, <=256KiB |
| receipt.json | training_result | input/source/runtime/seed/optimizer/loss/timing/artifact hashes, <=2MiB |
| predictions.npz | predictions | bounded train/validation restore witness, <=8MiB |
| summary.json | result_summary | closed generic summary, <=64KiB |

Weight schema/inventory belongs in receipt and explicit model format; server never
loads it. Restore binds these trained artifacts and unchanged dev input bytes
with existing `{kind:run_artifact, run_id, artifact_name, expected_sha256}` inputs.
Tuldok captures only eligible completed/drained same-catalog artifacts, checks hash
custody and independently publishes successful outputs. Weights are interpreted
only by the lab restore recipe after schema/source/shape/dtype/finiteness checks.
Restoration must not regenerate splits, refit normalization or execute optimizer.

Progress events use `tuldok_progress_v1` with bounded stage/counts and <=8 finite
numeric metrics. Completion requires Run custody/publication, not an event. Summary
uses only `format,title,metrics,scope,checkpoint?`; <=24 metrics and8 scope strings.
Display actual qualified cases, rejection/PCG/reset rates, residual/divergence,
train updates and elapsed cost. Scope says CPU smoke/failed or qualified/negative
speedup as measured; it must not imply global solver or GPU qualification.

## Existing CLI and UI path

Canonical contracts at the merged Tuldok pin:
[project CLI](https://github.com/MrScripty/Tuldok/blob/f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6/docs/contracts/project-cli.md),
[external recipes](https://github.com/MrScripty/Tuldok/blob/f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6/docs/contracts/external-recipes.md),
[captured CPU runtime](https://github.com/MrScripty/Tuldok/blob/f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6/docs/contracts/captured-cpu-runtime.md),
[projects](https://github.com/MrScripty/Tuldok/blob/f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6/docs/contracts/projects.md).

Use an owner-prepared local catalog/root alias and qualified captured runtime,
private AgentClient config/token and inspect/edit/execute scopes. Requirements
or file presence do not prove a runtime. An owner reviews exact preflight material
and supplies the existing one-use execution grant; this workflow-specific grant
is imposed by Tuldok, not a new approval invented by the lab plan.

```sh
# Existing commands; identifiers and hashes below must be actual returned values.
AGENT_PY=/path/to/qualified-agent-python
AGENT_CONFIG=/private/tuldok-agent.json
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" open --root research --manifest project.json
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" inspect --project-id PROJECT_ID
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" recipe --project-id PROJECT_ID --reference-id train --expected-sha256 RECIPE_SHA256
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" prepare --arguments train-graph-inputs.json
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" preflight --graph-id GRAPH_ID --revision 1
# Owner grants this exact digest to the selected actor in Pipeline.
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" submit --arguments approved-submission.json
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" status --run-id RUN_ID --progress
"$AGENT_PY" project_cli.py --config "$AGENT_CONFIG" download --run-id RUN_ID --name weights.npz --expected-sha256 WEIGHTS_SHA256 --out ./new-trained-weights.npz
```

Train graph inputs bind train shards, validation shard and split/config references
by exact hashes. Restore graph binds trained Run artifacts and dev dataset refs.
Eval binds the reviewed freeze, weights and one suite dataset only after exposure
is authorized. Submission carries request_key, graph_id, graph_revision,
expected_digest and approval_id from actual preflight/grant. On uncertain admission,
inspect saved state and reconcile the same body/key; do not create a second attempt.
Changed material needs fresh preflight/grant. Credentials stay out of recipes/Git.

UI: `/workspaces` → Open project (`research`, `project.json`) → inspect references →
Current local catalog → Pipeline → select saved external graph → review preflight →
Grant this material to an actor → explicit submit → Runs → actual progress/results →
verified checkpoint/artifact download. Reopen catalog/service and confirm same Runs
and hashes. Observe actual train/restore/eval at desktop and narrow widths; a mockup
or direct-local run cannot satisfy E6. This task does not send review requests or
messages outside the authorized lab.

## Next implementation handoff

1. Review this design; resolve native API/geometry feasibility before model code.
   Inspect the exact pinned Rheon source, forward-port only the reviewed sealed
   warm-start semantics in an isolated dependency, and test exact/zero/poor/shifted/
   nonfinite/wrong-length guesses, scratch reuse and cancellation for both PCGs.
2. Implement deterministic reference generation/verifier and tiny independent
   anisotropic hand-oracles. Measure reference feasibility, memory and shard sizes;
   audit reservation disjointness against all historical seeds/lineages.
3. Implement the fixed model/differentiable full stencil, finite-difference gradient
   and global-dependence checks. Admit exact dependency/runtime/source inventory
   and output bounds through generic Tuldok readers before actual Run submission.
4. Execute one bounded train and fresh restore-dev. Preserve all failure receipts
   and per-case native metrics. A failed usefulness result cannot admit more steps.
5. Independently review source, trained-weight restore and proposed freeze. Commit
   freeze covering generator/evaluator/model/patch/binaries/weights/config/partition/
   normalizers/method order/metrics/timing/runtime hashes before generating test data.
6. Generate all reserved suites once, evaluate unchanged methods, and publish complete
   quality/cost/conservation/fallback evidence in the lab. Full GPU remains separately
   conditional. Implementation PR must define its actual allowed paths and gates.

BC AI packet after evidence exists: source/runtime pins, reservation/freeze, full
input/artifact hashes, licensed generation/recovery procedure, trained weights,
restore recipe, per-trajectory outcomes including failures, timing samples and
clear unsupported domains. Offer partner tasks for independent restoration,
operator/geometry validation or a separately preregistered model comparison.
No partner identity, contact, invitation or public announcement is invented/sent.
Training Your Own Models gets the workflow and negative-result lesson; measured
solver gains can enter only after corresponding gates pass.
