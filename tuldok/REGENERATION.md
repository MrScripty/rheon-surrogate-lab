# Regenerate inputs in the executor's own checkout

Generated `runs/` paths from another execution environment are not shared.
Use a fresh lab checkout at the parent-selected reviewed commit. Preserve the
selected commit and dependency/runtime versions in new receipts; do not copy a
historical receipt over newly generated evidence. No release tag is needed.

## Native-verified captured CNN inputs

Prerequisites: Git, Rust/Cargo1.92.0, CPython3.12 and the explicitly installed
NumPy2.3.5 CPU environment. From the clean lab root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tuldok/requirements-cpu.lock
cargo build --release --locked
target/release/rheon-surrogate-lab generate data/executor-cnn 4
.venv/bin/python tuldok/cnn_project.py package --dataset data/executor-cnn --destination runs/executor-cnn-data
sha256sum runs/executor-cnn-data/dataset.npz runs/executor-cnn-data/splits.json
python3 tuldok/cnn_project.py assemble --package runs/executor-cnn-data --runtime-snapshot-sha256 ACTUAL_MANAGED_SNAPSHOT_SHA256 --destination runs/executor-cnn-project
```

All destinations must be new. The original generator pins Rheon34235d33 and
creates18cases:8train,4validation,4test and2checks. Packaging invokes actual
native verification, binds unchanged manifest bytes and captures lossless
float64 RHS/pressure pairs for the16noncheck cases. Splits remain whole original
seed/family/trajectory identities. Test arrays are not loaded by the trainer.

The previously recorded tensor NPZ hash is
`38cd317657d222145b6007792ee68eb2bdf70ba92ed2735f34cb9daa6ebbd9cf`.
On the recorded platform, exact regeneration can reproduce those tensor bytes.
`splits.json` additionally binds generator commit and original manifest hash;
its hash changes when those provenance identities change. Use the newly measured
split hash in graph bindings. Cross-platform bitwise equality is not promised.

The runtime hash must identify the executor's actual managed captured CPU snapshot.
The earlier inventory-only hash is not a registered snapshot. Assembly merely
checks its syntax; generic preflight owns actual runtime admission.

Configure the newly created project root and use the paired CLI at Tuldok
`cdbe5136e05f2b49c43b9884f3ed03dcc8c7025c` or a parent-verified compatible pin:

```sh
# Start the existing app with:
# --project-root cnn=/absolute/lab/runs/executor-cnn-project
AGENT_PYTHON project_cli.py --config PRIVATE_AGENT_CONFIG open --root cnn --manifest project.json
AGENT_PYTHON project_cli.py --config PRIVATE_AGENT_CONFIG inspect --project-id RETURNED_PROJECT_ID
```

Prepare recipe reference `train` with exact `dataset`/`splits` project references,
their newly observed hashes and `steps:200`. Review the returned preflight
material and obtain its one-use grant through the existing owner workflow.
After completion, download and verify `weights.npz`, `splits.json`,
`normalization.json`, `predictions.npz` and `receipt.json`. Prepare `restore`
with `dataset` from the same project and the other five inputs from that completed
Run via generic `run_artifact` bindings. Preflight/approval must bind that exact
new graph; do not guess catalog/graph/Run/actor/material IDs.

For a separate direct local observation of the same assembled sources:

```sh
mkdir runs/executor-cnn-train runs/executor-cnn-restore
.venv/bin/python runs/executor-cnn-project/cnn.py train --dataset runs/executor-cnn-project/inputs/dataset.npz --splits runs/executor-cnn-project/inputs/splits.json --output runs/executor-cnn-train --steps 200
.venv/bin/python runs/executor-cnn-project/cnn.py restore --dataset runs/executor-cnn-project/inputs/dataset.npz --splits runs/executor-cnn-train/splits.json --checkpoint runs/executor-cnn-train/weights.npz --receipt runs/executor-cnn-train/receipt.json --normalization runs/executor-cnn-train/normalization.json --predictions runs/executor-cnn-train/predictions.npz --output runs/executor-cnn-restore
```

This proves direct captured-data execution, not managed Run custody or CLI
download/reopen. Train performs200real original Adam updates; restore performs
zero and must reproduce retained train/validation predictions exactly. The
radius-one CNN still qualified0/16native physics predictions in the recorded
evaluation. Loss decrease and checkpoint restore do not imply pressure quality.

## Separate modal pressure parity

Use `project.py build` for the separate trajectory native pin2209012b, then
assemble the pressure project with an actual managed snapshot and measured build
receipt as documented in [README](README.md). It restores the committed original
checkpoint; it does not use the CNN dataset/checkpoint above. Modal fitting is
closed-form with zero optimizer steps, distinct from200-step CNN training.

The complete original32³/4seed/120frame/3repeat CPU comparison has its own
resident timing scope and cold full-field audit. Generate it in new destinations;
never substitute a tiny CI run or a training-loss receipt for this benchmark.
The historical8.20x result is not guaranteed on another CPU/runtime. CI checks
native contracts, small real trajectories and captured-data CNN restore; it does
not claim to rerun that full performance experiment or managed Tuldok acceptance.
