# Reusable local CPU pressure pipeline

This namespace owns fresh experiment configuration, data lineage and local
bootstrap. It reuses the historical GroupNorm model, preprocessing, finite-volume
loss, independent physics oracle and native diagnostic arithmetic without
changing their source. Historical Runs, hashes and negative results retain their
original meaning. GPU support and final suites are outside this CPU profile.

The [local CPU acceptance record](../../docs/plans/reusable-local-cpu-pipeline/completion.md)
records two completed configurable lineages, including the default fresh prepare
command. Successful execution qualifies the pipeline; native learned predictions
still require fallback in every observed case and establish no acceleration.

`configs/development.json` selects fresh whole-parent seeds, development geometry,
forcing, Adam schedule and checkpoint observations. It contains no host paths,
runtime hash, packet hash or originating Run ID. The packet, source witness and
runtime manifest record measured identities when material is created. Change the
experiment ID and development reservations for another lineage; do not reuse
historical or final seeds. The supported capture is a serial 16³ sealed domain,
24 frames per complete parent, two or more train parents and one or more
validation parents, at most 16 parents. The final configured update is the only
candidate. Other checkpoints are observations and omit optimizer resume state.
The default retains the historical 200-update/batch-two/seed/Adam settings and
interleaved train-parent/frame sampler. Data identities and host setup vary;
the GN architecture, loss and default optimization policy are unchanged.

## Host setup

Use Linux x86-64, Python 3.12 and Rust/Cargo 1.92.0, plus PostgreSQL 16–18. WSL2
requires separate environment qualification. Use a fresh venv with the CPU lock
and local application/CLI dependencies; the bootstrap never installs packages:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r tuldok/requirements-cpu.lock -r experiments/local-pressure-v1/requirements-local.lock
```

Make clean task checkouts of the selected Rheon and Tuldok sources. Preserve any
other working trees with dirty files or in-progress merges. The tested Tuldok
generic owner is `f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6`; the new native
lockfiles select Rheon `43ed26b645dd42de906d0fac7d1ec7d5ab6e351a` with the
retained lab pressure patch. Every native build verifies Cargo actually selected
the prepared source and records its revision, patch, file inventory and binary.

Copy `configs/host.example.json` to an ignored local file under `runs/` and supply
the actual paths and selected source revision. Relative host paths resolve from
that file's directory. Use a fresh, short, private PostgreSQL directory; the
helper creates no system service. Reserve at least 6 GiB beyond measured runtime
capture, plus scientific outputs, native caches and transient source copies.
Existing venvs, runtime snapshots and credentials are never transfer inputs.

## Prepare, inspect, execute

Commit qualified source on your experiment branch before immutable project
assembly. Launch with the interpreter selected in the host file:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/local-pressure-v1/bootstrap.py prepare \
  --config experiments/local-pressure-v1/configs/development.json \
  --host runs/local-pressure-host.json --output "$PWD/runs/local-pressure-workspace"
```

Preparation applies the retained pressure patch to committed Rheon source,
builds capture, generates and independently qualifies **one train frame first**,
builds/tests the native residual/diagnostic operator, then generates/verifies the
complete development capture and lossless numeric packet. The fixture is
ineligible for training. Packing verifies every field byte and residual row,
whole-parent split coverage and cross-split fingerprints. Normalization uses
training parents only.

Preparation runs the unchanged GN numerical suite and new portability tests
with zero skips and no optimizer construction. It initializes a new local
catalog/owner, measures/captures/verifies the installed CPU runtime, qualifies
initialized model arithmetic and complete output budget, assembles a new project
and saved Graph, and runs generic Tuldok preflight. `prepared.json` is published
only after valid preflight and confirmation of zero Runs. No training grant is
created. The private PostgreSQL service stops after successful preparation.

Inspect `preflight.json`, `project/source-witness.json`,
`project/qualification.json` and `runtime-inspection.json`, then execute:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/local-pressure-v1/bootstrap.py run \
  --workspace "$PWD/runs/local-pressure-workspace"
```

Execution uses the existing generic project CLI/HTTP driver and exact one-use
owner grants. Train and independent restore each complete and drain in the
captured CPU runtime. Restore consumes every configured checkpoint, weights,
history, predictions, scales, splits and originating receipt as typed artifacts
from one completed Run. It constructs no optimizer, preserves trained bytes and
requires fresh exact prediction parity. Tuldok's captured cold custody validator
remains the authority; downloaded files alone do not establish managed custody.
The driver reopens the catalog and verifies every artifact download again.

Finally, native evaluation observes frames 0/23 of every parent using the same
qualified residual operator. It retains the original residual, divergence,
wall-flux, net-flux and Courant gates, previous-pressure baseline, zero baseline,
learned warm work and every fallback. It journals responses before validating
them. Successful `completed.json` means the CPU pipeline completed; it does not
mean the learned solver accelerated anything. Teacher states do not establish
autonomous rollout or final generalization. Stage clocks overlap and cannot be
added to claim end-to-end speedup.

Every stage refuses existing output directories. A failed attempt remains
inspectable and is never automatically replayed or silently resumed. An
interrupted preparation may leave its private PostgreSQL service running; stop
it with the pinned helper and selected `--directory` after inspecting the failure.
Never delete an existing catalog to make preflight pass.

For separately qualified generation material, `prepare --material FILE` accepts
an explicit object with `fixture`, `capture`, `packet`, `capture_binary` and
`native_binary` paths. Each is reverified against the selected experiment and
native revision; this skips generation, not qualification or managed preflight.

## Where results appear in Tuldok

Bootstrap creates a **normal Tuldok catalog** at `WORKSPACE/catalog`, using the
configured private PostgreSQL service. This fresh catalog isolates qualification
from an existing application's data and Runs; it is a bootstrap choice, not a
Tuldok requirement. The prepare command currently requires this fresh-catalog
workflow and checks that dry preflight created zero Runs. It does not attach to
the catalog served by an already open Tuldok UI.

Generation and verification finish before training starts. Captures remain under
`WORKSPACE/capture`; the numeric packet is frozen into Project data references.
Checkpoints, history, weights and predictions are published as **Run artifacts**
when managed execution completes. They are not streaming imports into a Data
collection. Changing the UI's catalog alone does not turn those references into
collection items.

To inspect this catalog in a UI, use the qualified Tuldok checkout and its
application dependencies, start the configured PostgreSQL service with its
`scripts/local_postgres.py` helper, and load that helper's `env` configuration.
Serve `app.py --data WORKSPACE/catalog --project-root qualification=WORKSPACE/project`
on a free loopback port. For paths containing spaces, quote each path and the
whole `qualification=...` argument. Keep the existing instance's catalog,
database configuration and port intact. The local owner's credential belongs to
this new catalog; do not reuse or copy another catalog's token. Stop the private
service only after closing this UI and any managed execution. Integration into
an existing catalog or collection is a separate operation.

## Verification

```sh
TULDOK_CUSTODY_SOURCE=/absolute/clean/Tuldok/recipe_process.py \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python experiments/local-pressure-v1/qualify.py
```

Native Rust tests and Clippy run during each qualified build. Pure packet tests
use synthetic numeric lineage and do not claim physical qualification. The real
bootstrap and managed Runs provide the environment and custody evidence.

[Implementation plan](../../docs/plans/reusable-local-cpu-pipeline/plan.md).
