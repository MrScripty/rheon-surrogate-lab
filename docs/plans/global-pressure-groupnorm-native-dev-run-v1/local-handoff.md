# Local development handoff

The intended clean start is source → synthetic development simulations → verified
numeric dataset → a newly frozen Tuldok Project → training → independent restore
and physics evaluation. Existing weights and simulation arrays are optional
reproduction material. Keep them outside Git. This document prepares transfer;
it does not run anything on the user's machine or admit another scientific Run.

[PR8](https://github.com/MrScripty/rheon-surrogate-lab/pull/8) is merged into
main at `03c05e8`; it publishes the source and this handoff.
The inventory below records the earlier preparation snapshot.
The later [complete native result](reports/measured-results.md) found rawGN0/32
and fallback32/32, with no demonstrated solver acceleration.

## Source and transfer inventory

| Repository | Exact accepted pin and purpose |
| --- | --- |
| `https://github.com/MrScripty/rheon-surrogate-lab.git` | Main `28c65ac868678e496603d248c5c1e350ffc0fab3`; GN scientific source `fd3a6009e5805978a29ed1df7389ad226a40a914` |
| Same lab, active implementation branch | `feat/groupnorm-native-development-evaluation`, corrected source `135ae25786f72ff37848b768aafa27d331164596`; not remotely published at inventory time |
| `https://github.com/MrScripty/Tuldok.git` | Generic project/runtime owner `f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6` |
| `https://github.com/MrScripty/Rheon.git` | Native base `2209012bc5f902693d0f695ec1c2d854126af317`; lab-owned patch SHA `b851b38d2d03280220bc4648cd4d721b2e796786fb18d8188dc335336fe2a998` |

Git contains the generator/verification/build adapters, native dependency patch
and license, model/training code, fixed configuration, project/recipe assembler,
dependency locks, evaluation contracts, tests and historical compact evidence.
`project.py` generates a machine-specific `project.json` and train/restore recipe
JSON; the actual assembled project, data and checkpoint files are ignored under
`runs/`. A new host assembles its own project from its own qualified material.

The [measured inventory](reports/local-handoff-inventory.json) distinguishes
retained payload bytes from filesystem allocation:

| Optional executor-only scientific material | Files | Bytes |
| --- | ---: | ---: |
| Complete development raw capture | 3,555 | 75,695,918 |
| Packed development data and provenance | 6 | 51,811,126 |
| GN train outputs, including all 11 states | 18 | 20,607,920 |
| GN zero-update restore outputs | 18 | 20,609,575 |
| Total | 3,597 | 168,724,539 |

Final weights alone are 1,500,186 bytes; accepted final weights/checkpoint200
SHA256 is `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147`.
The retained interpreter environment is about900MB and is not a portable venv.
The separate historical captured CPU runtime has870,900,515 payload bytes; its
private catalog/session is excluded from transfer. Neither runtime is necessary
as a copied artifact for a fresh local lineage.

A verified source-only Git bundle is prepared at
`/tmp/rheon-local-source-135ae25.bundle`: 3,969,708 bytes, SHA256
`1aac2261602e0a3978543c60600529b65b1ed279d3ed97b0e03750cac73ac79e`.
It contains committed lab source/history through135ae25, without ignored runs,
catalogs, sessions, credentials or this later handoff document. No hosted download
URL or external artifact publication was created. The normal delivery route is
the public repository's reviewed source branch once pushed; this local bundle
can be delivered through an authorized task-file download route if the host UI
supports it. Do not claim that `/tmp` is a URL accessible from the user's laptop.
Optional old scientific artifacts can later be exported from an explicit
allowlist with file hashes and boundary review of provenance/custody metadata; they are not prerequisites and no bulk bundle was
created during this handoff. No whole `runs/` or catalog directory export.

## Local prerequisites and executable generation paths

Use Linux x86-64 or a qualified WSL2 Linux environment for the current process
adapter. Its Linux pidfd/subreaper/proc/drain tests do not establish macOS/Windows
support. Existing CPU evidence uses CPython3.12.14, NumPy2.3.5, Torch2.8.0+cpu,
Rust/Cargo1.92.0, one thread and deterministic algorithms. Start isolated local
environments from `tuldok/requirements-cpu.lock` and Tuldok's own pinned application
and agent requirements; record actual installed packages and file identities.
For Tuldok, configure the local PostgreSQL/catalog/owner as its pinned README and
contracts describe. Do not copy executor credentials. No GPU dependency is
installed by these recipes. Reserve space for native build caches, a captured
runtime and transient copies in addition to roughly169MB of scientific payload;
the payload size alone is not a sufficient disk budget.

These are the existing source entrypoints, shown with explicit local paths.
Build/capture generation is implemented; it has not been executed on the user's
hardware. The final pack command below shows its historical interface and
REJECTS a fresh capture until the new-lineage packer/reader is implemented:

```sh
git clone https://github.com/MrScripty/rheon-surrogate-lab.git lab
git clone https://github.com/MrScripty/Tuldok.git Tuldok
git -C Tuldok checkout --detach f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6
git clone https://github.com/MrScripty/Rheon.git Rheon-base
git -C Rheon-base checkout --detach 2209012bc5f902693d0f695ec1c2d854126af317
# Check out the delivered lab branch/pin, then work in a NEW local experiment branch.
cd lab
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3.12 experiments/global-pressure-v1/prepare_dependency.py \
  --source ../Rheon-base --destination "$PWD/runs/global-pressure-v1-dependency/rheon"
python3.12 experiments/global-pressure-v1/capture.py build \
  --cargo /absolute/local/cargo \
  --prepared-source "$PWD/runs/global-pressure-v1-dependency/rheon" \
  --output "$PWD/runs/global-pressure-v1-build/local-dev-build"
python3.12 experiments/global-pressure-v1/capture.py generate \
  --preset smoke --split development \
  --binary "$PWD/runs/global-pressure-v1-build/local-dev-build/rheon-global-pressure-capture" \
  --prepared-source "$PWD/runs/global-pressure-v1-dependency/rheon" \
  --output "$PWD/runs/global-pressure-v1-capture/local-dev-capture"
python3.12 experiments/global-pressure-v1/verify_capture.py "$PWD/runs/global-pressure-v1-capture/local-dev-capture"
python3.12 experiments/global-pressure-v1/training/evaluate_native.py build \
  --cargo /absolute/local/cargo \
  --prepared-source "$PWD/runs/global-pressure-v1-dependency/rheon" \
  --output /tmp/local-dev-native
# Historical interface ONLY: rejects new origin/source identities today.
python3.12 experiments/global-pressure-v1/training/data.py pack \
  --capture "$PWD/runs/global-pressure-v1-capture/local-dev-capture" \
  --native-binary /tmp/local-dev-native/rheon-global-pressure-training \
  --output "$PWD/runs/global-pressure-v1-training/local-dev-packet"
```

First qualify a one-frame train-only capture using the documented
`--split train --trajectory smoke-train-100000 --fixture-frames 1` path before a
complete generation. Every destination must be fresh. A fixture is ineligible as
a complete training dataset. `smoke --split development` generates16 whole
trajectories/384frames:12train parents/288frames and4validation parents/96frames.
Full/test generation is currently rejected by the capture adapter; final suites
remain sealed. Reusing the old deterministic development seeds does not create a
new independent holdout.

## Missing clean-start bootstrap and exact managed paths

This section describes the historical interfaces below. Fresh-host portability
work now lives in the separate [reusable local CPU pipeline](../../../experiments/local-pressure-v1/README.md)
and [its implementation plan](../reusable-local-cpu-pipeline/plan.md). That new
namespace owns configurable reservations and measured new identities while the
historical readers, trainers and evidence remain unchanged. Consult its current
acceptance report before treating it as a qualified end-to-end procedure.

Cloning is insufficient for an accepted fresh training pipeline today. The
historical packer first rejects a fresh capture: `training/data.py read_origin()`
pins its old manifest SHA and source commit before any residual generation. The
historical GN trainer deliberately pins packet2 byte hashes, native-residual
origin, train-only normalization, original helper source and runtime/custody
identities. Fresh capture JSON includes measured timing/provenance and will have
a fresh identity even if the numerical seed outputs agree. The historical
qualification/budget also bind that packet. Do not edit hashes until it passes
or label fresh outputs as the old Run.

The next local agent implementation must create a NEW experiment namespace with
an explicit fresh manifest/config, implement a packer/reader bound to a newly verified capture origin and native
residual anchor, generate and verify its development inputs,
fit preprocessing on train parents only, produce honest qualification/budget
reports, freeze the source, and assemble against the actual captured local CPU
runtime. Preserve historical source and all negative evidence. Supply a tested
bootstrap that configures a new local catalog/runtime/project and exposes a dry
preflight before execution; this GN-specific bootstrap does not yet exist.

Existing runtime CLI is `runtime_snapshots.py inspect|create|verify --data CATALOG`.
Inspection needs an explicit `--config`; creation additionally binds
`--expected-inventory-sha256` and `--quiesced`. The CLI requires an existing local
owner/catalog and must not be given an invented runtime hash. After a new lineage
is qualified, the existing project assembly interface is:

```sh
python3.12 experiments/global-pressure-v1/groupnorm-v1/project.py \
  --package /absolute/new-qualified-packet \
  --runtime-manifest /absolute/new-catalog/runtime-snapshots/ACTUAL_SHA/manifest.json \
  --native-build-receipt /tmp/local-dev-native/build-receipt.json \
  --qualification-report /absolute/new-qualified-report.json \
  --output-budget /absolute/new-qualified-budget.json \
  --tuldok-source /absolute/clean-pinned-Tuldok \
  --output "$PWD/runs/global-pressure-v1-project/new-local-project"
```

That literal historical assembler is not a promise of fresh-packet acceptance;
the new namespace must own its new identities. Current generic
`project_cli.py --config LOCAL_PRIVATE_CONFIG` paths are `open`, `inspect`,
`recipe`, `prepare`, `preflight`, `submit`, `status --progress`, `download`.
Use the actual project references/Graph revision/material digest from preparation;
the exact owner grant/submission commands depend on those returned values.
The UI path is Workspaces → open Project → Pipeline recipe/saved Graph/preflight
→ owner admission → Runs progress/results/artifact download. Desktop/narrow UI
observation is a separate check, not established by a CLI-only local setup.

The recipe invokes the child as `train.py train` with required `--train0`,
`--train1`, `--validation`, `--splits`, `--origin`, `--config`,
`--runtime-manifest`, `--source-witness`, `--input-evidence`, `--output-dir`.
Restore uses `train.py restore` with the same closure and all17
`--trained-*` typed originating-Run inputs declared by `TRAINED_INPUTS` in
`groupnorm-v1/train.py`. Tuldok supplies input evidence and the byte-pinned cold
custody validator; a shell invocation with downloaded files and fabricated
evidence is not an accepted managed restore. Restore is zero-update exact
final200/history/prediction parity, not training continuation. The current code
has no resume path, and numeric observational states omit Adam optimizer state.
No complete fresh-host train/restore command can be supplied honestly until that
bootstrap/new material is implemented and preflighted.

For existing-artifact evaluation, the exact supported adapter CLI is documented
in `experiments/global-pressure-v1/groupnorm-native-dev-v1/README.md`: `verify`
or `evaluate` plus training/restored artifact roots, capture, packet, binary,
source freeze/hash and fresh direct `/tmp` output. Its current freeze binds this
executor's absolute paths/native/runtime file hashes. It is NOT portable as-is;
a local reproduction requires a separately reviewed new host freeze/admission.
Do not run the original old-model `training/evaluate_native.py evaluate` against
GN receipts; it is the preserved negative comparator adapter.

## GPU boundary and current evidence

CPU source explicitly rejects CUDA tensors/packages. Moving the network alone to
CUDA cannot work: inputs, coordinates, float64 finite-volume operator/loss and
diagnostics/restore all assume CPU. A modest new hybrid experiment could keep CPU
preprocessing/float64 physics loss and transfer CNN inputs/results across a
differentiable CUDA bridge. A full GPU path additionally needs same-device tensor
construction/contracts, CPU serialization, synchronized timing, deterministic
kernel qualification and GPU-runtime receipts. Use a new lineage/profile; do not
weaken historical CPU restore checks or call external local GPU evidence a
managed Tuldok CPU Run. No CUDA execution, compatibility, memory capacity or
speedup has been tested. The prior `local-gpu.md` is a conditional plan, not
working GPU implementation. Retain all transfer, CPU loss, inference, native PCG,
verification and fallback cost. Qualify on train-only fixtures before full work.

The initial native attempt stopped after its first snapshot because Python3.12
compensated `sum` disagreed with native serial squared accumulation:
0.006704438615123662 versus0.006704438615123659. Corrected source135ae25 uses
the existing serial norm helper and keeps exact equality and physics gates.
61fixture tests, byte-only verification and both renewed independent reviews
passed. The failed run/journal remains intact. Raw GN and warm32 failed physics
on that case; fresh previous-pressure fallback qualified in145iterations, the
same iteration count as full previous-pressure-only and zero-start. All644
iterations across methods are charged. The separately frozen corrective full32
pass subsequently completed: raw GN and all warm32 attempts failed; fresh
previous-pressure fallback qualified32/32. GN added1,024 iterations to the
5,226-iteration previous-pressure baseline, about19.6% more work. The failed
prefix and corrective action total23,763 diagnostic iterations. These are
iteration counts, not an end-to-end timing ratio; no acceleration is established.
The [complete results](reports/measured-results.md) own those observations.

Preserve original small-CNN0/16 native failures, original global correction0/32
failure/384-row collapse, GN candidate/history and the failed adapter prefix.
Final unseen whole trajectories/geometries/regimes stay sealed. Any new local
model must retain group-level split/leakage checks, trained-state restore,
original residual/divergence/wall/flux/Courant gates, autonomous rollout and
total cost including failed warm work/fallback. Existing teacher snapshots do
not establish rollout, tracer/momentum/energy conservation or final generalization.
