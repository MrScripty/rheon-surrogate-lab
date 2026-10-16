# Source and compatibility assessment — 2026-10-16

This is a bounded consumer assessment, not a review of every upstream line or
runtime migration acceptance. Full fetched first-parent histories and changed
paths are retained in [upstream-changes.json](upstream-changes.json). Actual local
and live identities are in [versions.json](versions.json).

## What was pulled

| Repository/source | Before | Fetched target / action |
| --- | --- | --- |
| Primary Rheon main | 43ed26b645dd42de906d0fac7d1ec7d5ab6e351a; dirty | Fetched bc2f472787060d9a69cb619ed2c2fea10fd754b6,150 commits ahead. Primary working files preserved; target checked out detached at `runs/dependency-migration-v1/Rheon` |
| Primary Tuldok main | f58880df84d701ef66ffe21615ae787d80ce8a5e; clean | `git pull --ff-only origin main` to300f12df6f48e028fd9379911703f7c89337952a,422 commits ahead; clean afterward |
| Lab main | 03c05e82383ce0a18ded838803ce1f8f4ea1cd09 | Fetched; origin/main unchanged. Created separate planning branch |

Rheon changes 187 files, 54826 insertions/501 deletions relative to primary old
main; Tuldok changes 486 files, 94181 insertions/1641 deletions. These include large
documentation/test records; counts are not measures of migration complexity.
Rheon's actual experimental engine was separately taken from
`0fbadbe733b5b521e192dc8a2aa980d6d69a1e01`, not primary main. That branch and new
main diverge from43ed26b6. Do not assume its four local commits were merged merely
because the new implementation provides similar features.

## Rheon: relevant new behavior

The fetched history includes configurable pressure providers, accepted Session
composition, diagnostic policies/recordings, configured desktop workers, saved
comparison integration and independent domain/preview density controls. Read the
exact target documents:

- [Pressure context / adapters](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/pressure-context.md): immutable geometry/rho/dt/RHS/prior lineage, named inputs, package support, process ownership and full-pressure/correction reconstruction.
- [Configured accepted Session](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/configured-pressure-session.md): authoritative post-viscosity context, accepted history, native fallback and transactional publication.
- [Diagnostic experiments](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/diagnostic-experiments.md): teacher history, independent stop-on-reject, same-request reference and separate diagnostic persistence.
- [Configured desktop](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/configured-desktop-session.md) and [saved comparisons](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/configured-saved-comparison.md): real Worker/ModelProcess ownership, typed runtime outcomes, cold field comparison and provenance limits.
- [Domain density](https://github.com/MrScripty/Rheon/blob/bc2f472787060d9a69cb619ed2c2fea10fd754b6/docs/api/domain-density.md): cubic density constructor and higher application edit limits; allocation and model bounds remain separate. General `GridGeometry::new` still permits explicit spacing, so cubic cells remain an enforced adapter/data invariant.

The old residual probe includes only desktop `engine.rs` and `scene.rs` through
its build script. The new engine imports `model_process::PredictionFailure` and
typed recording/context owners. Its request exposes `context`, `initial_guess`,
`residual`, `viscosity`; the old probe reads `step`, `scene`, `base`, `residual`.
Changing only `RHEON_COMPARISON_SOURCE` will therefore not migrate this consumer.
This is source evidence, not a completed compile test. Prefer an owned supported
boundary over expanding the lab's private-module include graph.

Upstream still lacks the lab's `solve_rhs_with_initial_pressure` entry point.
The historical b851b38 patch passes `git apply --check` against the new tree;
it was not applied or built. Keep the historical patch provenance immutable.
Determine its remaining real consumers before either preparing a separately
identified new variant or replacing it through an adequate public API.

The legacy v2 seven-channel deployment and new generic v1 adapter are separately
supported by the target loader. Generic output reconstruction can add prior for
a correction; the old bridge already returns full pressure. Double addition
would be a migration bug. Neither port declarations nor fixture tests qualify
the later iterative residual model automatically.

The target's measured-bundle handoff says the original aa82 deployment was not
available in its implementation environment. It **is** present locally at
`runs/live-ai-comparison/deployment-2`. We verified the descriptor, checkpoint,
normalization, bridge, history and receipt hashes against the descriptor, and
the installed interpreter exists. See [availability receipt](measured-bundle-availability.json).
We did not rehash the full 5.74 GB installed runtime, execute that model or claim
new parity/accuracy. Its known negative numerical results remain historical.

## Tuldok: relevant new behavior and missing local features

New main adds operation negotiation, inert project-v2 extensions, bound authored
contributions, Model/RecipeMaterial/Study/Trial owners, task evidence/checkpoints,
candidate packaging and experiment progression. These can replace some of the
lab's manual progress comparison, but need a truthful pressure task/evaluator.
An opaque external Run artifact is not automatically a qualified checkpoint.

Exact target contracts:

- [Model development](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/model-development.md) and [experiment lineage](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/experiment-lineage.md).
- [Project versions](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/projects.md), [extensions](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/extensions.md), and [operation negotiation](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/agent-operations.md).
- [Captured CPU profile](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/captured-cpu-runtime.md): CPython3.12 CPU runtime with a 2 GiB snapshot ceiling; this cannot be used as a new name for our measured CUDA installation.
- [Explicit storage migration](https://github.com/MrScripty/Tuldok/blob/300f12df6f48e028fd9379911703f7c89337952a/docs/contracts/workspace-storage.md): recognized schema 8→9→10 transactional upgrade; stopped writers and complete verified backup; no startup migration/downgrade.

The running service is PID 11464, port 8091, using
`runs/live-integration/Tuldok/app.py` at d3514c60, and the primary Tuldok `.venv`.
We changed neither running source, installed environment, process nor catalog.
That service has four commits after its f7543fb7 base:

- 9eeb6535: live project activity and source refresh.
- b881ebd2: measured CUDA/CPU managed Torch execution.
- d852e219: dispatch host identity selection.
- d3514c60: aggregate runtime inventory hashing budget.

Their runtime/activity modules are absent from new main. Real inert calls to
the new reader reject the current lab manifest's `activity_path` and both
`managed_torch_python_v1` recipes with `runtime_device`. Removing that field
still rejects the runtime profile. This is a concrete incompatibility, not
an assumption about GPU availability. See [contract probes](contract-probes.json).
Historical saved project/material reads must be tested as well as new writes.

The existing schema 8 catalog was read through its current Metadata owner:
ID a7da4a8de53541a2a0e2aad08005bb71,120 Projects, 160 Runs:
158 completed, one interrupted, one failed. No queued/running state was observed
in those documents; this observation does not fence writers during a future
backup. New tables start empty after upgrade. Do not invent old Study/Trial
ancestry or automatically backfill numerical acceptance into them.

Legacy agent discovery bytes match between the running source and new main,
digest cb070a69ebd35b48e4f95a57ca01647c75bd7567521429e49efc3a3a1daebae8.
Nine upstream negotiation tests passed in 23.57 s using the existing lab Python3.12
environment with jsonschema4.26.0 and host loopback fixtures. The first test
attempt using the app environment failed because its new JSON Schema dependency
was missing, and sandboxed fixture sockets were disallowed. Both outcomes/logs
are retained; no packages were installed. Provision a separate new application
environment rather than modifying the live process's environment in place.

Project v2 is not a drop-in switch: its static extension snapshot currently
does not offer v1's data-import, recipe-creation and external-evidence-preview
actions. Preserve current live v1 behavior before opting into a new extension
snapshot. New model workflows and runtime migration are separate acceptance
claims; CPU scalar examples do not establish pressure-model or CUDA support.

## Recommended sequence

The canonical milestones and gates are in [the plan](../plan.md): preserve and
qualify CUDA/live-project support on new Tuldok first; migrate the lab/Rheon
boundary with one recorded frame; rehearse retained-state upgrade; then cut over
and adopt truthful typed model/experiment inspection. Keep the old service until
these gates pass, and preserve source/model identities in every new report.
