# Bounded global pressure model and generic Tuldok smoke

**Plan status:** `Active`

**Current phase:** Completed scientific smoke and independent evidence review

**Next slice:** Parent review of draft PR; diagnose collapsed correction before proposing more training

**Acceptance status:** `pending`

**Execution ledger:** [execution-ledger.md](execution-ledger.md)

**Issues:** [issues.md](issues.md)

## Objective

Implement the reviewed small global pressure-correction model as a generic Tuldok
CPU project, fixed200Adam updates and exact trained-byte fresh-process restore.
Use only verified complete slice1 development trajectories; retain sealed final
reservations and all historical failures. Parent accepted exact PR3 dec82a2 and
separately authorized this slice. No final/full data, expensive/GPU training,
upstream Tuldok/Rheon edits or solver-gain claim without measured qualification.

## Objective Acceptance

| ID | Criterion | Kind | Environment | Mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Measured flat shard/input bounds and source/capture lineage; cross-shard duplicate/relabel/crop/test exposure rejected | integration | required-real verified384-frame CPU development capture | automated | satisfied | [Data](reports/data.md), [receipt](reports/data.json) |
| T2 | Fixed10channel architecture below1M parameters/8MiB; full stencil/gauge/loss gradients and distant dependence independently qualified before training | focused | required-real eager Torch CPU and hand/native oracles | automated | satisfied | [Model](reports/model.md), [qualification](reports/model-qualification.json) |
| T3 | Actual immutable CPU runtime captured in a task-local catalog; generic project/recipe/preflight binds exact sources/data/runtime and one-use grant | integration | required-real Tuldok generic catalog/service/CLI | automated | satisfied | reports/runtime.md |
| T4 | One200-update Adam Run using train-only scales/sample order, bounded resources; actual drained successful artifact custody and200-update final weights | system | required-real captured CPU fresh child | automated | satisfied | reports/training.md |
| T5 | Fresh restore uses exact trained Run artifact bytes, zero optimizer steps and bitwise predictions; validation perturbation cannot change normalization/training | system | required-real same captured CPU runtime | automated | satisfied | reports/restore.md |
| T6 | Development-only teacher-state native correction/fallback diagnostics preserve every case and report quality/cost/poor outcomes honestly; no final rollout qualification | system | required-real pinned isolated native CPU | automated | satisfied | reports/physics.md |
| T7 | Independent numerical/standards source/artifact/evidence review, affected checks and exact pushed draft PR | release-artifact | representative independent GPT6.1SolHigh and real GitHub | either | pending | reports/review.md |

## Binding Decisions

Allowed tracked writes only `experiments/global-pressure-v1/training/**`,
`probes/global-pressure-v1-training/**`, `docs/plans/global-pressure-training-v1/**`
and new `.github/workflows/global-pressure-training.yml`. Slice1 capture/API
source/patch/inventory, original design/reservation, historical probes/receipts,
Tuldok and upstream Rheon remain unchanged. Generated data/runtime/catalog/private
credentials/native targets/receipts stay task-created ignored runs or /tmp.

Data owner: training/data.py and test_data.py. Model owner: training/model.py,
train.py and test_model.py. Main owns generic project/CLI/config/README/new CI,
plan/evidence/Git and actual runtime/catalog/admission execution. Native evaluator
owner uses only the new isolated probe/evaluator namespace. Reviewers read-only.
No source overlap, no worker commits/push, main serializes shared publication and
qualified training. Actual code and compact receipts are committed before claiming
revision-based numerical provenance; no private token, full runtime or raw data in Git.

Immutable development source is source56b2507 plus its verified complete manifest
SHA256dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d.
Only12train+4validation ×24frames, existing16³sealed rectangles. Pack losslessly
original RHS/reference pressure/previous accepted reference pressure and physical
conditioning, retain exact raw lineage and whole-history identities across shards.
Each flat source input/output <=32MiB, inputs aggregate <=256MiB, total capture
<=320MiB, no compression-based admission. Training outputs aggregate <=32MiB;
weights <=8MiB. Actual measured sizes, source/native hashes and leak checks precede
recipe assembly/admission. No cropping, unreserved lineage or final references.

Architecture/optimizer/loss/normalization exactly follow accepted design
[data-and-model.md](../global-pressure-correction-v1/data-and-model.md): eager
multiscale3D encoder[8,16,24,32],48bottleneck with pooled broadcast, decoder
[32,24,16,8],10channels, gauge-fixed float64 correction/full operator loss,200Adam
updates/batch2/seeds20261014/20261015, no hyperparameter/step/checkpoint selection.
Validation remains observational. Positive train RMS provenance only; labels/current
pressure never enter inputs. No JIT/compiler/network/GPU in captured child.
Torch/BLAS one thread;600s child wall enforced by existing custody. The4GiB
RAM requirement is a measured process high-water stop/acceptance budget: check
RUSAGE_SELF before updates, every20 updates and before successful publication.
It is not an instantaneous hard or aggregate/adversarial containment claim. Pre-training
finite-difference/operator/global-dependence qualification is required. A disposable
small arithmetic fixture may check validation invariance without submitting another
scientific200-step train; full train follows one fixed sequence only.

Use actual Tuldok mergedPR58 exactf7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6
from isolated read-only reference; preserve existing /workspace/Tuldok checkout.
Generic readers/client/runtime capture/run-artifact custody own execution, never
copied executors or mandatory Rheon adapters. Owner-local real catalog/service
scope is explicit, not hosted/cloud qualification. Independently review exact
preflight material before owner one-use grant; current user authorization permits
this bounded campaign action, no invented extra confirmation gate. If the real
catalog/runtime/grant is unsupported or refused, stop training and report exact
blocker while finishing source/data/model checks.

Development solver diagnostic predeclares32 teacher-state cases: frame0 and23
of every16developmenttrajectory, matching bounded prediction witness. Each case uses reference RHS/prevelocity and
previous reference pressure; candidate gets <=32warmPCG then one reset FROM p_base<=3000,
unchanged residual/divergence/wall/flux/Courant gates and full timed fallback cost.
This does not qualify autonomous rollout stability or finaltest/generalization.
Model training usefulness may fail; preserve outcomes and do not add steps/restarts.
Neither standalone prediction quality nor a fallback-completed case qualifies a
speedup. Global model training may complete even if physics diagnostics are poor.

## Simplicity And Ownership Review

**Applicability:** `applicable`

- Independent concepts and dimensions: Data custody, eager model/loss, generic execution and native physics own separate boundaries.
- State, identity, value, time, policy, and mechanism: SHA256/commit select bytes; trajectory selects split; physical dt and elapsed execution differ; fixed200updates is scientific policy.
- Caller and composition-root knowledge: Project binds actual source/input/runtime references; model never needs catalog credentials or native executor internals.
- Representative change paths and forced owners: Packing changes requalify data; model/stencil changes requalify gradients; recipe/run changes requalify generic admission.
- Stable Interfaces versus hidden knowledge: Flat numeric arrays, closed metadata, eager state inventory, explicit generic argument/output declarations and native initial-pressure API remain visible.
- Independent evolution, testing, failure, and replacement: Shard reader refuses leaks; restore refuses byte drift; native solver refuses invalid state; independent oracles never repair outputs.
- Necessary complexity and containment: Lossless lineage, typed inventory, captured runtime and artifact custody are required, confined to new namespaces.
- Deletion and cumulative machinery result: No new registry, trainer scheduler, executor, native runtime requirement or schema is added to Tuldok.

## Milestones

| Unit | Write owner/set | Gate | State |
| --- | --- | --- | --- |
| Data | Worker: newtraining/data.py/test_data.py only | T1 | Complete |
| Model | Worker: newtraining/model.py/train.py/test_model.py only | T2/T4/T5 | Complete; negative learned result retained |
| Generic execution | Main: newtraining project/CLI/config/README and tasklocal runtime/catalog | T3–T5 | Complete |
| Native diagnostic | Worker: newisolated probe/evaluator only | T6 | Complete; all32 require fallback |
| Review/publication | Independent reviewers read-only; main docs/newCI/Git | T7 | Planned |

## Blockers

No implementation, data, runtime, admission or restore blocker remains. One fixed
200-update Run and one zero-update restore completed/drained with exact trained
artifact parity and actual catalog reopening. Model usefulness failed: all32
predictions equal previous pressure; every short solve needs full fallback.
Independent final evidence review is clear. Exact pushed draft PR/CI and parent
integration review remain the publication gate.
Final trajectories, geometry/regime suites and full/GPU training remain sealed;
this smoke does not authorize those next steps.

## Re-Plan Triggers

Bounds exceeded, incompatible generic runtime/catalog/grant, bad source/hash,
failed stencil/gradient/global dependence, accidental final exposure, invalid
reference/native commit or resource overrun stops affected work. Preserve exact
failure/partial evidence; no step/architecture/physics tolerance retuning.

## Repository Isolation

Parent accepted PR3 exactdec82a2 with finalCI; main marked ready and normal merged
at21a4d234b2c97ca9e60a4eba43100b5e59726bd2 on2026-10-14T10:24:23Z.
New branch feat/global-pressure-training-v1 derives from that remote main. Main
owns writes/push/draft; parent owns later integration/merge. No worktree/history
rewrite/tag/outreach/upstream edits. Retain branch until parent disposition.

## Final Acceptance

- Acceptance status: `pending`
- Final status: `Active`
