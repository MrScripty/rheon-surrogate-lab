# Proposed local native development evaluation

Status: **planning only; implementation and execution are not authorized**.

This document was prepared on `feat/global-pressure-groupnorm-native-dev-v1`,
stacked on accepted PR6 head `9c873a0061ba398822d56977798b24377d33baf0`.
PR6 remains draft and unmerged. Automatic approval review denied the draft-to-ready
transition because the standing merge-ready instruction did not clearly authorize
promoting a draft. Main owns the explicit approval, readiness, merge and eventual
retarget to `main`. Branch availability authorizes this document, not those actions.

The candidate is the already trained GroupNorm **final200**, scientifically frozen
at `fd3a6009e5805978a29ed1df7389ad226a40a914`. Training Run
`d73bc2976c4f442f8aefc8f7242bd252` and cold restore Run
`265c9574c06048d8bc91daa1cf8eab02` completed and drained. The restore performed
zero new updates and passed its source-enforced 32-prediction byte-parity check.
The eleven states are observations; none becomes an alternative candidate here.
See the retained [training report](../global-pressure-groupnorm-v1/reports/training.md)
and [restore report](../global-pressure-groupnorm-v1/reports/restore.md).

## Read-only findings and available local tools

No model import, forward, replay, optimizer, native invocation/solve, service,
preflight, Graph, grant, Run, final/GPU access or credential operation was performed
for this handoff. Inspection used source text, SHA-256, JSON, ZIP/NPY headers and
raw payload hashes. Existing scientific sources, artifacts, evaluators, Tuldok and
Rheon were not edited.

The current local files are sufficient for a later, separately admitted diagnostic:

- Python `/workspace/task-torch/bin/python`, whose retained qualification records
  identify Python 3.12.14 and NumPy 2.3.5. The proposed adapter needs stdlib/NumPy;
  it must not import Torch or a model/trainer module or recompute predictions.
- Qualified executable
  `/tmp/global-pressure-training-native-build6/rheon-global-pressure-training`,
  with adjacent `build-receipt.json`, `binding.json` and `cargo-metadata.json`.
  Its bytes were inspected, not executed. Archived identity declares one native
  thread. Prepared patched source remains under
  `runs/global-pressure-v1-dependency/rheon`.
- Immutable original Python evaluator
  `experiments/global-pressure-v1/training/evaluate_native.py` and its source/build
  dependency inventory. Its `evaluate` command hardcodes the **original** receipt
  schema and therefore cannot directly admit the GroupNorm receipt.
- Retained artifact directories
  `runs/global-pressure-groupnorm-v1/training-stage/artifacts` and
  `runs/global-pressure-groupnorm-v1/restore-stage/artifacts`, plus their already
  collected manifests/custody and accepted public inventories.
- Raw reference capture
  `runs/global-pressure-v1-capture/committed-development-complete`. Packet2 NPZ
  inputs alone are insufficient: native evaluation also needs the original
  preprojection MAC faces, which are absent from the packed training fields.
- Original packet2 at `runs/global-pressure-v1-training/development-packet2` can
  supply the authoritatively bound native input-residual witness for an explicit
  zero-residual retention rule. This is data inspection, not a model forward.

The actual restored predictions archive has precisely two numeric members. Its
pressure is `<f8 [32,16,16,16]`; indices are `<i8 [32]`, ordered
`0,23,24,47,...,264,287,288,311,312,335,336,359,360,383`. All 38 numeric weight
members match the retained descriptor and raw tensor hashes. Restored weights
equal checkpoint200, and all 16 inherited artifacts match training bytes exactly.
These are file-integrity observations; they are not a new replay or live catalog
custody check.

## Exact inputs to freeze

Paths are local locators, not authority. Before any implementation execution,
create a reviewed, immutable execution-freeze document binding every path to its
bytes, approved source identity, role and resource limit. At minimum it binds:

| Object | Exact SHA-256 |
| --- | --- |
| Training receipt, 80,551 bytes | `88abf1b2eaff46da583588c422166f0db4e63eb8aaa7790ed585f88d1a5af1ed` |
| Restore receipt, 82,195 bytes | `f83242bf89a3db7da305781c5a880c7b08d4e7a0090e8469276b67ee88cc0a0a` |
| Final weights/checkpoint200, 1,500,186 bytes each | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |
| Predictions archive, 1,049,356 bytes | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` |
| Raw prediction pressure, 1,048,576 bytes | `cc70f931c156c96f53fff8c606ffe039a7e3f294320731e09ac0ae44d9f99cc7` |
| Raw row indices, 256 bytes | `17acc08ccda81f2e47be3a0229106f0b522dc3f941a72015f8a677c314fc2eb7` |
| Physical normalization | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` |
| Whole-history splits | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` |
| Original complete capture manifest/origin | `dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d` |
| Source witness recorded by both receipts | `549ad91612e7e2607e902bec3ccbf66366f6df1910c5b1755383d691b1f025f6` |
| GroupNorm model source | `8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c` |
| GroupNorm trainer source | `f46ef3a589a0e4e38913d683c9c0b1d68c3b4c98a7f04f03d03c8939ac160dbb` |
| Existing native binary, 952,728 bytes | `3a06964ed4121cde5394e114304abd3f50a569ed3345682d5dccd20c740ba3df` |
| Existing native build receipt | `70b0e33f5ec13c289e5afd0c38df783655936e06c2dfd32f0509e94d2aef8307` |
| Existing Python evaluator | `76fefe5da899f27ba0999a57b4d62e64fa41eea241dd131a8600c408573cd6a0` |
| Previously captured CPU runtime | `6367d0fa7eb989d53cc49d6ed47f937a4f63e68655de9acaf84c7f7d94838c77` |

Also bind the new adapter/test/source commit and hashes; GroupNorm configuration
hash `e5f6ac584ffe9eb067bbb5c6cb192ac8d399bb78848beafeb1d2aa5588998683`;
both complete artifact inventories and historical custody/manifests; all 17
originating training-artifact bindings recorded by restore; eleven checkpoint
hashes; the capture's reservation and complete 16-parent/24-frame mapping;
all raw arrays actually consumed; native binding/metadata/source inventory;
native base `2209012bc5f902693d0f695ec1c2d854126af317`, patch
`b851b38d2d03280220bc4648cd4d721b2e796786fb18d8188dc335336fe2a998` and prepared tree
`5a5de4635d3970c9b99e620d9fff2d9a23c55307a9e335f7473e7f95a0c7a4ab`.
An absent artifact, moved unsupported build closure, hash drift or missing
authorization stops the affected work. It does not authorize download, rebuild,
retraining or a substitute candidate.

For the zero-residual rule, also bind the original reader source
`experiments/global-pressure-v1/training/data.py`, SHA
`c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd`, all five
packet2 files already recorded in both receipt closures, and native residual
attestation anchor
`26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8`.
The packet's train shards have hashes
`6b3d0c4a57df0198b4fe13f7e887b8fa238fbf3fd34877f9dc598c157d2b55dc` and
`d6d7980a51c17b79a6567c870aca9001108779798cee490101429e00814a9f02`; validation is
`17a9faa84723208d1bd9d6a3a8b1f5921806e93b85d8983d2041404fb295855d`.
Splits/origin hashes are in the table above. A self-declared residual hash alone
does not establish this anchor or consistency with the current raw capture.

## Minimal proposed lab adapter

Proposed new source location:
`experiments/global-pressure-v1/groupnorm-native-dev-v1/evaluate.py`.
**This file does not exist yet and is not implemented by this planning slice.**

The lab owns the new GroupNorm artifact decoder, local launcher and report identity.
It should reuse the byte-pinned original evaluator's population/capture/NPY
decoding and request conventions where sufficient, and the unchanged qualified
native binary for all physics and PCG. Pin helper bytes before executing an import.
Do not modify the original evaluator to accept another schema, fabricate an
original-schema receipt, change native physics, add a Tuldok executor or introduce
a model registry. Model weights are decoded for identity/integrity only, never
loaded into a model.

Admission must enforce:

1. Closed, duplicate-free JSON with finite numbers and exact primitive types,
   including true integer update/frame/count fields and explicit extra-field
   rejection. Both receipts must use
   `rheon-global-pressure-groupnorm-training-receipt-v1`, status complete and
   architecture `rheon-global-pressure-groupnorm-model-v1`, including fixed GN4,
   epsilon `1e-5`, affine false and unchanged 372,377-parameter descriptors.
   Training is mode train/200 new and 200 trained updates; restored is mode
   restore/0 new and 200 trained updates. Reject original-model receipts.
2. Same frozen source/config/runtime/data/scales/splits/sample order/witness closure
   in the two receipts, allowing only their documented input-evidence difference.
   Restore's originating training Run/catalog/custody must match the accepted
   training evidence. Its own accepted restore Run/custody remains a separate
   historical fact. Do not infer a new managed Run from local paths.
3. Complete historical artifact manifests and exact inherited bytes. Final weights
   must equal checkpoint200; all eleven states remain observations, without
   model loading, selection, resume or validation comparison. Check final weight
   NPZ against the approved 38-tensor descriptor inventory and per-tensor hashes.
4. Bounded immutable-byte NPZ decoding: exact member names/count, unique ZIP names,
   uncompressed/unencrypted members, bounded headers and expanded lengths,
   true-integer shapes, declared little-endian numeric dtypes, C-order bodies,
   no pickle/object payload, complete raw hashes and exact byte lengths. For this
   candidate require pressure `<f8 [32,16,16,16]` and rows `<i8 [32]`; do not weaken
   the admitted witness to the older helper's optional int32 case.
5. Fixed population derived independently from the reservation/capture:
   train parents `smoke-train-100000` through `100011`, validation parents
   `smoke-validation-100100` through `100103`, each whole history 24 frames.
   Only frames 0 and 23 form the existing 32 witnesses. Bind their textual IDs,
   split/geometry/time/units, exact numeric indices and pressure raw hashes.
   The 32 snapshots are not 32 independently reserved trajectories. No filtering,
   replacement, alternate frame, subset or final/full population flag is allowed.
6. Manifest identity and required raw data hashes: original RHS, reference pressure,
   preprojection MAC u/v/w and actual previous accepted pressure (initial pressure
   at frame0, frame22 pressure at frame23). Preserve lineage checks. The source
   capture's complete whole-parent eligibility is retained; native execution
   consumes only these predeclared snapshots, not a new 384-case solve.
7. Before first native request and after final completion, verify the existing
   binary/build receipt/binding/Cargo metadata/prepared-source/full source closure.
   A later evaluation may invoke its existing `--identity` interface with a 10 s
   bound. Planning and the proposed offline verify mode invoke no native program.

The existing native `corrected` function has no explicit base-retention branch
for an exactly zero input residual. Do not claim it supplies that branch, and
do not require a native rebuild merely to add a lab input-disposition rule.
Propose the following narrow adapter-owned construction and focused regression:

- Read the original packet through its byte-pinned NumPy/stdlib reader with the
  required approved residual anchor; do not import its model or trainer. Match
  each selected packed RHS/base/reference and physical fields to the current
  capture and actual previous-pressure lineage before inspecting its residual.
- If the bound `native_input_residual` is exactly zero in every cell, require the
  retained full prediction to equal that actual previous pressure **byte-for-byte**.
  The accepted GN predictor's zero-input bypass makes this a checkable hypothesis;
  it is not assumed solely from a schema name. Then supply that same base pressure
  as the candidate by construction and record `zero_residual_base_retained` as an
  **adapter** disposition, separately from the unchanged native outcome.
- The native result must still reconstruct original RHS from current MAC faces,
  qualify the reference and check all current full-original-b/velocity physics.
  For a zero-residual disposition require the returned raw temporal residual to
  be exactly zero and raw candidate/temporal quality to agree. A mismatch stops
  for review; base retention is not a substitute for those checks.
- A focused contract fixture must prove zero residual retains identical base,
  reject a different stored candidate, reject a missing/unbound/stale attestation,
  and leave a nonzero residual on the ordinary candidate path. Fixture proof is
  not an additional scientific population or a historical zero-input observation.

If these proofs cannot be established, stop and report unavailable conformance.
Do not silently rewrite a different prediction, shift pressure/RHS, waive native
quality, select another case or declare native6 itself changed. No new native API
or solve is authorized in this planning phase.

Future source tests must exercise isolated malformed receipts, old model identity,
bool/int aliases, extra/duplicate NPZ members, shape/dtype changes with identical
body bytes, unknown/misaligned witnesses, wrong previous-pressure lineage, path
escape/overwrite, missing native closure, and deadline/stream/drain failure. They
must not train or substitute scientific cases. Independent review must clear the
exact new adapter and freeze before any native evaluation is requested.

## Proposed local modes and limits

The proposed CLI has exactly two modes. `verify` reads and validates frozen local
bytes only: no model or native invocation. `evaluate` repeats admission, performs
one serial pass over the same 32 native requests and records every outcome. It
does not compute new predictions, advance state, probe amplitudes or infer rollout
stability. Post-training amplitude replay remains a separate pending model-forward
authorization, not an implicit part of this artifact-only adapter.

| Boundary | Proposed fixed limit |
| --- | --- |
| Offline verify work | 60 s total, identity report at most 1 MiB |
| Local evaluate workflow | 600 s total, including admission, child work, drain and final publication |
| Native identity / each teacher child | 10 s / 60 s, also bounded by remaining workflow deadline |
| Concurrency | One adapter, one owned native child at a time, one native CPU thread |
| Memory | 4 GiB observed process high-water stop for adapter and native children; no OS/aggregate-containment claim |
| Input bytes read | At most 256 MiB across declared local input inventory; no unrelated artifact traversal |
| Native request / captured streams per child | 1 MiB request, 512 KiB stdout, 64 KiB stderr |
| Evaluation output | 32 MiB aggregate: input proof 1 MiB, population 64 KiB, journal 8 MiB, final report 16 MiB, failure evidence 1 MiB |
| Candidate policy | Exactly final200 and all 32 fixed cases; no retries or checkpoint search |

Use explicit `RAYON_NUM_THREADS=1`; set NumPy/BLAS environment thread limits to one
before import and record the observed runtime. The old wrapper uses
`subprocess.run` with timeout and post-hoc stream limits; those alone do not prove
bounded streaming or owned descendant cleanup. The new local launcher must use
one owned child group, an enclosing watchdog, bounded stream collection, observed
child/parent RSS and explicit terminate/drain on deadline or stream violation.
This is a lab diagnostic lifecycle boundary, not a new service/scheduler.

Require canonical regular inputs, no symlink traversal, and a fresh output direct
child of `/tmp`, matching the old local publication boundary. Refuse overwrite
before work. Publish the independently frozen population first, append/fsync every
case and preserve errors/partial records. A terminal error or incomplete population
must not have a `complete_diagnostic` report. Hash sources/inputs before and after;
record the new adapter and native identities with the enclosing clock and observed
resource scope. Later archival/publication of these local files is another explicit
handoff; it does not create generic Run custody.

The following is **proposed syntax only**. The adapter and reviewed freeze are
absent. `GN_NATIVE_FREEZE_SHA` must later be replaced by the exact independently
reviewed freeze hash; these commands are not authorized to run now.

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/task-torch/bin/python \
  experiments/global-pressure-v1/groupnorm-native-dev-v1/evaluate.py verify \
  --freeze docs/plans/global-pressure-groupnorm-native-dev-v1/reports/execution-freeze.json \
  --freeze-sha256 "$GN_NATIVE_FREEZE_SHA" \
  --training-artifacts runs/global-pressure-groupnorm-v1/training-stage/artifacts \
  --restored-artifacts runs/global-pressure-groupnorm-v1/restore-stage/artifacts \
  --capture runs/global-pressure-v1-capture/committed-development-complete \
  --packet runs/global-pressure-v1-training/development-packet2 \
  --binary /tmp/global-pressure-training-native-build6/rheon-global-pressure-training \
  --output /tmp/global-pressure-groupnorm-native-dev-v1-verify-reviewed1

RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/task-torch/bin/python \
  experiments/global-pressure-v1/groupnorm-native-dev-v1/evaluate.py evaluate \
  --freeze docs/plans/global-pressure-groupnorm-native-dev-v1/reports/execution-freeze.json \
  --freeze-sha256 "$GN_NATIVE_FREEZE_SHA" \
  --training-artifacts runs/global-pressure-groupnorm-v1/training-stage/artifacts \
  --restored-artifacts runs/global-pressure-groupnorm-v1/restore-stage/artifacts \
  --capture runs/global-pressure-v1-capture/committed-development-complete \
  --packet runs/global-pressure-v1-training/development-packet2 \
  --binary /tmp/global-pressure-training-native-build6/rheon-global-pressure-training \
  --output /tmp/global-pressure-groupnorm-native-dev-v1-evaluate-reviewed1
```

## Unchanged scientific paths and cost interpretation

Native requests retain original RHS and full finite-volume row0, original MAC
reconstruction, gauge, units and native correction precision. Reference-physics
failure stops for parent review. Thresholds remain residual L2
`<= max(1e-12, 1e-8*||b||2)`, predicted divergence `<=1e-7 s^-1`, actual corrected-f32
divergence `<=1e-5 s^-1`, exact zero wall-normal flux, net flux
`<=max(1e-12,1e-8*absolute_flux)` and Courant `<=0.5`. No source mean shift, RHS
renormalization, tolerance change, failed-case omission or scenario tuning.

The unchanged native evaluator provides raw reference/candidate/temporal/zero;
candidate direct acceptance or warm Jacobi-PCG at most32 followed, if necessary,
by exactly one fresh full previous-pressure reset at most3000; bounded temporal
diagnostic plus a distinct fresh full-budget temporal attempt at most3000; and
zero-start PCG at most3000. Worse-than-temporal or numerically invalid proposals
are recorded, skip warm work and use the reset path. A failed candidate remains
in the journal; it is not replaced by its fallback pressure and called raw success.
Native results must declare `state_advanced=false`.

There is **no distinct analytic-start baseline** in this qualified binary's
`--evaluate` contract. Report the actual comparator inventory. An additional
analytic-pressure construction/comparison would require a separately reviewed
scientific protocol and authorization; do not relabel zero or invent another
native mode. Preserve the original small-CNN 0/16 and original-global 0/32 negative
results and the prior full-reset costs as historical comparisons, not fresh runs.

Report each raw quality failure, proposal rejection, warm/reset solver failure,
iterations, initial-guess hash, correction/verification/solver/child wall and all
comparison work. Full-budget temporal is the fair temporal baseline, not only its
32-step diagnostic. The local enclosing clock includes input verification,
serialization, all owned native work, cleanup and publication. It excludes the
already completed training/restore and fresh model inference, because none is
performed. Recorded training/restore inference totals (`0.30471505499008344 s` /
`0.32481544799520634 s`) are historical separate scopes. Adding them to a new
native clock does not establish a fresh end-to-end solver speedup. Lower loss,
nonzero outputs, reduced iterations and successful fallback are not acceleration,
conservation or autonomous-generalization qualification.

The current binary exposes no separate initial reference/RHS/raw-candidate
physics-check timer. Mark that stage cost **unavailable**, rather than deriving
it by subtracting PCG timers. Its `native_elapsed_s` and the enclosing child wall
include **all** requested methods/comparators. The learned `corrected` path timer
begins after raw quality checks; it is not a complete candidate wall clock.
No candidate-only wall speedup follows from these fields. The recorded restore
inference cost is cross-execution accounting from an older cached result, not
fresh resident-model inference in the proposed local pass.

## Local artifact authority versus managed Tuldok execution

After separate implementation review and explicit local evaluation authorization,
the two modes above can operate entirely on already retained local bytes and the
existing binary, without renewing Tuldok credentials. Their report is a local lab
diagnostic with historical accepted custody references. It must not claim new
generic Run registration, captured execution, live owner access or fresh Tuldok
artifact custody. Local success does not consume or resurrect an old grant.

A managed evaluation is a different action. It would require a reviewed external
recipe with exact source/runtime/input/output capture, truthful typed originating
`run_artifact` bindings for trained/restored artifacts, authorized existing-owner
access, actual generic project/Graph/revision/preflight, independent exact-material
review and a new action-time execution grant. Existing sessions/credentials were
revoked and prior grants were consumed; they cannot be reused. This plan requests
none of those operations. Missing local files would stop local admission, not
implicitly authorize a catalog reopen, download, credential renewal or a managed
Run. Scientific training/restore reruns, reserved final suites and GPU/full work
remain outside this boundary.

## Pending handoff

Main owns PR6 ready/merge approval and plan retargeting. Parent must separately
authorize adapter implementation; reviewers then inspect exact source, negative
contract tests, lifecycle/limits and complete execution freeze. Only a subsequent
explicit authorization for the frozen **local32-case evaluation** may execute it.
Managed capture/access/grant/Run or any model-forward amplitude probe needs its
own stated scope and permission. None is implied by this planning document.
