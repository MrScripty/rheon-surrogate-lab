# Frozen candidate and native development-input preservation

Disposition: **local planning inputs available and metadata verified**. This audit performs no model import/load/forward, replay, native binary execution/solve, new data generation, training/selection or physical candidate evaluation.

## Base and instruction scope

The planning branch is `feat/global-pressure-groupnorm-native-dev-v1`, stacked on accepted PR6 head `9c873a0061ba398822d56977798b24377d33baf0`. Scientific candidate source is `fd3a6009e5805978a29ed1df7389ad226a40a914`. PR6 remains draft and unmerged: main reports that automatic approval review denied draft-to-ready promotion because standing authority to merge ready PRs did not authorize promoting this draft. No retry or credential renewal is part of this audit. This report makes no merge claim.

All **273 prior tracked file blobs and executable modes** match the accepted head, checked before and after this audit. No `/AGENTS.md`, `/workspace/AGENTS.md`, lab/tracked `AGENTS.md` was found. The adopted standards remain `188beda1fa477d21d576c233dd8c7f4c4c267d23`; no authoring skill or standards engine attestation is implied. Detailed identities, proof contracts and evidence limits are in [preservation.json](preservation.json).

## Exact frozen candidate and restore proof

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| Final weights, also exactly checkpoint200 | 1500186 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |
| Restored predictions | 1049356 | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` |
| Restore receipt | 82195 | `f83242bf89a3db7da305781c5a880c7b08d4e7a0090e8469276b67ee88cc0a0a` |
| Original physical normalization | 2112 | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` |
| Persisted development splits | 70478 | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` |

Final weights/checkpoint200, predictions, physical scales, splits and diagnostics match originating training bytes exactly. Only final200 is the candidate; intermediate states cannot become alternate selections. The current scientific model/train/config/qualification files match their source-commit blobs. Their three original helper dependencies remain byte-pinned. Stdlib ZIP/NPY inspection verified all **38 weight tensor headers/raw-body hashes**, pressure `<f8[32,16,16,16]`, row indices `<i8[32]` and their receipt-bound body hashes. No tensor was loaded into model state; pressure values were not evaluated physically.

Retained completed/drained restore Run `265c9574c06048d8bc91daa1cf8eab02` reports zero new updates and 200 trained updates from originating Run `d73bc2976c4f442f8aefc8f7242bd252`. Fresh prediction parity is supported by the successful historical execution of frozen `validate_prediction_parity` after recomputation and before completion/publication, plus exact restore receipts/source/custody evidence. Archive equality by itself is not a fresh-forward proof, and this audit performs no second inference. See the unchanged [actual restore review](../../global-pressure-groupnorm-v1/reports/restore-review.md).

## Development-only population and available raw files

The frozen population is **32 teacher snapshots**: frame000 and frame023 from each of 12 training and four validation parents, in the original 384-frame packet2 lineage. There are 24 training cases and eight validation cases. Receipt case IDs and decoded integer row metadata agree exactly. Packet2's five input file sizes/hashes, whole parent identities, origin and split hashes and native residual anchor `26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8` remain unchanged.

All **192 selected raw array references** are present with exact bytes/hashes: original RHS, current reference pressure, previous accepted reference pressure (`p_base`) and all three preprojection MAC velocity components for every case. Geometry/density/dt originate in the immutable manifest. At frame000, `p_base` is the initial zero field; frame023 uses the preceding accepted reference pressure. These are teacher states, not autonomous candidate history. No reserved final arrays are needed, read or generated.

## Unchanged native dependency and fair baseline aliases

Qualified local native6 binary:

`/tmp/global-pressure-training-native-build6/rheon-global-pressure-training`

SHA256 `3a06964ed4121cde5394e114304abd3f50a569ed3345682d5dccd20c740ba3df`; build receipt `70b0e33f5ec13c289e5afd0c38df783655936e06c2dfd32f0509e94d2aef8307`. The actual Cargo metadata hash, nine-file lab source closure and all **746 prepared native file contents/Git modes** match their immutable declarations. Native base `2209012bc5f902693d0f695ec1c2d854126af317`, patch `b851b38d2d03280220bc4648cd4d721b2e796786fb18d8188dc335336fe2a998` and prepared tree `5a5de4635d3970c9b99e620d9fff2d9a23c55307a9e335f7473e7f95a0c7a4ab` remain pinned. This verification reads files only and does not execute the binary/helper.

Source inspection confirms full-budget temporal PCG starts fresh from `request.previous_pressure` with a 3000 limit. Learned eligible warm work has a 32 limit, followed at most once by a fresh 3000-limit reset from the same `p_base`. The separately named zero-Jacobi baseline starts from zero. The temporal32/reset path remains explicitly labelled a bounded diagnostic, not the full-budget temporal comparator. Full original-b action includes physical cell zero; native physical residual/divergence/wall/flux/Courant gates remain unchanged. These source facts do not predict the GroupNorm candidate's acceptance or cost.

## Required new adapter and proof contracts

The unchanged original Python evaluator deliberately accepts only `rheon-global-pressure-training-receipt-v1`; the candidate uses `rheon-global-pressure-groupnorm-training-receipt-v1`. A future **new lab adapter** must validate the GroupNorm model/receipt/source/final200/restore proof and exact prediction/body/row hashes before calling the unchanged native case protocol. The original evaluator must not be edited or relabelled to accept this candidate.

The proposed file contracts must reject source/artifact/lineage corruption before scientific execution; retain all 32 cases and every proposal/warm/reset/terminal outcome; bind original RHS/reference/p_base/preMAC/units/geometry; and recheck actual native content before/after execution. Native scratch must reset while preserving original-b tolerance scale and p_base semantics. True temporal full-budget and zero starts remain fair separate comparators. Future raw predictions, matched-quality native correction, inference/verification/fallback/setup and enclosing costs must be reported completely. Teacher snapshots cannot establish autonomous stability, conservation, held-out generalization or acceleration.

## Availability, preserved negatives and limits

Candidate archives, packet2, selected raw arrays, qualified native6 binary and prepared source are **currently readable locally without credentials**. No credential/session/API/database/Graph/preflight/grant/Run action occurred. The public checkout alone lacks ignored bulk weights/data/native build products; another machine requires a separately authorized exact hash-bound archive transfer and dependency/build qualification. This is local material availability, not a public reproduction claim.

The original CNN remains **0/16 native cases**. The original global model remains **0/32 raw**, with all 32 corrected cases using reset fallback and 288 training plus 96 validation zero corrections in its accepted diagnosis. Original failed weights, reports and reservation bytes are preserved. Lower GroupNorm development loss, active features and cold restore remain distinct from its still-unperformed physical qualification.

This audit supports a reviewable **planning** packet. It grants no execution authority, runs no model/native evaluation and claims no solver usefulness or acceleration. Native qualification, rollout/conservation, reserved final suites and GPU/full training remain separate gates.
