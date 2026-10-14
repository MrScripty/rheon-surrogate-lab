# Actual zero-update cold restore

The separately reviewed, action-time-approved restore completed in a fresh captured CPU process with **zero new optimizer updates** and drained custody. Authoritative preflight matched the full offline candidate exactly before the one 600 s grant. There was one stable request key, one new Graph, one grant and one Run; no retry or second replay.

## Exact identity and lifecycle

- Restore Run: `265c9574c06048d8bc91daa1cf8eab02`; source training Run: `d73bc2976c4f442f8aefc8f7242bd252`.
- Scientific source: `fd3a6009e5805978a29ed1df7389ad226a40a914`; actual restore material: `4265f5157c7cfaf1759ae2f501956cefc9912b430a93f1a90d9706792674360e`.
- Restore Graph: `dfbca06c9b9a42fdb1f435990de2a394`, revision 1; recipe `1668672332b506f75f942e1bc64fab06c48fcdf375650ac1ccd712aa087b03c4`.
- Existing CPU snapshot: `6367d0fa7eb989d53cc49d6ed47f937a4f63e68655de9acaf84c7f7d94838c77`; existing catalog `619034b1109f45faac099835ee9a9065`.
- Catalog Run/grant/Graph counts each changed 5→6. The exact 600 s grant was consumed once.
- Temporary existing-owner session `f17cd4fafc5e47dab2c2fc8c086b013f` had the pinned 28,800 s lifetime and was revoked after Dataset/supervisor closure/drain. A separate PostgreSQL READ ONLY transaction verifies **zero live sessions and zero usable credentials across all actors**. No new actor, agent or bearer was issued. Consumed grant remains a non-reusable audit record.

## Exact restore acceptance

All **18 artifacts / 20,609,575 bytes** were collected through the actual generic artifact API and matched by hash, bytes, role and per-file caps. All **16 inherited artifacts** match source bytes exactly: weights, predictions, splits, physical scales, all 11 numeric states and diagnostics. Final weights still equal checkpoint200 byte-for-byte; checkpoints remain observations with no selection/resume.

The frozen restore source loads only final200, constructs no optimizer, recomputes the 32 retained teacher predictions, and requires closed-key/dtype/shape/body parity through `validate_prediction_parity` before successful child completion and accepted Run publication. Pressure is float64 `[32,16,16,16]`, row indices int64 `[32]`; every recomputed array byte must equal the retained output. This execution passed that contract. The external read-only audit checks published bytes/source/custody and runs no second model replay. Local staging copies the predictions archive before parity; successful publication still requires parity success.

`receipt.json` and `summary.json` are new truthful restore artifacts. The receipt reports mode restore/status complete,0 new updates / 200 trained updates, original source Run/custody, unchanged final weights/checkpoint metadata and expanded input-evidence hash `9411df6a9d99461465dcb207d1512d1ca002db4a8c36e299f143c9297316321c`. Its role is `evaluation_result`. Summary title is “GroupNorm pressure correction: restore”. They are intentionally not byte-identical to training receipt/summary. Validation/history/384-row diagnostics are inherited training observations, not freshly evaluated restore measurements.

## Cost and resource scope

Enclosing Run: **12.929004 s**. Created-to-finished including the queued interval: **13.119643 s**. Recipe clock: **1.1796124529973895 s** after Python imports/bootstrap and before final receipt/publication. Owner procedure: **25.059683751002012 s** through artifact collection and **25.177940639001463 s** through final shutdown/revocation, excluding later separate access audit/docs. Recorded 32-case inference totals **0.32481544799520634 s**, excluding native verification/fallback. These clocks establish no solver gain.

Observed process peak RSS: **473,665,536 bytes**. Actual Python 3.12.14 / Torch 2.8.0+cpu / NumPy 2.3.5, one Torch/interop thread and deterministic CPU settings match the approved runtime. Child wall 600 s, observed 4 GiB high-water,256 MiB input / 320 MiB capture / 32 MiB output policies remain binding; RAM is not OS containment. Actual capture is76,332,833 bytes / 35 files and inputs 72,204,703 bytes. Outputs fit the conservative 21,116,430 byte restore envelope.

Completed/drained custody records spawn closed, ECHILD, GO received, exit 0 and empty stderr. Full actual identities, output inventory, cost scopes, custody, access metadata and source/evidence digests are in [restore.json](restore.json). The [exact actual material](restore-material.json) and unchanged [independent pre-admission review](cold-restore-admission/independent-review.md) are public reviewable metadata. Bulk weights/data and full preflight remain in the private task-local catalog/archive; their public inventories are hash-bound.

| Artifact | Bytes | SHA-256 | Inherited bytes equal |
| --- | ---: | --- | --- |
| `checkpoint-000.npz` | 1500186 | `1b4b77cf299d798b6b787bbd12486bc8d4301d7534972c66e9dddce41e93f737` | yes |
| `checkpoint-020.npz` | 1500186 | `61ce49c91030ae2107b8b63986a5a11e4c10d941a3ac5ce542927e7e2c98bd1f` | yes |
| `checkpoint-040.npz` | 1500186 | `1ed46be82d8d6848e915921803dc71d9605e27ae44f08b315345dbd7ddaf0703` | yes |
| `checkpoint-060.npz` | 1500186 | `a48a17923ee7554b7ecb8f6a6361fc1d7903ffd0319b0c60a1b71f77206c1403` | yes |
| `checkpoint-080.npz` | 1500186 | `20bcbe8faadc5214560ab202b4daf60136d7e34f734f92b78af43976cf0280b4` | yes |
| `checkpoint-100.npz` | 1500186 | `ea033f4aba7c1d44c1b3cb3fc8e5583ff869a34c05a8a484ea0ad037d2f8b392` | yes |
| `checkpoint-120.npz` | 1500186 | `6141585f2ef4c75f8c45ee8073d38c86a1412e6179ad588c0072976b0029a46e` | yes |
| `checkpoint-140.npz` | 1500186 | `8c1d4d792ad8171c96945af5d5a5a04a3087f370f3c80b207b2f11562dada201` | yes |
| `checkpoint-160.npz` | 1500186 | `199029c43d45348f8da8120bc50d24f0350abd0203f1049f772627999bebfd3b` | yes |
| `checkpoint-180.npz` | 1500186 | `d5149bbca9e3a7850438ea7292d0336bdf4b5102bae90740ab7b9bee65efbd1a` | yes |
| `checkpoint-200.npz` | 1500186 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` | yes |
| `diagnostics.json` | 1402428 | `a94a30492f675b40fdd3c2a5ec92419dbc798f01f42c6219a3719ff7bee858b0` | yes |
| `normalization.json` | 2112 | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` | yes |
| `predictions.npz` | 1049356 | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` | yes |
| `receipt.json` | 82195 | `f83242bf89a3db7da305781c5a880c7b08d4e7a0090e8469276b67ee88cc0a0a` | new restore record |
| `splits.json` | 70478 | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` | yes |
| `summary.json` | 774 | `40cb30370999e2ccea9c8baf45d9ca99150839c57850e0af2f2e0eeacff9d553` | new restore record |
| `weights.npz` | 1500186 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` | yes |

## Remaining solver gates

Cold restore acceptance is complete. Original small CNN 0/16 native failures, original global 0/32 raw failures/constant-output diagnosis, full reset fallback work and historical modal scope remain unchanged. No post-trained amplitude probe, native candidate original-b residual/divergence/wall/flux/Courant acceptance, full fallback cost, conservation/autonomous stability, reserved final suite or GPU/full training evaluation occurred. This model is not a qualified accelerated solver.

The next proposed boundary is independent review of a new lab adapter for this model’s final prediction/receipt identity and the unchanged qualified native evaluator. No further execution is admitted here. See [handoff](../handoff.md) and [independent actual restore review](restore-review.md).
