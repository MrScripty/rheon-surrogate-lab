# Independent training outcome review

Status: **ACTUAL RUN / ARTIFACT / CUSTODY AUDIT CLEAR**. No blocking finding remains within this outcome-review scope. This review is not a restore, native physics, acceleration or final-evaluation qualification.

## Exact admitted observation

Run `d73bc2976c4f442f8aefc8f7242bd252` completed and drained in catalog `619034b1109f45faac099835ee9a9065`, with exactly 200 recorded optimizer updates and no retry or additional Run. It binds scientific source `fd3a6009e5805978a29ed1df7389ad226a40a914` and the admitted material `4e8b58899d7c03f074b2b5632911c539f183bb2cf57024e3809c1a07055f0668`. The fresh preflight and actual terminal Run material match that digest byte-for-byte after canonical encoding.

The reviewer used only retained JSON, hashes, standard-library ZIP/NPY-header/struct reads and Git/source inspection. No model module was imported, no trained state was loaded into a model, and no forward pass, optimizer, credential action, grant, Run submission, restore, native evaluation, GPU or final-data operation was performed. Only this review's Markdown/JSON files were written; no commit or push occurred.

## Artifact and numerical-byte checks

All **18 artifacts / 20,607,920 B** match the Run artifact list, artifact manifest, download verification, recipe roles and individual caps. The complete inventory fits the 32 MiB aggregate. Diagnostics are 1,402,428 B under 2 MiB; the receipt is 80,551 B under 512 KiB. Eleven checkpoints are each 1,500,186 B under 2 MiB. All artifact paths are canonical regular files, with no omitted or additional output.

Independently parsed every uncompressed numeric archive: **458 NPY arrays** across all eleven checkpoints, final weights and predictions. Closed ZIP names, no duplicate/encrypted/compressed members, bounded headers, C-order, little-endian dtype, exact positive integer shapes, body sizes and raw tensor hashes all pass. **4,599,596 floating values are finite**. This is byte/header verification, not reconstructed inference.

Each checkpoint has the complete 38-tensor / 372,377-parameter / 1,489,508-raw-byte inventory. Checkpoint000 matches the pretraining initialization mapping and original seeded tensor hashes. All eleven state archive hashes are distinct. Final weights are an exact byte copy of checkpoint200 and use its inventory. Only update200 remains the scientific candidate; intermediate states were neither selected nor resumed.

Predictions retain exactly float64 pressure `[32,16,16,16]` and int64 row indices `[32]`, matching the ordered original frame0/23 development witnesses. The 17 prepared typed `run_artifact` references match names/hashes from this one completed/drained Run. They are references for a future restore material, not evidence that an actual cold restore has occurred.

Final weights SHA256: `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147`. Receipt: `88abf1b2eaff46da583588c422166f0db4e63eb8aaa7790ed585f88d1a5af1ed`. Diagnostics: `a94a30492f675b40fdd3c2a5ec92419dbc798f01f42c6219a3719ff7bee858b0`. Predictions: `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86`.

## Scientific closure and observations

Config and executable source hashes remain identical to the reviewed capture. The new GN architecture/source/receipt identities, five exact executable sources, original packet2 inputs, train-only physical scales, whole-history split, native residual anchor and CPU snapshot are unchanged. Initialization seed20261014, sample-order seed20261015, batch2, fixed Adam policy/clipping and 200 updates remain bound by the captured config/source. Normalization reproduces the original 2112 bytes.

All 200 sampled row pairs select only the 288 training rows. Reconstructed int64 order bytes match `1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9`. Ten fixed20..200 history entries link their exact sampled rows and checkpoint hashes. Diagnostics contain twenty before/after batch observations, all384 ordered teacher frames and eight validation gradient cases; complete typed loss, activation, ten-channel gradient and component-gradient metadata pass. Validation remains observational only at0/200.

Stored observations show nonconstant raw heads and nonzero corrections on **384/384 frames**, with nonzero pooled/global gradients and all ten input-channel gradients on every frame. The previously observed exact constant-output shutdown is absent in these retained update200 records. Composite validation loss is **7.158427100451592 → 0.0605612329423709**, using one half of each normalized squared pressure/full ungauged-residual term. These are retained development measurements, not independent re-evaluation or a useful-solver conclusion.

The public training report's identity, artifact inventory, evidence-file hashes, population component means, activation ranges and inference-time sum agree with retained evidence. Original **233 tracked files and executable modes**, four original failed-model artifacts, the all384 zero-output diagnosis and final reservation hashes remain unchanged. Existing negative native results and historical modal scope are not replaced by the new conditioning observation.

## Custody, access and costs

Completed/drained custody has spawn closed, ECHILD, GO received and leader return code zero. Receipt SHA `105b9e57257ca275dae8b171fe6f30f835551fc9d4c29b02144f0d1bff3a5c1b` and request SHA `16e575dede8e9e1b07baa2f8464da0f15abfb018b95e7648472430ab9c807acc` match the artifact manifest and Run.

Saved read-only access metadata confirms the exact600 s grant was consumed once, Run/grant counts each changed4→5, the authorized existing-owner session was revoked after drain, and zero live owner sessions remain. Its recorded session lifetime is28,800 s. A consumed grant's null `revoked_at` is expected: pinned Tuldok rejects revocation after consumption and rejects reuse. The required lifecycle operation was **session revocation**, not revocation of that consumed audit record. No fresh credential or mutation was needed for this review. The reviewer inspected saved records and pinned access semantics; no live authentication/database write or authorization probe was performed.

Recorded costs remain distinct: recipe work **19.989755383001466 s** before final receipt/publication; enclosing started-to-finished Run **31.357075 s**; owner procedure **40.85638581600142 s** through artifact collection, before final shutdown/session-revocation cleanup; summed32-witness inference **0.30471505499008344 s**, excluding native verification/fallback. Observed process RSS **525,803,520 B**, one Torch/inter-op thread, Python3.12.14/Torch2.8.0+cpu/NumPy2.3.5 and deterministic CPU execution agree with the receipt. Wall/output bounds pass; RAM is observed self high-water, not OS or aggregate containment. These numbers establish no solver speedup.

## Remaining gates

No actual cold restore, trained forcing-amplitude response, native residual/divergence/wall/flux/Courant acceptance, fair fallback/total cost, autonomous conservation/stability, final-suite or GPU/full-training evidence is supplied by this audit. Lower composite loss and transmitted gradients cannot close those gates. Any future restore must bind exact typed artifacts, perform zero updates in a fresh process and preserve final/history bytes; it requires separate material/access/execution authorization.

See [training.md](training.md), [training.json](training.json), [preflight.json](preflight.json), [standards-review.md](standards-review.md) and [preservation.md](preservation.md).
