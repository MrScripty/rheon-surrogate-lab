# Input preservation audit for the bounded GroupNorm native development evaluation

The fixed final200 candidate, restored prediction witness, development packet, selected raw captures and qualified native binary are present locally and match their declared SHA256 hashes. No missing or drifted input was found. This is an input and preservation audit; no model import, inference, replay, native solve, training, data generation, credential operation or service call was performed.

The portable role/path/size/hash inventory and execution limits are declared in [execution-freeze.json](../execution-freeze.json), SHA256 `92d20132c0ae3d28aebee7fb729e98b29d3aaced5e7628c06b99e7b187a91fa9`. [input-preservation.json](input-preservation.json) records the audit disposition and complete case order. Exact new implementation source, effective launch environment and independent numerical/standards reviews remain pending in the separately authored `reports/source-freeze.json` contract. The inventory is not permission to run before those gates complete.

## Candidate and retained lineage

The scientific source is `fd3a6009e5805978a29ed1df7389ad226a40a914`. The final200 weights and restored prediction archives match the training artifacts, including the retained update200 checkpoint. Training and restore receipts bind their original source witness, configuration, input evidence, split order, normalization and tensor inventories; the restore performed zero optimizer updates and declares 200 trained updates. This audit checks retained bytes and receipt relationships, without loading model state or regenerating predictions.

| Input | SHA256 |
|---|---|
| Final200 weights/checkpoint | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |
| Restored predictions archive | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` |
| Physical pressure body | `cc70f931c156c96f53fff8c606ffe039a7e3f294320731e09ac0ae44d9f99cc7` |
| Row-index body | `17acc08ccda81f2e47be3a0229106f0b522dc3f941a72015f8a677c314fc2eb7` |
| Train-only normalization | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` |
| Immutable splits | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` |
| Qualified native6 binary | `3a06964ed4121cde5394e114304abd3f50a569ed3345682d5dccd20c740ba3df` |

The prediction archive contains numeric pressure `[32,16,16,16]` and row indices `[32]`; canonical textual identifiers are bound by the receipts and splits. The frozen population is 32 teacher-state snapshots: frames 000 and 023 of each of 12 training parents and four validation parents, ordered training first and then validation. Their original 288/96 development rows remain unchanged. Every selected case has its original RHS, reference pressure, previous pressure `p_base`, and three pre-projection MAC velocity components. All 192 selected raw file locators match original lineage hashes, totalling 4,816,896 bytes. Frame 000 uses the initial pressure; frame 023 uses frame 022 pressure. No final partition arrays are needed or admitted.

The native source dependency remains base commit `2209012bc5f902693d0f695ec1c2d854126af317`, patch SHA256 `b851b38d2d03280220bc4648cd4d721b2e796786fb18d8188dc335336fe2a998`, and prepared content tree `5a5de4635d3970c9b99e620d9fff2d9a23c55307a9e335f7473e7f95a0c7a4ab`. The 746 prepared source files and nine lab native-source files match their retained inventories. Native residual provenance remains anchored at `26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8`. Existing build and oracle records are retained evidence; this audit did not rerun an oracle or native qualification.

## Input bytes, runtime and read accounting

| Scope | Declared bytes | Locators |
|---|---:|---:|
| Scientific inputs and retained proof/source files | 111,120,136 | 279 |
| Separate minimal interpreter, NumPy and platform-library qualification | 90,846,557 | 907 |
| Scientific input limit | 268,435,456 | — |
| Remaining scientific admission budget before new source/proofs | 157,315,320 | — |

The 256 MiB scientific input budget counts every declared provided-file locator, including identical copies in training, restore and historical directories. The inventory includes the whole immutable development packet and historical proofs as well as the 192 selected raw inputs; it does not deduplicate identical content to reduce the declaration. New adapter source or additional proofs must fit the remaining budget.

A final instrumented hash pass reread all 1,186 declared scientific/proof and minimal-runtime locators: 201,966,693 bytes. Every size and SHA256 matched. Earlier metadata and protected-tree reads are additional and were not instrumented as a complete audit I/O total. Two full scientific verification passes would read 222,240,272 bytes. This is an estimate, not a measured execution cost or a bound on all I/O. Before/after hashing, receipt parsing, archive header/member reads, extraction and repeated reads incur real costs. The implementation must record actual read operations, provided bytes and decompressed-member reads separately, and identify how its input limit is enforced. Runtime qualification and verification I/O are not disguised as inference or scientific throughput.

The selected local interpreter is `/workspace/task-torch/bin/python`, resolving to the retained Python 3.12 executable; venv metadata reports Python 3.12.14. NumPy dist-info reports 2.3.5. The current interpreter, venv metadata, 902 NumPy code/library/metadata files and three conventional platform library files were byte-hashed without initializing that runtime or importing NumPy. Native `readelf` metadata identifies `libgcc_s.so.1`, `libc.so.6` and `ld-linux-x86-64.so.2`; effective process mappings were not observed. The new implementation must qualify its actual launch and environment.

The historical captured-runtime manifest SHA256 is `6367d0fa7eb989d53cc49d6ed47f937a4f63e68655de9acaf84c7f7d94838c77`; it describes 21,073 files and 870,900,515 bytes. This audit reads that manifest and the minimal current runtime closure, rather than traversing all 870 MB again. Historical managed/captured admission and drained artifact custody remain historical evidence. No new managed runtime or custody result is asserted. All inventoried assets are currently available without credentials; a moved or missing locator requires exact-byte qualification and must fail admission rather than trigger regeneration, download or renewal.

## Preserved evidence and bounded execution contract

All 273 tracked files at main merge `28c65ac868678e496603d248c5c1e350ffc0fab3` and all 286 tracked files at accepted PR7 plan head `d4be7adf034e06d047c5d3161efc4f7e5d142673` preserve their Git blob bytes and modes. Their inventory SHA256 values are respectively `9a63308a3d7868e7ee3ea0cecbe8e86436fcacaf0f053a3966205560b8712584` and `b338d4ad225eb5384597423019cdc4eaf0ae00873cf83061c85133d865de73a1`. The audit adds only the new run-plan reports and freeze.

The original CNN result of 0/16 committed cases, original global candidate raw acceptance of 0/32 with all 32 resets, and 384 zero-correction diagnosis remain preserved. The sealed reservation SHA256 `1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5` is included as metadata; no reserved final arrays were opened or generated. GroupNorm training loss, active features and a restored pressure witness cannot establish native acceptance, autonomous rollout stability or accelerator performance.

The freeze retains one serial 32-case pass, zero scientific retries, 60 seconds verification, 600 seconds evaluation, at most 60 seconds per native case, one native thread, 32 MiB aggregate reports and a 4 GiB observed process high-water stop. The memory stop is not an OS containment claim. Learned proposals receive at most 32 warm iterations and then at most one fresh workspace solve from original `p_base` with 3000 iterations. The primary temporal comparator starts once from the same `p_base` with 3000 iterations; zero Jacobi and the bounded temporal diagnostic retain their distinct labels. Strict original-RHS residual, divergence, wall flux and Courant gates remain fixed. Corrupted identity fails before numerical fallback. Both-zero residual handling, timer availability and fail-stop publication must be verified against the exact new implementation.

No input-preservation blocker remains. Execution stays pending exact implementation binding, effective runtime qualification and independent source reviews. This report supplies no new physical evaluation evidence or gain claim.
