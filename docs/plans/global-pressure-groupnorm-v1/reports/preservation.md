# Pretraining preservation and inherited native arithmetic

Status: **satisfied for the preservation scope**. This is independent of the candidate's GroupNorm derivative/global controls, output budget, generic preflight and final independent acceptance review.

Against accepted diagnosis merge `16a0656830d66f50318516bb174aa0f74f28e3ed`, all **233 prior tracked files** retain their exact Git blob bytes and executable modes, checked before and after this observation. This includes original source/tests, native probe, reservations and scientific reports. The new candidate's three imported dependencies match the explicitly pinned original data/model/train hashes. GroupNorm `full_operator` and `fit_normalization` are the exact original function objects; its sample-order helper is likewise inherited unchanged. Detailed file identities, packet hashes, observed candidate hashes and runtime are in [preservation.json](preservation.json).

## Data and physical preprocessing

The bounded original reader admitted the same five packet2 inputs and independently recomputed their lineage/numeric inventories: **288 training frames from 12 whole trajectories and 96 validation frames from four whole trajectories**, 24 ordered frames per parent, with disjoint identities. No cropped histories or substituted rows were admitted. Origin `dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d`, split `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a`, native residual anchor `26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8` and reservation `1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5` are unchanged.

Recomputing train-only normalization and serializing with the original trainer's canonical JSON plus trailing newline reproduces all **2112 original bytes**:

- Normalization SHA256: `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f`.
- Fixed 200-by-2 sample order: `3200 little-endian int64 bytes`, SHA256 `1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9`.

Every sampled index is in the 288 training rows. Perturbing all validation RHS/reference/base/residual values in a disposable in-memory copy changes neither normalization bytes nor sample-order bytes. Original packet files remain unchanged.

## Native arithmetic evidence

Using the inherited float64 full operator on all **384 original previous-pressure snapshots**, in bounded 16-frame batches with `N,1,Z,Y,X` tensors, recomputed `b-Ap_base` differs from the immutable native-generated residual by at most **4.3368086899420177e-19 m³/s²**. The tiny anisotropic [3,2,1] native6 oracle matches exactly, including its nonzero physical cell-zero action. Pressure gauge handling does not replace that physical row.

Native binary SHA256 `3a06964ed4121cde5394e114304abd3f50a569ed3345682d5dccd20c740ba3df` and build receipt `70b0e33f5ec13c289e5afd0c38df783655936e06c2dfd32f0509e94d2aef8307` were verified through the unchanged prepared-source helper before and after observation. This reuses the retained qualified native residual/oracle evidence; it runs **zero new native pressure solves**. It is not a new model quality or rollout result.

## Preserved negative result

The original trained artifact bytes match the before/after diagnosis anchors. The original 32 teacher-state cases retain **0/32 raw qualified predictions**, raw predictions equal temporal pressure, **32/32 corrected qualification through reset fallback**, and no acceleration qualification. The accepted diagnosis retains zero corrections on all **288 training and 96 validation frames**. It retains only ten every-20-update logs, with no intermediate weights/Adam states inspected or reconstructed. Original failed weights and reports were not replaced or relabelled.

## Procedure and limits

Observation used Git tree/blob/mode comparisons, the pinned bounded packet reader, exact canonical-byte and numeric-order comparisons, function-object identity, current CPU arithmetic against all stored native residuals, and the retained native6 oracle. Two initial observation-script attempts stopped on serialization/tensor-adapter assumptions: the accepted normalization has a trailing newline and the operator requires `N,1,Z,Y,X`. The observation adapter was corrected; no subject source or data changed.

Python `3.12.14`, NumPy `2.3.5`, Torch `2.8.0+cpu`; one Torch and one interop thread. This scope constructed no optimizer, performed **zero optimizer steps**, created no grant or Run, evaluated no trained-model predictions and read/generated no final or GPU data. It establishes preservation and inherited arithmetic, not GroupNorm efficacy, managed training, autonomous stability or solver acceleration. Final review must bind the final candidate source hashes after other owners finish their changes.
