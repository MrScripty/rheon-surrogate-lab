# One admitted CPU training observation

The single approved GroupNorm Run completed with exactly 200 Adam updates and drained custody. The fresh preflight matched the independently reviewed source, inputs and CPU runtime exactly. This report reads retained outputs; it runs no model, optimizer, restore or native evaluation.

## Execution and access

- Run: `d73bc2976c4f442f8aefc8f7242bd252`; catalog: `619034b1109f45faac099835ee9a9065`.
- Scientific source: `fd3a6009e5805978a29ed1df7389ad226a40a914`; exact train material: `4e8b58899d7c03f074b2b5632911c539f183bb2cf57024e3809c1a07055f0668`.
- Catalog Run/grant counts each changed from 4 to 5. The 600 s one-shot grant was consumed once by this Run.
- The approved existing-owner session lifetime was 28,800 s. It was revoked after completion/drain; a separate read-only metadata check found zero live owner sessions. No new bearer, actor or agent was issued.
- A consumed grant remains an audit record with `revoked_at=null`; unique consumption prevents reuse, and Tuldok explicitly rejects revoking a consumed grant. This is distinct from the required, verified owner-session revocation.

## Retained scientific observations

Validation **composite** relative squared pressure plus full ungauged finite-volume residual loss changed from **7.158427100451592** at initialization to **0.0605612329423709** at update 200. Each term has weight 0.5 and a safeguarded reference squared norm; pressure is `p_base + gauged correction`. This is observational development validation over 96 frames from four whole trajectories. It is not residual-only, an autonomous rollout, or a final geometry/regime evaluation.

All 384 retained teacher states have a nonconstant raw head, nonzero correction, nonzero pooled/global gradient and nonzero native-residual input gradient. The last decoder pre-SiLU range across these states is −8.073195457458496 to 7.441528797149658; no state has an entirely zero pre-SiLU input. The previously diagnosed exact constant-output shutdown was therefore not observed at update 200 on this population. A single run does not establish a general remedy or useful pressure solver.

The fixed candidate is update 200. Eleven numeric states at 0 and every 20 updates remain observations; none was selected or resumed. Diagnostics retain all 200 sampled row pairs, ten before/after observation logs, the final 384 states and eight validation derivative cases. Original initialization mapping, physical scales, split hashes and sampled-order hash are bound in [the machine-readable report](training.json).

| Population | Frames | Mean pressure loss term | Mean residual loss term |
| --- | ---: | ---: | ---: |
| train | 288 | 0.0188383463318574 | 0.0408316730187454 |
| validation | 96 | 0.0203085869622447 | 0.0402526459801262 |

## Cost and custody

The enclosing Run lasted **31.357075 s**. The recipe reported **19.989755383001466 s** before final receipt/publication; the owner procedure recorded **40.85638581600142 s** through access, preflight, capture and artifact collection, before its final shutdown/session-revocation cleanup. Recorded 32-case inference totaled **0.30471505499008344 s**, excluding native verification and fallback. These scopes cannot be used as a solver speedup.

Observed process peak RSS was **525,803,520 bytes**, below the observed 4 GiB stop budget. CPU Torch and interop thread counts were one. Child wall/input/capture/output budgets remained 600 s / 256 MiB / 320 MiB / 32 MiB. The memory policy observes process high-water; it is not OS containment. The receipt records Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5 and deterministic CPU execution.

All 18 outputs were downloaded through the actual generic Run artifact API and independently matched by SHA-256 and bytes: **20,607,920 bytes** total. Custody proves completed, drained, spawn closed, ECHILD and return code zero. The final weights exactly equal checkpoint 200 byte-for-byte. The complete hashes, numeric states, closure, terminal access metadata and 17 typed restore bindings are in [training.json](training.json). Large artifacts remain in the private task-local catalog/run archive; this public report is their hash-bound inventory.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `checkpoint-000.npz` | 1500186 | `1b4b77cf299d798b6b787bbd12486bc8d4301d7534972c66e9dddce41e93f737` |
| `checkpoint-020.npz` | 1500186 | `61ce49c91030ae2107b8b63986a5a11e4c10d941a3ac5ce542927e7e2c98bd1f` |
| `checkpoint-040.npz` | 1500186 | `1ed46be82d8d6848e915921803dc71d9605e27ae44f08b315345dbd7ddaf0703` |
| `checkpoint-060.npz` | 1500186 | `a48a17923ee7554b7ecb8f6a6361fc1d7903ffd0319b0c60a1b71f77206c1403` |
| `checkpoint-080.npz` | 1500186 | `20bcbe8faadc5214560ab202b4daf60136d7e34f734f92b78af43976cf0280b4` |
| `checkpoint-100.npz` | 1500186 | `ea033f4aba7c1d44c1b3cb3fc8e5583ff869a34c05a8a484ea0ad037d2f8b392` |
| `checkpoint-120.npz` | 1500186 | `6141585f2ef4c75f8c45ee8073d38c86a1412e6179ad588c0072976b0029a46e` |
| `checkpoint-140.npz` | 1500186 | `8c1d4d792ad8171c96945af5d5a5a04a3087f370f3c80b207b2f11562dada201` |
| `checkpoint-160.npz` | 1500186 | `199029c43d45348f8da8120bc50d24f0350abd0203f1049f772627999bebfd3b` |
| `checkpoint-180.npz` | 1500186 | `d5149bbca9e3a7850438ea7292d0336bdf4b5102bae90740ab7b9bee65efbd1a` |
| `checkpoint-200.npz` | 1500186 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |
| `diagnostics.json` | 1402428 | `a94a30492f675b40fdd3c2a5ec92419dbc798f01f42c6219a3719ff7bee858b0` |
| `normalization.json` | 2112 | `8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f` |
| `predictions.npz` | 1049356 | `bb0290ca1855f841d3fe99260790d9cfdd39e8b7982e054ea01791b3a848ee86` |
| `receipt.json` | 80551 | `88abf1b2eaff46da583588c422166f0db4e63eb8aaa7790ed585f88d1a5af1ed` |
| `splits.json` | 70478 | `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` |
| `summary.json` | 763 | `5befc3ef4435080132e5a427ecaf9696fead9016db88f29bd5320c2597859111` |
| `weights.npz` | 1500186 | `4adf55ca940c2013b6352edc4a6716492e0a697484331c65cdf80495312db147` |

## Qualification boundary and handoff

The original small CNN failure (0/16 native cases), original global constant-output failure (0/32 raw cases), full-reset fallback work and historical modal scope remain unchanged. Lower composite loss and active features do not qualify an accelerated solver. Cold trained restore, post-training amplitude response, native divergence/residual/wall/flux/Courant acceptance, conservation, rollout stability and total cost including fallback are unperformed. Reserved final suites and GPU/full training remain sealed.

The next gate is a separately reviewed restore material built from the 17 recorded typed `run_artifact` inputs from this one completed/drained originating Run. It must perform zero updates and reproduce final predictions exactly in a fresh process while republishing all inherited history byte-for-byte. No restore is launched or admitted here. Revoked sessions cannot be reused; any future owner access or execution requires separate authorization. See [handoff](../handoff.md) and [independent training review](training-review.md).
