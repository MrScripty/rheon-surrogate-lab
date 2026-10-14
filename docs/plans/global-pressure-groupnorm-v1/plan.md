# Fixed GroupNorm global pressure experiment

**Plan status:** `Accepted`

**Current phase:** One admitted CPU training and exact zero-update cold restore completed/drained

**Next slice:** Separately review a model-specific lab adapter and native cost/gate contract; no native candidate, further Run or training is authorized.

**Acceptance status:** `complete` for A1–A7 pretraining admission and R1–R6 cold restore; solver qualification remains pending

**Execution ledger:** [execution-ledger.md](execution-ledger.md)

**Issues:** [issues.md](issues.md)

## Objective

Make one concrete, reviewable CPU experiment that tests whether within-sample feature normalization prevents the measured activation shutdown in the original global pressure model. This pretraining slice establishes executable contracts and qualification, not a remedy, solver qualification or benchmark gain.

Diagnosis PR5 was accepted at `ec48499cce5bf1c3e63e9fb4baac76e9c9241528` and merged normally as `16a0656830d66f50318516bb174aa0f74f28e3ed`. The original training sources, failed weights, 384-frame diagnosis and all negative native results remain immutable. The [diagnosis proposal](../global-pressure-diagnosis-v1/next-experiment.md) supplies the hypothesis; the initial authorization permitted implementation and pretraining review. The parent subsequently admitted exactly one fixed 200-update CPU Run at the frozen source/material, with a 600 s one-shot grant and owner-session revocation after drain. That Run completed; [training observations](reports/training.md) record its limits.

## Objective acceptance

| ID | Observable criterion | Kind | Environment / mode | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| A1 | Exact original convolution/head initialization mapping; fixed GN placement; unchanged parameter inventory; distinct model identity | construction and focused | pinned CPU / automated | satisfied | [captured model arithmetic](reports/captured-model-arithmetic.json) |
| A2 | Finite 16³/32³ and edge probes, batch independence, train/eval identity, gauge/bypass, meaningful derivatives and global controls | numerical | pinned CPU / automated | satisfied | [captured model arithmetic](reports/captured-model-arithmetic.json) |
| A3 | Original train-only normalization, 288/96 whole-history split, immutable lineage and final reservations | data contract | retained packet / automated | satisfied | [preservation](reports/preservation.md) |
| A4 | Eleven numeric checkpoints and every other declared output fit measured caps and 32 MiB aggregate; exact final-only restore contracts | contract | pinned CPU / automated | satisfied | [output budget](reports/output-budget.json) |
| A5 | Existing captured CPU runtime supports GN; generic CLI can open/prepare/preflight exact source/input/output closure without grant or execution | actual integration | owner-local snapshot/catalog / automated | satisfied | [preflight](reports/preflight.json), [no execution](reports/no-execution.json) |
| A6 | Independent numerical and standards reviewers clear the frozen material | independent review | exact source/material / read-only | satisfied | [numerical](reports/numerical-review.md), [standards](reports/standards-review.md) |
| A7 | Parent admits the exact source/material protocol before any grant or scientific execution | owner admission | frozen packet / review | satisfied | exact parent admission; [training](reports/training.md) |

## Cold restore acceptance

The parent independently reviewed the offline packet and approved the exact one-session sequence with action-time user confirmation. Authoritative preflight matched material `4265f5157c7cfaf1759ae2f501956cefc9912b430a93f1a90d9706792674360e` completely before grant. Run `265c9574c06048d8bc91daa1cf8eab02` completed/drained. See [actual restore](reports/restore.md); these gates close reproducibility only.

| ID | Observable criterion | Status | Evidence |
| --- | --- | --- | --- |
| R1 | Actual source/recipe/runtime and full material match independently reviewed candidate before one exact-material grant | satisfied | [restore JSON](reports/restore.json) |
| R2 | One fresh CPU process loads only final200; no optimizer, selection, resume or new update | satisfied | frozen source and actual restore receipt |
| R3 | All 18 outputs collected; all 16 inherited artifacts match bytes; 32 recomputed predictions pass exact keys/dtype/shape/body parity | satisfied | [actual restore](reports/restore.md) |
| R4 | New receipt/summary identify restore/zero new/200 trained updates and original Run/custody honestly | satisfied | [restore JSON](reports/restore.json) |
| R5 | Resource bounds, completed/drained custody and session revocation after cleanup; zero usable credentials verified independently | satisfied | [restore JSON](reports/restore.json) |
| R6 | Independent read-only outcome audit agrees with source, published artifacts, custody and lifecycle | satisfied | [restore review](reports/restore-review.md) |

## Binding experiment decisions

New identity: `rheon-global-pressure-groupnorm-model-v1`. Add `GroupNorm(4, C, eps=1e-5, affine=False)` after each Conv3d in the four encoder blocks, bottleneck and four decoder blocks, before its SiLU. The head is unchanged. No learned normalization parameters or running dataset statistics are introduced. Explicit mapping must prove byte-identical original seeded convolution/head values, despite changed sequential indices. Reuse byte-pinned original helper sources as captured dependencies; fail closed on dependency drift.

The same ten channels, f32 eager network, f64 gauge/full finite-volume operator, composite loss, train-only physical scales and native residual lineage apply. Group normalization can remove useful physical amplitude: report forcing-amplitude and distant/global sensitivity rather than assuming benefit.

The admitted smoke used exactly 200 Adam updates, batch 2, initialization seed 20261014, order seed 20261015, lr .001, betas (.9,.999), epsilon 1e-8, weight decay 0 and clipping 1. There is no sweep, restart, early stopping, checkpoint selection or additional baseline training. Validation is observational at 0/200. Retain numeric weights at initialization and every 20 updates, sampled row identities, and before/after layer ranges, zero fractions, correction variation and pressure/residual component gradients. Checkpoints are observations; only update 200 is the candidate. No optimizer state resume is supported.

The admitted run used one CPU thread, existing 600 s child wall, observed process high-water stop budget 4 GiB, 256 MiB input / 320 MiB capture and 32 MiB output aggregate. The RAM policy is an observed process limit, not OS containment. Restore must use typed final artifacts from one completed/drained Run, accept only the new identity, perform zero updates and reproduce final predictions exactly in a fresh process. The pretraining slice executed no optimizer steps and created no grant or Run; its historical reports remain unchanged. The separately admitted training Run performed exactly 200 updates and published all 18 artifacts. One separately admitted fresh zero-update restore completed and drained; all inherited bytes and fresh prediction parity passed. No additional restore is authorized.

## Evidence boundaries and future interpretation

Reuse the exact 384 development frames (288 training / 96 validation, whole trajectories). Native anchor `26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8`, split SHA `fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a` and reservation SHA `1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5` remain binding. No reserved final arrays are generated, read or tuned against.

The completed training report separates conditioning from physical usefulness. For a separately reviewed native evaluation, keep all 32 development teacher cases, unchanged original-b residual/divergence/wall/flux/Courant gates, failed warm work (<=32 iterations) and one fresh full p_base reset (<=3000). Compare full-budget temporal, zero and analytic starts fairly. Report inference, verification, fallback, runtime/setup, training/checkpoints/restore and enclosing cost. Nonzero corrections and lower losses do not qualify acceleration. All failures remain visible.

Autonomous held-out rollouts must later assess divergence, residual, conservation and stability before any solver claim. The separate local-GPU/full-training plan remains behind a new review gate; neither GPU training nor final evaluation is authorized here.

## Simplicity and ownership review

**Applicability:** `applicable`

**Composed-design review:** `applicable`

The lab owns scientific model/configuration and numerical artifact semantics. Tuldok owns generic recipe capture, runtime identity, preflight, grants, scheduler, custody and artifacts. Rheon owns the qualified native operator. New model/receipt/source identities prevent accidental compatibility with original failed weights. Captured pinned helper modules reuse the existing reader, stencil and serialization instead of duplicating them; the new trainer owns the changed checkpoint and observation policy. Generic recipe output declarations expose every file. No model-specific upstream executor, compatibility shim or hidden output is introduced.

The independent concerns are the fixed representation intervention, scientific state/history, immutable data lineage, and generic execution custody. They are evaluated in the lab on CPU because the previous decoder shut down; the later exact owner admission interleaved them in the single completed Run. Model identity, checkpoint time and final-candidate policy belong to the new trainer; runtime/material identity and grant lifetime remain Tuldok-owned. The assembler must know exact source names, input slots and declared outputs, while model callers use the accepted tensor/normalization interface. Changing the scientific architecture touches this experiment's model, qualification and identity; changing output policy touches trainer, budget and generic declaration; neither change requires editing Tuldok or Rheon.

Pinned original helper dependencies expose the already accepted reader/loss/serialization representations explicitly; no hidden ambient import ordering is permitted. Model numerics and generic capture can fail and be checked independently, but a replacement scientific source invalidates qualification and material hashes. Deleting the wrapper would spread pinned-source and identity checks across callers; deleting the assembler would spread recipe closure knowledge; deleting the budget owner would hide retained history cost. These small owners contain required complexity. No general registry, alternate executor or compatibility hierarchy is retained.

## Milestone: one pretraining admission packet

**Allowed write set:** `experiments/global-pressure-v1/groupnorm-v1/**`, `docs/plans/global-pressure-groupnorm-v1/**`, and new `.github/workflows/global-pressure-groupnorm.yml` for zero-update pretraining checks. Ignored task-local material lives under `runs/global-pressure-groupnorm-v1/**` and a new direct child of the existing project namespace. Prior source, evidence, Tuldok and Rheon are read-only.

- [x] Implement candidate, exact mapping and future train/restore contracts.
- [x] Qualify numeric controls, preservation and complete output budget without updates.
- [x] Commit source, assemble immutable generic project, inspect actual runtime and CLI preflight.
- [x] Obtain independent numerical and standards review; freeze protocol for parent admission.

**Acceptance gate:** A1–A7 above. **Status:** `Accepted`. This closes pretraining admission, not solver qualification. Separate admitted stages are recorded in [training](reports/training.md) and [restore](reports/restore.md); native solver gates remain pending.

## Concurrent work and repository isolation

Main owns project assembler, plan, runtime/preflight and Git publication. development_capture owns model, trainer, configuration and focused qualification. native_warmstart owns output-budget measurement. numerical and standards workers independently review the frozen material. All writers have separate files; original evidence is shared read-only. Branch `feat/global-pressure-groupnorm-v1` isolates the new experiment from the accepted diagnosis; no history rewrite or workspace removal is planned.

## Replan triggers and blockers

Stop admission if exact initialization/derivatives fail, runtime closure changes, checkpoint diagnostics exceed caps, restore accepts old identity, data/reservations drift, or independent review identifies a material flaw. No grant/training retry repairs a failed pretraining gate. Current technical blockers: none. A1–A7 pass. Exactly one separately admitted training and one zero-update restore completed/drained; no retry, additional update/Run, native candidate evaluation, final suite or GPU run is authorized. Future material/access/execution gates remain separate.

## Primary sources

[PyTorch 2.8 GroupNorm](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html) defines per-sample grouping and train/eval input statistics. [Wu and He](https://arxiv.org/abs/1803.08494) motivate grouping for small batches; their vision benchmarks imply no pressure-solver gains. [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html) motivates explicit f32 controls. The [original research](../global-pressure-correction-v1/research.md) retains other practical architectures and primary solver sources.
