# Current-dependency migration results

Status: verifying; original service cutover and final live acceptance remain
pending. This report records migration evidence, not a pressure-model promotion.

## Method and invariants

Start the lab integration from the complete water-training lineage and merge the
accepted migration plan without rewriting either history. Integrate fetched
Rheon `bc2f472787060d9a69cb619ed2c2fea10fd754b6` and Tuldok
`300f12df6f48e028fd9379911703f7c89337952a` in separate worktrees/branches.
Select source, build artifacts and installed runtime independently through the
reusable configuration. Record dirty source bytes as such, not as the unchanged
upstream commit. Preserve original weights, normalization, numerical sources and
scientific results.

Qualify one captured frame before whole trajectories. Reproduce all 74 consumed
baseline cases, then compare the current build on the same cases with unchanged
physical/stopping limits and independent sparse pressure checks. The 26b residual
checkpoint is the regression control; the desktop keeps the original aa82 model.
The latter's known pressure failures remain visible. No untouched scenes or
checkpoint selection are part of this migration.

CUDA float32 training and CPU float32 neural inference remain separate. The
pre-cutover execution smoke performs one optimizer update; its fresh CPU restore
performs none and does not initialize CUDA. The later live typed Trial/restore
is a second separately gated smoke. Neither replaces deployed weights.

## Completed evidence

| Check | Observed result | Retained evidence |
| --- | --- | --- |
| One captured frame | 15 native prefix steps, one AI step, six actual network calls; input/output bytes match the reference | [Probe receipt](evidence/lab-probe.json) |
| Saved-input and operator parity | Exact model preprocessing/reconstruction and zero-residual behavior; independent operator/gauge checks pass | [Scientific parity](q4-scientific-parity-v1.md) |
| Original 74-case reproduction | All 74 cases and 8,523 AI steps reproduce recorded calls and pressure metrics; no native fallback/refinement | [Original reproduction](q7-old-reproduction-v1.json) |
| Current 74-case qualification | All 74 current cases, 74 native controls and 233 independent sparse same-RHS samples pass; every per-case pressure maximum and actual call sequence equals the reproduced baseline | [Current scientific qualification](q7-scientific-qualification-v1.md) |
| Tuldok CUDA/CPU pipeline | One CUDA update and fresh zero-update CPU restore, both completed/drained; checkpoint and weight bytes match | [Full pipeline receipt](evidence/q3-pipeline.json) |
| CUDA lifecycle | Five real hashing/training cancellation, deadline, exception and cold-recovery cases pass with zero updates | [Lifecycle receipt](evidence/q3-lifecycle.json) |
| Rheon lifecycle | Real child startup/cancel/reap/restart/shutdown; restarted 16-step run makes one prediction after 15 native steps | [Lifecycle receipt](evidence/rheon-q5-real-lifecycle-03.json) |
| Visible desktop rehearsal | Native pressure and signed AI-minus-native voxels/legend/probe work; model rejection preserved; schema-3 saved result opens with package unavailable | [Desktop receipt](evidence/rheon-visible-desktop-rehearsal.json) |
| Post-A7 desktop enablement | Same bounded 16-step comparison and package-hidden cold inspection pass; current read-only observer remains visible | [Enablement receipt](evidence/rheon-post-a7-enablement.json) |
| Schema-only restore | Exact original 31 tables, sequences and 38,598 managed entries retained through restore and explicit 8-to-10 upgrade/re-entry | [Schema rehearsal](evidence/schema-rehearsal02-receipt.json) |

## Timing and resources

These are different workloads; their elapsed times must not be added as a serial
project duration or interpreted as training throughput.

| Workload | Elapsed time | Scope |
| --- | ---: | --- |
| CUDA update plus fresh CPU restore | 271.887 s | Includes runtime verification, current material/custody setup, execution and artifact checks |
| Schema-8 restore and schema-only upgrade/re-entry | 329.666 s | Restore itself 316.815 s; peak RSS 1,234,776 KiB |
| Visible 16-step teacher comparison | 13.429 s | Model startup 11.090 s and one prediction 0.031 s included; GUI interaction time excluded |
| Independent saved-field pressure oracle | 0.044 s | 4,096 cells; no model or simulation rerun |
| Matched advisory timing | 540.83 s | Twelve serial measurements, two reversed-order repetitions; independent raw quality checks pass |

Use at most two one-thread simulation workers, one GPU optimizer and the measured
CPU affinity, retaining the 8 GiB available-memory reserve. Full installed
numerical-runtime verification binds 25,765 files, 1,047 symlink aliases and
5,744,383,179 file bytes. Runtime verification/startup are recorded separately
from steady-state inference. [The timing report](q7-advisory-timing-v1.md) records
first versus steady calls, spread, native/model/full-step scopes and uncontrolled
warm-cache limitations; it makes no performance acceptance or speedup claim.

## Historical preservation and failures

The source archive has 120 Projects and 160 Runs. Selected external closure is
3,314 files / 1,481,169,445 bytes. Thirty-six exact pinned files were recovered
from retained evidence; 137 pre-existing unavailable references remain explicitly
unavailable, never replaced with similar current bytes. Large historical JSON is
exposed as lossless bounded fragments through current evidence readers.

Prepare02 stopped before publication on a terminal interrupted Run whose original
phase remained `prepared`. The original owner permits this interruption before
GO, with no worker or custody identity. A narrowly qualified archive-only
disposition preserves that exact state; unknown custody still blocks conversion.
The failed stage remains intact; retry03 uses a fresh stage/spec.

Attempt03 then published and reconciled its exact accounting successfully. A
deeper current recipe check found four captured-CPU declarations unsupported by
the selected Project-v2 contract and 231 inspectable managed declarations pinned
to the obsolete runtime descriptor. The owned migration policy exposes all 235
as historical evidence, with exact bytes and pins preserved. The correction
passed 15 real PostgreSQL tests; fresh restore03 and stage04 rehearse it without
patching published state. Attempt03 remains a superseded rehearsal, not final Q2
acceptance.

Other retained failures include parent-directory timestamp false conflicts before
Run admission, early lifecycle cold-prefix clock disagreement and the first
GUI evidence decoder attempt. Their fixes and successful attempts are linked
from the owning receipts. No failed Run or rejected pressure is relabeled passed.

## Evidence custody and remaining gates

[The evidence index](evidence/index.json) hashes small exact original receipts
retained in Git. Full command logs, SQL backups, source inventories, raw arrays,
model bundles and datasets remain under `runs/dependency-migration-v1/` or their
original historical paths. Credentials and large scientific assets are excluded
from Git. A receipt that names a pending check is not evidence of its completion.

Current blockers to final acceptance are corrected domain publication/cold reads,
completed-current backup/restore, current-project inspection through the clean
deployment, then the gated original-service CUDA/CPU/typed browser smoke.
The corrected ordinary CPU/managed Trial browser paths and fresh deployment
artifact have passed their isolated tests. See the active plan and
independent Standards and Spec reviews for their current dispositions.

## Source commits and active resources

Rheon branch `integration/lab-current-contracts-20261016` is committed through
`759d928448dec8b160f4c113e35bf2bf2bd1b9d3`; its public diagnostic integration is
`054e4cc1ded96e1cd22d80f6473a678a33482f17` and observation UI is a separate
commit. Tuldok's permanent host integration is
`824665aefdc68a458b5510a0b33d6cae1eb4887d` on
`integration/rheon-lab-current-20261016`; offline conversion/retirement remains
a separate pending commit. No push or history rewrite was performed.

The three integration worktrees under `runs/dependency-migration-v1/` remain
protected active sources owned by this migration task for Jeremy. Their branches
retain their commits. Synchronization is explicit after assessment, never an
automatic pull into qualified source. Keep these paths while live Project
references, review and deployment depend on them. Their next disposition is
review/merge, relocation of active references/deployment, then a separate clean
task-owned worktree retirement assessment with commit protection verified.
The exact post-commit heads and reachability are retained in the operations
receipts. The original primary checkouts and unrelated files remain preserved.
