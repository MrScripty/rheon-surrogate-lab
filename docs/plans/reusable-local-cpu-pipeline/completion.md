# Local CPU pipeline acceptance

The reusable pipeline completed on two fresh development configurations with
unchanged GroupNorm architecture, preprocessing, finite-volume loss and default
optimization policy. Configuration selects lineage; new packet, source, runtime
and Run identities are measured. Historical dataset hashes and result bytes were
not replaced. The latest source is `d7978fd` on
`feat/reusable-local-cpu-pipeline`, based on merged lab main `03c05e8`.

The [compact measured evidence](acceptance.json) includes both configurations,
Run IDs, source witnesses, fixture/capture/packet receipts, runtime identity,
artifact/download parity, costs and evaluation journals. Bulk evidence remains
under ignored `runs/`; the record is a local observation, not a transferable
runtime or credential bundle. The [runbook](../../../experiments/local-pressure-v1/README.md)
owns setup and commands.

## Completed workflow

| Observation | Default development lineage | Configuration-only reuse lineage |
| --- | --- | --- |
| Whole parents / frames | 12 train + 4 validation / 384 | 12 train + 4 validation / 384 |
| Seeds | 210000–210011; 210100–210103 | 310000–310011; 310100–310103 |
| Executable source | `216daa5` | `d7978fd` |
| Preparation | Explicit independently qualified material | Default command; no `--material` |
| One-frame qualification | Passed before complete capture | Passed before complete capture |
| Dry generic preflight | Valid; zero Runs | Valid; zero Runs |
| Training Run / updates | `de66b697512e426691d8172c911639a2` / 200 | `0f923b6401914fae9d9885b6fa06b9b9` / 200 |
| Independent restore Run / updates | `aa32b8295a23471ea33ac7b4a97d792d` / 0 | `07373561747041c5ad0fdc2c700c6dec` / 0 |
| Managed custody | Both completed and drained | Both completed and drained |
| Artifact publication / catalog reopen | All 18 per Run, exact verified downloads | All 18 per Run, exact verified downloads |
| Numeric restore | All 16 numeric artifacts byte equal | All 16 numeric artifacts byte equal |

Restore consumes 17 typed originating artifacts, including the training receipt,
and validates every configured checkpoint. Its own receipt/summary identify
restore; weights, predictions, history, splits, normalization and checkpoint
bytes match training exactly. There is no optimizer construction or update in
restore. Final update 200 is the configured candidate; intermediate checkpoints
are observations, not selection/tuning or optimizer resume state.

Only experiment ID, trajectory IDs and seeds changed for the second lineage;
model, geometry/forcing policy and optimization settings stayed the same. Its
fresh source-to-preflight route includes clean dependency preparation, build,
fixture, oracle, full capture, packing, numerical checks, runtime capture and
project assembly. It then uses Tuldok's actual generic CLI/HTTP driver, scoped
one-use grants, captured CPU custody and independent catalog/service reopen.

## Physics and work

| Native teacher-state diagnostic | Default lineage | Reuse lineage |
| --- | ---: | ---: |
| Raw learned qualified | 0/32 | 0/32 |
| Failed learned warm attempts | 32 | 32 |
| Qualified previous-pressure fallbacks | 32/32 | 32/32 |
| Learned method total iterations, including fallback | 6449 | 6474 |
| Full previous-pressure baseline iterations | 5425 | 5450 |
| Wasted learned warm iterations | 1024 | 1024 |
| Short temporal warm plus fallback iterations | 6449 | 6474 |
| Zero-start baseline iterations | 5583 | 5590 |
| Unknown completed-iteration attempts | 0 | 0 |

The learned method adds about 18.9% and 18.8% iterations respectively over the
full previous-pressure baseline. No solver acceleration is established. The
original residual, divergence, wall-flux, net-flux and Courant gates remain;
native diagnostic arithmetic is byte identical to the historical evaluator.
Every failed warm attempt and fallback is counted. The two evaluations' native
method clocks are 1.453 s and 1.365 s; enclosing evaluation clocks are 15.336 s and
14.786 s. Restore inference clocks are separate overlapping scopes. These do not
include preparation or managed training/restore and cannot establish end-to-end
speedup. Teacher states establish neither autonomous rollout nor final unseen
geometry/regime generalization. No GPU support or final-suite access was added.

## Environment, checks and retained boundaries

Actual environment: Linux x86-64, installed Python 3.12.3, Torch 2.8.0+cpu,
NumPy 2.3.5, PostgreSQL 16 and Rust/Cargo 1.92.0. Tuldok `f7543fb` captures and
verifies 22,501 runtime files totaling 825,750,560 bytes; runtime manifest
`ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7`.
Rheon `43ed26b6` is the latest fetched main with the unchanged reviewed pressure
patch. The external Rheon merge and Tuldok app.py edits were preserved; clean
ignored task checkouts supply the selected sources.

Qualification passes 33 unchanged GN numerical tests plus 48 configuration,
lineage, leakage, publication and process lifecycle tests, zero skips; native
builds pass 3 capture and 5 diagnostic tests, Rustfmt and Clippy with warnings
rejected. Git comparison with 03c05e8 confirms unchanged historical executable,
reservation and measured-result paths. All implementation commits are portability
and configurable lineage work; none changes model or native numerical policy.

Failed runtime-permission preparation, project-root open, input-evidence Run and
oracle-timeout preparation remain retained, with causes and fixes in the
[ledger](execution-ledger.md). The failed managed Run
`af85e2415b734f299b5799179c226527` completed cleanup before any optimizer update
and published no artifacts; its real elapsed costs remain observable. No attempt
was silently replayed or relabeled. The private PostgreSQL service is stopped
after successful runs; catalogs, snapshots, source copies and downloads are
retained deliberately for local review. No push, merge or external worktree
cleanup occurred.

These are normal Tuldok catalogs at each `WORKSPACE/catalog`. Fresh-catalog
isolation was an implementation choice for qualification, not a user requirement
or Tuldok limitation. The owner's existing UI serves `data-postgres` on port 8091
and was not modified. Generation finishes before training; frozen packets are
Project references and outputs are completed Run artifacts, not streaming Data
collection items. The runbook describes opening the experiment catalog separately.
Using the existing catalog or importing collection items remains a separate
integration operation.
