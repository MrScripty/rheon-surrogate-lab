# Execution ledger

2026-10-11 — Inspected exact Rheon pin and Coding-Standards, read pressure/operator
contracts and tests. Separate reference checkout used; existing Rheon worktree
untouched. Creation lookup 404, createRepository denied by integration (PUB-1).
Independent read-only review confirmed sign, units, gauge and distinct predicted
versus actual f32 divergence gates.

Implemented fixed 8×7×6 reference data generation, strict schema/typed raw arrays,
completion and SHA-256 checks, full persisted requalification and deterministic
replay. Four nontrivial seed/family distributions and two check-only cases.
`cargo test --locked`: 16 passed. `cargo fmt --all --check` and
`cargo clippy --locked --all-targets -- -D warnings`: passed. Real CLI smoke is
pending a clean source commit. No upstream files changed.

2026-10-11 — Generator committed as cc2820646601f3db51fade707bfac5a521653257.
Real release CLI generated and independently reverified 18 cases, 144 arrays,
262,944 bytes. Max actual f32 divergence 1.909211277961731e-7 s^-1; max predicted
6.203842994878528e-10 s^-1; max 77 iterations. Receipt: reports/generator-smoke.json.
Push to exact requested remote failed: Repository not found. No alternative used.

Source review found missing predicted-divergence gate in raw prediction evaluator
and row/plane wrap in a test oracle. Both fixed with regressions before checkpoint;
16 Rust tests pass. Added optional NumPy 3D CNN with train-only normalization,
cell-zero gauge, Adam, and raw prediction evaluation using Rust/Rheon. Four Python
tests pass including finite-difference gradients and normalization leakage probe.

Independent CPU source review confirmed gradient/order/split handling and found
missing evaluator provenance. Added explicit evaluator build metadata and binary
hash to model receipts, with compatible schema/pin and clean-build checks. All
16 Rust and 4 Python tests, fmt and Clippy pass after that repair. CPU smoke run
will use a fresh release build of the resulting clean commit.

2026-10-11 — CPU path committed as e45a4ceefc3ecc89531a2dcf859935c8459f652a.
Release rebuild identifies that clean commit and pinned schema/Rheon revision.
400-step one-case fit: final/initial normalized MSE 0.00010008604693783139.
200-step 8-case fit: ratio 0.03651220558054541. Held-out test mean relative pressure
RMSE 7.664304193007792; max corrected f32 divergence 17.457689086596172 s^-1.
Zero of 16 model predictions qualify in either run. This accepts the runnable
training/evaluation path and explicitly rejects a model-accuracy claim.
Compact receipts (no weights or bulk arrays) record source and evaluator hashes,
normalization, optimizer, losses, splits and timing limits. Independent reviewer
confirmed the evaluator-provenance repair and no remaining material code finding.
Publication remains blocked; no remote success is claimed.

2026-10-11 — Parent authorized further local numerical/model diagnostics while
publication remains paused pending user creation of the exact public repo.
Preserved initial bundle and source. Admitted fixed development-only baselines,
unchanged-checkpoint inspection, zero new optimizer steps and one structural
nonlocality witness (ADR 0002). No create/push retry or upstream mutation.
New model runs default to train/validation evaluation; reserved unseen seeds
9000–9015 for future frozen-model evaluation without generating their targets.

Diagnostic implementation passes 18 Rust tests and 6 Python tests, formatting,
Clippy and Python compilation. Tests cover development-only output (no test
prediction files), all reference baseline acceptance, local-invariance witness,
and consumer filtering before optional test-array reads. Independent review
accepted numerical definitions and documented the full-integrity-check exception.
Clean-commit comparison pending; fixed comparison does not train new models.

2026-10-11 — Committed diagnostic source cf0deca791add57277f98f5fe9692b8188ebc4bb.
One clean-build comparison took 0.339 s total / 0.214 s baseline workflow, zero
new optimizer steps or candidate models. Both PCGs qualified 12/12 development
cases; all approximations/CNNs qualified 0/12. Validation mean relative residual:
zero 1.000, diagonal 0.7581, existing eight-case CNN 8.6226. Measured structural
locality failure (-0.00661994 Pa reference response vs zero CNN response), nonzero
zero-input output (4.245 Pa RMS), and polynomial contribution 93.7% of zero-predictor
training squared error. Existing checkpoint predictions replay exactly; no data
ordering/normalization/gauge bug found. Report/compact receipt accepted. No new
test prediction evaluation, no reserved-test generation, no upstream/API change.

2026-10-11 — Parent authorized one global-context candidate and conditional local
warm-start patch if initial-guess quality improves. ADR 0003 fixes architecture,
335 positive modal gains, 500 Adam steps/lr0.05, equal-case pressure/residual loss,
and one final validation comparison before any training run. No analytic inverse
or eigenvalues supplied. Exported A comes from pinned public apply_full only.
19 Rust and 8 Python tests pass, including full-operator export and loss-gradient,
zero/linearity/gauge and per-case amplitude-invariant loss checks; fmt/Clippy pass.

2026-10-11 — Single preregistered run at cd6f61adceca5fca9fb479790b2931fa81782362
completed exactly 500 Adam steps in 0.225573 s, 8 training cases and one final
4-case validation evaluation; no new test predictions. Raw train 8/8, validation
1/4 qualified; validation mean relative residual 1.3426866694e-10, worst linear
gate ratio 1.7280411872, maximum actual divergence 1.1765708526e-7. Initial-guess
signal passed. No analytic inverse gains supplied, no sweep/restart/tuning.
Admitted isolated sealed-domain initial-pressure API patch and ADR 0004's fixed
20-pair per-method probe. Lab source remains pinned; reserved tests ungenerated.
Independent API review found no material defect and requested exact failure-type
assertions, now fixed. Probe review confirmed physics and timing scope and caught
a preparation-timing documentation ambiguity, now corrected. Full regressions
and actual timed probe are pending. 19 lab Rust / 9 Python tests and Clippy pass.

The isolated patch is committed locally as 61182a4c4b2ec49cbc19a952e6b1d0b86f590d21.
All 309 no-default-feature Rheon tests pass (8 new), fmt and strict Clippy pass.
Exported patch/base/tree/license/test manifest under patches/; probe driver
requires the reviewed source tree, allowing a new git-am committer timestamp.
No upstream publication and no root dependency change. Clean-build timing is next.

2026-10-11 — Clean probe source0ec9ee6166b0679e425891f5da3c060cfd730d71 plus
local patch61182a4c4b2ec49cbc19a952e6b1d0b86f590d21 completed the fixed20-pair
per-preconditioner probe. All960 repeated corrected solves qualify. Per12-case
batch: Jacobi zero920 iterations / warm5; SGS zero359 / warm3. Median full measured
batch wall: Jacobi21.127/3.523ms; SGS14.422/3.349ms. Maximum warm relative residual
8.399e-11, predicted divergence2.992e-10, actual f32 divergence1.909e-7. Inference,
file transfer/cleanup, native child startup/read/hash/RHS build/allocation, solve,
correction and qualification are inside wall time. Preparation0.052056s measured
separately; parent startup/imports, training, generation and builds excluded.
A8/A9 satisfied as bounded local development evidence; no acceleration/general CFD
claim. Patch dry-run applies to34235d3. Original Rheon and reference checkouts
remain clean, pins unchanged, reserved tests ungenerated, publication paused.

2026-10-11 — Parent authorized a bounded independent final evaluation to close
experiment1: fresh API review, analytic spectral baseline, frozen candidate/code/
tolerances/protocol, then reserved seeds9000–9015. No retraining/test tuning or
upstream/GitHub publication. Fresh review of exact patch61182a4 independently
reran23 initial-pressure/pressure/operator tests, all passing; no new issue.
ADR0005 fixes six paths, same physical gates,12 balanced Williams rows per each
fresh-Python and resident-model regime; all overhead and cold/cache distinctions
are explicit. Added final schema with fixed seed plan/freeze hash, preserved
v1 contract; metadata tests materialize no reserved fields. Archived the exact
2950-byte candidate checkpoint as the only tiny final-checkpoint exception.
Analytic modes agree with a hand chain and public Rheon A on geometry only.
20 Rust and13 Python tests pass; implementation review/development smoke precede
freeze. Reserved data is still absent; publication remains paused.

Development smoke at38b06f0 completed72 quality and288 timed projections on12
existing train/validation cases. Analytic and all fourPCG paths qualify12/12;
raw learned9/12 retains the known three validation failures. Fresh process costs
are larger for learned/spectral paths than zero-PCG because initialization is
included; resident-model calls are lower in this tiny scope. This did not change
the candidate or protocol. Independent review found no material blocker and
verified receipt hashes/summary/timing accounting. Metadata/wording corrections
and a source/binary/checkpoint tamper-rejection test precede the committed freeze.

2026-10-11 — Final admission complete:20 Rust/14 Python tests, fmt and strict
Clippy passed; independent API/source/development review approved. Clean source
44c470178a7411cea9b812fe485020af59094ff6 built the exact frozen binaries.
Freeze-only commit af6d2a10cd94b2a69b44bc70b470ea611ecc069c preceded any reserved
field generation; independent reviewer verified source/checkpoint/binary hashes
and the absent final output directory. Then `python3 scripts/final_evaluate.py
generate` and `python3 scripts/final_evaluate.py run` each completed once.

Generated and verified16 final-test cases/128 arrays/233728 raw bytes, seeds
9000–9015; manifest620cb224547e93c67e2b54a8aa28250eabcd0f26430d63d5d16baae99a2d3192.
Final comparison:96 quality+2304 timed projections. Analytic and both zero/warm
PCG paths qualify16/16; raw learned15/16 fails seed9005's linear gate by3.70%.
Each warm preconditioner needs1 total iteration; zero Jacobi1210 andSGS472.
Maximum actual f32 divergence1.6142924619e-7 s^-1. No solver errors, no tuning,
no source/checkpoint/tolerance changes or further optimization after test access.

Fresh Python/model medians per16-casebatch: analytic81.702ms, zeroJ59.700ms,
warmJ83.866ms, zeroSGS51.117ms, warmSGS81.277ms, raw82.834ms. Resident medians:
5.197,27.975,5.477,19.723,5.458,5.191ms respectively; full startup/quality/call/
teardown amortization is separately reported. Cold OS caches are not claimed.
No learned advantage over the analytic baseline or simulator acceleration is
established. Independent read-only final audit recomputed hashes, numerical
fingerprints, acceptance, balanced order, statistics and accounting: no material
blocker. Final receipts/report/outcome preserve both positive and negative results.
Experiment1 is closed; final seeds exposed. Publication stays paused after the
known403, root dependency unchanged, isolated API unpublished, original Rheon
checkout untouched. A new recoverable source bundle and separate local evidence
archive preserve closure without adding bulk data or binaries to git.
