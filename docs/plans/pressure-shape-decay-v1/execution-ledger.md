# Execution ledger

2026-10-17 America/Vancouver: admitted the recorded C64–C66 follow-up on a new
branch from 90bf4e1. Preserved pre-existing dirty files. Training schedule is the
only proposed learning-policy change. Gradient measurement and qualification are
pending; no new CUDA job admitted yet.


2026-10-18 06:39 UTC: both CPU gradient workers completed and were reaped.
C53/C66 used the same C66 four-pass objective, zero optimizer updates, unchanged
inventories and float32 stored vectors with float64 dot products. C53 has 12/28
opposing pairs (2.486 s); C66 has 16/28 (2.482 s). At C66, case 0 versus cases 6/7
has cosines -0.830/-0.908. These are local Euclidean gradients, not Adam dynamics.
The 16 MiB guard and an initial config identity comparison caught two preflight
issues; fixes and original failure logs are retained in issues.md.

C67 passed full bank qualification plus matched-control proof. Captured numerical
sources, normalization, cases, order, seeds and work match C66. Only the schedule
changes. The no-matching-proof admission guard rejects before any child startup.
A fresh visible Tuldok Project was prepared (d23b16aca66a4233a9d1522e954fb4b3),
then a single managed CUDA execution admitted (e17624f982b54e858698207227010410).
Existing lifecycle owns cancellation/reaping, restore and artifact readback.

2026-10-18 06:56 UTC: C67 completed CUDA training and fresh CPU restore.
Training Run e17624f982b54e858698207227010410 and restore Run
849bdddaeff3476a88b37813e311da4a both completed/drained. The controller exited 0
and reaped its child; no retry or recovery was required. All five published
artifact-parity checks passed. All 37 artifacts were verified, including nine
verified local-byte reuses. First 800 CUDA updates / ten bank observations
(320 case/pass measurements) exactly replay C66 before the schedules diverge.
Actual work matches: 1,600 updates, 6,400 network forwards, 7,424,000 forward cells,
200 exposures per case. All 370,016 parameters retain the same architecture.

Actual timings: CUDA trainer 72.386 s (optimizer loop 68.233 s); managed workflow
1,223.625 s; local CPU evaluation 36.147 s. Peak allocated CUDA memory 19,713,024
bytes. Managed time includes admission, preflight, polling, transfer and restore;
it is not GPU training time. Restore preflight took 102.654 s, admission 210.781 s.
CPU snapshot timing was serial per checkpoint; three one-core fitting/diagnostic
workers followed. Every owned CPU worker returned 0 and was reaped.

Restored fitting gate FAILED: four-pass 6/8 cases qualify (C66: 5/8), but the worst
residual is 3.058697653e-5 (C66: 2.294755891e-5, 33.3% worse). First grid improves
about threefold to 7.694e-6 but still misses 1e-6; case 7 (16×7×11) worsens to
3.059e-5. Eight-pass fitting is 8/8. Case 7 also regresses against C53 at one pass
(+41.0%) and two (+39.6%). Thus later decay alone does not solve shared fitting.
The conditional C68 gate rejects this actual failed decision before preparation;
no C68 Project or Run was created, and no longer training was attempted.

| CPU population / passes | C53 | Retained C66 | C67 |
| --- | ---: | ---: | ---: |
| Eight fitting cases, 4 passes ≤1e-6 | 0/8 | 5/8 | 6/8 |
| Fitting worst relative residual, 4 passes | 1.60637e-3 | 2.29476e-5 | 3.05870e-5 |
| 128 real requests, 4 passes ≤1e-6 | 7/128 | 3/128 | 0/128 |
| Real worst relative residual, 4 passes | 9.78553e-5 | 2.41706e-4 | 2.34703e-4 |
| Real requests, 8 passes ≤1e-6 | 128/128 | 128/128 | 128/128 |
| Real worst relative residual, 8 passes | 1.45222e-7 | 6.74760e-7 | 6.57849e-7 |
| 240 manufactured cases, 4 passes ≤1e-6 | 0/240 | 25/240 | 30/240 |
| Manufactured worst, 4 passes | 1.60652e-3 | 2.82715e-3 | 2.80505e-3 |
| Manufactured cases, 8 passes ≤1e-6 | 230/240 | 215/240 | 210/240 |
| Manufactured worst, 8 passes | 2.59751e-6 | 7.79526e-6 | 8.02978e-6 |

C53's 512 real-request quality rows replay exactly. No untouched scenes were
consumed. Real-request measurements use fixed inputs, not accumulated CFD drift.
Retained C66 timings come from an earlier CPU phase and support no matched speed
claim. C53 remains deployed; C67 is not promoted. C67 weights identity (weights
remain local): 0e32d48fc84fa555459559198bc782cb6d1680c2953dc1e7883a3b71e93885b8.

Evidence verification passed: source identities, finite CPU-only measurements,
actual work, exact shared CUDA prefix, recalculated fitting gate, completed/drained
custody, artifact parity and reaped workers. The interactive report links from
cohort progress and compares both CUDA traces, shape gradients, fitted results,
real/manufactured regressions, time and memory. The final browser receipt verifies
its exact HTML hash, selected cases/pass counts, retained scalar labels, keyboard
access and mobile overflow. Historical results and unrelated dirty files remain
unchanged. Commands and environment: commands.md. Follow-up proposal: next-steps.md.

Closure review: prior C64–C66 historical evidence verification still passes for all
three screens. Reviewed the complete experiment range from explicit base
90bf4e12789afed7a8166206cae65140ec580ce3, including the qualified preparation,
reporting and final evidence slices. Scope is the declared new-study directories,
model-ID reservation and owned report link only; the historical dashboard data
payload is unchanged by these commits. No weights, datasets, credentials or
binary screenshots are staged. No history rewrite, push or merge is performed.
The private branch is retained for Jeremy's review under the plan's existing
integration/retirement contract; no worktree was created. Tuldok activity was
finalized as completed with the failed gate and unchanged deployed C53 stated.
