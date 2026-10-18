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
