# Results ledger

Completed: C64 combined bank, C65 axis isolation, C66 shape isolation.
C53 retained; no model promoted. Target relative equation residual <=1e-6.

| Model | Fitted cases | Four-pass passing | Four-pass worst | Eight-pass passing | Eight-pass worst | Full fitting gate | CUDA trainer s | Managed workflow s | CPU phase s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| C64 combined-fundamentals | 24 | 15 | 3.2972074e-05 | 24 | 2.6550862e-09 | fail | 207.273 | 1228.379 | 37.047 |
| C65 axis-isolation | 3 | 3 | 6.1844269e-08 | 3 | 7.2336955e-12 | pass | 29.971 | 1599.302 | 36.740 |
| C66 shape-isolation | 8 | 5 | 2.2947559e-05 | 8 | 8.2277401e-09 | fail | 72.830 | 1855.143 | 36.739 |

C53 on the full 24-case bank passes 0/24 at four and 22/24 at eight, with
worst residuals 1.6063694e-3 and 2.5973784e-6. C64 reduces those worst errors
48.7× and 978×, but three cases regress at one pass and two at two passes.
C65 passes its first-grid axis fitting gate. C66 still fails three of eight
shapes at four passes and has early-pass regressions.

C66’s worst fitted case is the original 13×15×16 X fundamental: 2.2947559e-5
at four passes. That same case is part of C65’s passing bank and was fitted by
C62. Shared shape training is therefore a useful optimization target; this does
not prove that geometry alone caused failure. Population size and interactions
also differ. One seed and 200 exposures per case are bounded evidence.

## Broader regression guards

| Model | Real 4-pass /128 | Real 4-pass worst | Real 8-pass /128 | Real 8-pass worst | Manufactured 4-pass /240 | Manufactured 8-pass /240 | Manufactured 8-pass worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| C53 | 7 | 9.7855327e-05 | 128 | 1.4522194e-07 | 0 | 230 | 2.5975074e-6 |
| C64 | 1 | 2.6926827e-04 | 128 | 7.9572892e-07 | 75 | 240 | 1.5049574e-07 |
| C65 | 5 | 5.5120410e-04 | 119 | 1.2567273e-05 | 15 | 175 | 1.1196693e-04 |
| C66 | 3 | 2.4170597e-04 | 128 | 6.7476013e-07 | 25 | 215 | 7.7952552e-06 |

C64’s improved manufactured spectrum does not preserve real-request worst-case
accuracy. C65 specializes strongly to one geometry. No candidate is promoted.
No untouched scenes, new CFD captures, long rollouts, thresholds, architecture
or precision changes were used. Default deployment remains two passes.

## Work, memory and identities

- C64: 4,800 updates, 19,200 forwards, 22,272,000 forward cells; peak allocated CUDA 20,347,904 bytes, peak trainer RSS 2,414,972,928 bytes. Exactly 200 exposures per fitted case; teacher zero.
  Weights `36a03244adbe8758b4a864f41a09ccac0ca6fa4979a9bfc392c91f4e73ad4413`. Training Run `2097763eed3249c186119e920eeee788`, CPU restore `622caf928da24d3c8b0f2d83433b953d`; both completed/drained, published parity exact.
- C65: 600 updates, 2,400 forwards, 7,488,000 forward cells; peak allocated CUDA 17,261,056 bytes, peak trainer RSS 2,343,317,504 bytes. Exactly 200 exposures per fitted case; teacher zero.
  Weights `bdd4ea01dd7667f84e78924c4372e540da8b36a2b779a88f23e9c51fba4c139d`. Training Run `06df7e568a8d4cf7bd2190e82c7e531a`, CPU restore `019534bd13974678ab4c2eb11eda5de2`; both completed/drained, published parity exact.
- C66: 1,600 updates, 6,400 forwards, 7,424,000 forward cells; peak allocated CUDA 19,713,024 bytes, peak trainer RSS 2,414,211,072 bytes. Exactly 200 exposures per fitted case; teacher zero.
  Weights `69b49daa62ea6945726571dcf01eecc6d0084a89c40acd5531c5bd6607260655`. Training Run `c42bb8e6d3144cd38d140663cbda90c7`, CPU restore `0b8f06c8e62d4dd481c6b5c2183c2b5b`; both completed/drained, published parity exact.

Managed time includes admission, queueing, approval, transfer, readback and fresh
CPU restore. C65/C66 workflows overlapped; their elapsed times are not an isolated
pipeline speed comparison. Actual trainer times total about 310 seconds. C66’s
1,855-second managed time includes its client timeout and read-only recovery,
with zero new submissions, restores or retraining. Eight exact-byte matches were
reused locally and reverified by the original collector. Its initial API profile
covers only the failed initial attempt; recovery evidence is retained separately.

CPU snapshots were serial across checkpoints, one thread on the same logical
CPU; fitted/manufactured diagnostics followed on three distinct logical CPUs.
Every C53 replay reproduced all 512 historical quality rows exactly. All owned
controllers, monitors, recovery, CPU and browser workers are reaped. No weights
or full simulation fields are committed. Detailed per-case and learning scalars
remain in each immutable arm result; the aggregate HTML is a generated projection.

Next proposal: see [next-steps.md](next-steps.md).
