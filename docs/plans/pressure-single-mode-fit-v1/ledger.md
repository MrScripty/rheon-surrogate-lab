# Results ledger

## C62 capacity screen

Exact C53 initialization, one manufactured fundamental-X field, counts 13×15×16,
cubic spacing 0.0625 m, original train-only amplitude/scalars. Fresh Adam, 2,000
CUDA float32 updates, four shared corrections, no teacher or companion losses.
Settings and effective objective: [method](method.md),
[configuration](../../../experiments/pressure-single-mode-fit-v1/config.json).

Training Run `f4011890960040359b6ab83db245dd6e` and fresh CPU restore Run
`456cdcccb54c4e99b2bdebc19393f991` completed and drained. Artifact parity is exact.
Weights SHA256: `ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5`.
Actual optimizer work: 2,000 examples, 8,000 network forwards, 24,960,000 voxel-forwards.
No teacher forwards occurred. Trainer elapsed 76.528 s; complete managed workflow
990.457 s; local CPU screen 33.800 s. Peak CUDA allocation 16.337 MiB.

| CPU passes | C53 fitted residual | C62 fitted residual |
| --- | ---: | ---: |
| 1 | 2.1848176067e-1 | 1.8067672892e-4 |
| 2 | 4.0820314653e-2 | 2.2860897926e-6 |
| 4 | 1.6063694048e-3 | 8.8232528619e-9 |
| 8 | 2.5973783547e-6 | 2.5454202517e-12 |

The capacity/optimization gate passed. The CUDA trace crossed the four-pass target
by update 100. CPU and CUDA residuals are measured separately; exact published
artifact parity does not assert identical CPU/CUDA floating-point evaluations.

This specialist regressed outside its fitting case. On 128 consumed real requests,
C62 qualified 1/128 at four passes and 111/128 at eight (C53: 7/128 and 128/128).
Worst residuals became 7.6301924563e-4 and 8.3755798932e-6. On 240 manufactured
cases, eight-pass qualification fell from 230 to 155, with worst residual
1.2846496251e-4. C53's 512 historical scalar quality rows replayed exactly.
C62 is not promoted. The result establishes that the network can learn the tested
fundamental correction; it does not establish generalization or imply that the
mixed training problem is solved. No original untouched holdout was consumed.

The accepted next slice is C63: retain C60's budget, seeds, objective, original
banks and outer sampling order, changing only supplemental spectral exposure.
The immutable [capacity decision](capacity-decision.json) gates its preparation.
