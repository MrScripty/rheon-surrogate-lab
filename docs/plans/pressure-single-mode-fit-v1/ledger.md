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

## C63 matched frequency-curriculum screen

C63 starts from C53 and matches C60 except supplemental pattern content. Static
qualification and actual training both confirm unchanged original banks, outer
sampling order, normalization, inputs, seeds, objective and work: 1,500 CUDA
updates, 30,000 optimizer-loss network forwards, 229,629,992 voxel-forwards.
Actual supplemental draws: 579 fundamental, 542 other low, 379 high. C60 had
131 fundamental draws. All 1,024 supplemental labels passed independent operator
checks; combined resident tensors were 567,504,896 bytes.

Training Run `ef4298f2e657491080732e7aaf089895` and CPU restore Run
`00b8fcac2b364acfa5b370cef3c8d3ad` completed and drained, with exact artifact parity.
Weights SHA256: `0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2`.
Trainer elapsed 343.401 s; full managed workflow 1,378.779 s; local CPU screen
36.487 s. Peak CUDA allocation 855.985 MiB. CPU inference uses one thread per
worker, with three independent diagnosis workers after isolated timings.

| Checkpoint | Four-pass qualified /128 | Four-pass worst residual | Eight-pass qualified /128 | Eight-pass worst residual |
| --- | ---: | ---: | ---: | ---: |
| C53 | 7 | 9.7855326615e-5 | 128 | 1.4522193519e-7 |
| C60, matched control | 9 | 1.2951174047e-4 | 128 | 2.4265158816e-7 |
| C63 | 8 | 1.1172466071e-4 | 128 | 2.0561491919e-7 |

C63 improved worst real-request residual versus C60 by 13.73% at four passes
and 15.26% at eight, but remained 14.17% and 41.59% worse than C53. Four-pass
median residual was 7.8306618012e-6, about 19.10% lower than C53; this median gain
does not satisfy the worst-case guards. The same-phase median four-pass CPU times
were 13.177 ms for C53 and 13.166 ms for C63: essentially unchanged. Historical
C60 and earlier C62 timings remain labelled by their recorded stage.

On 240 manufactured cases, eight-pass qualification improved from C60's 200 to
C63's 210, below C53's 230. C63's worst was 3.6682336388e-6. No manufactured
case qualified at four passes. Median paired four-pass residuals improved by
about 10.1%, 14.3%, 16.0% on fundamental X/Y/Z modes versus C60; the 4,4,4 mode
regressed by about 22.7%. At eight passes those fundamental improvements were
14.6%, 28.1%, 25.6%, while 4,4,4 regressed 22.0%. Complete mode/shape/amplitude
rows are retained in [results](results.json).

The predeclared selection gates failed. No model was promoted, no independent
scenes or longer training were started, and no original untouched holdout was
consumed. C53 remains baseline. The result supports spectral exposure as a useful
training variable under this matched seed, while showing an unresolved transfer
and competing-objective problem; it does not prove a universal causal mechanism.

Next proposed test: fit the 24 fundamental grid/axis combinations (eight grids,
three axes) with full fitting weight and no competing losses, keeping the cubic
operator and C53 initialization. First establish shared fitting capacity; then
test retaining that accuracy when real-frame feedback returns. If the 24-case
fit fails, separate axis transfer on one grid from shape transfer on one axis.
Do not extend the current mixed fit or change architecture before that diagnosis.
