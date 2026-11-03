# Pressure correction work on the frozen 128 requests

Twelve captured checkpoints were executed on CUDA at their saved default and at
explicit one-, two-, four- and eight-correction budgets. All 7,680 saved pressure
fields were independently rehashed and rescored with a separate NumPy finite-volume
stencil. The largest relative disagreement with the worker metrics was
1.4622e-15, within the stated 1e-12 relative / 1e-30 absolute comparison tolerance.
No weights changed, no optimizer ran and no physical simulation time advanced.

C37 saves one correction by default. C44, C53 and all nine fine-tunes save two;
the fine-tunes trained through four corrections. Default-to-default comparisons
describe the deployed proposals, but C37 versus its fine-tunes also changes work.
At matched two/four/eight corrections, all three C37 fine-tunes improve both worst
equation and pressure errors. At one correction they improve worst equation error
but worsen worst pressure error. All three C44 and all three C53 fine-tunes are
worse than their own parents on both worst metrics at every explicit tested budget.
This is a conditional result on these consumed requests, not an identification of
the sole training failure or an independent generalization result.

| Checkpoint | Pressure error, 2 corrections | Pressure error, 8 corrections | Equation error, 8 corrections |
| --- | ---: | ---: | ---: |
| C37 | 0.01500141 | 3.066736e-05 | 2.289315e-05 |
| C37-U | 0.01129882 | 8.698292e-06 | 2.057524e-06 |
| C37-A | 0.01109277 | 2.662652e-06 | 2.395167e-06 |
| C37-D | 0.01118007 | 4.639894e-06 | 2.5996e-06 |
| C44 | 0.01345072 | 6.910625e-07 | 3.944669e-07 |
| C44-U | 0.02205282 | 2.570321e-06 | 1.452541e-06 |
| C44-A | 0.02138458 | 2.999841e-06 | 1.685612e-06 |
| C44-D | 0.021524 | 7.228318e-06 | 2.724462e-06 |
| C53 | 0.006021537 | 3.117936e-07 | 1.4522e-07 |
| C53-U | 0.01796591 | 2.262638e-06 | 6.747607e-07 |
| C53-A | 0.01238743 | 2.409944e-06 | 9.14918e-07 |
| C53-D | 0.01247961 | 1.973248e-06 | 8.320025e-07 |

Errors are worst relative L2 per request; pressure error is mean-free. Full
precision and density-separated maxima are in the independent report. Grid density
is 8 or 16 cells/m; domain dimensions are the original 128 case shapes, at most 16 cells
on an axis. These results do not qualify larger-grid, temporal or own-history behavior.

The unnamed C53-U training checkpoint has the same weight archive as C66. Loading
C53-U through its own captured source and original training receipt gives byte-exact
C66 default outputs on all 128 requests. This establishes tested inference equivalence,
not identical training histories, a new cohort alias, or a fabricated CPU restore.

No inference timing is claimed for this matrix. Correction count is actual root
network-call count per pressure equation, not simulation steps or generated frames.
The existing C106/C107 timing pilot has a different, explicitly measured complete
GPU pressure-call boundary. Full-state and K-output latency remain separate.

[Independent numerical report](reports/pressure-correction-work-01.json),
[preregistered protocol](../../../runs/model-portfolio-migration-v1/pressure-correction-work-protocol-01.json).
