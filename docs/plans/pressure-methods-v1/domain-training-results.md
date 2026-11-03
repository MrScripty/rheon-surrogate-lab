# Matched domain-data comparison

Both arms start from C53 and run the same 1,600-update two-pass objective. The small-only arm uses 16 original parents; the expanded arm retains them and adds four 32³/64³ parents at 8/16 cells per metre.

At eight passes and equation tolerance 1e-6:

| Population | Small-only qualified | Expanded qualified | Small-only worst residual | Expanded worst residual |
|---|---:|---:|---:|---:|
| consumed128 | 128/128 | 123/128 | 2.32039e-07 | 6.22023e-06 |
| domain32 | 0/4 | 4/4 | 4.79212e-05 | 1.93069e-08 |
| domain64 | 0/4 | 4/4 | 0.000221824 | 6.23699e-08 |
| domain-odd | 2/4 | 0/4 | 1.58952e-06 | 0.000753903 |

The larger cubes improve strongly, while the original small grids and new larger rectangular grids regress. This is a promising domain-specific result, with no model promotion. Native larger-domain acceptance/drift and pass-count/shape follow-ups remain required.

The full numerical spectral control qualifies every request, including every new larger-domain request. Its prepared-input cost remains much lower on these sealed uniform grids. The measurements include final residual evaluation and cannot be compared directly with the original chart’s timing scope. CPU compilation and other campaign work ran concurrently; small runtime differences require a quiet repetition.

[Raw summaries and evidence](domain-training-results.json).
