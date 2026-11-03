# Detached correction training outcome

The nine-fit screen completed on 2026-10-18. **Retain C53.** No fit met the full
capacity gate, no candidate qualified for independent scenes, and this experiment
established no reduction from eight to four corrections. All training and fresh
CPU restore workflows completed successfully.
[All results](execution/results.md), [scalar evidence and gates](execution/results.json).

## Detached results from each starting checkpoint

The target is relative equation residual at most `1e-6`. The fitting population
has eight manufactured X-fundamental cases; the real regression population has
128 consumed requests. U is unchanged four-pass fitting, A adds attached
auxiliary supervision, and D detaches that auxiliary history.

| Starting checkpoint | U fitted at four passes | A fitted at four passes | D fitted at four passes | D real at four passes | D real at eight passes |
| --- | --- | --- | --- | --- | --- |
| C53 | 5/8 | 6/8 | 5/8 | 2/128 | 128/128 |
| C44 | 6/8 | 5/8 | 5/8 | 2/128 | 121/128 |
| C37 | 4/8 | 4/8 | 4/8 | 0/128 | 120/128 |

Every fit passes all eight fitting cases at eight corrections, but every fit
fails at least one early-pass retention check as well as the four-pass fitting
target. This is why a lower final loss or a higher success count does not satisfy
the capacity gate. The original C53 achieves 7/128 real requests at four passes
and 128/128 at eight. The original C44 also achieves 128/128 at eight; all three
C44 fine-tunes regress on that gate. C37 starts at 112/128 and improves to
119/128, 121/128 and 120/128 for U, A and D respectively, while still missing
the required 128/128. That improvement is not specific to detachment.
[Per-case and population checks](execution/results.json).

## What detachment changed

Compared with the matched attached arm, D reduces the worst four-pass fitted
residual by 18.7% from C53 and 48.4% from C37, but makes it 2.04 times as large
from C44. Compared with U, D makes that worst fitted residual 13.6% worse from
C53 and 22.5% worse from C44; it improves it by 9.5% from C37. These are different
effects across starting checkpoints, not a consistent detachment benefit.
[Paired comparisons](execution/results.json).

C53 illustrates the tradeoff. D improves the `13×15×16` fitted case from
`2.29e-5` under U to `5.15e-6`, but worsens the `16×7×11` case from `1.89e-5` to
`2.61e-5`. Both still miss `1e-6`. On real requests, C53 D improves the median
four-pass residual by 13.6% relative to U while worsening the worst residual by
6.2%. Against the original C53, its worst real residual is 2.62 times as large.
The data support a tradeoff between cases; they do not identify its gradient or
optimizer cause.
[Raw fitting measurements](../../../runs/pressure-detached-correction-v1/cpu-01/c53-d-fitted.json),
[control fitting measurements](../../../runs/pressure-detached-correction-v1/cpu-01/c53-u-fitted.json),
[paired real comparisons](execution/results.json).

All variants also regress against C53's manufactured eight-pass count of
230/240. The detached variants achieve 205/240, 200/240 and 140/240 for C53,
C44 and C37 respectively. Specialist fitting improvements did not preserve the
broader diagnostic behavior. No model was promoted and no unused whole-scene
holdouts were consumed.
[Regression evidence](execution/results.json).

## Verification and scope

The C53 U weights exactly reproduce retained C66, providing a numerical control
for the wrapper. All nine fits use the same cases, order, schedule, precision and
normalization, with 1,600 fresh-Adam updates each. Actual work totals 14,400 updates,
67,200 network forwards and 77,952,000 forward cells. The summed trainers took
737.05 seconds, and complete managed workflows took 2,383.66 seconds. A and D have
matched forwards; their measured wall times remain separate from training work.
[Execution records](execution/execution.json), [per-arm work](execution/results.json).

Independent checks recompute the gates from raw scalar measurements and verify
all 18 managed terminal Runs completed and drained. Four reporting and worker
lifecycle tests pass. [Verification](execution/verification.json),
[test evidence](execution/report-tests.json).

Evaluation used the common train-calibrated bank after fixing the older C44
loader's mixed-development-view incompatibility. The corrected helper exactly
reproduces all 32 original C53 case/pass fitting measurements. Training was not
repeated, and successful CPU results and the failed attempt were preserved.
[Recovery evidence](execution/evaluation-recovery.json),
[execution ledger](execution-ledger.md).

This is a single-seed screen for each of three starting checkpoints. It tests an
auxiliary loss of weight `0.05` on depths zero through three while retaining full
four-pass training. Those states already occur in the original training chain.
The outcome does not test wholly detached training, extra states beyond that
chain, or the paper's separate denoising carry. It does not establish a general
failure of Fixed-Point Forcing.
[Experiment definition](training-experiment.md), [paper adaptation](training-method-notes.md).

## Next step

Do not lengthen these fits or start the conditional broad retention stage.
The declared capacity condition did not pass. Before another training proposal,
use the retained per-shape and per-pass results to investigate why improvements
on one hard shape accompany regressions on another and on early corrections.
Any new test should change one mechanism and preserve the same controls and
physical gates. A longer run alone would not isolate that tradeoff.
