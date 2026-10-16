# Earlier pressure corrections: findings

The matched CUDA fits and trajectory comparison are in progress. No checkpoint
is promoted from the explanatory diagnostics below.

## Amplitude and frequency

The answer differs for isolated modes and mixtures. On three manufactured grids,
globally reducing a pure low spatial mode from1Pa to1e-8Pa preserved approximately
the same relative error. In a mixture, a stronger high mode created low-mode
error that overwhelmed a weak low signal. That is consistent with an effective
error floor caused by learned cross-band leakage; it does not establish that
float32 cannot represent weak modes.

For the frozen late model after8 single evaluations on9×11×33:

| Input | Low-mode coefficient error | Relative to the supplied low coefficient |
| --- | ---: | ---: |
| Pure low,1Pa | -0.001838Pa | 0.184% |
| Pure low,0.001Pa | -1.838e-6Pa | 0.184% |
| Pure low,1e-8Pa | -1.838e-11Pa | 0.184% |
| High only,1Pa | -0.018253Pa | Low input absent |
| Low0.001Pa + high1Pa | -0.018256Pa | 1826% |

The float32-rounded exact high-only pressure's low coefficient error was about
1.23e-10Pa, far below the learned leakage on this case. Physical pressure is
accumulated in float64 even though network computation is float32, so that
comparator is not a universal inference precision floor. Leakage declines with
additional learned corrections: high-only low error is3.066Pa after1 call,
-0.3052Pa after4, -0.01825Pa after8 and-0.001096Pa after12. This is a stationary
manufactured test, not an independently advanced fluid trajectory.

The temporal test replays16 consumed recorded teacher trajectories,49 states
apiece. Both signal and error use cell-wise demeaned periodic Hann32 windows,
hop8, identical gauge removal and PSD unitsPa²/Hz. At the lowest positive bin,
1.875Hz, late-model8-call median relative RMS error across weak/middle/strong
signal tertiles is3.10e-7 /4.99e-7 /2.71e-7. Those tertiles do not show monotonic
weak-signal degradation. Pooled windows/scenes are correlated; associations do
not identify a causal noise floor. These short windows cannot resolve slower
cycles. Temporal frequency and spatial cosine mode index are distinct.

No fitting examples or loss weights were selected from these observations.
The already admitted first-correction ablation remains unchanged.

[Methods](execution-ledger.md) · [Raw scalar probes](reports/amplitude-late.json) ·
[Temporal summaries](reports/amplitude-baseline-summary.json) · [Current status](reports/current.json).
