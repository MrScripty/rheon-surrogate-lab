# Native pressure follow-up and periodic numerical correction

All planned workers exited successfully and reaped their native children.
A worker may report a scientific pressure rejection; that is kept separately
from infrastructure failure and is not counted as a complete trajectory.

| Policy | Complete repeats | Median diagnostic loop s | Median AI track s |
| --- | ---: | ---: | ---: |
| native-control | 3/3 | 18.119 | 10.985 |
| full-numerical | 3/3 | 16.798 | 9.651 |
| neural | 3/3 | 43.171 | 36.001 |

| Policy | Outcome | Accepted AI steps | Worst pressure trajectory error |
| --- | --- | ---: | ---: |
| periodic-4 | completed | 497 | 6.672442443700344e-07 |
| periodic-16 | completed | 497 | 6.473864079349802e-07 |
| periodic-64 | completed | 497 | 6.91385820942694e-07 |
| before-4 | pressure_rejection | 0 | None |
| after-4 | pressure_rejection | 0 | None |

Timing includes the instrumented dual-track diagnostic loop and IPC. Repeated
comparisons use the same scene and instrumentation in a predeclared shuffled
order. The report retains reference/candidate work separately. This does not
establish a single-track production speedup.

Pressure drift thresholds of 0.1% and 1% are diagnostics, not user-specified
application limits. The present wire does not expose velocity/tracer drift.
Those measurements and the remaining trained cohorts are still required.

[Per-run results and evidence hashes](followup-results.json).
