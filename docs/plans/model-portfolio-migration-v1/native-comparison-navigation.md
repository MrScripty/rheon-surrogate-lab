# Read the migrated full-state comparisons

Open the [native project UI](http://127.0.0.1:37645/project-files), then find
**Retained measurements** and choose **Load measurement sources**. The project
contains both pressure-only and full-state sources. Select compatible sources
within one population; pressure proposals and complete simulation outputs are
different tasks.

The seven new full-state sources are identified by these packet hash prefixes:

| Population | Select these sources together | Subjects | Cases per subject |
| --- | --- | --- | --- |
| Expanded diversity | `314667992e14`, `f8f381a67197`, `0a529e14841c`, `f33aaebc0e30`, `4e4bf7959fa6` | Cell-coordinate baseline and eight residual/execution variants | 2,120 |
| Temporal schedules | `c8f4de804e56`, `4ae64e3a99da` | Fine-reference, half-pass and single-pass variants | 180 |

There are now 19 measurement sources. Two additional sources contain freshly
measured complete-generator GPU blocks:

| Source prefix | Executions | Physical block cases |
| --- | --- | ---: |
| `96bd9889339a` | Cell bound and cell coordinate | 384 |
| `846e98fb97c7` | Temporal eager/graph, each with fine/half/single schedules | 72 |

Select one of these sources and choose **Median complete GPU time per physical
output · ms/output** or **Worst in K-output block · Full-state pressure error**.
Filter **Grid density**, **Grid X/Y/Z**, **Physical outputs per block** and
**Simulation output cadence** independently. All K output leads contribute to
the accuracy maximum; timing includes the whole block and divides only by K.
Full block median, per-output p95 and maximum are available separately. Actual
internal advances are observation attributes, not another physical-output count.

The immutable packet context incorrectly describes the temporal masters as one
training and one development scene. Both original captures and all native case
split fields say **train**. Treat both as consumed training scenes; an explicit
correction is adopted in the portfolio overview and its actual UI read passes. The numerical rows and
their case split fields are correct. No held-out generalization or600-output
drift follows from these block comparisons. The seven older accuracy-only
sources below retain their original unavailable timing cells.

For the measured C106/C107 complete
GPU pressure calls, select only source `1e2f77bceefc`, then load it. Choose
**Complete GPU pressure-call case median · ms/request**. Each observation is
the median of 40 calls for one request; the plot/table show the maximum of
those case medians, rather than a single-call tail or the model-wide median.
Both models have 128 requests, separately filterable by cells per metre and
grid dimensions. **Mean-free pressure field error (relative L2)** shows the
corresponding retained accuracy. The native maximum inspector links the exact
case and original scalar evidence. These pressure-call times cannot fill the
unavailable full-state timing slots described below.

Use the multiple-selection control to select the appropriate group, then
**Load selected sources**. Select a metric and press **Compare metric**.
The chart and maximum table use every matching observation, independently of
the 50-row observation page. A worst relative RMS error is the largest case
ratio, not the largest individual voxel error.

For a matched comparison of the diversity baseline and residual variants, open
**Filter declared attributes**, find **Exact cell/control shared population**,
and choose **Exact value → true**. All nine subjects then have the same 1,018
measured cases. In the unfiltered 2,120-case population, the baseline has 1,102
unavailable cases; those cases must not be treated as accurate or omitted from
the coverage count.

Grid density is **cells per metre**. Domain X/Y/Z cells are separate attributes.
For example, select density 100, X/Y/Z 32 and K4 to examine four-output blocks on
a 32 × 32 × 32 grid at 100 cells/m. Physical output duration and outputs per simulated
second are time-discretization attributes, not wall-clock generation speed.
A K4 block at 30 outputs/s spans 4/30 seconds; at 120 outputs/s it spans 4/120 seconds.

Click a subject's **inspect maximum** button to see its exact case, execution
attributes, scalar value, source packet and original evidence. Select by model
name rather than assuming a particular ordering. Internal advance counts and
residual-network calls are recorded separately from K physical outputs.

Select **Complete GPU latency per physical output** or **Complete K-output GPU
block latency** to inspect timing availability. These imported captures have
no fresh qualified latency: the chart correctly shows no measured values.
Historical anchor timings and browser/API request durations cannot fill those
slots. They require separately qualified GPU measurements.

“Measured” means a finite value was recorded. It does not mean an accuracy
target passed. Numerical measurement states and the recorded diagnostic-pass
attribute are distinct; application-specific drift limits remain undecided.

The [benchmark contract](benchmark-contract.md) owns metric and timing scope.
The [API qualification](reports/native-full-state-comparison-API-01.json)
checks 24 real native reads against the original packets. The latest
[execution state](work-state-71.json) records the separate GUI acceptance and
remaining portfolio work.
