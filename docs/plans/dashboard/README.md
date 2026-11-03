# Offline evaluation dashboard

Open `../cohort-progress.html`. It remains a self-contained snapshot with no
runtime CDN, web server or build tooling required by the viewer.

Edit these sources instead of editing generated JavaScript inside the HTML:

| Source | Responsibility |
| --- | --- |
| `shell.html`, `panels.html`, `pressure-controls.html`, `styles.css` | Page layout, analysis panels and styling |
| `measurements.js` | Pure comparison eligibility, Pareto frontier, weighted preference and population-safe measurement selection |
| `comparison.js` | Pressure accuracy–cost and independent training-progress charts |
| `multistep.js` | Full-state generated-step comparison and measurement inventory |
| `navigation.js` | Grouped navigation, mobile analysis selector, per-tab filter persistence and reset |
| `app.js`, `filters.js`, `pressure.js` | Shared chart helpers, setup, cohort selectors and complete hybrid comparisons |
| `domains.js`, `drift.js`, `frequency.js`, `policies.js`, `budgets.js`, `history.js`, `studies.js` | Retained analyses grouped by evaluation question |
| `../full_ai_study_ui.js` | Campaign pressure executions, temporal accuracy and causal-request diagnostics |
| `../pressure-precision-drift-v1/chart.js`, `section.html` | Precision diagnostics |
| `../evaluation_measurements.py` | Explicit CPU pressure timing-capture/population catalog derived from retained records |
| `../dashboard_build.py` | Assemble the standalone HTML, preserving all experimental records |

Run `python3 dashboard_build.py` from `docs/plans` after presentation changes.
`python3 refresh-cohort-progress.py` imports completed retained cohorts and then
uses the same renderer, so a refresh preserves the layout. Neither operation
trains models or runs simulations. Experimental artifacts are read-only inputs.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_dashboard.py test_full_ai_chart.py test_best_error_metrics.py test_drift_pressure_metrics.py`.
Node is used to verify the same pure JavaScript interface used in the browser.
For desktop/mobile interaction and reload checks, run
`python3 dashboard_check.py --browser /path/to/chromium` from `docs/plans`.
The harness uses a temporary profile and snapshot, without modifying user preferences.

Comparison rules:

- Historical original proposal timings stay exploratory. A shared accuracy
  population does not establish identical timing conditions.
- Repeated pressure comparisons use one timing capture and population. The
  default is the recorded C107/C125/C126/C127 warm benchmark; choose historical
  mode to see every captured full-AI pressure cohort.
- Diagnostic frontier filters reference this fixed common pressure benchmark
  plus the separately measured complete V5 and V6 hybrid frontiers. They do not
  inherit a different tab's selection. Diagnostics default to their available
  data, since many newer pressure cohorts have no accepted rollout capture.
- Best-case accuracy still excludes grids with all axes at most six cells.
  Raw minima and original worst scores remain in the records.
- Never copy historical best scores or case counts into a fresh population.
- Pressure correction stages, accepted pressure solves, full simulation steps
  and blocks of generated states have different cost scopes.
- Full-state temporal fits retain joint accuracy failures and missing latency.
  Failed fits cannot establish an accepted simulation speedup.
- Repeat ranges are ranges of repeat medians, not confidence intervals.
- GPU training does not establish GPU inference timing. Pressure comparisons
  show missing GPU timing rather than substituting CPU measurements. The
  Expanded CPU & GPU models tab uses actual full-state GPU measurements, with
  resident, event and complete-entry clocks kept separate.
- The frozen October 19 audit covers the original chart snapshot. It remains
  linked as evidence; refactoring the display does not rewrite that report.

The Multi-step prediction tab includes completed endpoint/reuse sharing fits
and the step-scaled, full-state MSE, increment MSE and curl-increment follow-ups.
Timing-capture selection isolates the original versus sharing CPU measurements;
fitting follow-ups without timing remain accuracy-only. Request-component
selection keeps the causal and factored request predictors separate from full-state models.

Pressure-expert and decoder full-state fits retain inherited training context.
Their recorded rollout accuracy is shown without a latency claim; incomplete
intervals use hollow endpoints and dashed segments and report observed errors only.

Coupled, continuation-control and provisional-supervision fits have complete-block
CPU timings in the `temporal-provisional-timing-02` capture. Tensor-only timings
and instrumented stage profiles do not enter the comparison. Continuations
record 4,000 additional updates and the shared 10,000-update coupled parent.

Displacement transport includes the matched `temporal-displacement-timing-01`
CPU capture with coupled and provisional-supervision baselines. Multiple timing
captures for one fit remain distinct points and evidence links. The inherited
972-parameter point-sampling calibration is a component, not an extra full-state fit.

Fixed-viscosity primaries K=1/2/3/4 and completed replicas are trained fits.
The calibrated-viscosity assembly and fused/fast/float32 execution policies are
listed separately in the selector. Byte-qualified executions may inherit their
parent accuracy; float32 policies use their own actual evaluations. Training
budgets and inherited component work remain explicit. Source normalization for
these policies lives in `../full_ai_execution_chart.py`.

The Expanded CPU & GPU models tab imports recorded original-divisor CPU and GPU
measurements, the expanded-data shared-pressure component, fusion variants and
the numerical repair. Exact grids, density, internal substeps and K are separate
filters. Times per published step divide the complete block cost by K only.
Complete-entry timings retain their matching qualified output evidence; they
are not assigned to later repairs. Original larger-domain failures and each
completed repair remain visible. Generic-transform, allocation, integer-sampler,
affine-fusion, sampler-table and 64-cube layout variants keep separate labels.
Full-history byte equivalence requires every actual output hash to match the
native-qualified baseline and retains both evidence links. The layout repair
now has all 24 development histories; its own capture remains accuracy-only.
Later native-entry executions retain their separately measured costs.
The workspace capture uses its verified offline
finalization, preserving the original terminal serialization error in evidence.
Normalization lives in `../expanded_models_chart.py`.

The October 21 follow-ups are normalized in `../expanded_followup_chart.py`.
They include scalar/gather/geometry/native-entry trials, guarded arithmetic,
two managed temporal students, untrained macrostep controls, absolute-pressure
composition, nested/bound/cell-coordinate interpolation, rejected FP32
interpolation and the accurate but slower FFT pressure trial. Training diagnostic
blocks remain separate from development blocks. Candidate advances and teacher
substeps are retained separately; reducing internal advances does not increase K.
Untimed and rejected trials remain visible in the block-error graph and table.
The all-development rollout selector includes all 24 scenes, while the combined
selector also includes larger training-domain cases. Primitive and capacity
diagnostics do not become full-model accuracy or inference latency records.

Later follow-ups include exact alpha reuse, the accuracy-rejected compensated
interpolator, internal-residual S1 and loss-conditioned S2, and S1 plain/fused
graph execution. Both residual fits ran 1,566 updates with 10,876 parameters;
S2 selected the initial epoch-zero checkpoint. Full 2,120-output native block
failures stay visible for each execution. Residual subset timings lack
per-timed-call output hashes and use a separate diagnostic cost graph without
an accuracy coordinate. Three graph timing captures retain their own eager
controls; only the final paired graph capture supports its incremental fusion
comparison. The compensated-interpolator complete-call measurements have exact
output qualification but retain their diagnostic scope and failed accuracy.

Residual S3/S4 add separately recorded physical-loss and delta-MSE fits, fused
execution accuracy and diagnostic timing. Optimizer updates (7/24) and closure
evaluations (18/51) are distinct fields; neither trial gains a native accuracy
pass. Earlier residual-fit and capture records remain intact.

The Temporal pilots & cadence tab uses `../temporal_pilot_chart.py`, `pilots.html`
and `pilots.js`. It keeps eight temporal captures separate across 30/60/120
physical outputs per second, fine/grouped schedules, eager/graph execution and
untrained shared widths 4/8. Every timing row retains five wall and CUDA-event
pipeline samples and matching per-output native errors. Pipeline events are not
kernel-only timing. Physical duration is K/cadence; cost divides by K only.
Stationary-prefix repairs retain recomputed metrics, while the earlier exact
batched-force capture retains its explicitly verified original metric source.
The initial eight model captures use four consumed training masters. The separate
24-train/12-development reference readiness report, its four finite-F32
recoveries and portable objective qualification do not establish a model fit,
full-36 model accuracy, temporal convergence or independent reserved testing.

`../temporal_successor_chart.py` adds six later captures: dualwidth development,
initial/full control-guide ablations, initial/full predicted fine readouts and
paired zero-head folding. The development capture includes both widths and both
selected/last checkpoints over all 12 development masters. Both fits ran 960
Adam updates and selected epoch zero. The failed original probe packaging and
managed recovery remain explicit; recovery adds zero fits. The objective graph
retains the initial loss and all eight epoch scores without inventing unavailable
per-window training history. Training-design captures cover 24 training masters
and do not borrow development scores. Folded cost comparisons retain actual
three-repeat timings and inherit only the matched byte-qualified original's
accuracy; the 16-zero-head certificate is not a general nonzero-network folding
claim. New captures have no CUDA-event samples: the clock selection shows them
as unavailable while retaining their wall times and accuracy tables. Guided RMS
zero-head/gradient readiness is separately labeled as zero new fits.
