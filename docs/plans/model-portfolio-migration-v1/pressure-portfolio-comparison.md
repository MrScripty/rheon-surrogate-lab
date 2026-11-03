# Frozen pressure-portfolio comparison

All 127 available named pressure checkpoints were measured on the same 128 consumed native requests, with 20 warm samples per request across two process passes. Independent reduction verifies 325,120 timed calls, original input/output hashes and every common accuracy row. These are diagnostic comparisons on repeatedly consumed scenes, not fresh independent acceptance.

Each timing includes native CPU input validation/packing, transfers, the captured GPU pressure policy, reconstruction, an independent CPU output and synchronization. Setup, warmups, the instrumented first call, hashing and scoring are excluded. Units are **GPU ms per pressure request**; no simulation time advances and no K division is applied.

| Checkpoint | Worst pressure relative L2 | Worst equation relative L2 | Median of case medians, ms | Worst case median, ms |
| --- | ---: | ---: | ---: | ---: |
| C37 | 0.0538221 | 0.0489171 | 2.0604 | 2.2291 |
| C44 | 0.0134507 | 0.00398262 | 4.3745 | 4.8990 |
| C53 | 0.00602154 | 0.00390012 | 4.5317 | 5.5548 |
| C106 | 0.0155048 | 0.00678757 | 6.0868 | 7.1966 |
| C107 | 0.00815329 | 0.00102944 | 6.0643 | 6.9721 |
| C130 | 8.59009e-05 | 0.000225521 | 7.2864 | 8.4306 |

C107 improves both worst error metrics over C106 while showing essentially the same complete-call speed: the median paired C107/C106 latency ratio is 0.9955, with C107 faster on 67 of 128 cases. C130 has much smaller common pressure error and higher call latency. These observations compare deployed policies; they do not isolate a single architectural cause.

C53 is among ten models whose reverse-pass timing falls in a conservative possible overlap window with a stopped metadata checker. The timing effect was not measured. The reduction retains both process summaries, flags these subjects, and gives a separately labeled frontier sensitivity excluding C52–C61. Another, explicitly GPU-contended C102 block was excluded entirely and replaced under the registered continuation; its uncontended replacement alone enters the pooled result.

The quoted full-state anchors are different tasks and populations. Fresh complete-generator timings reproduced 9.2500e−17 pressure error at 0.2327 ms per physical output for the cell-coordinate anchor, and 9.1585e−16 at 0.4114 ms for the temporal graph anchor. Both quoted grids have 15,360 cells, with different shapes, densities and integration work. The N8-labelled cell anchor takes its N1 branch; the quoted temporal case takes two fine advances per output. A 6 ms pressure-request row therefore does not demonstrate that either full-state anchor regressed.

Actual temporal coarsening still causes real accuracy loss: reducing the fine-reference integration work changes the generated field even with unchanged parameters. Keep integration work, physical timestep/output cadence, case identity and pressure/equation/velocity/tracer metrics separate. The frozen old-Rheon epoch remains the reference throughout this comparison.

Two native tables are imported and independently checked: 127 subjects, 128 cases, two error metrics and three latency statistics, totaling 81,280 measured scalars. They retain all raw timing samples and distinguish grid dimensions, cells per metre, physical timestep, GPU identity, precision and known host conditions. Actual rendered acceptance passes 78 actions and 16 complete HTTP200 requests: both tables together, separate density/domain filtering, exclusion of known-host-overlap subjects, C106/C107 comparison and original timing/case inspection. All nine screenshots were inspected. The project now contains 21 measurement sources. See [native custody](reports/native-pressure-portfolio-GPU-timing-custody-01.json) and [actual UI](reports/native-pressure-portfolio-GPU-timing-UI-01.json).

The complete 205-fit / 524-parameter-state portfolio is not finished. Missing named checkpoint identities remain explicit; unnamed fits, remaining state/adaptor coverage, model reviews, clean reproduction and the 600-output own-history evaluation still need completion.

Evidence: [exact pressure reduction](reports/pressure-portfolio-GPU-timing-01.json), [native packet qualification](reports/pressure-portfolio-GPU-timing-native-preparation-01.json), [complete-generator timing](reports/frozen-full-state-GPU-timing-01.json), [host-activity disclosure](reports/pressure-timing-host-activity-01.json).


## Training feedback findings from retained source reviews

These comparisons use the same 128 consumed pressure requests and unchanged
checkpoint-specific deployment policies as the common benchmark above. All rows
below request two internal pressure corrections. They are not two generated
physical frames, and latency must not be divided by two. GPU timing includes the
complete native CPU-input/output boundary. These diagnostics do not establish
held-out scene generalization, large-domain accuracy or trajectory drift.

| Checkpoint | Intervention | Worst pressure relative L2 | Worst equation relative L2 | GPU median of case medians (ms/request) |
|---|---|---:|---:|---:|
| C40 | Intermediate residual contraction | 0.014911085 | 0.0083245411 | 4.430799 |
| C41 | Extra existing-bank proposal | 0.011980936 | 0.011665529 | 4.420489 |
| C42 | Extra targeted-grid proposal | 0.010405588 | 0.010799692 | 4.429167 |
| C43 | Early captured directions | 0.0078238244 | 0.004725959 | 4.423661 |
| C44 | Late captured directions | 0.013450717 | 0.0039826222 | 4.374484 |
| C45 | Expanded late-bank; first-pressure weight0 | 0.0097331973 | 0.0056079029 | 4.360084 |
| C46 | First-pressure weight1 | 0.013275492 | 0.0040508705 | 4.341350 |
| C47 | First-pressure weight0.1 | 0.0075183027 | 0.0029634214 | 4.330769 |
| C48 | First-pressure weight0.25 | 0.009792512 | 0.0043981399 | 4.408052 |
| C49 | Low-mode weight0.1 | 0.009369155 | 0.0036985233 | 4.397011 |
| C50 | Low-mode weight0.01 | 0.0063411958 | 0.0040036748 | 4.462554 |
| C51 | Low-mode weight0.03 | 0.0076667189 | 0.0029722563 | 4.430329 |

C42 improves both worst metrics over C41 at the same optimizer/proposal budget,
but changes grid coverage and exposure per grid together. C43 versus C44 trades
pressure accuracy against equation accuracy. The expanded-bank C45 is not a
one-factor objective comparison against C44: it includes 48 extra manufactured
long-axis directions. Within C45–C48, weight0.1 has lower values of both worst
metrics than weights0/0.25/1. Training supervision adds no inference correction.

C49–C51 share initial tensors, executed core/trainer source and sample streams;
weight0.01 favors the pressure tail, while0.03 favors the equation tail. The
low modes are spatial DCT-II modes, not temporal frequencies. Their retained
history key `validation` is explicitly a diagnostic on training parents:
16trajectories/32frames, zero validation trajectories and zero held-out
evaluations. A separate native evidence wrapper emits an already-consumed
validation probe after fitting. Neither is independent held-out evidence.
The wrapper keeps the original trainer profile at `training-profile.json`;
its SDK-owned `profile.json` is a different artifact. Both identities must be
kept distinct when auditing or reproducing these native fits.

Source evidence: [budget/coverage](reports/captured-pressure-budget-coverage-method-reviews-01.json),
[supplemental streams](reports/captured-targeted-supplement-method-reviews-01.json),
[directions/first-correction](reports/captured-direction-pressure-method-reviews-01.json)
and [low spatial modes/native diagnostic scope](reports/captured-low-mode-feedback-method-reviews-01.json).
These are retrospective source/receipt reviews; clean retraining remains
unqualified. Original recipes perform CPU neural forwards and require a
separately qualified GPU-only reproduction path under current policy.
