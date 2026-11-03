# Learned pressure reuse, mixed-query refinement and CPU efficiency

The learned causal pressure-reuse prototype passes all 40 consumed development scenes for K1–4. Its frozen independent check passes 15 of 16 new whole scenes. The maximum 1.2ms per generated step target remains unmet on one, two and four cores. No joint target or later-phase gate is declared complete.

## Managed training and data

All144 original native training pairs were retained byte for byte, with 128 mixed-mode reference queries added across the same four geometries. These inputs improve modal signal coverage using only training data. Numerical pressure references produce offline labels only; they never enter candidate inference. The four minimum modal RMS/global ratios rose from approximately 4.69e-6,4.57e-5,1.06e-3 and 1.28e-7 to 0.405,0.449,0.471 and0.353.

Managed CUDA training Run `dda357a46d4c4ebeaeac30bbb65488f3` performed 80 LBFGS calls. Exact typed CPU restore Run `80bdc4c7e25048f49d47c67c4fc84039` reproduced all five archived artifacts. The resulting weighted objective was 2.26634e-25. The full generator retains 40,017 learned parameters; all new parameter fitting ran on managed CUDA. Raw checkpoints are unchanged. The inference head uniformly rounds its 48 managed-restored parameters to 2^-40.

The initial recipe declared 64 MiB per input, above the actual 40 MiB contract. The largest shard is 34,418,718 bytes. An actual decoder red/green check isolated that declaration; the version 11b project changes only the declared cap to 40 MiB. Both training and restore recipes passed before admission. Earlier evidence remains preserved.

## Why pressure history mattered

Direct learned pressure still failed two of40 consumed scenes. Even a numerical pressure oracle in that direct formulation failed two scenes. The native reference uses the supplied current pressure as a starting point. The new candidate expresses that causal skip path using only existing learned flux, tied learned momentum and learned pressure modules: form the learned residual from the current state, predict its pressure correction, add current pressure and regauge. It performs no numerical pressure solve or analytical inverse, makes no teacher resets and sees no future labels. It required no new parameter fit.

| Three-scene control | Pressure role | Small-grid worst equation |32³ worst equation |
| --- | --- | ---: | ---: |
| direct | learned | 1.50073e-14 | 8.6415e-05 |
| direct | OFFLINE | 2.85543e-05 | 8.6415e-05 |
| reuse | learned | 1.22495e-14 | 5.16817e-14 |
| reuse | OFFLINE | 2.03503e-14 | 9.95963e-14 |

OFFLINE rows are diagnostic oracles, excluded from AI candidates, latency and frontier claims.

## Frozen independent accuracy

The selected model, inference sources, raw managed evidence and equivalent execution variants were frozen before generating the 16 previously reserved reference trajectories. Every complete pressure, velocity and tracer state is scored, including every intermediate K2–4 output. Own-history tests run 240 actual simulation steps from the initial state.

| K | Grid regime | States | Worst equation | Worst pressure | Worst velocity | Worst tracer | All pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | small | 2880 | 1.59338e-14 | 1.27648e-15 | 0 | 0 | True |
| 1 | 32cubed | 960 | 9.90576e-05 | 4.20895e-07 | 1.35977e-07 | 3.37392e-08 | False |
| 2 | small | 2880 | 1.59338e-14 | 1.27648e-15 | 0 | 0 | True |
| 2 | 32cubed | 960 | 9.90576e-05 | 4.20895e-07 | 1.35977e-07 | 3.37392e-08 | False |
| 3 | small | 2880 | 1.59338e-14 | 1.27648e-15 | 0 | 0 | True |
| 3 | 32cubed | 960 | 9.90576e-05 | 4.20895e-07 | 1.35977e-07 | 3.37392e-08 | False |
| 4 | small | 2880 | 1.59338e-14 | 1.27648e-15 | 0 | 0 | True |
| 4 | 32cubed | 960 | 9.90576e-05 | 4.20895e-07 | 1.35977e-07 | 3.37392e-08 | False |

Failed independent scenes: pressure-successor-fresh-12. These results describe the frozen candidate; any successor design based on them must treat the scenes as consumed. Corresponding own-history physical outputs match byte for byte across K1–4. K here packages recurrent generation and does not establish that multiple future states are computed in parallel.

## Single-core efficiency and measured scaling

Finite donor min/max selection instructions replace equivalent comparisons and selects. Signed-zero operand order is preserved. For each allocation, 336 component cases and 64 complete states matched the qualified serial implementation byte for byte before timing. Alternating v7/minmax measurements share learned tensors and FFTW owners. The clock includes input checks, forcing, all learned stages and every intermediate output. One core is the primary efficiency metric; multicore results remain separate measured options.

| K | Physical cores | Median ms/step | P95 ms/step | Maximum ms/step | Median block ms | Speedup vs1 core | Core efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 9.095 | 10.818 | 10.841 | 9.095 | 1.000 | 1.000 |
| 1 | 2 | 7.826 | 8.569 | 8.751 | 7.826 | 1.162 | 0.581 |
| 1 | 4 | 8.453 | 14.555 | 14.643 | 8.453 | 1.076 | 0.269 |
| 2 | 1 | 9.022 | 10.947 | 10.991 | 18.044 | 1.000 | 1.000 |
| 2 | 2 | 7.482 | 8.786 | 9.362 | 14.964 | 1.206 | 0.603 |
| 2 | 4 | 7.565 | 9.022 | 9.670 | 15.131 | 1.193 | 0.298 |
| 3 | 1 | 8.966 | 10.847 | 10.941 | 26.897 | 1.000 | 1.000 |
| 3 | 2 | 7.339 | 8.795 | 8.806 | 22.018 | 1.222 | 0.611 |
| 3 | 4 | 7.606 | 8.629 | 8.712 | 22.817 | 1.179 | 0.295 |
| 4 | 1 | 9.315 | 10.417 | 10.432 | 37.260 | 1.000 | 1.000 |
| 4 | 2 | 7.172 | 8.406 | 8.530 | 28.688 | 1.299 | 0.649 |
| 4 | 4 | 7.533 | 8.410 | 9.043 | 30.134 | 1.236 | 0.309 |

These are 32³ consumed-development measurements on CPU 6, CPUs 6/4 and CPUs 6/4/5/7, with verified placement. Larger allocations are not consistently faster. In this sample, two cores have the lowest median for each K and lower maximum latency than four cores. The maximum, rather than the median, must meet the declared 1.2ms target. Historical CPU 23 efficiency-core results are separate. This experiment does not remeasure a matched Rheon application comparison.

The build-only registration retains inactive CPU, pair and old implementation-description fields inherited from its earlier template. Actual qualification and timing use their explicit resource protocols and the frozen pressure-reuse factory. [The scope clarification](minmax-execution-metadata-clarification-12.json) records which fields the builder consumes; all bound files remain unchanged.

## Next work and preserved gates

The next speed experiment should fuse construction of learned pressure features into compiled execution to remove repeated tensor indexing and temporary arrays, followed by complete-block fusion to reduce Python dispatch and thread synchronization. Coefficients must remain inherited learned tensors. Require exact agreement with this frozen graph before measuring cost. Keep single-core cost and measured two/four-core scaling visible. Do not add local fitting, numerical pressure solvers or future reference features.

The 1.2 ms and 1e-5 joint gates remain open for every K. Densities 50/100, random-aspect larger domains and 600-step 30/60 Hz drift optimization remain behind their ordered gates. Registered sources and original evidence were audited without modifying bound files.

[Machine-readable results](pressure-history-and-mixed-query-summary-12.json) · [Frozen selection](pressure-history-frozen-selection-12.json) · [CPU resource policy](cpu-resource-policy-04.md)
