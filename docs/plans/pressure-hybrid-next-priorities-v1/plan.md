# Next priorities for the hybrid pressure models

Date: 19 October 2026  
Status: Deferred; current priority is the [full AI model plan](../full-ai-next-priorities-v1/plan.md)

The next priority is to reduce the expensive numerical work required by each hybrid solve. First measure the faster implementation of the older model at large grids, then improve the hierarchy and the corrections that determine how many hierarchy cycles are needed. Further reductions in neural inference cost alone now have limited scope to improve the complete fluid step.

This plan concerns hybrids that combine a learned pressure correction with a numerical solver. Spectral remains the direct numerical comparison wherever its grid and boundary assumptions apply. The practical goal is to move the measured accuracy and time Pareto frontier, while preserving the existing acceptance checks.

## Evidence guiding the priorities

The completed study measured these median times per complete fluid step on the same CPU configuration. Each value summarizes three serial timing repeats.

| Method | 64³ | 128³ | 256³ |
| --- | ---: | ---: | ---: |
| Selected 9 parameter hybrid with fresh solves | 0.541 s | 5.354 s | 59.936 s |
| Previous 648 parameter hybrid | 0.825 s | 5.789 s | 56.063 s |
| Numerical hierarchy without learned correction | 0.575 s | 5.775 s | 63.656 s |
| Direct spectral solver | 0.401 s | 3.373 s | 30.613 s |

At 256³, the selected hybrid spent approximately 4.06% of total measured step time in the neural correction path, including its feature conversion and reconstruction. Eliminating that path's cost entirely, with all other work held fixed, would save only about 4% of step time. The hierarchy alone had a median cost of 21.82 s per step. Preparation took 24.58 s and the complete pressure solve took 34.20 s; these component medians need not sum exactly to the total median.

Across the twelve timed 256³ steps, the smaller model required 48 hierarchy cycles and the older model required 39. This motivates investigating correction effectiveness as well as execution cost. The models have different training histories, so this result does not isolate parameter count as the cause.

Fixed reuse of four directions reduced solver work but produced only modest median time improvements, with overlapping timing ranges. The adaptive policy admitted no reuse on the evaluated populations. The long trajectory tests completed 256 steps on two smaller grids, with diagnostic drift comparable to the numerical control. The 256³ timing tests covered four steps per repeat; they do not establish long trajectory behavior at that size.

Sources: [completed report](../pressure-methods-v6/results.html), [machine readable results](../pressure-methods-v6/results.json), [completion audit and limitations](../pressure-methods-v6/completion-audit.json), and [reuse calibration](../pressure-methods-v6/reuse-calibration.json).

## Priority 1 Measure the strongest existing hybrid at large grids

**Question:** Can the older model retain its lower hierarchy count while using the already qualified fused or streaming implementation?

Compare the following configurations without changing their weights or solver tolerances:

1. Previous 648 parameter model using its current Torch execution.
2. The same checkpoint using fused native execution.
3. The same checkpoint using streaming native execution.
4. Selected 9 parameter model using its current execution.
5. The 27 parameter stencil as an intermediate alternative.
6. The numerical hierarchy control.
7. The direct spectral control.

The native implementations were qualified for numerical equivalence through 256³, but their large grid end to end benefit still needs measurement. Recheck equivalence after integration, especially boundaries, gauge handling, odd dimensions, and the residual of the actual returned pressure.

Screen on identical saved pressure requests at 128³ and 256³, then time complete native fluid steps for configurations that pass. Use three balanced serial repeats initially, and five repeats for the leading pair if the result is close. Include a rectangular grid in qualification. Record cold and warm steps separately, hierarchy cycles, neural calls, operator applications, peak memory, and complete step time.

**Decision:** Keep the fastest qualified implementation of each checkpoint. A rule that chooses different checkpoints for different grid sizes is a candidate only if calibrated on development scenes, frozen, and tested on fresh scenes. A size threshold inferred from the completed study needs that new validation.

**Proposed practical target:** At least a 5% reduction in complete step time against the corresponding existing checkpoint execution, with the repeat results supporting the direction of improvement. Smaller gains can be retained as implementation findings without driving a new training campaign.

## Priority 2 Reduce the cost of each hierarchy cycle

**Question:** How much of the remaining pressure time can be removed without changing the learned correction?

Profile the winning existing hybrid from Priority 1 and its numerical control. Split hierarchy time into restriction, smoothing, coarse solves, prolongation, allocations and copies. Measure pressure operator work and the exchange between the native host and the solver separately. Preparation is also a large cost, so identify which part belongs to the hybrid integration and which part is common to every fluid method.

Use two distinct comparisons:

- **Equivalent execution:** Reuse scratch buffers and cached geometry, reduce conversions and memory passes, and fuse operations where this preserves the numerical method. The existing fast hierarchy implementation is the starting point. Demonstrate an additional cost reduction before retaining another optimization.
- **Changed hierarchy:** Separately test a small set of smoothing schedules or coarse stopping levels. These can change convergence, so measure cycle count and final accepted runtime as well as the cost of one cycle.

Hold the model and acceptance checks fixed. Select at most two implementation changes and two hierarchy schedules after profiling, rather than sweeping their full Cartesian product. Apply promising changes to the numerical control too, so common numerical improvements are visible.

**Decision:** Advance changes that reduce complete accepted solve time. A cheaper cycle that causes enough extra cycles to erase its savings is unsuccessful. Aim for a 10% complete step improvement over the strongest existing hybrid as a useful research target, not an assumed outcome.

## Priority 3 Learn corrections that avoid another hierarchy cycle

**Question:** Can a modest change in training or architecture remove the large grid convergence disadvantage of the smallest model?

Use the residuals actually encountered after hierarchy corrections and during later solver iterations. On development scenes, identify which residual structures persist in the cases requiring an additional cycle. Compare the old 648 parameter model and the 9 and 27 parameter models on those same requests before choosing a new architecture.

Screen at most three training arms:

| Arm | Change | Main comparison |
| --- | --- | --- |
| Matched training control | Retrain the current small correction with balanced examples from early and late solver iterations, including the diagnosed hard cases | Current correction with matched training budget |
| Correction trained on a coarse level | Train a small model for the residual and discrete operator at one selected hierarchy level; prolong its output and retain fine grid acceptance checks | The same hierarchy with its numerical correction, and the previous coarse placement experiment |
| Small correction with wider spatial reach | Add a restricted coarse branch or another modest increase in receptive field, chosen from the residual diagnosis | Matched training control at comparable measured correction cost |

Coarse placement and solver unrolling have already been tried. The new hypothesis must be specific: training on the actual coarse operator or on the residual patterns causing extra large grid work. Simply moving the existing network to a coarse level or repeating the previous three iteration objective would repeat completed experiments.

Start with one seed per arm for screening. For up to two promising arms, run three seeds in total per arm before drawing conclusions about training reliability. Keep data splits, initialization rules, normalization, optimizer budget, and training compute explicit. Include a matched control whenever both architecture and training history would otherwise change together.

Train on existing fields first. Add a bounded set of 256³ development captures only if the residual diagnosis shows missing examples at that scale. Training has so far reached 128³; larger captures should answer a specific generalization question. Keep smaller grids and multiple cells per metre in the mix.

**Decision:** Advance a trained correction only if it saves enough hierarchy or operator work to repay its inference and transfer cost. Report complete accepted solve time, worst same request pressure error, and the distribution of cycle counts. Lower training loss alone is insufficient.

## Priority 4 Revisit reuse only with a cheaper representation

**Question:** Can a small amount of temporal information save more work than it costs to retain and apply?

Keep this behind the first three priorities. The previous adaptive policy declined reuse on the measured populations, while fixed reuse produced small and uncertain time gains.

If profiling still shows a plausible benefit, compare fresh solves with retention of one or two useful directions. Consider retaining a representation on a coarse level before retaining several full resolution fields. Measure collection, storage, orthogonalization, application, and admission overhead explicitly. Four full resolution direction and operator image pairs in float64 require about 1 GiB at 256³, before other solver storage.

Calibrate any admission rule on new development episodes. Require its expected savings to exceed measured overhead by a declared margin, and include episodes that exercise both admission and rejection. A policy that never invokes reuse is a valid fresh solve control, but does not establish a benefit from reuse.

**Decision:** Retain reuse only when repeated complete step timing supports a net benefit without worse acceptance behavior. If it remains marginal, use fresh solves and spend the next training budget on better corrections.

## Priority 5 Validate sustained behavior and larger grids

Validation follows each winning change; the expensive extensions follow selection of a small number of finalists.

1. **Freeze finalists on development evidence.** Treat the completed v6 scenes as consumed evidence. Reserve new scenes, forcing histories, and seeds before selection, and keep their pressure fields and trajectory results out of fitting and tuning.
2. **Separate cell count from physical spacing.** Include 128³ and 256³ cubes plus an odd rectangular grid. Use paired cases that hold physical extent fixed while changing cells per metre, and other pairs that hold cells per metre fixed while increasing grid dimensions. Keep fluid density and timestep separately declared.
3. **Run sustained coupled comparisons.** Use at least two new 256³ scenes for 256 accepted timesteps per finalist and the numerical controls, subject to a measured resource budget declared before launch. Measure pressure, staggered velocity, tracer drift, divergence and runtime at matching physical times. Record every rejection and stop condition. Keep the current drift levels diagnostic; application limits remain deferred as requested.
4. **Scale to 512³ after memory and cost qualification.** Begin with pressure requests and a short native trajectory. Advance to longer runs only after those checks pass. Keep 1024³ as a later production target; extrapolated timing or memory does not qualify it.

Every result should identify whether it demonstrates better numerical accuracy on the same discrete grid, lower trajectory drift, or better physical accuracy under refinement. These are different claims. A spectral reference on a given grid does not by itself establish continuum accuracy.

**Decision:** Recommend a hybrid for integration only after it improves repeated complete step timing on the intended workload, passes the existing numerical gates, and has an acceptable documented trajectory result. Application acceptance requires application limits when those are set.

## Experiments to defer

- **Further inference compression as the main project.** The 9 parameter correction already accounts for a small fraction of total step time at 256³. An inference change is useful if it also preserves stronger convergence or reduces expensive solver work.
- **Repeating the same fitting recipe or adding wavelength examples indiscriminately.** Use observed residual failures to choose new data.
- **More expensive fluid rollout training.** Revisit it if sustained tests show a specific drift problem exceeding the numerical control. The completed small grid trajectories do not currently justify making it the next major cost.
- **Higher quality temporal targets.** Keep the proposed two reference substeps per model timestep as a later controlled data experiment. It needs matched physical endpoints and a consistent definition of the pressure target; it should not be conflated with changing the model timestep. Run it if temporal discretization appears to limit the intended accuracy measure.
- **An input for cells per metre without a demonstrated conditioning gap.** First test generalization across the existing spacing treatment. Add explicit conditioning if those paired tests expose a failure the current scaling cannot handle.
- **512³ or 1024³ training as the first response.** Increase training cell count when smaller controlled experiments identify the missing behavior and the capture and fitting costs are justified.
- **Irregular geometry as an assumed advantage.** Establish the intended application geometries and the operators the native system actually supports before starting that branch. On the current regular grid problems, spectral is the strongest measured comparison.

## Execution order and deliverables

The first wave should contain **no retraining**: register the new development and reserved evaluation sets, benchmark the seven Priority 1 configurations, and profile the winning existing implementation. Publish a short report showing which execution changes save complete step time and where the remaining time goes.

The second wave should test the bounded hierarchy changes from Priority 2. Start the three training arms from Priority 3 only after the residual diagnosis identifies the correction behavior to improve. Reuse remains an optional branch with its own measured cost justification.

The final wave should validate frozen finalists on new scenes and sustained 256³ trajectories, then qualify a bounded 512³ evaluation. Before each wave, declare its run count, resource estimate, numerical checks, selection rule, and stop conditions. Freeze one configuration per finalist before collecting its reserved evaluation results.

For each wave, write the configuration and source identities, raw timing and accuracy records, a complete step runtime breakdown, cycle count distributions, and updated accuracy versus time figures. Publish numerical and spectral controls alongside hybrids. Report failed attempts and unfavorable results as well as improvements.

Use new experiment, run, and publication directories. Preserve the completed studies and their frozen hashes. This file schedules research priorities; it does not start fits or change an application model.
