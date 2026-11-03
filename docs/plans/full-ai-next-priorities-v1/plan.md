# Full AI pressure tuning and simultaneous simulation prediction

Date: 19 October 2026  
Status: Proposed plan; full AI models are the current research priority

Run two tracks: improve C107 at approximately 4 ms CPU pressure inference and at most 7e-4 worst equation error, and develop models returning 2, 3, or 4 future simulation states together, targeting less than 2 ms per generated step and at most 4e-3 worst error. Begin C107 tuning first while establishing the temporal data contract. The multi-step track can proceed independently once that contract is qualified; it need not wait for a C107 successor to reach its target.

The [hybrid priorities](../pressure-hybrid-next-priorities-v1/plan.md) are deferred. Numerical solvers remain offline teachers and comparison controls. Candidate inference must not invoke a spectral, multigrid, Krylov, or other numerical pressure solve. Track A replaces pressure calculation within the native simulator; Track B predicts future simulation states. These have different inputs and timing scopes.

## Targets and measurement definitions

| Track | Required output | CPU timing target | Worst error target |
| --- | --- | --- | --- |
| A | Pressure for one prepared pressure request | Aim for median at or below 4.0 ms | Pressure equation relative residual at or below 7e-4 |
| B2 | All states at the next 2 timesteps from one evaluation | Less than 2.0 ms per generated step | At or below 4e-3 at every output horizon |
| B3 | All states at the next 3 timesteps from one evaluation | Less than 2.0 ms per generated step | At or below 4e-3 at every output horizon |
| B4 | All states at the next 4 timesteps from one evaluation | Less than 2.0 ms per generated step | At or below 4e-3 at every output horizon |

**Confirmed timing target:** less than 2 ms per generated physical step, with the full block latency reported alongside it. The arithmetic limits are:

| Future steps K | Complete model inference for a block must be |
| --- | ---: |
| 2 | Less than 4 ms |
| 3 | Less than 6 ms |
| 4 | Less than 8 ms |

These are target budgets, not predicted runtimes. Define amortized latency as measured block inference time divided by K, counting encoding, every output head, field reconstruction and required model preprocessing. Also publish a separate tensor-only measurement and complete simulator integration timing. Neither a cheap final head nor an endpoint-only prediction represents the cost of returning all K states.

**Working accuracy interpretation:** preserve the chart's equation residual for Track A. For Track B, evaluate predicted pressure against the reference pressure equation at each future time, and separately measure pressure, velocity and tracer field errors. The proposed stronger research criterion is that both the equation residual and the maximum of those field errors remain at or below 4e-3. If the requested 4e-3 applies only to equation residual, retain the field curves as diagnostics and label that narrower result explicitly. These interpretations should be settled in the run registration before fitting, rather than selected after seeing results.

The primary millisecond comparisons use the existing small grid regime and CPU setup: one CPU thread, batch size one, the recorded CPU affinity where available, and no GPU inference. The historical chart contains dimensions of 5 to 16 cells per axis and both 8 and 16 cells per metre. Report 32³ and larger grid results separately; a 4 ms or 2 ms claim on the small grids does not imply that latency at production grid sizes.

## C107 baseline and what changed from C106

| Checkpoint | Parameters | Median CPU inference | Worst equation relative residual |
| --- | ---: | ---: | ---: |
| C106 | 93,758 | 4.462 ms | 6.788e-3 |
| C107 | 93,758 | 4.208 ms | 1.029e-3 |

Both are 50 percent width models initialized by slicing the same C101 checkpoint. Each received 2,400 additional optimizer updates with two learned pressure correction stages. C106 used a frozen teacher correction loss with weight 1; C107 used weight 0. C107 is the non-distillation control, rather than a continuation trained from C106. The paired result supports starting with C107's direct physical and equation supervision; it does not prove all forms of distillation are harmful.

C107 retains a learned global branch: 63 low mode gains and a shape dependent head, together with the spatial network. Its cosine basis and equation checks do not perform an analytical pressure inverse. Preserve that distinction when identifying the full AI candidates.

Reaching 4.0 ms from the current chart median requires about 4.9 percent less inference time. Reaching 7e-4 requires about 32 percent less worst residual. Only one of the 128 chart requests exceeds 7e-4: frame 63 of validation trajectory 25, a 9×11×11 grid at 16 cells per metre. That is a useful diagnostic clue, but the population has already been consumed and cannot establish fresh generalization.

The chart measures one `predict_pressure` call per case, including the model's two correction stages. It excludes setup, sample assembly, independent verification and fallback. Its 4.208 ms value is a median across heterogeneous requests, not a maximum latency guarantee or a complete fluid step time. Repeated measurements are needed before interpreting a small timing difference.

Sources: [C107 measurements](../chart-diagnostics/pressure-methods-v4/v4-student50-control/validation.json), [C106 measurements](../chart-diagnostics/pressure-methods-v4/v4-student50-distill/validation.json), [C107 case records](../chart-diagnostics/pressure-methods-v4/v4-student50-control/cases.jsonl), [paired student preparation](../../../experiments/pressure-methods-v4/prepare_student.py), and [learned prediction path](../../../runs/pressure-methods-v4/v4-student50-control-project-01/physical_residual.py). The accompanying [baseline evidence](baseline-evidence.json) records source hashes and compact baseline values.

## Track A Improve C107 accuracy and CPU speed

### A1 Reproduce and diagnose the current checkpoint

Restore the exact C107 weights and normalization. Reproduce the 128 request equation errors, then measure C106 and C107 in a shared timing harness with explicit warmup and five balanced serial repeats. Preserve the historical chart records and publish these new timings as a repeated benchmark.

Profile the two correction stages, spatial convolutions, learned global branch, pressure operator calls, conversions, allocations and reconstruction. Compare residual patterns for the hardest train and development examples across aspect ratios, spacing and stage index. Inspect boundary errors and the residual's low and high spatial frequency components. Do not add the known worst validation snapshot to the training set.

First screen changes that keep the weights and prediction policy fixed: cache shape dependent basis and learned head values for a loaded checkpoint, reuse temporary buffers, and compare suitable CPU memory layouts or a compiled execution path. Cached values must be invalidated when weights, shape, spacing or other dependencies change. Qualify each implementation against the original output, gauge and equation residual, including odd rectangular grids. Keep all required work inside the declared timing scope.

**Decision:** retain equivalent execution changes that repeatedly reduce CPU time without worsening the error target. The purpose is to find the modest speed reduction independently of accuracy training.

### A2 Tune the existing architecture before increasing its size

Screen these four fits from the same restored C107 checkpoint. Use 1,200 additional updates each, with a checkpoint at 600 and 1,200. Match the sampled request stream and optimizer work where the comparison permits it. Keep the original normalization, two correction stages, direct physical labels and zero distillation weight.

| Arm | Change | Question answered |
| --- | --- | --- |
| A control | Low rate continuation: base learning rate 3e-5, reduced to 1e-5 after update 900; retain the existing head multiplier of 10/3 | Does additional conservative fitting help? |
| A head | Same continuation, with head multiplier 1 instead of 10/3 | Is the global branch moving too aggressively relative to the fine network? |
| A tail | Same continuation, with bounded train-only emphasis on difficult residual families | Can worst error improve without sacrificing the broad population? |
| A combined | Combine the head change and difficult-family emphasis | Do the two changes help together? |

For difficult-family emphasis, rank training requests using the frozen initial model, preserve the existing small and larger spatial tier proportions, and cap any request's sampling multiplier at 4. Normalize the probabilities within each tier and freeze them before fitting. Reweighting must use training errors only. Use multiple train trajectories from a problematic shape or residual family rather than memorizing a single request.

Rank checkpoints by worst development error subject to measured latency and broad regression checks. Include median, upper quantiles and worst errors so an improvement in one outlier cannot hide wider degradation. More optimizer updates are a hypothesis to test, not an assumption that the target will be reached.

### A3 Make bounded architecture changes only if needed

If A2 leaves a measurable gap, screen at most two additional architecture fits:

- A modest redistribution of channels between expensive fine levels and cheaper coarse levels. Profile candidate shapes first, maintain valid channel and normalization contracts, and compare against the unchanged architecture at matched training work.
- A model trained to produce the pressure result in one learned correction stage, using direct equation and physical supervision. Compare it with the two stage model at the same final error; dropping a stage without retraining is only a diagnostic.

Do not assume the native kernels written for the tiny hybrid models can execute C107's encoder and decoder. Any new implementation must cover C107's actual network and learned global branch.

For up to two winning training recipes, run two additional seeds, giving three seeds in total per recipe. Record that these are continuations from a fixed checkpoint: seed variation tests the continuation process, not independent pretraining. Select on development evidence, freeze the prediction implementation and weights, then evaluate new reserved pressure requests and native trajectories.

**Track A success:** approximately 4 ms median inference and worst residual at or below 7e-4, reproduced on the historical comparison and tested separately on a fresh population. Report any mismatch between those populations. A chart target pass does not establish the native simulator's tighter pressure acceptance or trajectory validity; retain those checks and publish failures separately.

## Track B Generate 2 3 and 4 future simulation states together

### B1 Establish the state and training data contract

Follow the [multi-step prediction research](../multi-step-prediction-research.md), with K fixed to the requested 2, 3 and 4. A model must map current state and available history to every future state from t+dt through t+K·dt. Two pressure correction passes on one request do not advance two simulation timesteps. Batching K requests whose future inputs were prepared by a reference simulation is a throughput control, not an online future-state predictor.

Audit the native integrator before collecting data. Inputs must include the staggered velocity fields, tracer, any required pressure and integrator history, grid spacing, timestep, material parameters, boundary and geometry information, and prescribed forcing over the interval. Outputs must include the state required to restart the next block, with an explicit timestamp and staggering convention for every field. Future reference velocities, pressure requests or endpoints must never become inference inputs.

The research note predates the completed v6 captures. There are now spectral reference trajectories with 256 consecutive pressure, staggered velocity and tracer exports on 64³ and 65×49×33 grids. Audit those files for complete restart information, forcing, history and phase alignment before reuse. They are consumed diagnostic scenes, so they can support data engineering and declared development work; they are not new blind evidence. Exported fields alone do not prove the native simulator can resume from them exactly.

**Data qualification:** reproduce at least eight subsequent native reference steps after restoring a saved state, within a predeclared numerical reproducibility tolerance. Verify full array shapes, timestamps, field hashes and forcing. Extend the state export contract if required hidden integrator state is missing.

Proposed first capture bank: four grid groups, two spacing values, and four independently seeded trajectories per group and spacing, giving 32 trajectories. Use three groups within the existing 5 to 16 cells per axis regime, including a cube and odd rectangles, plus a separate 32³ extension. Use 8 and 16 cells per metre. Within each group and spacing, assign two trajectories to training, one to development and one to reserved evaluation before capture: 16 train, 8 development, 8 reserved.

Capture the initial state and every subsequent step. Start with 128 steps for train and development and 240 for reserved trajectories. Split whole scenes before making overlapping windows. Use identical physical timesteps and reference solver settings across temporal horizons. This bank is a bounded feasibility pilot; use a learning curve if data quantity limits the result rather than claiming a failed pilot rules out temporal prediction.

### B2 Build a shared spatial network with joint future outputs

Start with a compact spatial encoder and a shared decoder that returns K outputs through inexpensive horizon heads. Share as much spatial computation as possible. Each output predicts the increment from the current physical state at its declared lead time. Include velocity and tracer in the state outputs, and pressure for comparison with the requested error measure. Reconstruct staggered fields explicitly.

Train four primary configurations with the same data split and training work accounting:

| Configuration | Future states returned | Purpose |
| --- | --- | --- |
| B1 control | 1 | Establish the full state prediction task and its sequential rollout baseline |
| B2 | 2 simultaneously | First temporal sharing test |
| B3 | 3 simultaneously | Test the requested intermediate bundle size |
| B4 | 4 simultaneously | Largest requested bundle and greatest sharing opportunity |

Use a shared architecture family with separate models for the first comparison. Only test a single horizon-conditioned model after this establishes whether the horizons need different capacity. Existing C107 weights may initialize compatible internal tensors, but its pressure residual input is different from a physical state input; its checkpoint cannot be treated as an unchanged full-state predictor.

Return every intermediate state in one joint evaluation. Do not count repeated full spatial model calls, reference interpolation, or a final-state-only model as simultaneous generation. A neural temporal decoder may have internal computation, which must be documented and timed. All candidates must run without an iterative numerical pressure solve or corrective projection. Numerical operators may be used in training losses and independent evaluation.

### B3 Train for every horizon and for continuation across blocks

Use equal initial weight for every output horizon. Normalize velocity, tracer and pressure losses with training-only scales; remove the arbitrary pressure gauge. Add declared divergence, boundary and tracer conservation losses, measuring their training cost separately. These losses encourage physical consistency but do not guarantee it.

First fit short blocks. Then expose the model to its own predicted history across at least two consecutive blocks, with the same procedure applied to the K=1 control. Preserve supervision of every intermediate output. Count added model calls and state transitions in the training budget. Do not substitute ground truth history between blocks during rollout evaluation.

Start with one seed for each of the four configurations. Replicate up to two promising bundled configurations with two further seeds each, and replicate the one-step control when making a confirmed advantage claim. Evaluate all three requested K values even if the first two-step result is unfavorable; one failure does not determine the other horizons.

If one horizon dominates the errors, change its training weight or capacity in a separately registered follow-up. Do not drop that horizon or report only the best endpoint to meet the target.

### B4 Evaluate actual future prediction and sustained CPU cost

For each configuration, measure both a single block from reference history and repeated blocks using only predicted history. Compare at 12, 48 and 240 physical steps; these horizons align across K=1, 2, 3 and 4. Use matching forcing and interaction times. Unannounced user or boundary changes must invalidate unconsumed predicted states and trigger a new prediction; count discarded work in the interactive benchmark.

Report these errors by scene and output horizon, retaining the worst case:

- Pressure equation residual `||A(p_predicted) - rhs_reference||₂ / max(||rhs_reference||₂, 1e-30)` at the corresponding future reference time. This uses the reference equation offline and must not be passed into the model. On a rolled out trajectory it also reflects disagreement with the reference dynamics.
- Mean-free pressure relative L2 error, joint staggered velocity relative L2 error, and tracer relative L2 error against the matching reference state. Report absolute errors as well. Declare treatment of zero and near-zero reference fields from training evidence before evaluation; do not tune denominator floors on held-out results.
- Divergence, boundary flux, tracer mass and bounds, energy behavior, and time to each diagnostic drift threshold. A small pressure residual does not by itself establish accurate velocity or tracer evolution.

For the proposed stronger 4e-3 field criterion, require every nondegenerate field error and every output horizon to pass on the declared evaluation interval. Declare absolute tolerances for degenerate reference fields in the registration. If this is not achieved, report which criterion fails; do not silently substitute a mean error or a shorter interval.

Use five balanced serial CPU timing repeats after explicit warmup, with inference mode and batch size one. Publish median, p95 and maximum block latency; amortized latency per generated step; peak memory; and complete application time per physical second advanced. Measure initial encoding, history updates, all decoded fields, transfers and required checks. Keep loading, compilation and initial cache preparation separately visible, and include their amortization in short interactive episodes.

Compare the bundled models with sequential B1 predictions, the native simulator using C107 pressure prediction at every step, and numerical references advancing the same physical interval. Numerical methods are controls rather than candidate inference components. If a candidate requires fallback, publish its raw failures and its sustained cost including fallback and discarded blocks; a post-correction output cannot establish the raw full AI accuracy claim.

**Track B success:** all K requested states are available, measured CPU latency meets the registered interpretation of 2 ms, and worst error stays within the registered 4e-3 criterion across the declared scenes, horizons and rollout interval. Report a short block success separately from a 240 step rollout success.

## Sequence and bounded initial work

1. **Register benchmarks and restore C107.** Preserve the original evidence, use the confirmed per-step timing target, resolve the multi-step accuracy interpretation, and record the existing execution path and exact weights.
2. **Run A1 and the state contract audit.** Find CPU savings while checking whether existing full-state captures can support temporal windows and native restart.
3. **Screen the four A2 fits.** Apply qualified execution improvements consistently. Add the two A3 architecture fits only if a gap remains.
4. **Qualify and capture temporal data.** Freeze scene splits and the model state contract before producing windows.
5. **Screen B1, B2, B3 and B4.** Measure sharing and worst error on the same development trajectories. Start with short blocks, then continuation across blocks.
6. **Replicate finalists and freeze them.** Evaluate fresh pressure requests for Track A and reserved temporal scenes for Track B, then publish the two separate accuracy and time frontiers.

The initial model screen contains four C107 continuation fits and four temporal fits. Up to two additional C107 architecture fits and the specified seed replications are conditional work. Set update counts for temporal fits and finite time, memory and storage limits after the state contract and per-update pilot are measured. Match and report physical state targets, forward calls and processed cells; a common optimizer update count alone does not equalize work across K.

Deliverables are an experiment registration, exact baseline identities, profiling report, trajectory data manifest, training receipts and CPU restores, per-case and per-horizon errors, repeated CPU timings, rollout traces, and a report showing successful and failed configurations. Use new experiment and run directories; keep the completed studies immutable. This request produces the plan and does not launch training.
