# Causal transport attribution and readout execution 01

The learned modal pressure branch is composed with the actual K=1/2/3/4 generators. Their pressure accurately satisfies their own generated RHS, but that RHS remains different from the native reference. Complete-state short-block diagnostics still fail the new criteria. No full-generator target or frontier pass is claimed.

Each comparison uses eight consumed development scenes and starts 0/120, scoring every intermediate horizon. There are 160 physical outputs per policy, with actual pressure/velocity/tracer outputs and generated RHS preserved. These are not 240-step own-history studies or fresh confirmation. Every retained parent and pressure tensor remains unchanged.

| K | Modal equation error, small | Modal equation error, 32³ | Double preparation, learned shifts, 32³ | Offline exact shifts + double preparation, 32³ |
| --- | --- | --- | --- | --- |
| 1 | 1.3692e-3 | 1.1923e-4 | 8.8240e-5 | 1.8627e-6 |
| 2 | 1.6768e-3 | 1.4764e-4 | 1.2464e-4 | 3.9882e-6 |
| 3 | 1.8206e-3 | 1.4518e-4 | 1.2122e-4 | 5.6576e-6 |
| 4 | 1.9961e-3 | 1.4663e-4 | 1.2233e-4 | 7.1399e-6 |

The parent flux already performs its learned convolution in double; recasting it was not a useful new hypothesis. Actual captured q tensors confirm that precision. Mean-free pressure and carrier errors remain explicitly reported. On small grids, current learned shifts still produce worst tracer errors up to about 4.52e-5 in four-step blocks.

Moving carriers to physical units and preserving double point accumulation helps the 32-cubed residual modestly and does not repair small-grid transport error. The offline exact-displacement control reduces error sharply. Native preparation precision matters too: known forcing adds in double before float32 storage, and the same learned viscosity coefficients accumulate/update in double before one float32 store. With both changes, every tested oracle short block passes all four 1e-5 criteria. **The oracle invokes teacher displacement generation and is not a full-AI candidate.**

This evidence prioritizes a more expressive, cheaper learned transport representation. A [bounded learned recurrent-sampling prototype](recurrent-sampler-prototype-01.md) is specified next, with cost qualification before fitting. Every coordinate proposal must depend on learned parameters. Native backtrace routines cannot enter candidate inference.

## Fixed neural field readout costs

The first native-ISA/redundant-floor variant was numerically exact but slower; it is a rejected speed hypothesis. Explicit vector queries alone helped core6 but not core23. The second vector variant uses contiguous loads only when donor indices are provably consecutive, otherwise gathers; it removes double donor clipping before final float32 materialization.

Every original 280-case output byte and three guard behaviours passed on both cores. Another 84 adjacent/extreme/subnormal donor controls per core matched canonical old-kernel bytes and tensor numerical values. Tensor signed-zero bytes are separately recorded.

| Core | Old readout median at 32³ | Vector-v2 median | Old maximum | Vector-v2 maximum |
| --- | --- | --- | --- | --- |
| 23 | 2.2471 ms | 1.1044 ms | 2.3911 ms | 1.2493 ms |
| 6 | 2.3096 ms | 0.9310 ms | 2.5447 ms | 1.0516 ms |

These are paired component measurements on 64 captured cases, with one warmup and five balanced serial repeats per variant. Core23 remains the comparable reference; core6 is a separately labelled exploratory core class. Each speedup uses the baseline on its own core. The readout alone still consumes most or all of the step budget, so this is not a complete-generator latency pass. Direct staggered-array execution and fusion remain cost hypotheses for the next architecture.

No new fitting occurred in these diagnostics. Total new managed learning remains the earlier 800-update pressure-component fit. Higher-density and 600-step campaigns stay gated.

## Evidence

- [Causal pressure composition](../../../runs/full-ai-tight-targets-v1/causal-composition-01/results.json), with 480 scored outputs across three policies.
- [Physical carrier/point arithmetic](../../../runs/full-ai-tight-targets-v1/physical-transport-01/results.json), with 320 scored outputs.
- [Offline exact-displacement control](../../../runs/full-ai-tight-targets-v1/oracle-displacement-01/results.json), with 160 scored outputs.
- [Double preparation and separately flagged oracle](../../../runs/full-ai-tight-targets-v1/double-preparation-01/results.json), with 320 scored outputs.
- [Rejected first speed variant](../../../runs/full-ai-tight-targets-v1/fast-readout-qualification-01.json).
- [Vector-v1 core23](../../../runs/full-ai-tight-targets-v1/vector-readout-cpu23-qualification-01.json) and [core6](../../../runs/full-ai-tight-targets-v1/vector-readout-cpu6-qualification-01.json).
- [Vector-v2 core23](../../../runs/full-ai-tight-targets-v1/vector-readout-v2-cpu23-qualification-01.json) and [core6](../../../runs/full-ai-tight-targets-v1/vector-readout-v2-cpu6-qualification-01.json).

Exact source/evidence hashes and completed process state are recorded in [work-state-03.json](work-state-03.json); each raw result binds its preregistered implementation and inputs.
