# Repaired all-neural family: K1/2/3/4

The K2 zero-update composition passes all eight consumed development scenes and
every pressure/equation/velocity/tracer criterion through own240. Its worst error
is 0.0033941. This motivates a bounded fresh family with the same 243 managed
viscosity coefficients frozen from initialization. All other trainable cell
weights initialize freshly at a common seed. The original failed parent and
successful composition remain intact; no single-factor claim compares this
fresh family with the post-training composition.

| K | Optimizer updates | Single-block phase | Block forwards | Physical state targets |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20,000 | 4,000 | 36,000 | 36,000 |
| 2 | 10,000 | 2,000 | 18,000 | 36,000 |
| 3 | 6,667 | 1,334 | 12,000 | 36,000 |
| 4 | 5,000 | 1,000 | 9,000 | 36,000 |

Every later update uses two own-history blocks. Common train-only scenes,
normalization, sampling seed, initialization seed, objective, base learning rate
and 80% schedule transition are retained. Different update counts and horizon
windows do not represent equal FLOPs. The existing managed CUDA wall limit
remains 1,800 seconds. Equal physical-target work uses the measured K2 training
cost to bound other horizons; actual per-update and setup costs must be reported.

Both actual CPU/CUDA constructors must pass all 64 cases, with repeated active
gradients through two deployed blocks, frozen viscosity and the same 6,600
trainable coefficients. The finite admission batch stops on failure and never
retries. Only after qualification does it freeze fit sources/configuration,
capture Projects and start ordinary managed CUDA fits plus typed exact CPU
restores. No local parameter learning is allowed. All complete development
outputs and own240 are evaluated serially for K2, K3, K4 and the K1 control.

Inference remains entirely neural. Native displacement labels are training-only;
no native viscosity, pressure solve or external correction enters the model.
CPU execution variants require qualification against each new trained model.
Complete block limits remain below 4/6/8 ms, with every intermediate state and
no internal-stage cap. Fixed-window timing does not establish application timing.
Replicate qualifying recipes and the one-step control, freeze them, then capture
the eight reserved temporal scenes. No reserved scene has been consumed.

[Qualification protocol](fixed-viscosity-qualification-protocol-01.json) ·
[Finite admission sequence](fixed-viscosity-admission-batch-specification-01.json) ·
[Prior all-field accuracy](neural-viscosity-results.md)
