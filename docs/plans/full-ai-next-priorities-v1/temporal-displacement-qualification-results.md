# Actual displacement model qualification

CPU and CUDA each pass all 64 registered constructor cases: K=1/2/3/4, four geometries, both spatial densities and cold/warm states. The model contains 100,456 parameters, with 6,843 trainable parameters.

The checks exercise the actual trainer and objective through two consecutive blocks of predicted history, repeated gradients, device parity, frozen learned coefficients and active trainable paths. Each block executes one shared encoder and K local/displacement/pressure/flux/momentum readouts. Training-label code is disabled during inference checks. Passive tracer, zero/constant fields, donor bounds and entering-velocity staging pass.

These are zero-update implementation checks. The diagnostic gradient tests activate separate parameter copies; initial error records use selected training windows. Neither qualifies held-out accuracy, own240 behavior or CPU inference latency. Full-state fitting, exact CPU restoration and the complete development evaluation are the next gates.

[Evidence and source hashes](temporal-displacement-qualification-results.json) · [Frozen protocol](temporal-displacement-qualification-protocol-01.json) · [Implemented architecture](temporal-displacement-plan.md)
