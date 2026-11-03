# Global pressure primitive and precision preflight 01

Completed zero-update controls on consumed development states 1 and 121, at all eight scenes, K=1/2/3/4 and float32/float64. Each transform covers every cell-centred mode of every axis. Two warmups precede five timed serial repeats. Native teacher pressure and identity multipliers are used; no inverse coefficients, learned predictor or causal candidate result is claimed.

| Precision | K | Median primitive block | Maximum primitive block | Maximum per output | Worst equation relative L2 |
| --- | --- | --- | --- | --- | --- |
| torch.float32 | 1 | 0.4694 ms | 0.7709 ms | 0.7709 ms | 0.00013102 |
| torch.float32 | 2 | 0.6510 ms | 0.8760 ms | 0.4380 ms | 0.00013116 |
| torch.float32 | 3 | 1.1689 ms | 1.4552 ms | 0.4851 ms | 0.00013116 |
| torch.float32 | 4 | 1.5113 ms | 1.9563 ms | 0.4891 ms | 0.00013116 |
| torch.float64 | 1 | 0.6890 ms | 0.9432 ms | 0.9432 ms | 1.0908e-12 |
| torch.float64 | 2 | 1.2948 ms | 1.7039 ms | 0.8519 ms | 1.0933e-12 |
| torch.float64 | 3 | 1.9631 ms | 2.9716 ms | 0.9905 ms | 1.0933e-12 |
| torch.float64 | 4 | 2.3027 ms | 4.4463 ms | 1.1116 ms | 1.0933e-12 |

These timings include the forward/inverse real DCT-II, pressure reconstruction and materialization. They exclude forcing, input encoding, learned mixing, dynamics, velocity/tracer generation and history updates. They are therefore neither complete model latency nor a frontier point. Prepared bases and input copies are setup.

The separate float64 normalization/reconstruction control has worst equation residual 1.62089e-13. Full double transforms preserve the control accurately; the float32 transform control fails 1e-5 despite its small pressure L2 error. Double transform/output arithmetic is the next baseline precision policy. These controls are not evidence that a learned multiplier will achieve this accuracy.

The dense axis-transform implementation leaves little full-block latency margin, especially under observed maxima. A faster fixed transform backend and shared intermediate decoding are bounded follow-up cost hypotheses; unchanged dynamics/readout costs must not be ignored. Qualify any alternative transform against the same native precision controls before fitting.

Raw evidence: [global-pressure-preflight-01.json](../../../runs/full-ai-tight-targets-v1/global-pressure-preflight-01.json). Protocol: [global-pressure-preflight-protocol-01.json](global-pressure-preflight-protocol-01.json).

Source SHA-256: `0be736c2afdf753945d1ca40378f4a2f593394d7c08cd5d5c444aa4d81054481`.
