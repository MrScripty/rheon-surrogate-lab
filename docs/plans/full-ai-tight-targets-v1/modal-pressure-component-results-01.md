# Learned pressure component result 01

The restored learned modal pressure branch passes both 1e-5 pressure criteria on all **1,920 consumed development states**. An independent audit recomputed every preserved physical output against native pressure/RHS using a NumPy edge-flux stencil. This is an offline teacher-RHS component result, not a causal full-state or own-history generator pass.

| Population | States | Worst equation relative L2 | Worst mean-free pressure relative L2 |
| --- | --- | --- | --- |
| small | 1440 | 1.9419979e-10 | 8.0065644e-12 |
| 32cubed | 480 | 1.0694166e-09 | 4.7301084e-12 |
| all | 1920 | 1.0694166e-09 | 8.0065644e-12 |

All 16 training scenes contributed 144 original native double pressure/RHS pairs. The managed CUDA fit made exactly 800 Adam updates on 38,673 double log-gain parameters; DC outputs are fixed to zero. Every non-DC mode had native training observations. Final supervised log-gain loss was 1.89951794e-13. No analytic pressure eigenvalues/inverse coefficients or local parameter fitting were used. The deployed operation is a fixed DCT, learned diagonal multiplication and inverse transform. Tables cover only the four registered geometries; this is not unseen-shape evidence.

Training Run: `2b498bb65aae42f5bb834a1d0152c60c`. Exact CPU restore Run: `09f441e79aa9486ea5eb917e309ef25a`. Both completed and drained. All weights, initialization, history, normalization and complete CPU pressure-probe artifacts matched byte for byte; restore made zero updates and did not initialize CUDA. Parameter inventories remain `<f8` throughout.

The loaded CPU FFTW implementation agrees with the restored Torch pressure graph to at most 2.785184e-15 relative L2 across these states. Double physical output avoids the previously measured float32 codec residual floor. All 1,920 actual double pressure outputs are preserved (about 141 MB compressed).

This removes the observed frozen-pressure approximation error on the declared diagnostic population. It does not establish velocity/tracer accuracy, predicted-RHS accuracy, autonomous drift, fresh-scene confirmation, complete inference time, or any K=1/2/3/4 target. No candidate frontier point is claimed. The [execution controls](execution-path-qualification-01.md) still show that encoding/forcing/readout and the rest of the temporal architecture need work.

Next: compose this exact learned branch with the existing causal full-state checkpoints in a separate zero-update diagnostic, preserving double physical pressure and recording all changed arithmetic. Measure how much error remains in the model-generated RHS and transport rather than continuing to tune the replaced pressure CNN. Use that evidence to register the compact full-state architecture and its full-block CPU budget before training. Higher-density and 600-step phases remain gated.

Evidence: [evaluation and all output bindings](../../../runs/full-ai-tight-targets-v1/modal-component-evaluation-01/results.json), [independent raw-output audit](../../../runs/full-ai-tight-targets-v1/modal-physical-output-audit-01.json), [managed fit/restore receipt](../../../runs/full-ai-tight-targets-v1/modal-pressure-pair-01/receipt.json), [fit registration](modal-fit-registration-01.json).
