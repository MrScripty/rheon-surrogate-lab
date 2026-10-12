# CPU-only pressure-model comparison

The frozen 32,767-parameter checkpoint from trajectory-v2 runs entirely on CPU
in this experiment. No fitting or checkpoint changes occurred. CUDA devices are
hidden before PyTorch import; all model parameters and buffers are on CPU.
Every method uses the same Rheon binary and eight native CPU threads. PyTorch
inference also uses eight CPU threads with one interop thread. Native physics
and model inference occur sequentially, respecting this per-stage thread limit.

Three rotated-order repetitions across seeds 900–903 measured complete 120-frame
rollouts (two simulated seconds), including CPU inference, pipes, all physics
stages, qualification, native startup and teardown. Model/import setup, explicit
warmup, field capture and rendering are separately scoped.

| Path | Median wall time | CPU-only speedup |
| --- | ---: | ---: |
| Rheon Jacobi | 15.21 s | reference |
| Rheon SGS | 9.39 s | reference |
| Learned pressure + Rheon | 1.86 s | 8.20× / 5.06× |

All 4,320 timed frames met stored divergence ≤1e-5 s⁻¹ and observed strict relative
residual ≤1e-8. AI frames performed zero PCG iterations. Independent full-field
replay over four seeds against both references found worst velocity relative L2
error 9.00e-8. These already exposed seeds compare execution backends; they are
not a new untouched model-selection test.

`freeze.json`, `result.json` and `verification.json` retain device/thread settings,
runtime and source/checkpoint hashes, timings, per-frame gates and accuracy.
Bulk numeric viewer fields remain in the ignored CPU run directory. The viewer
defaults to CPU-only replay and labels the earlier GPU recording separately.

Reproduce with a new destination:

```sh
python3 scripts/pumas_trajectory.py cpu-compare runs/trajectory-v2 runs/trajectory-v2/cpu-only-replication --threads 8
```

The model supports the declared uniform sealed 32³ box with prescribed forcing
and passive tracer. An analytic spectral solver also exists for this operator;
this comparison establishes speed relative to Rheon PCG, without establishing
superiority over that nonlearned alternative.
