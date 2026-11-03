# Temporal conditioning follow-up

The first two-step endpoint fit underfits training scenes: median warm pressure error is0.266 and velocity prediction remains close to persistence. The two-step retained-feature fit also misses the development target. Their other K values continue unchanged.

Training-only diagnostics show prescribed forcing at RMS0.02135 in the encoder’s current velocity units. One-step velocity and tracer increments have RMS0.01519 and0.001679 respectively, while the output heads currently use full-state units. A fixed train-only conditioning change is a concrete next experiment; these observations do not prove it will fix the error.

| Physical horizon | Velocity increment RMS / current velocity scale | Tracer increment RMS / current tracer scale |
| --- | ---: | ---: |
| 1 | 0.015188946 | 0.0016787425 |
| 2 | 0.030349376 | 0.0033536896 |
| 3 | 0.045442236 | 0.0050248082 |
| 4 | 0.060433156 | 0.0066920755 |

The follow-up divides known forcing by its fixed aggregate training RMS before encoding, and converts learned velocity/tracer output increments using the fixed per-horizon training RMS. Pressure remains in its original units. This changes conditioning with no added parameters; original physical losses, error floors, targets, scenes, normalization, initial parameter tensors, optimizer schedule and own-history curriculum remain unchanged. No future query labels enter these conversions.

Both endpoint and retained-feature K2 fits are registered for10,000updates each. CPU qualification covers all K values, all grid groups, identical initial physical predictions and active deterministic two-block gradients using the actual trainer constructor. The captured Projects match those sources and statistics. CUDA qualification and both managed fit/restore/evaluation pairs are queued after the existing six-model comparison and its clean serial CPU timing finish. The queue never resubmits a failed or unknown Run.

Every predicted state remains subject to pressure, equation, velocity and tracer errors at or below0.004. A future latency claim still requires complete block timing below4ms forK2. Qualifying K2 evidence would support K3/K4 extensions and subsequent replication; no target success is claimed from this preparation.

The first scale diagnostic mixed pressure gauges. Its corrected successor mean-centres both pressure fields; all velocity/tracer and offline cold-start control results are retained. Candidate evaluations were already correctly mean-free and are unaffected.

[Frozen registration](temporal-step-scaled-registration-01.json) · [Evaluation protocol](temporal-step-scaled-evaluation-protocol-01.json) · [Corrected training diagnostic](../../../runs/full-ai-next-priorities-v1/temporal-sharing-scale-diagnosis-02.json) · [Train-only statistics](../../../runs/full-ai-next-priorities-v1/temporal-step-scale-statistics-01.json)

## Completed result

Both managed fits and CPU restores complete. Their control audits pass. Pressure
improves, but velocity remains near persistence and both fail full-state
accuracy and 240-step own-history. No recipe qualifies and no new CPU latency
claim is made. See [all conditioning outcomes](temporal-step-scaled-results.md)
and the [next matched supervision test](temporal-increment-supervision-plan.md).
