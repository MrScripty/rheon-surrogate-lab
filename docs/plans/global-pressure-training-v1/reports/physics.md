# Negative development pressure diagnostic

The predeclared 32 teacher-state cases completed: frames 0 and 23 for all 12
training and 4 validation trajectories. Exact trained receipt, weights and pressure
witness bytes were bound to the qualified native binary and unchanged original
384-frame capture. Every case is retained in [the full diagnostic](physics.json).
These states use reference histories; no autonomous rollout was advanced.

All 32 learned pressures equal previous reference pressure byte for byte. Raw
learned pressure passed 0/32 cases, with maximum relative full residual
1.0545499785 and actual divergence 0.1407383351 /s. All 32 proposed warm solves
exhausted their 32 iterations and failed residual/divergence gates. Each reset
once from the previous pressure and completed a full-budget native fallback.
The accepted fallback passed 32/32: maximum full relative residual 9.9598038e-9,
actual float32 divergence 2.8808912e-7 /s, predicted divergence 1.3681407e-9 /s,
net flux magnitude 1.2334494e-19 m³/s; exact closed-wall flux and Courant checks
passed. Reference, zero-Jacobi and fair full-budget temporal PCG also passed 32/32.

| Observed cost over 32 fixed cases | Seconds | PCG iterations |
| --- | ---: | ---: |
| Learned warm solve plus complete reset/verification | 0.547990100 | 1,024 + 5,226 |
| Recorded model inference | 0.347912624 | — |
| Fair temporal PCG, one full attempt from previous pressure | 0.440765174 | 5,226 |
| Zero-Jacobi, one full attempt | 0.438621068 | 5,393 |

The learned solve plus recorded inference sums to 0.895902724 seconds, over twice
the separately measured temporal baseline sum. These sequential diagnostic stage
times are descriptive and include all failed warm/fallback work. They do not form
a repeated enclosing end-to-end benchmark. The evaluator wall was 2.710659 seconds,
including every baseline, native startup/JSON and verification; summed native
process walls were 2.117309 seconds. Generic custody, data packing, runtime capture,
training and restore costs are separately recorded and must not disappear from
any future total-cost comparison. No positive speedup is claimed.

This model has no demonstrated solver benefit. A qualified fallback does not
qualify its learned proposal as an accelerated solver. Full trajectory stability,
sealed geometry/regime holdouts and engineering-flow transfer remain untested.
Do not materialize them or run full/GPU training based on this result. A next
reviewed slice should diagnose representation/output collapse on development
arithmetic and compare a narrow alternative (for example a supervised low-mode
correction) before authorizing another scientific training configuration. Keep
these weights and all original failures as immutable evidence.
