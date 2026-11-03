# Frozen-model numerical correction: offline screen

All three workers completed. Pure-neural residuals exactly reproduce the prior
128-request measurements at every tested pass count. No new training or
unused scenes were consumed. This completes the first offline screen, not
the native follow-up or the eight-cohort campaign.

| Checkpoint | Policy | Real requests at 1e-6 | Worst real residual | Median policy ms |
| --- | --- | ---: | ---: | ---: |
| C53 | neural-4 | 7/128 | 9.78553e-05 | 11.626 |
| C53 | before-4 | 24/128 | 2.46268e-05 | 12.189 |
| C53 | after-4 | 15/128 | 5.82089e-05 | 12.239 |
| C53 | neural-8 | 128/128 | 1.45222e-07 | 23.040 |
| C44 | neural-4 | 0/128 | 0.000147907 | 11.656 |
| C44 | before-4 | 6/128 | 6.85428e-05 | 12.036 |
| C44 | after-4 | 5/128 | 5.60443e-05 | 12.177 |
| C44 | neural-8 | 128/128 | 3.94466e-07 | 22.921 |
| C37 | neural-4 | 0/128 | 0.00148807 | 11.315 |
| C37 | before-4 | 2/128 | 0.000163888 | 11.699 |
| C37 | after-4 | 0/128 | 0.000187299 | 11.678 |
| C37 | neural-8 | 112/128 | 2.28931e-05 | 22.355 |
| No network | Full numerical spectral | 128/128 | 5.51138e-15 | 0.535 |

These timings start from prepared physical tensors and end after the final
equation check. They include numerical transforms, reconstruction and neural
recurrence; they exclude loading, reference metrics, IPC and simulation.
They must not be compared directly with the existing HTML chart times.

Numerical low-mode success on pure modes inside its subspace follows by
construction. The 240 manufactured cases and 32 mixed cases are separate
populations in the raw reports; pure-mode success does not establish
generalization. Per-ordering/per-checkpoint tradeoffs remain explicit.

The full numerical solver is promising under the current sealed uniform-box
contract. Follow up with independent native trajectories and full-policy
timing; this does not demonstrate a learned hybrid or arbitrary geometry.

[Scalar comparisons and evidence hashes](cohort-02-results.json).
