# Pressure-floor and sampling-head precision diagnosis

The frozen decoder-compatible candidate passes all short blocks and 14 of 16 independent own240 scenes. Its two equation-criterion failures remain real target failures. Consumed-scene controls identify one repair and narrow the remaining pressure issue; none is a new independently qualified candidate.

## Failure timing

| Consumed scene | First carrier byte difference | First 1e-5 criterion failure | Worst equation difference |
| --- | ---: | ---: | ---: |
| modal-precision-fresh-03 | 18 | 51 | 2.35746e-05 |
| modal-precision-fresh-15 | 56 | 71 | 6.75336e-05 |

Pressure, velocity and tracer errors remain below 1e-5 in both failures. The equation criterion includes accumulated differences from the independent native trajectory. Tiny carrier changes can grow into a failing equation difference, so isolated pressure residuals and training loss cannot replace own-history confirmation.

## Matched controls on the two consumed scenes

| Control | Pressure role | 16-cubed worst equation | 32-cubed worst equation |
| --- | --- | ---: | ---: |
| frozen_learned_baseline | managed learned pressure | 2.35746e-05 | 6.75336e-05 |
| OFFLINE_numerical_pressure | OFFLINE numerical oracle | 2.35746e-05 | 1.31898e-13 |
| forcing_gaussian_learned | managed learned pressure | 2.35746e-05 | 6.75336e-05 |
| forcing_gaussian_OFFLINE | OFFLINE numerical oracle | 2.35746e-05 | 1.31898e-13 |
| forcing_scalar_learned | managed learned pressure | 2.35746e-05 | 6.75336e-05 |
| forcing_scalar_OFFLINE | OFFLINE numerical oracle | 2.35746e-05 | 1.31898e-13 |
| head_m30_learned | managed learned pressure | 8.3312e-14 | 6.75336e-05 |
| head_m30_OFFLINE | OFFLINE numerical oracle | 1.11973e-08 | 1.31898e-13 |
| head_m40_learned | managed learned pressure | 8.3312e-14 | 6.75336e-05 |
| head_m40_OFFLINE | OFFLINE numerical oracle | 1.11973e-08 | 1.31898e-13 |
| head_m40_tensor-multiplier | managed learned pressure | 8.42287e-14 | 6.75336e-05 |
| head_m40_tensor-division | managed learned pressure | 8.43152e-14 | 6.75336e-05 |
| head_m40_fftw-division | managed learned pressure | 8.35045e-14 | 6.75336e-05 |

Every control predicts causally through all 240 steps from the initial physical state. Future labels enter only offline scoring. Oracle rows deliberately replace learned pressure with the numerical reference and are excluded from AI candidates, frontier claims and latency results. All other pressure rows use only exact managed learned parameters or declared arithmetic/precision transforms of those parameters. No new parameter fitting occurred in these diagnostics.

The numerical pressure reference removes the 32-cubed failure while leaving the original 16-cubed failure. Replacing factorized Gaussian forcing with a scalar Gaussian, then also using scalar sine metadata, changes neither failure. These forcing changes are not adopted.

Uniform rounding of all 48 restored query-head parameters to either 2^-30 or 2^-40 repairs the 16-cubed learned-pressure trajectory: worst equation difference becomes about 8.33e-14 and velocity difference is zero. The raw managed checkpoint is preserved, and no algorithmic coefficient is assigned. The head controls keep the original forcing implementation; an auxiliary execution clarification records an unused inherited forcing-policy field in their registrations.

After head rounding, explicit cosine multiplication, explicit cosine division and FFTW division by derived learned values all leave the 32-cubed failure near 6.75e-5. These tests narrow the remaining issue to learned-pressure behavior in this scene; they do not establish whether supervision conditioning, parameter representation or another pressure detail is the cause.

## Next bounded experiment

Keep the uniform head-precision repair as a candidate inference policy, then improve pressure supervision before spending another independent bank. The next preparation should audit modal signal conditioning and add bounded mixed-mode pressure-query supervision from the offline reference, preserving the original native training pairs. A directly learned double divisor is a separate representation to compare with log/exp gains. It must be fitted on managed CUDA from data; no analytical divisor or inverse-coefficient labels may be inserted.

Before either fit: qualify the actual objective on CPU and CUDA, freeze its finite update budget and sources, and require exact managed CPU restoration. Sixteen new whole scenes have now been reserved, with seeds disjoint from all listed prior reservations. Their references remain uncaptured. Full K1–4 development checks and candidate/source freezing must precede any new confirmation.

The speed requirement also remains unmet. Preserve the independently measured one/two/four-core costs and the single-core efficiency metric. Reduce complete execution work while keeping raw checkpoints and numerical checks intact. Densities 50/100 and 600-step drift testing remain behind the ordered joint gates.

A setup path typo in the conditioning build registration was corrected before compilation or numerical execution; its observation is retained. All completed controls closed their owners, and the registered source audit passed.

[Machine-readable controls](pressure-floor-and-head-precision-summary-10.json) · [Frozen candidate independent results](decoder-compatible-pressure-results-09.md) · [New confirmation reservation](pressure-successor-fresh-scene-reservation-10.json)
