# Eight-cohort pressure-method results

All eight bounded cohorts and their selected follow-ups are complete. Twelve managed fits each ran 1,600 CUDA updates and a fresh CPU restore, with exact parity of the five retained artifacts. C53 remains the baseline; no model was promoted.

The full spectral numerical control is the strongest prepared-pressure baseline on these sealed, uniform grids. The neural/hybrid candidates below are useful comparisons against the existing learned policies. These timings include numerical work and final residual evaluation; the historical chart uses a different scope and was left unchanged.

| Cohort | Result |
|---|---|
| 1. larger-training-domains | Mixed: strong 32/64 cube gains; small and larger-rectangular regressions |
| 2. frozen-model-numerical-correction | Promising: C53 before-six, C44 before/after-seven and C37 before/after-ten qualify all standard requests |
| 3. finer-teacher-timesteps | Small matched improvement at half-step; no monotonic quarter-step improvement and no gain over original C53 at eight passes |
| 4. numerical-coarse-neural-fine | Promising: 103/128 at four passes, 128/128 at five; eight-pass worst residual about 9.07e-10 |
| 5. sparse-numerical-anchors | No general advantage from this exact-anchor construction; standard and large-grid regressions remain |
| 6. revised-learned-coarse-fine | Positive gated low-mode branch is finite and trainable; Improved standard-request worst residual |
| 7. spacing-training-and-conditioning | Explicit spacing does not improve independent-spacing worst error over its matched zero-feature control |
| 8. periodic-numerical-correction | Eight-pass periodic policies remain below declared drift limits over the tested horizons; periodic correction does not rescue rejected four-pass requests |

Eight-pass accuracy on the common populations (qualified requests / total, equation tolerance 1e-6):

| Training arm | Standard 128 | New 32³ | New 64³ | Larger rectangular | Worst standard residual |
|---|---:|---:|---:|---:|---:|
| domain-small-control | 128/128 | 0/4 | 0/4 | 2/4 | 2.32e-07 |
| domain-expanded | 123/128 | 4/4 | 4/4 | 0/4 | 6.22e-06 |
| teacher-coarse | 128/128 | 0/4 | 0/4 | 2/4 | 2.32e-07 |
| teacher-half | 128/128 | 0/4 | 0/4 | 2/4 | 2.07e-07 |
| teacher-quarter | 128/128 | 0/4 | 0/4 | 2/4 | 2.28e-07 |
| numerical-coarse | 128/128 | 4/4 | 4/4 | 2/4 | 9.07e-10 |
| sparse-anchors | 124/128 | 4/4 | 0/4 | 0/4 | 9.95e-06 |
| learned-coarse | 126/128 | 4/4 | 4/4 | 1/4 | 1.71e-06 |
| spacing-data-control | 128/128 | 0/4 | 0/4 | 3/4 | 2.36e-07 |
| spacing-zero-feature | 128/128 | 0/4 | 0/4 | 2/4 | 2.36e-07 |
| spacing-conditioned | 128/128 | 0/4 | 0/4 | 2/4 | 2.11e-07 |
| coarse-head-rate | 128/128 | 4/4 | 4/4 | 2/4 | 3.87e-08 |

Selected quiet comparisons: same CPU, three repetitions per request, after owned campaign workers ceased. Geometry caches are warm; loading and one-time geometry setup are excluded. First-invocation measurements are retained in the primary evaluation files.

| Model/policy | Neural passes | Qualified / 128 | Worst equation residual | Median pressure-policy ms |
|---|---:|---:|---:|---:|
| C53 / none | 7 | 128/128 | 7.42e-07 | 19.400 |
| C53 / none | 8 | 128/128 | 1.45e-07 | 22.160 |
| C53 / before | 6 | 128/128 | 8.9e-07 | 17.098 |
| C53 / before | 7 | 128/128 | 1.73e-07 | 19.845 |
| C53 / full-numerical | 0 | 128/128 | 5.53e-15 | 0.360 |
| C44 / before | 7 | 128/128 | 4.93e-07 | 20.005 |
| C44 / after | 7 | 128/128 | 4.97e-07 | 19.932 |
| C37 / before | 10 | 128/128 | 4.28e-07 | 27.799 |
| numerical-coarse / none | 5 | 128/128 | 8.38e-07 | 16.268 |
| numerical-coarse / none | 6 | 128/128 | 8.57e-08 | 19.627 |
| coarse-head-rate / none | 7 | 128/128 | 1.9e-07 | 22.685 |

Frozen C53 before-six completed all 3,585 post-warmup steps in a 3,600-step simulation (60 seconds total). Its maximum pressure/velocity/tracer relative L2 drift was 9.11e-07 / 1.54e-07 / 1.95e-07. No declared 1% drift threshold was crossed. This establishes a tested lower bound, not a prediction of eventual failure time.

The finer-teacher experiment preserves a fixed model pressure equation and uses matched coarse-history initial guesses. Its rescaled instantaneous pressure requests are not a claim of exact fine-integrated coarse flow labels. The coarse-control fit exactly reproduces the small-only control weights.

The spacing ablation varies cells per metre separately from cell counts; mass density remains 1000 kg/m³. Both two-channel arms have identical initialization, with the added convolution column zeroed. All independent-spacing scenes are excluded from fitting.

These are controlled development results from one training seed per new arm. The larger-data arms use balanced spatial tiers under the same optimizer-update budget, changing per-parent exposure and total forward-cell work. The full numerical control uses separable dense cosine transforms; no production FFT scaling claim is made. Coupled larger-domain testing here uses 32³, while the 64³ pressure requests and training gradients were tested offline. Arbitrary geometry and production sizes 256³–1024³ remain future work. Native runtime includes two tracks, diagnostic reference work and IPC; a single-track application benchmark is still needed before a deployment speed claim.

[Full results and hashes](results.json) · [Execution ledger](execution-ledger.md) · [Domain comparison](domain-training-results.md) · [Drift execution](drift-execution.json)
