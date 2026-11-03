# Recurrent sampler follow-up: CPU efficiency and precision

The owner permits up to four physical CPU cores while retaining single-core latency as a primary efficiency metric. Additional cores are optional; every measured regression remains visible. The 1.2 ms maximum per generated step and 1e-5 worst-error gates remain unmet.

## Controlled CPU scaling

The existing managed-trained and exactly CPU-restored model is unchanged. A new strict OpenMP implementation divides independent rows in transport and sparse learned stages; pressure transforms and prescribed forcing remain serial. It preserves per-voxel operation order, guards and every intermediate physical output. No training occurred in this CPU experiment.

Performance-core configurations use CPU 6, CPUs 6/4 and CPUs 6/4/5/7. Historical efficiency-core CPU 23 is measured separately. Worker placement is verified in fresh processes; Torch and BLAS remain single-threaded. Passive workers prevent an idle worker pool from intentionally spinning during serial controls. The comparison includes checks, conditioning, all model stages and every complete intermediate output.

Each resource allocation passed 336 donor-address controls and 64 complete four-step state comparisons. Across all four allocations, 2,560 alternating timed complete blocks were byte-identical to the serial implementation. These are development measurements, without new held-out scene consumption.

| Steps/pass | One P core, serial ms/step | Two P cores ms/step | Four P cores ms/step | Four-core p95 | Four-core maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.273 | 6.208 | 5.820 | 8.104 | 8.286 |
| 2 | 9.333 | 6.181 | 5.539 | 12.217 | 14.462 |
| 3 | 9.070 | 6.071 | 4.829 | 5.519 | 8.838 |
| 4 | 8.965 | 6.043 | 4.906 | 6.015 | 6.259 |

All values are for 32 cubed. The first three columns are medians. For K steps, multiply a per-step value by K to obtain full-block latency; the machine-readable evidence also retains full-block median, p95 and maximum observations.

Within the alternating measurements, two cores provide about 1.56–1.59x speedup over the paired serial path and four provide about 1.67–2.03x. Speedup divided by allocated cores is about 0.78–0.79 and 0.42–0.51 respectively; this is a resource-efficiency proxy, not a CPU utilization measurement. These experiments do not isolate memory bandwidth, cache sharing, scheduling or synchronization as the cause of sublinear scaling.

The OpenMP implementation is slower when constrained to one core, so the original serial path remains the single-core implementation. CPU 23 measurements retain their separate hardware label. Small-domain timing tails also worsen with extra cores, including one roughly 30 ms/step four-core observation. Extra cores improve some medians but do not establish a maximum-latency guarantee. No timing outliers were removed.

## Matched Rheon application controls

Both 32-cubed development scenes, densities 8 and 16 cells per metre, steady forcing, 24 accepted steps, three timed repeats plus a separate 14-step warmup. Each configuration compares K1–4 generators with native PCG and the spectral reference control. Native processes and Rayon workers receive the same physical core allowance as the generator. Parent pressure-exchange transforms remain single-threaded; native process affinities are checked. The one-core generator uses the faster original serial implementation.

| Core allowance | Method | Completed timed episodes | Median application ms/accepted step |
| --- | --- | ---: | ---: |
| 1 | sampler-k1 | 6/6 | 9.753 |
| 1 | sampler-k2 | 6/6 | 10.186 |
| 1 | sampler-k3 | 6/6 | 9.785 |
| 1 | sampler-k4 | 6/6 | 9.443 |
| 1 | native-pcg | 6/6 | 215.738 |
| 1 | spectral | 6/6 | 53.917 |
| 2 | sampler-k1 | 6/6 | 7.038 |
| 2 | sampler-k2 | 6/6 | 7.115 |
| 2 | sampler-k3 | 6/6 | 6.816 |
| 2 | sampler-k4 | 6/6 | 6.586 |
| 2 | native-pcg | 6/6 | 221.194 |
| 2 | spectral | 6/6 | 55.284 |
| 4 | sampler-k1 | 6/6 | 5.347 |
| 4 | sampler-k2 | 6/6 | 5.298 |
| 4 | sampler-k3 | 6/6 | 5.155 |
| 4 | sampler-k4 | 6/6 | 4.951 |
| 4 | native-pcg | 6/6 | 208.822 |
| 4 | spectral | 6/6 | 52.409 |

The application clock includes prediction or numerical advance, physical checks, state exchange and final verification/close; setup and pure offline exports are separate. All 108 timed and 36 warmup episodes completed, with every native owner reaped. Neural consumers perform zero native numerical advances. Native control physical gates remain unchanged. Physical outputs are retained for deferred accuracy analysis; this timing experiment does not newly certify long-history accuracy or replace the original independent 240-step evaluation.

## Pressure precision investigation retained from this follow-up

A bounded managed CUDA refinement of the existing 38,673 learned modal gains completed 80 LBFGS calls and an exact typed managed CPU restore. The original attempt completed its updates but failed publication because its 1,218,858-byte normalization artifact exceeded a declared 1 MiB output cap. A separately registered attempt corrected only that declaration to 2 MiB and completed train/restore custody. Both attempts and the red/green publication regression remain preserved.

On 1,920 native-RHS development requests, refined learned pressure reaches approximately 1.23e-12 worst equation residual, versus roughly 1.07e-9 previously. However, the complete generator still reaches about 1.07e-4 reference-equation difference over 240 own-history steps at 32 cubed. The same-request pressure improvement does not establish an improved trajectory. All K1–4 complete own-history paths were evaluated; no target pass is claimed.

Changing only the fixed transform implementation to explicit tensor cosine transforms reduces the small-domain own-history equation difference to about 1.18e-13 with zero velocity difference, but the 32-cubed difference remains about 1.06e-4. Uniform local/head coefficient rounding and the physical-coordinate arithmetic control do not resolve it. An explicitly offline numerical pressure oracle was a diagnostic only and is excluded from candidate/frontier results. No analytical inverse was inserted into candidate inference.

The refined pressure checkpoint is a development result. The 16 newly reserved precision-confirmation scenes remain unconsumed. The original fully independently tested prototype remains available.

## Next work

Keep the faster serial path, treat two/four cores as explicit deployment options, and reduce complete work and timing tails before claiming a speed pass. Continue narrowing the remaining 32-cubed trajectory differences with bounded controls. Freeze any selected successor before independent confirmation. Higher-density training and the 600-step phase remain behind the existing ordered gates.

[CPU resource policy](cpu-resource-policy-04.md) · [Machine-readable results](recurrent-sampler-followup-summary-02.json) · [Original independent prototype results](recurrent-sampler-results-01.md)
