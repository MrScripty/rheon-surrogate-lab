# Exact timestep/spacing scalar reuse

Passing the already validated timestep/spacing ratio into the N8 sampler preserves all measured output bytes, but does not provide a consistent speed improvement. The cell-coordinate research parent and selected baseline remain unchanged.

| Steps per call | Parent median ms/step | Prototype median ms/step | Prototype max ms/step | Prototype median block ms |
|---:|---:|---:|---:|---:|
| 1 | 2.225 | 2.215 | 2.420 | 2.215 |
| 2 | 2.163 | 2.164 | 2.609 | 4.327 |
| 3 | 2.136 | 2.097 | 2.510 | 6.290 |
| 4 | 2.044 | 2.111 | 2.450 | 8.446 |

**GPU (RTX 5090 Laptop)** timings at 32³ and 100 cells per metre, with eight actual internal advances per physical step. Complete calls include native views/packing, H2D, known forcing, all learned work, every K output, reconstruction, D2H and synchronization. Divide only by K. Each table row has 20 measured calls. The complete balanced study has 3,840 calls across 24 development scenes, four starting windows, K=1–4 and five repeats. All 9,600 timed outputs and 11,520 warmup outputs matched their expected bytes. All 96 shape/density/K comparison groups and maxima are retained.

The existing caller already computes and validates binary64 `dt/h`. Build 24 accepts that same scalar in the old timestep argument position, eliminating a repeated GPU division while leaving every other operation in double precision. The matching `AlphaArgumentPressureGraph` wrapper is required. There is no new CPU learned computation or numerical pressure inverse. PTX inspection confirms no `div.rn.f64` in any of the ten transport entries.

Correctness checks passed: 1,048,576 CPU/GPU quotient bit comparisons, 2,048 exact-rational rounding checks, 24 physical scalar-parameter launches, compiled learned parameter bytes, 196,608 ordered point outputs, 393,216 gate/mix outputs and 104 consumed/random/zero/nonfinite transport patterns. All 1,018 native block outputs matched the parent bytes, worst equation error 2.17e-11. All 26 expanded histories matched 24,960 already native-scored outputs, and all 16 smaller-grid histories matched 15,360 outputs. No fresh teacher or reserved reference was needed or used for this every-output byte-inheritance proof. This does not replace the original independent-confirmation gates.

The single paired profile found 150 CUDA activities per N8 output for both versions. Transport took 2.941 ms across three prototype calls versus 2.942 ms for the parent, effectively unchanged. The specialized kernel used 52 registers per thread versus 56, but both had the same theoretical occupancy limit and zero reported local storage. Static limits do not establish achieved occupancy or spills; the complete-call wrapper overlaps child CUDA events, so family sums are not a runtime decomposition.

The 1.2 ms target remains unmet. The prototype is retained for reproduction and will not be retimed unchanged. No new fits occurred; the 52 independent reserved references remain unopened. The next priority is a richer nonlinear coupled spatial/temporal capacity investigation, building on the failed narrow readouts and conditioned feature screens. Compensated full-sampler arithmetic remains a feasibility question, not an implemented or rejected model.

Detailed bindings and observations: `alpha-argument-results-01.json`.
