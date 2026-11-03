# Compact FFT pressure execution

The compact full-AI execution prototype passes the registered accuracy gates but does not improve the measured GPU frontier at 32³, 100 cells per metre. It is retained for reproduction; the faster cell-coordinate parent and selected baseline remain unchanged.

| Steps per call | Parent median ms/step | Prototype median ms/step | Prototype max ms/step | Prototype median block ms |
|---:|---:|---:|---:|---:|
| 1 | 2.249 | 2.286 | 2.390 | 2.286 |
| 2 | 2.098 | 2.161 | 3.006 | 4.322 |
| 3 | 2.055 | 2.103 | 2.652 | 6.310 |
| 4 | 2.022 | 2.128 | 2.622 | 8.511 |

These are **GPU (RTX 5090 Laptop)** complete native-input-to-complete-output timings. CPU views/packing, CPU↔GPU transfers, known forcing, every internal advance, all K physical outputs, allocation, reconstruction and synchronization are included. Divide only by K physical outputs. Setup, plan preparation and warmups are separate. Each table row has 20 measured calls; the whole balanced study covers 3,840 calls, 24 development scenes, four starts, K=1–4 and five repeats. All 9,600 timed and 11,520 warmup outputs matched their own native-scored bytes. Grid dimensions and cells per metre remain separate in all 96 comparison groups.

The transform uses length-N real FFTs with even/odd ordering, fused phase/layout transitions, the unchanged learned response and fused pressure anchoring. It avoids a 3D eightfold extension, keeps F64/complex128 precision, and retains the original N1 path. It introduces no numerical pressure inverse.

Accuracy: 96 population controls and 16 changed-input graphs passed, including independent arbitrary-response checks, odd rectangles, singleton axes and 128³. Largest discrepancy was 8.50e-14. All 1,018 native block outputs passed, worst equation error 6.39e-8. Fresh uninterrupted native 240-step histories passed all 960 outputs across K=1–4, worst equation error 2.22e-14. All 26 consumed expanded histories matched 24,960 native-scored complete outputs, and all 16 smaller-grid regressions matched 15,360 complete outputs. These are consumed development/regression results, not independent reserved confirmation.

One failed component run is preserved. A constant-field fused-response comparison stopped because the explicit F64 cosine matrix leaked DC into its nominally zero answer. A minimal test at 64³ and 128³ showed both fused and unfused compact outputs exactly zero, while the matrix reference had tiny nonzero norms. The corrected verifier uses the exact analytic zero for zero/constant fused responses, records the old matrix leakage separately, and retains matrix checks for every nonconstant population and changed-input graph. Neither candidate source nor component/native thresholds changed.

The single post-study profile counted 238 CUDA activities per N8 output versus 150 for the parent. Across three calls, FFT kernels took 1.009 ms plus 0.331 ms of layout kernels; parent fixed transforms took 1.156 ms. This supports additional layout/activity erasing a modest core FFT saving. The trace includes an overlapping complete-call wrapper, so sums of families are not a runtime decomposition. Learned transport remained the largest named kernel family at about 0.98 instrumented ms per output.

The 1.2 ms target remains unmet. No new fit or reserved-reference consumption occurred. No unchanged candidate timing/profile rerun is authorized by these results. The next investigation should address learned transport, first distinguishing a cheap compensated-precision certificate from the earlier expensive F64 guard approach that regressed despite high fast-path coverage. Exact arithmetic/rounding bounds and finite candidate gates must be frozen before execution.

Detailed source, failed-control, native, timing and profile bindings: `compact-FFT-pressure-results-01.json`.
