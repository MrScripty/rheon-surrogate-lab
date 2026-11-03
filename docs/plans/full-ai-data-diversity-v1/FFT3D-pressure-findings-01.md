# Fused 3D FFT pressure execution

The new full-AI execution prototype passes the registered native accuracy checks but is slower than the parent at 32³, 100 cells per metre. It remains available for reproduction; it is not promoted.

The learned pressure response and all model coefficients are unchanged. Only the N8 pressure transform uses the new generic double-precision 3D FFT with fused layout kernels. The original N1 path is retained.

| Steps per call | Parent median ms/step | Prototype median ms/step | Prototype max ms/step | Prototype median block ms |
|---:|---:|---:|---:|---:|
| 1 | 2.225 | 2.547 | 3.235 | 2.547 |
| 2 | 2.071 | 2.397 | 3.021 | 4.794 |
| 3 | 2.034 | 2.360 | 2.910 | 7.081 |
| 4 | 2.019 | 2.349 | 2.754 | 9.397 |

These are **GPU (RTX 5090 Laptop)** complete-call timings, including native input views, packing, CPU↔GPU transfers, known forcing, every internal advance, all physical outputs and synchronization. Divide only by generated physical outputs. Each table row has 20 measured calls; the whole balanced study has 3,840 calls across 24 development scenes and four starting windows. All 9,600 timed outputs and 11,520 warmup outputs matched their previously scored bytes. All grid/density groups and observed maxima remain in the JSON report.

Accuracy evidence: 78 independent cosine-matrix/round-trip controls and 13 changed-input graphs passed; maximum component discrepancy was 4.77e-14. All 1,018 native block outputs passed, worst equation error 6.39e-8. A fresh uninterrupted native 240-step comparison passed all 960 outputs across K=1–4, worst equation error 2.22e-14. Replaying all 26 consumed expanded histories matched 24,960 native-scored complete outputs, and the 16 consumed smaller-grid regressions matched 15,360 complete outputs. These are development/regression results, not independent reserved confirmation.

The one diagnostic profile counted 190 CUDA activities per N8 physical output versus 150 for the parent and 486 in the earlier axis-by-axis FFT prototype. However, the 3D FFT processes an eightfold even extension: in this trace its FFT kernel family took more time than the fixed transforms. Unchanged gather timings varied too; the trace does not isolate every cause. The complete-call wrapper overlaps child CUDA events, so summed families cannot be treated as a runtime decomposition.

Two implementation failures were preserved and isolated before qualification: absent CUDA context at module loading, then an incorrect contiguous-layout assumption for the FFT result. The fixes initialize the context and pass actual tensor strides. Independent controls then passed on all 13 consumed shapes, including odd rectangles and 128³. No native solver or physical inverse was added to inference.

The 1.2 ms target remains unmet. No new parameter fitting or reserved-reference consumption occurred. Do not rerun the unchanged candidate to seek a better maximum. The next useful transform investigation is a compact separable FFT with fused phase/layout operations, testing whether avoiding the eightfold workspace reduces cost; its design and finite trial budget must be registered before execution. The 3D prototype has no measured larger-domain speed advantage.

Raw evidence and reproduction bindings: `FFT3D-pressure-results-01.json`.
