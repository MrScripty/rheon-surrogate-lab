# Supported physical-domain boundary change

After accepting step13, change the physical box size by switching8↔16cells per metre at fixed logical cell counts. The controller copies the current arrays, discards unused predictions and calls the same frozen model with the changed geometry. Fluid mass density and the closed boundary type remain unchanged. Every output through48 is evaluated using its matching native geometry; the externally imposed volume and mass mapping is recorded. Unsupported boundary types retain the separate explicit stop guard.

| Model | Scene | Cells/metre before→after | Discarded at event | Fresh continuation | Worst diagnostic error | Accuracy diagnostic pass |
| --- | --- | --- | ---: | --- | ---: | --- |
| b1-fixed-viscosity-fast-float64 | temporal-g0-cpm8-r2 | 8→16 | 0 | True | 0.0001015186 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g0-cpm16-r2 | 16→8 | 0 | True | 7.637552e-05 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g1-cpm8-r2 | 8→16 | 0 | True | 0.0001101502 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g1-cpm16-r2 | 16→8 | 0 | True | 3.899541e-05 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g2-cpm8-r2 | 8→16 | 0 | True | 0.0001284337 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g2-cpm16-r2 | 16→8 | 0 | True | 0.000290779 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g3-cpm8-r2 | 8→16 | 0 | True | 0.0003570101 | True |
| b1-fixed-viscosity-fast-float64 | temporal-g3-cpm16-r2 | 16→8 | 0 | True | 0.0003978269 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g0-cpm8-r2 | 8→16 | 1 | True | 0.0001030912 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g0-cpm16-r2 | 16→8 | 1 | True | 7.785822e-05 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g1-cpm8-r2 | 8→16 | 1 | True | 0.0001255881 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g1-cpm16-r2 | 16→8 | 1 | True | 3.916719e-05 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g2-cpm8-r2 | 8→16 | 1 | True | 0.0001283646 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g2-cpm16-r2 | 16→8 | 1 | True | 0.0002907651 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g3-cpm8-r2 | 8→16 | 1 | True | 0.0003570101 | True |
| b2-fixed-viscosity-fast-float32-map | temporal-g3-cpm16-r2 | 16→8 | 1 | True | 0.0003978269 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g0-cpm8-r2 | 8→16 | 2 | True | 0.0001015186 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g0-cpm16-r2 | 16→8 | 2 | True | 7.637552e-05 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g1-cpm8-r2 | 8→16 | 2 | True | 0.0001436343 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g1-cpm16-r2 | 16→8 | 2 | True | 3.899541e-05 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g2-cpm8-r2 | 8→16 | 2 | True | 0.0001284337 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g2-cpm16-r2 | 16→8 | 2 | True | 0.000290779 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g3-cpm8-r2 | 8→16 | 2 | True | 0.0003570101 | True |
| b3-fixed-viscosity-fast-float64 | temporal-g3-cpm16-r2 | 16→8 | 2 | True | 0.0003978269 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g0-cpm8-r2 | 8→16 | 3 | True | 0.0001015186 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g0-cpm16-r2 | 16→8 | 3 | True | 7.637552e-05 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g1-cpm8-r2 | 8→16 | 3 | True | 0.0001348526 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g1-cpm16-r2 | 16→8 | 3 | True | 3.899541e-05 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g2-cpm8-r2 | 8→16 | 3 | True | 0.0001284337 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g2-cpm16-r2 | 16→8 | 3 | True | 0.000290779 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g3-cpm8-r2 | 8→16 | 3 | True | 0.0003570101 | True |
| b4-fixed-viscosity-fast-float64 | temporal-g3-cpm16-r2 | 16→8 | 3 | True | 0.0003978269 | True |

This verifies supported-boundary invalidation and regeneration on actual independently trained horizons. It is a functional/accuracy diagnostic, with no new CPU speed, arbitrary-boundary generalization or application-acceptance claim.

[Actual receipts, all fields and source evidence](application-domain-change-results.json)
