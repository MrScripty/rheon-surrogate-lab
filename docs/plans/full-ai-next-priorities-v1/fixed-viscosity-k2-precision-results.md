# Corrected two-step model: paired execution precision comparison

Same managed neural weights, one CPU thread, core23, batch one and five balanced serial repeats. Complete cost includes input/forcing preparation and every intermediate physical field. Float32 accuracy is recomputed on all eight scenes and through240 own-history steps.

| Policy | Block ms | Per step ms | Worst own240 error | Combined median pass |
| --- | ---: | ---: | ---: | --- |
| fast-float64 | 4.0838 | 2.0419 | 0.003396646 | False |
| fast-float32-map | 3.5445 | 1.7723 | 0.003403181 | True |

This warm mixed-grid median is not a per-grid or worst-call guarantee. Tails,32³, setup/RSS, seed replications, reserved scenes and full application/event costs remain separate.

[Raw costs, errors, per-grid measurements and sources](fixed-viscosity-k2-precision-results.json)
