# Selected temporal recipes: three seeds and actual CPU costs

All added seeds are retained. Each policy has its own complete accuracy or exact execution proof. Five balanced serial repeats use one CPU thread, core23 and batch one, with complete/tensor scopes and matched sequential B1controls. Original primary seed remains fixed.

| Model | Seed ordinal | Block ms | ms/state | Worst own240 error | Combined median pass |
| --- | ---: | ---: | ---: | ---: | --- |
| b1-fixed-viscosity-fast-float64 | 1 | 2.3635 | 2.3635 | 0.003721229 | Control |
| b2-fixed-viscosity-fast-float32-map | 1 | 3.6663 | 1.8331 | 0.003403181 | True |
| b4-fixed-viscosity-fast-float64 | 1 | 6.8981 | 1.7245 | 0.003367723 | True |
| b1-fixed-viscosity-s2-fast-float64 | 2 | 2.3361 | 2.3361 | 0.003322732 | Control |
| b2-fixed-viscosity-s2-fast-float32-map | 2 | 3.5903 | 1.7951 | 0.003559253 | True |
| b4-fixed-viscosity-s2-fast-float64 | 2 | 6.9568 | 1.7392 | 0.003328111 | True |
| b1-fixed-viscosity-s3-fast-float64 | 3 | 2.3109 | 2.3109 | 0.004348945 | Control |
| b2-fixed-viscosity-s3-fast-float32-map | 3 | 3.5788 | 1.7894 | 0.003946324 | True |
| b4-fixed-viscosity-s3-fast-float64 | 3 | 6.9879 | 1.7470 | 0.003020889 | True |

These are warm mixed-small-grid medians. Per-grid/tails,32³, setup/RSS, reserved generalization and sustained application/event throughput are separate. Seeds vary the temporal cell and sampling, with shared frozen component pretraining.

[Every seed, raw costs, accuracy and sources](fixed-viscosity-replication-cost-results.json)
