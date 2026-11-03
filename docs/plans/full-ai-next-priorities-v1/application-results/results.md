# Full-state application and interaction measurements

Five balanced serial CPU repeats include complete neural generation, required checks, actual native state handoff, event reinitialization and discarded outputs. Offline reference comparisons follow the measured episode. Native numerical controls retain their original pressure gates; the neural state consumer is structural.

| Method | Scenario | Grid group | Complete episodes | App s / physical s | Model ms / generated state | Worst error | Accuracy scope |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| C107 | forcing | 32cubed | 0/10 | Unavailable | Unavailable | Unavailable | Native diagnostic checkpoints only |
| C107 | forcing | small | 0/30 | Unavailable | Unavailable | Unavailable | Native diagnostic checkpoints only |
| C107 | steady | 32cubed | 0/10 | Unavailable | Unavailable | Unavailable | Native diagnostic checkpoints only |
| C107 | steady | small | 0/30 | Unavailable | Unavailable | Unavailable | Native diagnostic checkpoints only |
| b1-fixed-viscosity-fast-float64 | forcing | 32cubed | 10/10 | 6.72965 | 111.053 | 0.000397827 | Every consumed forecast state |
| b1-fixed-viscosity-fast-float64 | forcing | small | 30/30 | 0.149811 | 2.30959 | 0.000290779 | Every consumed forecast state |
| b1-fixed-viscosity-fast-float64 | steady | 32cubed | 10/10 | 6.71422 | 111.032 | 0.000673313 | Every consumed forecast state |
| b1-fixed-viscosity-fast-float64 | steady | small | 30/30 | 0.135542 | 2.13797 | 0.00372123 | Every consumed forecast state |
| b2-fixed-viscosity-fast-float32-map | forcing | 32cubed | 10/10 | 6.02128 | 95.3337 | 0.000397827 | Every consumed forecast state |
| b2-fixed-viscosity-fast-float32-map | forcing | small | 30/30 | 0.114345 | 1.68119 | 0.000290765 | Every consumed forecast state |
| b2-fixed-viscosity-fast-float32-map | steady | 32cubed | 10/10 | 5.17089 | 85.3472 | 0.000883088 | Every consumed forecast state |
| b2-fixed-viscosity-fast-float32-map | steady | small | 30/30 | 0.106919 | 1.67828 | 0.00340318 | Every consumed forecast state |
| b3-fixed-viscosity-fast-float64 | forcing | 32cubed | 10/10 | 4.37825 | 67.7111 | 0.000397827 | Every consumed forecast state |
| b3-fixed-viscosity-fast-float64 | forcing | small | 30/30 | 0.116147 | 1.68218 | 0.000290779 | Every consumed forecast state |
| b3-fixed-viscosity-fast-float64 | steady | 32cubed | 10/10 | 4.51111 | 74.3873 | 0.000834351 | Every consumed forecast state |
| b3-fixed-viscosity-fast-float64 | steady | small | 30/30 | 0.11035 | 1.73502 | 0.00353177 | Every consumed forecast state |
| b4-fixed-viscosity-fast-float64 | forcing | 32cubed | 10/10 | 4.38775 | 66.5563 | 0.000397827 | Every consumed forecast state |
| b4-fixed-viscosity-fast-float64 | forcing | small | 30/30 | 0.116914 | 1.66082 | 0.000290779 | Every consumed forecast state |
| b4-fixed-viscosity-fast-float64 | steady | 32cubed | 10/10 | 4.31818 | 71.196 | 0.000732322 | Every consumed forecast state |
| b4-fixed-viscosity-fast-float64 | steady | small | 30/30 | 0.0989777 | 1.56068 | 0.00336772 | Every consumed forecast state |
| native-pcg | forcing | 32cubed | 10/10 | 13.1141 | Unavailable | 7.73557e-06 | Native diagnostic checkpoints only |
| native-pcg | forcing | small | 30/30 | 0.229193 | Unavailable | 7.39099e-06 | Native diagnostic checkpoints only |
| native-pcg | steady | 32cubed | 10/10 | 13.1421 | Unavailable | 0.000103255 | Native diagnostic checkpoints only |
| native-pcg | steady | small | 30/30 | 0.224779 | Unavailable | 2.3345e-05 | Native diagnostic checkpoints only |
| spectral | forcing | 32cubed | 10/10 | 3.37412 | Unavailable | 3.06364e-14 | Native diagnostic checkpoints only |
| spectral | forcing | small | 30/30 | 0.136299 | Unavailable | 6.00241e-15 | Native diagnostic checkpoints only |
| spectral | steady | 32cubed | 10/10 | 3.34116 | Unavailable | 3.06364e-14 | Native diagnostic checkpoints only |
| spectral | steady | small | 30/30 | 0.128585 | Unavailable | 6.1756e-15 | Native diagnostic checkpoints only |

Loading/learned-cache preparation, whole-population warmup, native setup, RSS, latency tails and every discarded state remain separately reported. Cold amortization is an arithmetic sum of measured components and is labeled accordingly. A rejected or incomplete interval has no completed throughput or full-interval worst-error claim. Application costs are distinct from the original model-only2ms/state budget.

[Complete results and source evidence](results.json) · [CSV](results.csv)
