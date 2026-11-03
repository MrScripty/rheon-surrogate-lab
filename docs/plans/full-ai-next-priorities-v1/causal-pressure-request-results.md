# Causal request component: first fit and diagnosis

The first 3,936-parameter causal component completes 10,000 managed CUDA updates and an exact independent CPU restore, but fails accuracy on training and development scenes. Native-file inference matches its managed CPU probe bit for bit, and the sampled stream matches registration exactly.

| Population | Queries | Median pressure error | Worst pressure error | Worst equation error |
| --- | ---: | ---: | ---: | ---: |
| train small | 48 | 0.16575 | 2.71848 | 1.81098 |
| train 32cubed | 16 | 0.578549 | 4.76719 | 6.42101 |
| development small | 1440 | 0.627802 | 5.21602 | 4.4042 |
| development 32cubed | 480 | 1.59944 | 6.93324 | 10.0752 |

The development evaluation covers all starts 0–239 on all eight scenes, using reference current velocity and known forcing; future RHS/pressure files are opened only after prediction. This is a pressure-only component, not autonomous full-state simulation. No reserved scene is used.

The sampled branch diagnosis shows a large spurious learned velocity-linear contribution. On warm development queries its median norm is approximately 0.745 times the true request norm. The learned nonlinear branch is approximately 0.282, versus approximately 0.177 for the actual nonlinear/viscous residual after the exact linear response. At startup, the nonlinear branch adds approximately 1% despite zero current velocity; its forcing inputs permit that unphysical term.

Numerical force-only/current-plus-force controls are not adequate candidates: worst sampled development request error is 1.067 on small grids and 0.738 on 32³. The nonlinear contribution matters. Replacing only the learned linear term improves some errors but leaves the already fitted nonlinear compensation. The true-future-request oracle stays accurate and is explicitly excluded from candidate frontiers.

The next registered fit learns a linear flux layer on native-qualified synthetic labels, freezes it, and then learns a velocity-only nonlinear residual with a more expressive decoder. The added 1,000 SGD calibration updates are counted separately from the matched 10,000 native-data Adam updates. Inference remains neural; no explicit flux teacher or analytical pressure inverse enters deployment.

[Managed training, parity and evaluation evidence](causal-pressure-request-training-result.json) · [Factored follow-up registration](factored-pressure-request-registration.json)
