# Learned displacement full-state results

The K2 model completes 10,000 managed updates, an exact CPU restore and every development output. Data, sampling and physical target work match the coupled control. Representation, auxiliary objective, initialization seed and learning-rate schedule differ, so the comparison does not isolate one cause.

| Regime | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- | --- |
| small | single_block | True | 0.005725132 / 0.01642461 / 0.000141967 / 1.950037e-05 | False |
| small | own_history12 | True | 0.0003520937 / 0.0009792085 / 0.0003678665 / 1.448828e-06 | True |
| small | own_history48 | True | 0.001564117 / 0.005108539 / 0.001441898 / 6.16859e-05 | False |
| small | own_history240 | True | 0.01289845 / 0.01991779 / 0.007352104 / 0.005567787 | False |
| 32cubed | single_block | True | 0.003956885 / 0.01182859 / 0.0003425625 / 9.037722e-07 | False |
| 32cubed | own_history12 | True | 0.0003217629 / 0.0008110122 / 0.0004581925 / 7.039517e-07 | True |
| 32cubed | own_history48 | True | 0.001318234 / 0.0039339 / 0.001643866 / 3.271071e-05 | True |
| 32cubed | own_history240 | True | 0.007685387 / 0.0208197 / 0.007450515 / 0.003695144 | False |

Incomplete trajectories retain their raw finite-prefix errors and failures, without a complete-interval pass. CPU latency is not established by this accuracy report. The eight reserved scenes remain unopened.

The unchanged evaluation core runs with the registered displacement loader. Its raw legacy loader stamp is preserved separately; actual loader, model source and restored weights are bound in the evidence.

[Results, controls and source evidence](temporal-displacement-results.json) · [Implemented model](temporal-displacement-plan.md) · [Original objectives](plan.md)
