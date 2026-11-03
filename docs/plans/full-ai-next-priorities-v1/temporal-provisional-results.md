# Matched provisional-supervision continuations

Both continuations complete 4,000 additional updates from the same 10,000-update checkpoint and exact zero-update CPU restores. The data, sample stream, initial tensors, optimizer schedule, two-block feedback, inference architecture and target work match. Only the training-only auxiliary weight differs.

| Regime | Interval | Arm | Complete | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- | --- | --- |
| small | single_block | control | True | 0.1041834 / 0.4287571 / 0.005228947 / 0.009962654 | False |
| small | single_block | auxiliary | True | 0.1028878 / 0.4468193 / 0.005071767 / 0.01001403 | False |
| small | own_history12 | control | True | 0.001015795 / 0.003886272 / 0.001016003 / 0.004814671 | False |
| small | own_history12 | auxiliary | True | 0.001078987 / 0.003953838 / 0.0009687678 / 0.004827013 | False |
| small | own_history48 | control | True | 0.02158643 / 0.0846948 / 0.02168388 / 0.04521673 | False |
| small | own_history48 | auxiliary | True | 0.0194306 / 0.08546634 / 0.02058777 / 0.04501733 | False |
| small | own_history240 | control | True | 1.339008 / 4.971978 / 0.3938186 / 0.8775298 | False |
| small | own_history240 | auxiliary | True | 1.674119 / 6.787298 / 0.442789 / 0.8740367 | False |
| 32cubed | single_block | control | True | 0.05139007 / 0.1771139 / 0.0008660229 / 0.00316353 | False |
| 32cubed | single_block | auxiliary | True | 0.05191939 / 0.1823018 / 0.0008302117 / 0.003150776 | False |
| 32cubed | own_history12 | control | True | 0.0004464677 / 0.001283988 / 0.0005606147 / 0.0007460778 | True |
| 32cubed | own_history12 | auxiliary | True | 0.0004788321 / 0.001290762 / 0.0005209233 / 0.0007457063 | True |
| 32cubed | own_history48 | control | True | 0.005066699 / 0.01775588 / 0.003361804 / 0.009507395 | False |
| 32cubed | own_history48 | auxiliary | True | 0.005375134 / 0.01772018 / 0.003144105 / 0.009560855 | False |
| 32cubed | own_history240 | control | True | 0.2109713 / 0.7345548 / 0.07115558 / 0.3475594 | False |
| 32cubed | own_history240 | auxiliary | True | 0.2028628 / 0.8030642 / 0.06781385 / 0.3448809 | False |

Every intermediate output and all eight development scenes are retained, including complete 12/48/240-step or explicitly incomplete own-history outcomes. Finite-prefix maxima cannot establish full-interval accuracy. No new CPU timing is claimed; the eight reserved scenes remain unopened.

[Source-bound results and fitting diagnostics](temporal-provisional-results.json) · [Frozen comparison](temporal-provisional-registration-01.json) · [Transport diagnosis](temporal-coupled-transport-results.md)
