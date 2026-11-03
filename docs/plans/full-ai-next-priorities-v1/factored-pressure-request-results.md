# Factored causal pressure-request component

The follow-up completed its managed CUDA fit, exact zero-update CPU restore and all 1,984 causal evaluation windows. It improves on the first request model but still fails the 4e-3 worst-error criterion. This is a pressure-only component, with no velocity/tracer outputs or own-history claim. It is excluded from temporal candidate frontiers.

The 4,385-parameter model first learns a linear flux convolution from native-qualified synthetic labels using 1,000 float64 SGD updates. It freezes those 81 learned parameters, then trains its velocity-only nonlinear residual for 10,000 Adam updates on the same native window stream as the first component. Total training is 11,000 updates; this is not a claim of matched total compute. Candidate inference uses only learned convolutions. Analytical inversion appears only in the offline training loss.

All five restore artifacts match bitwise, and native-file inference matches the published CPU probe bitwise across all four grid groups and steps 0, 8 and 120. No current or future pressure, future RHS or analytical teacher enters inference.

| Split | Grid group | Time range | Queries | Median pressure error | Worst pressure error | Worst equation error | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| train | small | all | 48 | 0.00347993 | 0.0859804 | 0.210843 | False |
| train | small | startup | 12 | 2.80984e-05 | 9.11376e-05 | 0.00047902 | True |
| train | 32cubed | all | 16 | 0.00315544 | 0.0316634 | 0.0996399 | False |
| train | 32cubed | startup | 4 | 0.000146534 | 0.000352748 | 0.000567798 | True |
| development | small | all | 1440 | 0.0598037 | 0.646783 | 0.942247 | False |
| development | small | startup | 6 | 2.13615e-05 | 0.000100862 | 0.000289461 | True |
| development | 32cubed | all | 480 | 0.0326569 | 0.491848 | 0.707312 | False |
| development | 32cubed | startup | 2 | 0.000169987 | 0.000262408 | 0.000396735 | True |

Worst small-grid development pressure error improves from 5.2160 to 0.64678, and the median improves from 0.62780 to 0.059804. All cold-start groups pass, but later nonlinear evolution still fails; the development worst equation error is 0.94225. These are reference-history queries, so autonomous errors remain unmeasured.

The initial Project failed its runtime-manifest material cap before any Run was submitted. That failed attempt is retained. A new Project corrected the cap without changing numerical sources, data or optimizer settings; fresh CPU/CUDA qualifications preceded admission. See [admission recovery](factored-request-admission-recovery-01.json).

Proceed with the explicit endpoint/intermediate-decoder and retained-feature full-state comparisons. Preserve this component as a promising but inaccurate factorization; do not substitute its startup pass for all-horizon success.

[Full evidence](factored-pressure-request-results.json) · [First component failure and diagnosis](causal-pressure-request-results.md) · [Sharing comparison registration](temporal-sharing-registration-01.json)
