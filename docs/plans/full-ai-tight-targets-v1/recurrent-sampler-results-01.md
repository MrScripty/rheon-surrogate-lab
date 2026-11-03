# Trained recurrent sampler: complete prototype results

The prototype is trained and independently tested. It is a promising complete-state accelerator, but **the active 1.2 ms / 1e-5 targets remain unmet**. No higher-density or 600-step phase has started.

It predicts pressure, three staggered velocity fields and tracer. Learned query gates and mixing drive fixed field-sampling layers, followed by prescribed forcing and learned viscosity, flux, modal pressure and tied momentum. The 40,017 learned coefficients include 48 newly fitted double transport parameters. Candidate inference invokes no native numerical advance or analytical pressure inverse.

One managed CUDA fit used 400 Adam and 40 LBFGS calls on 77,998 samples from 16 original training scenes. A separate managed CPU Run restored every artifact exactly and reproduced the complete transport probe bitwise. The final normalized training loss was about 2.99e-24. The restored full four-step implementation passed 64 complete-state comparisons against its tensor graph.

All costs below use one CPU thread, batch one, core 23. The trained version uses the qualified integer-address compilation optimization; its neural computation is unchanged.

## Independent complete-block cost at 32 cubed

| Steps per pass | Median full block ms | Median ms/step | Maximum observed ms/step |
| --- | ---: | ---: | ---: |
| 1 | 11.2527 | 11.2527 | 13.8695 |
| 2 | 21.7354 | 10.8677 | 13.5098 |
| 3 | 33.0645 | 11.0215 | 14.5731 |
| 4 | 43.9486 | 10.9871 | 13.5905 |

Construction and first warm calls are recorded separately. Complete warm calls include input validation, forcing, all neural stages, every intermediate output and internal state updates. This is an internally recurrent generator: increasing K gives little amortized speed improvement, and no parallel endpoint-decoding benefit is claimed.

## Independent physical accuracy

Sixteen new whole scenes were reserved before the fit. We froze the sole trained candidate and deployment sources before capturing their references; all existing shapes and densities of 8 and 16 cells per metre are represented. This confirms new scene parameters/seeds within supported shapes, not unseen-shape generalization.

| Interval | Domain group | Worst equation | Worst pressure | Worst velocity | Worst tracer |
| --- | --- | ---: | ---: | ---: | ---: |
| Short K1–4 blocks | small | 7.77108e-08 | 5.95965e-09 | 7.9819e-10 | 0 |
| Short K1–4 blocks | 32cubed | 8.92018e-07 | 5.34733e-09 | 3.78839e-09 | 0 |
| 240 own-history steps | small | 4.93118e-05 | 3.14064e-06 | 1.57021e-07 | 1.05448e-07 |
| 240 own-history steps | 32cubed | 9.05305e-05 | 6.01338e-07 | 1.45034e-07 | 5.28039e-08 |

Every intermediate field and actual own-history step was scored. All 320 fresh short-block states passed 1e-5. Through 240 steps, pressure/velocity/tracer stayed below 1e-5, while the equation criterion failed. That criterion compares predicted pressure with the RHS of the independently evolved native reference, so it includes accumulated history differences. It is not a measurement of the pressure branch in isolation.

## Matched complete application comparison

Both 32-cubed development scenes, both densities, steady/step 13 forcing-change scenarios, 48 accepted steps per episode, five balanced serial repeats. Four actual K generators and native PCG/spectral controls use the same CPU core and consumer interface. Setup is separate; timing includes model/solver work, state transfers, checks, event handling and final verification/close.

| Method | Scenario | Completed episodes | Median application ms/accepted step |
| --- | --- | ---: | ---: |
| sampler-k1 | steady | 10/10 | 11.8549 |
| sampler-k2 | steady | 10/10 | 11.8644 |
| sampler-k3 | steady | 10/10 | 11.8518 |
| sampler-k4 | steady | 10/10 | 11.6782 |
| native-pcg | steady | 10/10 | 225.9458 |
| spectral | steady | 10/10 | 58.1806 |
| sampler-k1 | forcing | 10/10 | 12.1014 |
| sampler-k2 | forcing | 10/10 | 12.8909 |
| sampler-k3 | forcing | 10/10 | 12.5521 |
| sampler-k4 | forcing | 10/10 | 12.7710 |
| native-pcg | forcing | 10/10 | 226.1170 |
| spectral | forcing | 10/10 | 58.6039 |

Physical application outputs were retained outside the clocks; exact repeated bytes are checked and accuracy is evaluated afterward. Candidate accuracy covers every consumed state, including forcing changes. Native-control accuracy retains its diagnostic-checkpoint scope. The neural consumer performs structural uploads with zero numerical advances; this is not native physical-acceptance certification. Original numerical control gates remain unchanged.

## Disposition

Retain this trained prototype as a useful research result. It improves the earlier complete-generator baseline substantially. The absolute 1.2 ms budget and own-history equation criterion still need work; these remain explicit failures rather than a reason to call the model useless. The user-prioritized prototype is complete before any different architecture is started.

[Machine-readable results and exact evidence](recurrent-sampler-summary-01.json) · [Completion scope](recurrent-sampler-completion-02.md) · [Practical-value triage policy](prototype-triage-policy-03.json)
