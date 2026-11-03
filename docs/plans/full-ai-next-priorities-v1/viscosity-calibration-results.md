# Managed neural viscosity calibration

The 243-coefficient neural layer completes 1,000 managed CUDA SGD updates from zero using exact native strain-update labels. The fresh CPU restore matches all five saved artifacts byte for byte.

Worst held-out synthetic error is 5.75e-16 in float64 and 3.878e-18 after float32 publication. All 24 native geometry/density checks pass: worst operator error is 7.241e-19; worst physical-update error is 1.567e-10.

No analytical stencil coefficients were assigned. Numerical viscosity remains a training/evaluation teacher; candidate inference uses the learned convolution. These are component results. The composed full-state model must separately pass every output, own240 and actual CPU timing.

[Receipts, weights and source evidence](viscosity-calibration-results.json) · [Frozen registration](viscosity-calibration-registration-01.json)
