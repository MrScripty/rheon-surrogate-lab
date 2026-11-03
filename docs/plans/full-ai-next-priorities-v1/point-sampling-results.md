# Learned point-sampling component

The 972-coefficient layer completes 1,000 ordinary managed CUDA SGD updates from zero. Its fresh CPU restore matches all five saved artifacts byte for byte. The worst error on 16 held-out synthetic patterns is 7.658e-16 in float64 and 2.493e-08 after float32 publication.

The restored float32 layer passes all 24 independent native point checks across four domain shapes, both spatial densities and zero/small/unit random fields. Worst native relative error is 2.513e-08; zero input remains exact.

No analytical sampling coefficients were assigned. This layer predicts negative point velocities through a learned convolution. It is an initialization component for the displacement model, with no complete-state accuracy, CPU latency or rollout claim.

The first evaluation reached a reporting-path error after native comparisons. Its failed batch and log remain preserved. Supplying an absolute input path completed the same evaluator and protocol; no weights changed and no fit was resubmitted. Both batch controllers reaped their children.

[Evidence and source hashes](point-sampling-results.json) · [Calibration registration](point-sampling-registration-01.json) · [Full-state representation qualification](displacement-transport-results.md)
