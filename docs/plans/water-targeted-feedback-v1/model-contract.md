# Model contract

The [retained one-channel contract](../water-coverage-feedback-v1/model-contract.md)
continues to own inference, physical settings and restrictions. Architecture and
float32 neural precision are unchanged: 370,016 parameters, one [N,1,Z,Y,X]
physically scaled residual input, scalar cubic h/rho/dt metadata, no coordinate or
voxel-size feature maps, gauge reconstruction and existing float64 pressure base.
Each CPU solve checks stopping after every actual network call, up to 32; two
training corrections do not mean two inference calls. Actual force enters the
native predictor/RHS every step; native history warm-up remains 15 steps.

The v7 change only adds supplemental training selection and its evidence/accounting.
Both paired recipes preserve the original stream; actual configs, managed Runs,
checkpoint/normalization hashes, file sizes and CPU restore proofs live in the
sealed trial records. New CPU-teacher preparation is future data, not current-fit
input or AI inference. The native application adapter and GPU inference are
unchanged; shape admission does not establish every shape's accuracy.
