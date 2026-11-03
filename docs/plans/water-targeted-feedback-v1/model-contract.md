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


The retained learned control stays 17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8.
The targeted diagnostic checkpoint is 5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe.
It has the unchanged 1,485,452-byte weights archive/370,016 parameters. All 50
required development and eight fresh final simulations qualify, while worst
pressure differences prevent promotion. The 127×5×5 exploratory failure remains.
Full identities and managed custody are in the trial/decision records.
