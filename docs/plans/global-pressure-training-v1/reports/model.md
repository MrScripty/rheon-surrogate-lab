# Pre-training model and stencil qualification

Exact architecture:10channels, encoder widths8/16/24/32,48bottleneck with pooled
broadcast, decoder32/24/16/8 and scalar head. Eager CPU float32 model with
float64 full finite-volume loss and gauge projection;372,377parameters and
1,489,508rawweightbytes. No current label enters inputs. Native residual is
precomputed; Torch stencil is independently qualified for differentiable loss.

Actual source tests:20focused model/normalization/gradient/checkpoint/custody tests
pass; combined data/model/evaluator Python suite45passes/no skips in5.455s.
Independent numerical reviewer repeated frozen20tests and all384 native residual
comparisons. All-axis hand values/full gauge row, analytic gauge gradient, central
finite differences, nonzero head-weight gradient and distant/global broadcast
paths pass.16³/32³ shapes pass. Exact executed metrics/source hashes appear in
[the typed qualification report](model-qualification.json), captured in the
project and checked before training. The report is evidence reviewed by owners,
not a cryptographic proof that tests ran.

Train-only scales use12trainparents/288frames. Pressure RMS means previous
accepted reference pressure; residual RMS and original RHS RMS are separate.
Constant rho/dt use a unit scale of1. Sample order is fixed private PCG64 and
200×2 train-only rows; validation remains observational. Disposable arithmetic
fixtures change validation values/labels and actually evaluate them before paired
single Adam updates: normalization/order/weight bytes stay identical. These
fixtures do not constitute another scientific200-update campaign.

Dry timings use synthetic arithmetic only: forward16³.01321s,32³.06991s,
backward+one disposable Adam update.04260s; observed process peak463,429,632bytes.
These are admission support, not model training/runtime gains. Scientific train
and fresh trained-byte restore remain separate actual Run gates.
