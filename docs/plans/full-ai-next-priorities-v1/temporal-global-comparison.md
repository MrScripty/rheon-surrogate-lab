# Shared global temporal capacity comparison

The compact K=1,2,3,4 models and the controlled loss follow-up all miss the
original accuracy objective. This comparison changes the neural representation
while preserving the original training inputs, objective and sampled work.
The [registration](temporal-global-registration.json) predates Project capture
and fitting. No reserved temporal scene has been opened.

The candidate uses a width-12 shared spatial encoder and two global blocks.
Each block transforms learned spatial features into up to eight cosine modes
per axis, applies learned channel mixing, then returns features to the spatial
network. A learned head conditions the mixing on grid shape, spacing and mode
coordinates. Both blocks feed all five output fields. One joint evaluation
returns every declared future state. There is no analytical inverse, pressure
solve, numerical projection or reference endpoint in candidate inference.

| Configuration | Parameters |
| --- | ---: |
| K=1 | 170,728 |
| K=2 | 170,874 |
| K=3 | 171,050 |
| K=4 | 171,256 |

First fit K=1 for 10,000 updates: 2,000 single-block updates followed by 8,000
updates across two blocks of predicted history. Retain the original log loss,
learning-rate schedule, sampling seed, 16 training scenes, normalization and
feedback encoding. This matches sampled work and optimizer updates, not FLOPs.
The new model starts with the same persistence output function as the original
control; its larger parameter initialization is necessarily different.

CPU/CUDA qualification covers every requested K and all four grid groups,
including odd rectangles and 32³. Complete-basis reconstruction and orthogonality
are checked independently. A separate nonzero probe exercises every trainable
parameter through two physical loss blocks, verifies repeated gradients bitwise,
and compares CPU/CUDA outputs. Those diagnostic probe weights are never used to
initialize fitting. The largest CPU/CUDA probe difference was 4.77e-7.

The untrained full-output CPU profile is feasibility evidence only. At 16³,
K=2/3/4 blocks take approximately 7.21/7.50/7.68 ms. At 9×11×11 they take
3.66/3.82/3.95 ms. This demonstrates why grid populations must remain separate;
it does not establish any accuracy or trained-model latency claim.

After the managed CUDA fit, require an actual exact CPU restore and audit
sampling, processed cells, normalization, static inputs, objective and training
loop against the original K1 control. Evaluate all development windows and
240-step own-history rollouts using the unchanged qualified evaluator. Diagnose
cold and warm training windows separately. Improve fitting before replicating
bundled models or opening the reserved scenes. The original K=2,3,4 accuracy and
latency objectives remain unchanged.

Evidence lives under `runs/full-ai-next-priorities-v1/`: the four
`temporal-global-*qualification-01.json` records, `temporal-global-feasibility-01.json`,
`temporal-global-batch-01`, `b1-global-project-01` and `b1-global-pair-01`.
