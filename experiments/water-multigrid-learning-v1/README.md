# Controlled multi-grid pressure experiments

The [plan](../../docs/plans/water-multigrid-learning-v1/plan.md) owns scope and
acceptance. Reuse the conditioning trainer's captured-source assembly and normal
Tuldok workflow. Historical `stability-modes-single-pass.json` is the single-grid
control. New `multigrid-smoke`, `multigrid-modes-single-pass` and conditional
`multigrid-modes-two-pass` configurations declare train-only analytical feedback.

Version 4 adds five explicit grids: 16³, 24³, 32³, 40³, and 24×16×12. At 16 cells
per metre, each voxel is a 1/16 m cube and physical extent follows slice count.
Five-grid shuffles give exactly equal exposure over each five updates. Each grid
has 32 patterns at four signed amplitudes; amplitude and normalization come only
from the real fitting frame. Generated labels use Neumann cosine eigenvalues,
independent of the implemented finite-volume stencil. No extra CFD simulation
or observation of the reserved population is implied.

The resident tensor cache is projected and checked against a 768 MiB limit;
network parameters, gradients and CUDA workspace are separate measured memory.
Historical schema-v3 sampling and tensor values retain their original behavior.
The 10-update smoke covers all five grids before a 5,000-update fit.

`learned_iteration.py --single-evaluation --proposals 32 --adaptive-profile ...`
checks the same equation/divergence targets after every shared network evaluation
for either a one-pass-trained or four-pass-trained checkpoint. Without that
option the original proposal width applies. The captured recurrence implementation
is reused without mutating checkpoint metadata. This is iterative CPU inference,
not a new one-pass success or a changed Rheon deployment contract.

`qualify_bank.py --config CONFIG --packet FIT_VIEW --legacy-source CAPTURE/analytical.py
--legacy-config ORIGINAL_CONFIG --output FRESH.json` checks every authored label
and exact parity with the retained single-grid implementation without optimizing.

`study.py --config STUDY.json --output FRESH --workers N` runs native controls
before candidate episodes. The JSON names restored checkpoints and scenes rather
than embedding historical hashes in executable logic. Use up to available cores
minus one for independent numerical jobs. Each worker owns a fixed core and one
Torch/native thread. Use `--workers 1` on an otherwise idle CPU for isolated,
alternating repeated timings. Native/learned retain 1e-8/1e-6 equation profiles.
