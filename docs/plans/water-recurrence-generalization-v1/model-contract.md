# Qualified laboratory checkpoint

The current generalization reference is cohort-two-shape-feedback, weights
`17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8`;1,485,452 bytes and 370,016 float32 trainable parameters.
Source capture: ff9ae119e7c28f50a8f69d4e3c9451b50fc023cf. Configuration:
experiments/water-recurrence-generalization-v1/configs/cohort-two-shape-feedback.json.

CUDA Run: 76090fe64e344e708b6080cfe9e18d20. CPU restore:
2331c80af17c4e4dac476127b8054c8c. Managed raw validation:
554b87f286114ecd804bea03fda52185. All complete/drain and reopen/redownload.
Local project/restore: runs/water-recurrence-generalization-v1/
cohort-two-shape-feedback/project and managed-full/stage-01/artifacts respectively.

The [preceding full interface contract](../water-multigrid-learning-v1/model-contract.md)
continues to own unchanged input, reconstruction and physical restrictions.
Normalization remains f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1.
Input is one float32 tensor [N,1,Z,Y,X], physically scaled residual divided by
the train-only 0.6197167573289654 Pa scale. There are no XYZ position, voxel-size,
density, timestep or previous-pressure feature maps. Scalar cubic spacing and
density remain required preprocessing metadata; actual previous pressure remains
required for reconstruction/history. Output scale remains22.254425460012698 Pa.
The existing physical operator and base accumulation retain float64 arithmetic;
learned parameters and corrections retain float32.

Training unrolls two shared evaluations. Qualified inference checks after each
individual evaluation, with at most 32 and the same equation/divergence gates.
Raw two-evaluation output remains unqualified; successful iterative inference
is a distinct result. Each physical step's residual includes that step's applied
force. Native warm-up is15 steps. No pressure cleanup is counted as AI success.

Fixed qualified fixture properties remain water density 1000 kg/m³, dynamic
viscosity 0.001 Pa·s, timestep 1/60s, origin 0, filled connected domains and sealed
impermeable free-slip walls. Grid extents follow counts times scalar cell size.
Native axis/cell/history/message limits and absence of obstacle/free-surface/
variable-density representations remain enforced by their existing owners.
Broader material/origin/timestep generalization is unqualified, even where a
model helper accepts positive scalar metadata. Native trajectories cap at 512
steps and 65,536 cells; accepted examples do not prove every admitted shape.

The final 16 novel simulations,8 development shapes,7 prior grids and 3 bounded
long trajectories pass. More detailed limits and regressions are in results.md.
This is a laboratory checkpoint; the existing Rheon application adapter requires
its separately owned one-channel/scaling/iteration/history/fallback migration.
No application deployment or GPU inference is introduced here.
