# Model and checkpoint contract

Both experiments retain the 370,016-parameter float32 network and its 19 state
arrays. Each weights.npz is 1,485,452 bytes. The sealed trial records in reports/
own checkpoint hashes, managed Run identities, parameter inventory, source
witnesses and CPU restoration evidence. The final decision owns promotion;
completion of training alone does not qualify a checkpoint.

The unchanged [one-channel contract](../water-recurrence-generalization-v1/model-contract.md)
uses float32 [N,1,Z,Y,X] input: physical pressure residual (rho/h times the native
residual), divided by the train-only 0.6197167573289654 Pa scale. No XYZ position,
voxel-size, density, timestep or previous-pressure feature maps are passed.
Scalar cubic spacing and density remain preprocessing metadata. Actual previous
pressure remains required for reconstruction and temporal history. Float32 neural
output is multiplied by 22.254425460012698 Pa, converted to float64, gauged at cell
000 and added to the previous pressure. The physical operator and pressure
accumulation retain their existing float64 arithmetic; neural precision stays
float32. The original normalization is reused rather than fitted on evaluation.

The selected correction network remains linear (biases and nonlinear activations
are removed by the existing linear variant). Late-residual training would target
residual directions/error modes rather than cure an assumed amplitude nonlinearity.
Training unrolls two shared network evaluations. The coverage experiment adds
three mixed odd grids to the analytical examples; the feedback experiment adds
an intermediate residual penalty without changing the grids. Neither changes
inference. CPU inference recomputes the residual and checks stopping after each
actual network evaluation, up to 32. Forces enter the native predictor and RHS at
every physical step. Native warm-up remains 15 steps. Cold-start inference,
native pressure cleanup and fallback are not credited as AI success.

Fixtures remain filled, connected water domains: density 1000 kg/m³, dynamic
viscosity 0.001 Pa·s, timestep 1/60 s, origin zero, sealed impermeable free-slip
walls, no obstacles or free surface. These are the qualified physical settings;
positive metadata accepted by a helper does not establish broader qualification.
Cubic spacing is enforced, with physical extents equal to axis counts times h.
Native trajectory axes are bounded to 5–128, total cells to 65,536, and duration
to 512 steps. Admission of a grid shape does not guarantee learned accuracy.
The existing history/message and physical/equation gates remain unchanged.

This is laboratory qualification. Rheon's deployed adapter is unchanged; its
one-channel/scaling/iteration/history migration and GPU inference qualification
remain separate work. No new weights or raw simulation arrays enter Git.
