# Checkpoint contract

Experimental checkpoint: `4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2`.
The sealed trial record owns its managed Run identities, source witness,
normalization, parameter inventory and CPU restore evidence. Qualification and
promotion are separate decisions recorded in reports/decision.json.

The unchanged [one-channel contract](../water-recurrence-generalization-v1/model-contract.md)
uses float32 [N,1,Z,Y,X] physically scaled residual input, scalar cubic spacing
and physical metadata, float32 neural output, and gauge reconstruction with
float64 previous-pressure base/operator accumulation. There are 370,016 learned
parameters and 19 state arrays; weights.npz is 1,485,452 bytes. The train-only
normalization matches the retained control. No coordinate or voxel-size maps
are reintroduced.

Two shared training corrections do not fix inference to two calls. Each CPU
inference correction uses a freshly computed residual, with stopping checked
after each actual network call, up to 32. Native warm-up remains 15 steps.
Water, sealed/free-slip boundaries, density, viscosity, timestep, origin,
capacity and history restrictions remain those documented in the linked
contract. No native pressure refinement counts as learned-solver success.

This phase changes the experimental checkpoint and evaluation evidence.
Rheon's deployed adapter is unchanged; application migration and GPU inference
qualification remain separate work.
