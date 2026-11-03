# Retained model and experimental checkpoint

The qualified laboratory control remains shape-two-reference, weights
17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8.
The frozen experimental candidate is shape-one-longer, weights
8cd2a0d4a694f44f393ebc8381a407e4195fd9361f1e115060104bd0df60ac8f.
It is not promoted after failing one final simulation. Exact source/project/Run
identities are in reports/final-selection.json and the sealed trial records.

Both use the existing [one-channel contract](../water-recurrence-generalization-v1/model-contract.md):
float32[N,1,Z,Y,X] physically scaled residual, scalar cubic spacing/physics
metadata, float32 learned correction, gauge reconstruction and float64
base/operator accumulation. Train-only normalization remains
f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1.
No coordinate, voxel-size, density or timestep feature maps are added.

The architecture,370,016 float32 parameters and 1,485,452-byte weights file are
unchanged. One training correction does not mean one inference call: CPU
inference checks each actual network evaluation and stops adaptively, capped 32.
Fifteen native warm-up steps and previous-pressure history remain required.
Lab failures retain their native verdicts; no native cleanup counts as AI success.

Water density1000 kg/m³, viscosity0.001 Pa·s, timestep1/60 s, zero origin,
uniform cubic cells and sealed free-slip boundaries remain fixed. No obstacles,
free surface, inlet/outlet or variable-density qualification is added. Native
admission allows5–128 cells/axis, at most65,536 cells,512 steps and 96 MiB history;
these bounds are enforced admission limits, not learned-accuracy guarantees.
Both checkpoints miss some127×5×5 tests; the candidate also fails16×23×25.
Accepting a tensor shape does not establish qualified pressure on that shape.

Rheon's deployed adapter is unchanged; the lab checkpoint contract and application
migration remain separate. No GPU inference or new native-versus-AI latency claim.
