# Targeted coverage and intermediate feedback

The sequence preserves the original eight-grid analytical stream and adds one
explicit supplemental sample per update. A private balanced RNG selects either
reused primary tensors (extra-example control) or six targeted cyclic grids.
Both recipes use intermediate feedback, 5,000 updates and unchanged real data,
normalization, initialization, physics and CPU inference. Extra example count
matches; cell work differs and is reported. Combined tensor residency remains
bounded by 768 MiB; legacy experiments keep their old closed contracts.

The v7 configuration adds supplemental_feedback: source=primary requires only
source, weight and seed; source=grids adds the existing explicit bank fields.
Primary tensor/order/RNG evidence is unchanged. CUDA update history records the
supplemental stream separately; trial accounting includes both streams.

run.py delegates to the existing coverage sequence owner using configs/sequence.json.
finalize.py uses the retained audit with this experiment's process/source scope.
The canonical plan and executable reproduction records are under
../../docs/plans/water-targeted-feedback-v1/. No weights/raw datasets in Git.
