# Pressure tensor visualization

The self-contained viewer/tensor.html explains `[batch, channel, Z, Y, X]` with
sample/channel selection, an isometric physical box, an XY heatmap, Z slicing,
numeric XYZ controls and all ten values at a selected voxel. Exact selected
float32 values accompany the rounded table. Output shape is explanatory only.
Rectangular grids preserve cubic cells; batch and channels are explicitly
nonspatial selectors. Default schematic data is clearly labelled illustrative.

scripts/export_pressure_tensor.py reuses the actual model preprocessing after
full packet validation and exact train-only normalization comparison. It exports
one to four same-parent frames into ignored JSON, without loading a network,
running inference or updating weights. Actual frames 0 and 23 of training parent
variable-train-510001 produced `[2,10,5,7,9]` with h=0.125 m and 6,300 numbers.
The exported float32 tensor SHA-256 is
`7f4c26eabb1e526e6cfbc02c04850b3fe72898198b6ed5ec29e4056aa9aa6d81`.

Real Chromium verification passes seven claims: rectangular cubic geometry;
sample/channel and pointer/keyboard selection; exact recorded values and flattened
offsets; invalid anisotropic input rejection without losing the current view;
390 px layout without horizontal overflow; direct offline HTML opening; and
native JSON file loading offline. No browser exceptions occurred. Supporting
files, screenshot and server receipt are in ignored runs/tensor-viewer/.
The local browser was opened with the actual export loaded; the server binds
only to loopback. The standalone file needs no server or external dependencies.

This is a UI/export commit, separate from model changes. The existing private
feat/variable-cubic-pressure-model branch is retained locally for continued review.
No push or merge. Model/training/native sources, weights, scientific results,
historical experiments, external app checkouts and managed Runs remain unchanged.
The exporter is reusable for other qualified packets and matching normalization;
viewer/TENSOR-README.md documents both offline and served workflows.
