# Pressure tensor explorer

Open [tensor.html](tensor.html) directly in a browser. It is a self-contained,
offline HTML page with deterministic, explicitly labelled illustrative inputs.
Select a batch entry, channel, Z slice and XY cell to inspect the five-dimensional
array. The current model has seven channels, with no voxel-size maps. Its single
cubic cell-size scalar remains physical metadata. Preserved ten-channel exports
still load with their original channels and an explicit historical label.
Numeric index controls provide the same cell selection as the canvas.
The physical box preserves cubic voxel sizes on rectangular and odd grids.

To inspect actual recorded inputs, export a qualified packet with its saved,
train-only normalization. This runs preprocessing only, with no neural inference
or optimizer updates. Run from the lab root, using a fresh output filename:

```sh
.venv/bin/python scripts/export_pressure_tensor.py \
  --packet runs/variable-pressure-v2-material-1/packet/dataset.npz \
  --manifest runs/variable-pressure-v2-material-1/packet/splits.json \
  --normalization runs/variable-pressure-v2-workspace-1/execution/stage-01/artifacts/normalization.json \
  --parent 1 --frames 0 23 \
  --output runs/tensor-viewer/seven-channel-inputs.json
```

Use **Load actual tensor** in the page to open that JSON. The export records exact
float32 input values, the original channel order, physical spacing, sample/frame
identity and measured packet/normalization/model/tensor hashes. Full packet
validation precedes export; normalization must match the packet's train-only
statistics. Select one to four frames from one parent, so each batch entry has
the same shape. Differently shaped training samples are processed separately.
Generated exports remain in ignored `runs/`, outside historical result files.

Alternatively serve the repository over loopback:

```sh
python3 -m http.server 8098 --bind 127.0.0.1
```

Open `http://127.0.0.1:8098/viewer/tensor.html?data=../runs/tensor-viewer/seven-channel-inputs.json`.
URL-loaded exports must share the page's origin. Local file loading also works
without a server. Invalid or oversized exports report an error while preserving
the current view. No external scripts, fonts, rendering libraries or services
are required. The output diagram explains the network's shape; it does not
present a computed pressure prediction or qualification result.

The browser smoke check uses Node's built-in WebSocket and a Chromium executable:

```sh
BROWSER=/absolute/path/to/chromium node scripts/test_tensor_viewer.cjs
```

It tests real pointer/keyboard interactions, recorded input values, malformed
input rejection, responsive geometry and direct-file operation. Generate the
example export above before running the recorded-input portion.

To export the preserved ten-channel model, add
`--model-profile variable-pressure-v1` and supply its original packet and
normalization. Both profiles retain their real channel counts in the JSON;
the viewer never pads or removes loaded tensor channels.
