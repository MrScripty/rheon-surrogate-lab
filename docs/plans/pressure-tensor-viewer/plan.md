# Interactive pressure tensor page

Scope: explain the current pressure model's five-dimensional input through a
self-contained HTML page, with actual recorded inputs as an optional export.
Keep all model, training, native, historical result and external-app sources
unchanged. Root agent owns this serial UI/export slice on the existing private
variable-model branch; retain locally for review, with no push or merge.

The page owns selection state only: batch, channel and XYZ cell. Pointer cell
selection and native keyboard-operated indices reach the same numeric values.
Canvas displays are accompanied by shape, address, geometry and numeric table
text. Responsive layout, visible focus and labelled controls support desktop
and narrow browser use; no broad accessibility-conformance claim is made.

Standalone defaults are explicitly illustrative. A reusable preprocessing-only
exporter validates the qualified packet and its train-only normalization before
building exact float32 model inputs. Generated field data stays ignored in
runs/. The viewer decodes its bounded input representation before replacing
state, excludes superseded asynchronous loads and preserves the current view
on errors. Output shape is explanatory, not a generated scientific prediction.

Apply the previously adopted Coding Standards 188beda frontend, accessibility,
verification and commit workflows. Verify the exporter against the actual saved
packet and a real Chromium browser: sample/channel switching, pointer and
keyboard indices, slice bounds, cubic geometry, exact exported values, invalid
input, narrow layout and offline opening. Record evidence before committing.

Status: complete. Seven real Chromium checks pass, including exact recorded
values, native keyboard/pointer selection, offline file loading and narrow layout.
The preprocessing export contains two frames of a qualified 9×7×5 parent:
shape [2,10,5,7,9], 6,300 exact float32 values, shared h=0.125 m.
The page is open over loopback with this recorded export selected. No inference,
optimizer updates, capture regeneration or external-app changes occurred.
See completion.md and ignored runs/tensor-viewer/browser-checks.json for evidence.
