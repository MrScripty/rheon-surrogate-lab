# Seven-channel model execution

The model receives `[N,7,Z,Y,X]`: pressure residual, previous pressure, physical
X/Y/Z coordinates, density and time step. No hx/hy/hz channel or equivalent
spacing-map intermediate is constructed. The first convolution has seven
inputs; the architecture has 371,729 parameters. One scalar `cell_size_m` per
sample remains available for physical coordinates and the unchanged pressure
operator. It is not supplied as a separate network conditioning feature.

The versioned model source was committed as 1db001b before managed recipe
freezing. All ten-channel variable-pressure-v1 sources, captures, weights and
results remain unchanged. The qualified native probe and physical gates are
also unchanged. New build receipts measure the v2 source selection; historical
hash constants were not replaced.

The six existing whole-parent captures were independently reverified before
training: 144 frames at two cubic cell densities and six grid shapes, with
96 train and 48 validation frames. All numeric array declarations and per-frame
hashes are identical to v1. The seven surviving preprocessed channels are also
byte-identical to the corresponding old channels. This is a controlled
development comparison using previously observed parents, not a new test set.

The normal Tuldok catalog a7da4a8de53541a2a0e2aad08005bb71 owns both completed,
drained Runs in execution project 497a1373e5654c75b9d067561f7d6298:

- Train 89e5e42c4563405d9afb3942907aea14: 200 updates, 18 artifacts;
  recipe elapsed 7.850 s, peak observed RSS 467,050,496 bytes.
- Restore cfe01464cb8d4444b546cc06b4fc304f: zero updates, 18 artifacts;
  all 12 predictions recomputed exactly and 16 shared artifacts byte-identical.

The existing captured CPU runtime was reused through its normal catalog owner.
Whole artifact downloads and repeated downloads pass SHA and byte checks.
No private catalog, new optimizer outside Tuldok, GPU execution or hidden retry.
Validation combined loss changed from 2.6075725674 initially to 0.1003719134
at update 200. The prior ten-channel run ended at 0.1197461126. These losses
do not establish strict pressure accuracy or a generalization improvement.

Actual restored predictions were evaluated on the same 12 first/final snapshots:
raw learned pressure qualified 0/12; all 12 conventional fallback solves passed.
Learned warm-start work totaled 1,691 iterations, including 384 wasted warm
iterations, versus 1,307 for the full-budget temporal baseline and 1,332 from
zero. There is no observed solver acceleration or autonomous rollout result.
Actual diagnostic pressure proposals and reference tracer fields were published
to the existing Rheon observer; they remain clearly labelled failed raw proposals.

The input viewer defaults to seven channels and loads the actual `[2,7,5,7,9]`
export. Historical ten-channel exports retain their actual channels and gain
an explicit historical label. The reusable exporter selects either preserved
model profile. Nine real Chromium checks pass, including exact float32 values,
no voxel-size controls, pointer/keyboard selection, narrow layout, offline file
loading and historical compatibility. Ten focused model checks cover variable
shapes, scalar geometry, unchanged physical loss, seeded-column removal,
zero-residual/gauge behavior and heterogeneous gradient accumulation.

Current source project d1135e6855c14aea95be802ee9f059e9 is registered in the
same Tuldok catalog. Its managed Runs live in the immutable execution project
listed above. Scientific evidence is retained in runs/variable-pressure-v2-*
and tensor-viewer artifacts in runs/tensor-viewer/. acceptance.json records
measured identifiers and hashes. Changes remain on the private feature branch
as separate model and UI commits; no push or merge was performed.
