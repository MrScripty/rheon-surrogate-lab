# Continuous water simulation observation

The previous observer was a sparse dataset preview, not simulation playback.
Each domain began with the same centered Gaussian marker; it published only
native indices 15 and 63 before switching domains. CUDA optimizer updates then
reused saved inputs without advancing any liquid. This presentation could remain
on a nearly spherical blob throughout training and was misleading as a flow view.

The native transport was checked separately. An independent analytical test
translates a compact tracer exactly one cell in X, Y and Z under a known velocity.
All three translations pass. In the actual first training scene, the marker's
small displacement also broadly agrees with its local tracer-weighted velocity.
No advection algorithm or physical forcing was changed to manufacture motion.

| Native replay | Grid | Simulated endpoint | Centroid travel | Axis width change |
|---|---|---|---|---|
| Original train-0000 prefix | 16 × 16 × 16 | 1.067 s | 0.227 voxel | 0.3–6.2% |
| Same scene, continued | 16 × 16 × 16 | 4.267 s | 2.056 voxels | 26–60% |
| Train-0022, different force | 11 × 5 × 16 | 4.267 s | 4.244 voxels | 310–666% |

Both grids have 0.125 m cubic cells. The rectangular domain is 1.375 × 0.625 × 2 m.
Centroids compare the first and last recorded state, not time zero. Width is a
concentration-weighted standard deviation. The marker is transported with native
semi-Lagrangian interpolation, which is diffusive and not mass-conservative.
Especially on these coarse grids, width changes do not establish physical
stretching or resolved turbulence. The marker is not an AI input or pressure
acceptance target.

`tracer_check.py --require-visible` reproduces a failed visibility diagnostic for
the short prefix (exit 2), and passes the two longer timelines (exit 0). Its
half-cell travel / 10% width rule is a presentation heuristic, not a new physics
or accuracy threshold. The regression specifically rejects value changes that
only dim an otherwise stationary blob.

The versioned preview probe emits complete native states or a declared decimated
timeline. Every native step still passes the existing solver gates. Each replay
reproduces all 11 original fields at both original retained indices byte for byte.
Steps beyond the original 64 are explicitly observation-only continuations; the
frozen captures and all training recipes remain unchanged.

Rheon branch `feat/live-training-force-preview`, commit `fe9d586`, adds
`rheon-training-viewer`. It shares the ordinary desktop's native voxel geometry,
water palette, mesh and camera. Choose a scene explicitly; playback stays pinned
to its domain and uses physical timestamps. Water velocity, passive tracer and
pressure magnitude have distinct views, adjustable translucency and a fixed
timeline scale. The force widget and scrollable inspector expose the actual
moving center, radii, direction, acceleration, pulse, swirl, fluctuations and
frequency. Velocity arrows use actual corrected native values. The user can load
another timeline by path. This is recorded native playback, not live AI dynamics.

The opened local viewer contains both scenes:

```bash
runs/live-integration/native-target/debug/rheon-training-viewer \
  --session "$PWD/runs/water-learning-sequence-v1/preview-cube-long/session.json" \
  --session "$PWD/runs/water-learning-sequence-v1/preview-rect-long/session.json"
```

Both timelines were exported through Tuldok's existing scientific-field format,
imported into the existing catalog and downloaded byte-exactly. In Data Collection,
search **continuous native timeline**. Cube record:
`cd09d485f04b4197a953de7e1248ab09`; rectangular record:
`36c3245618cb4f5989f08aa1f1b40e13`. Each stores 64 recorded states, pressure,
tracer, velocity components, speed and exact force provenance. The project
snapshot is `179bf8ae04184187b1e5d1895a4faf23`. Tuldok retains immutable manifest
snapshots, so new references receive a new project ID; the catalog is unchanged.

All 768 newly captured training simulations also have whole-scene records in the
same catalog. Original records remain intact. New records preserve both original
saved states, water/force provenance, pressure, tracer and staggered velocities.
Reservation-qualified family identities distinguish scenes with coincident local
names across cohorts. All 768 content hashes were reopened; six representative
bundles and 24 pressure/tracer slices were verified exactly. Auxiliary validation
and reserved scenes were excluded from this publication.

Verification: 3 native probe tests, 5 viewer tests including the actual egui
frame-selection/rendering path, 47 ordinary desktop tests passed (one preexisting
ignored), and warning-free Rust clippy. Python view transport tests (4) and tracer
diagnostic tests (4) pass. Measured receipts and motion rows are in `reports`.
The independent diagnostics import fix is a separate lab commit (`ce8729c`).

The isolated Rheon worktree under `runs/water-learning-sequence-v1/Rheon` is
retained with its named branch protecting the commit. Owner: this task for Jeremy.
It serves the active viewer and Tuldok source references while awaiting review.
Keep subsequent synchronization in forward commits. After accepted integration
and reference migration, remove the clean task worktree with accepted commits
still protected. Existing dirty original/serving checkouts remain untouched.
