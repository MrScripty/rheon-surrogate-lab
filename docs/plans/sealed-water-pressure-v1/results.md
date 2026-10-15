# Varied sealed-water training results

The model completed 2,000 CUDA updates on 256 different training simulations,
with 64 validation and 64 independent reserved test simulations. It learned a
closer pressure approximation, but **passed 0/128 reserved pressure checks**.
All 128 conventional fallback solves qualified. This checkpoint cannot replace
Rheon's pressure solver under the existing accuracy and physical gates.

[Acceptance evidence](acceptance.json), [scene coverage](reports/population.json),
[native evaluation](reports/native-evaluation.json),
[previous-run comparison](reports/comparison.json) and
[normal Tuldok dataset references](reports/collection.json) retain the measurements.
Complete local material, raw arrays, model checkpoints, managed Runs and repeated
downloads remain under `runs/sealed-water-v1-3` and in the existing Tuldok catalog.

| Property | Measured result |
| --- | --- |
| Scenes | 256 train / 64 validation / 64 reserved test; 384 distinct physical setups, force centers and seeds |
| Grids | 339 distinct shapes; 5–16 cells per axis; 8 or 16 cells/m, always cubic cells |
| Test grid coverage | 63 different test shapes; 48 absent from training |
| Physical lengths | 0.3125–2 m per axis, from integer counts × scalar spacing |
| Water | Density 1000 kg/m³; dynamic viscosity 0.001 Pa·s; engineering approximations |
| Boundaries | Filled sealed stationary free-slip boxes; no emitter, free surface or obstacles |
| Forcing | Distinct location, direction, amplitude, extent, rotation, temporal frequency, moving center and spatial fluctuations |
| Native evolution | 64 timesteps at 1/60 s per scene; all 24,576 timesteps qualified |
| Retained observations | Native steps 15 and 63; 512 training / 128 validation / 128 test snapshots |
| Model | Existing seven-channel CNN; 371,729 float32 parameters; unchanged architecture, physical operator and loss |
| Training | CUDA Adam, 2,000 updates, batch 2; 0.001 learning rate then 0.0001 from update 1,000 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU; actual CUDA batches, parameters, finite gradients and Adam moments |
| Precision | Float32 network; AMP and TF32 disabled; existing physical reconstruction/loss accumulation retained |
| Training elapsed | 80.580 s inside training function; managed runtime verification/startup excluded |
| Peak Torch GPU memory | 32,808,448 bytes allocated; 58,720,256 reserved |
| Validation loss | 55.8941818 → 0.00354147; validation parents only |
| CPU restore | Fresh managed Run, 3.063 s, zero updates, CUDA uninitialized; exact prediction inventory parity for 24 train/validation witnesses |
| Reserved inference | Fresh managed CPU Run, 3.038 s, zero updates, CUDA uninitialized; all 128 reserved snapshots |
| Artifacts | 12 train / 12 restore / 3 test; completed/drained Runs reopened and every whole artifact redownloaded with exact bytes/hashes |
| Accuracy | 0/128 under strict native gates; 0/128 under current 1e-6 float32/physical profile |
| Test equation residual | Minimum 0.316%, median 4.367%, maximum 25.018% relative L2 residual |
| Fallback | 128/128 qualified; 16,950 learned-plus-fallback iterations versus 12,854 full-budget temporal iterations |
| Existing Tuldok collection | 384 draft scientific field records; independently grouped whole scenes, with no invented human review/release |

The median residual is an error in the pressure equation, not a claim that every
pressure value has a 4.367% error. The target is 1e-6 (0.0001%), with all other
physical gates retained. Learned bounded warm attempts used 4,096 iterations
before falling back in every case; these iteration counts establish no speedup.

## Comparison with earlier attempts

| Same seven-channel model family | Updates | Median equation residual | Qualified proposals |
| --- | ---: | ---: | ---: |
| Previous managed CUDA portability qualification | 20 | 87.110% | 0/12 |
| Previous seven-channel CPU training | 200 | 38.841% | 0/12 |
| Current varied-water CUDA baseline | 2,000 | 4.367% | 0/128 |

These are descriptive measurements from different populations and budgets. The
old twelve development snapshots were already observed; this run uses independent
reserved parents. Material density, forcing, grids, retained times and train-only
scales differ. They do not establish a controlled 8.9× or 19.9× accuracy gain.
The original successful fixed-domain learned solver is a separate model and
experiment; its results remain unchanged. No architecture, loss or normalization
ablation was mixed into this expanded-data baseline. In particular, the existing
constant-unit density normalization policy was retained.

## Reproducibility and ownership

Training source: `11408e93995a9f3c3f59e171ccd0889906531027`.
Water diagnostics source: `7bf9f0f5f069063140f67f005da39c7a516c416d`.
Configuration is reusable; historical hashes/sources/results were preserved.
The reservation hash is
`f5175321d24f4cb897b854ec4b60841d5c907613859cc4e2a60dfbb4720b59fa`.
Final checkpoint weights are
`aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d`.
Normalization was fitted only to training parents. Reserved arrays never entered
normalization, optimization or checkpoint selection. The final update was the
predeclared candidate, before reserved inference; no held-out tuning or retraining
followed its results. CUDA/CPU pressure predictions differed by at most
0.000365 Pa across the 24 portability witnesses; fresh CPU predictions repeat
exactly. Cross-device bit identity is not claimed.

The normal catalog `a7da4a8de53541a2a0e2aad08005bb71` owns execution project
`bb9315b6e176491bb6c1eb29a5e9dc89`:

- CUDA training: `fa641802967e4d339ad2d0d82dd54b7c`.
- CPU restore: `630e12fa31d348e7af584ce125a819bc`.
- Reserved CPU inference: `4ee8b8c359444202bccf3d67b320e622`.

One-frame fixture and four-scene pilot passed before full generation. Five Python
contract tests, two native capture tests and seven native diagnostic tests pass;
both release builds, formatting and Clippy with warnings denied pass. The retained
snapshots' maximum independent relative reference residual is 9.9893e-9, actual
float32 divergence 7.1526e-7 s⁻¹, net boundary flux zero and Courant 0.3974.
The installed Torch runtime was measured and checked by Tuldok before/after each
managed Run; the host OS, driver and GPU remain external trusted-local dependencies.

Two concrete setup failures were retained and fixed without relaxing physics.
An infinite Gaussian tail produced subnormal velocities refused by viscosity;
a native regression demonstrated the failure, then compact smooth support fixed
the original scene. Failed captures remain in `runs/sealed-water-v1-1`; preformat
qualification remains in `runs/sealed-water-v1-2`. The historical diagnostic also
refused water at its original 1.25 kg/m³ admission guard. A separate measured
water-material evaluator preserves its pressure arithmetic, thresholds and
fallback policy, with a red-to-green material regression. The exact failed first
request replayed successfully; the model, checkpoint and reserved predictions
were unchanged. Diagnostic failure/replay evidence is retained.

## What the current interface shows

Tuldok contains the actual field records and three normal managed Runs. Real native captures and frozen model pressure proposals were published to the
Rheon egui observer file. A persistent observer window was not confirmed in the
final process check; this is not a live AI comparison qualification.
These results are matched teacher-state pressure checks, not autonomous rollouts.
Small grids with controlled fluctuating forcing do not establish resolved physical
turbulence, general arbitrary-resolution accuracy or free-surface fluid behavior.

The user's next requested capability is two fresh evolving simulations side by
side: Rheon's conventional solver versus the loaded AI checkpoint. Desktop
currently compares Jacobi PCG against SGS PCG only. Its live-project window is
an external observer. The library has a pressure-provider API, but the desktop
has no adapter for this checkpoint, and its current provider step excludes
viscosity. A fresh AI comparison therefore needs actual integration work and
explicit behavior when AI pressure fails qualification; recorded field switching
must not be presented as that feature.
