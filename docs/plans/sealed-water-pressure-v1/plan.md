# Varied sealed-water pressure training

Status: Active. Canonical operation: implement and execute the user's requested
water training experiment. Serial owner: root; no delegation. New lab branch
feat/sealed-water-pressure-training from a44584b. Keep all historical namespaces,
results, dirty original Tuldok and Rheon checkouts intact. No push or merge.

User outcome: train the existing seven-channel float32 pressure correction model
on genuinely varied simulations of sealed domains completely filled with water.
Domain proportions, force location/direction/strength, swirl and fluctuating
forcing vary. CUDA training and fresh CPU inference remain required. Observe
real progress/captures/results through the existing Tuldok catalog and Rheon egui.
User confirmed "turbidity" means turbulent motion and fluctuating forces.
Optical cloudiness is outside this pressure-training scope.

Write set: new experiments/sealed-water-v1 modules/configuration/tests, new
probes/sealed-water-v1 native capture, this plan/ledger/evidence; current README,
tuldok-project and scientific-work pointers. Reuse measured installed Torch
runtime and existing generic Tuldok workflow without runtime/model-architecture
changes. Reuse the qualified prepared Rheon dependency and independent sealed-MAC
operator/diagnostic oracle. Native viscosity is part of the reference generator.

Physical contract: constant rho=1000 kg/m3 and dynamic viscosity=0.001 Pa s,
explicit room-temperature engineering approximations, SI units. Incompressible,
filled, stationary impermeable free-slip box (Rheon's existing viscous boundary).
No mass emitter, free surface, pressure-storage/compressibility or optical model.
Per-axis integer cell counts determine physical lengths with scalar h=1/(cells/m).
Use 8 and 16 cells/m; never stretch individual voxels. Finite-support smoothly tapered localized directional
and rotational acceleration with time/spatial fluctuations; controlled forcing,
not a claim of resolved real-world turbulence. Same old physical acceptance gates.

Dataset decision: reproducible procedural reservation of 256 train, 64 validation
and 64 reserved-test parents. Distinct force setups and dimensions, not ID-only
copies. Each native trajectory advances 64 timesteps at 1/60 s and retains two
explicit snapshots (15 and 63). All native timesteps must pass reference gates;
retained snapshots receive independent pressure/RHS/divergence/gauge/wall/flux
checks. Reserve test parents before generation and exclude their arrays from
normalization, optimization and candidate selection. Test once on the frozen
final candidate. Modest grids and two snapshots bound the train/validation packet
to 32 MiB for the existing managed runtime; configuration preflights aggregate
cell/payload budgets. Native streams retain selected fields only. No large-runtime
extension or private catalog is needed. Disk free space is currently 7.4 GiB;
keep this slice below 1 GiB additional data with a 1 GiB reserve.

Model/training fidelity: unchanged 371729 float32 weights/architecture/physical
operator/loss from the qualified CUDA namespace. Dataset reader becomes variable
parent/frame/material aware. Preserve the existing normalization policy, including
constant unit scalar conditioning, for a clearly identified baseline. Fresh seed,
2000 CUDA Adam updates, batch 2, sparse checkpoints and explicit bounded rate
schedule; 600 s elapsed cap. Existing initial-checkpoint and typed artifact custody
proofs remain. Current 1e-6 float32 model-output threshold and all physical limits
stay fixed. Report learning failure honestly; successful execution is distinct
from accuracy/generalization. No architecture or normalization ablation in scope.

Composed-design review: applicable. Canonical owners remain Rheon native physics,
lab scene reservation/capture/packet/model experiment, Tuldok auth/material/Run/
artifact lifecycle, and observers for presentation only. Native producer validates
its bounded request against its compiled measured reservation, and inherited
process custodian owns it. Training consumes lossless bounded packets. Reuse
immutable source modules with explicit source capture instead of editing historical
code or replacing its hashes. New dataset/material contracts are versioned; old
recipes/results keep original meaning. No new standalone scheduler or catalog.

Acceptance: one-frame fixture first, then representative force/grid pilot;
force/material/topology and parent separation tests; native build/test/clippy;
all 384 scenes with measured variation and retained independent physical checks;
real catalog dataset references visible; managed CUDA training/CPU restore and
final reserved-test inference; complete artifact redownload/CPU parity; independent
native pressure acceptance per held-out case; durable honest results and commits.

Initial slice: implement reusable scene reservation and qualified native water
capture, then pass fixture/pilot before generating the full corpus.
Blockers: none. Replan explicitly if native water gates, bounded data transport or
managed execution cannot support the scope; retain failures and do not silently
relax physics or substitute CPU training. Adopted Coding Standards 188beda routes:
planning/discovery, implementation, code-design/contracts, verification and commit.


Capture verification update: initial fixture and four-scene pilot passed. First
full generation stopped after 276 scenes when the viscosity stage rejected 14
subnormal float32 values from the infinite Gaussian force tail. A one-step native
reproducer and regression demonstrated the exact failure. The generator now uses
a smooth taper to zero outside four force radii, explicitly versioned as
compact-gaussian-taper-v1. No viscosity, residual, divergence or flux gate changed.
The original failing scene now passes one and 64 timesteps. Native tests/clippy
pass. Failed material remains under runs/sealed-water-v1-1; regeneration uses
runs/sealed-water-v1-2. Exactly one next slice: complete full capture/packet
qualification, then freeze the managed training/CPU-restore/reserved-test project.

Qualification update: all 384 scenes / 768 snapshots passed in the second namespace.
Final source formatting requires rebuilding the measured producer and repeating
fixture/pilot/full capture in runs/sealed-water-v1-3; no physics changed. Five
Python contract tests pass. Exactly one next slice: freeze qualified sources and
assemble the three-stage CUDA training / CPU restore / reserved CPU test project.

Final material and managed stages: source frozen at 11408e9. Namespace
runs/sealed-water-v1-3 passed fixture, pilot and all 384 parents / 768 snapshots;
24,576 native timesteps qualified. Normal Tuldok collection imports contain
320 train/validation and 64 reserved-test field records in the existing catalog.
CUDA training completed 2000 updates; CPU restore and reserved inference both
completed and drained. All artifacts reopened/redownloaded with exact hashes.

Replan from concrete diagnostic evidence: the historical native evaluation binary
admits only rho=1.25, contrary to the initial assumption that its density argument
was generic. The unchanged captured first water request deterministically refused
"development density". A separate probes/sealed-water-evaluation-v1 and
experiments/sealed-water-v1/evaluation_native.py retain pressure arithmetic,
physical thresholds and fallback policy, but admit water at rho=1000. The new
material regression failed on the original guard, then all seven native tests,
release build, formatting and Clippy passed after the isolated guard correction.
The exact original request now accepts its reference and reports learned pressure
failure as data. Historical probe/source/binary/results are untouched. No training,
normalization, checkpoint selection or reserved predictions were rerun or changed.
Exactly one next slice: finish all 128 frozen reserved native diagnostics and
record measured accuracy, source/material/Run lineage and final visibility.
