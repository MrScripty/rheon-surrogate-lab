# Baseline inventory and executor handoff

Authenticated GitHub search+get verified exact public slug
`MrScripty/rheon-surrogate-lab`, repository ID1414801167, main
`e8a90bfef6b2955c502008ccc62443e29dd9f2ec`. It is distinct from `MrScripty/Rheon`.
Cloud HTTPS clone succeeds; no prior lab checkout existed. Saved Tuldok work
branch is clean; fetched origin/main `d856a98` without altering its working tree.
No AGENTS.md or .agents skills exist in the lab. Consulted Coding-Standards at
`188beda1fa477d21d576c233dd8c7f4c4c267d23`: Core/Router, planning, implementation,
verification, commit, build, contracts, ownership, dependencies, licensing,
performance, applicable Rust profiles. Its authoring skill excludes ordinary
application edits.

## Existing experiments

Original pressure-only dataset:8×7×6 sealed grid, spacing[.25,.3,.4]m,
density1.25kg/m³, dt.03125s, train white-noise/polynomial, validation blobs,
test oscillatory; disjoint seeds+families. Native generation/verifier pinned
Rheon34235d33. CNN failed heldout physics; frozen global335-gain experiment
and final16-seed pressure evaluation remain historical, with analytic baseline.

Current trajectory target:32³ sealed uniform unit box, spacing1/32m,
density1.25, dt1/60s; imposed three-component forcing/passive tracer;
no viscosity, obstacles, free surface or moving boundary. Rheon trajectory
source2209012b; parallel native remains sibling-path dependent historically.
Train100..111,8frames each=96 labeled RHS/pressure pairs[96,32768]float64;
validation500/501,60frames; tests900..903,120frames, already exposed.
Full-spectrum cosine basis with32767 positive fitted modal responses,
closed-form supervised least squares, zero optimizer steps, no analytic inverse
at inference, subtract gauge cell. Checkpoint
`25533a2fbaa5ab0f5fb574dc6307a2b8d6a686deb7db867d0d03338d4d70d5a9`
is tracked as experiments/trajectory-v2/model.pt. Training NPZ is untracked and
must regenerate; historical SHA3ca87734…b539184. No durable restored numeric
prediction arrays are committed. Historical result/verification receipts and
viewport screenshot are tracked; NPZ/viewer fields are ignored runs artifacts.

CPU historical control:8 native threads for all methods,8 Torch threads,
interop1,3balanced repeats×4seeds×120frames×3methods=4320 timed frames.
Median15.21s Jacobi,9.39s SGS,1.86s learned;8.20x/5.06x.
Resident-model wall includes native startup/teardown, full simulation, inference,
pipes/transfer, gates/reply parsing. Excludes imports/model setup/warmup, full-field
capture and rendering. Independent replay velocity relativeL2 max9.0048e-8.
Original runtime Torch2.14.0+cu132,Python3.14.7,NumPy2.5.3; CPU model tensors,
CUDA hidden. New cloud runs must record their own runtime and binary identities.

## Smallest generic contract

Refreshed Tuldok main already supports project schema1 (`data`, `recipes`,
`evidence`, `requirements`), `external_recipe_v1` structured input/execute/output,
`captured_cpu_python_v1`, native source/policy tokens, run-artifact input custody,
progress events and result summaries. Initial stale docs understated this.
No new Tuldok schema or physics registry is required for the lab project.
Use fixed lab-owned config as captured input, immutable checkpoint/freezes and
measured build receipt references; named artifact outputs are hashed by Runs.
Whole trajectory membership and RHS/pressure interpretation stay lab-owned.
The parent and paired worker own real catalog/runtime/grant/Run/UI admission.
Future imported training data can use existing scientific_field_sequence_v1;
no proposed field-name special cases in Tuldok. This first parity slice restores
original weights and regenerates original supervised labels for clean training.
