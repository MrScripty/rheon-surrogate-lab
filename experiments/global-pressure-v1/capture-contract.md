# Development native capture contract

This namespace implements slice1 of the approved global pressure correction
contract. It creates no model or final-test data. The canonical reservation is
`docs/plans/global-pressure-correction-v1/reservation.json`, unchanged.

`capture.py build` is the only qualified build entrypoint. It validates every
prepared native file/mode against the lab-owned patch inventory, resolves Cargo
with one explicit prepared-source patch override, checks the actual resolved
native manifest path, and runs the build with the identical manifest/configuration
arguments and a bounded environment. Rust/Cargo 1.92.0 and the source/config/lock
closure are embedded in the binary's build binding; pre/post verification catches
source changes during compilation. `build.rs` checks that binding, verifies the
prepared receipt and watches its complete directory/file inventory. Direct Cargo
builds lack a qualified binding and are unsupported. A reused binary is admitted
only against its build receipt, embedded identity, current complete source closure,
metadata bytes and actual prepared receipt. Source qualification is distinct from
physics qualification. Optional parallel compilation is available for separate
native investigations, but the capture caller admits only the serial binary and
sets `RAYON_NUM_THREADS=1` explicitly. An experimental parallel producer also
requires that exact runtime variable; absence never implies a one-thread Rayon
pool.

The generation CLI admits only `--preset smoke` and `--split train`, `validation`
or `development` (both). Optional `--trajectory` selects exactly one eligible
reserved ID. `--fixture-frames` permits 1–24 frames only for one explicitly selected
TRAIN trajectory. Such a result has `fixture_complete` status and is diagnostic;
it does not satisfy the complete development dataset gate. Full, test, pulsed,
unreserved and invalid requests fail before output creation, including when
binary/source arguments do not exist. Rust's direct producer boundary also checks
its strict request fields/types, reserved identity and development eligibility.

The fixed configuration declares physical units and exact stage order: retain
previous accepted state; native velocity advection from that velocity; native
tracer advection from previous tracer and previous velocity; add forcing to
advected velocity; build original RHS; zero-initialized Jacobi-PCG via the new
sealed initial-pressure API; stage f32 correction; gate then commit. Tracer
advection occurs before forcing/projection and there is no later injection/clamp.
The centered Gaussian tracer uses amplitude 0.25 and standard deviation 0.08 times
minimum physical extent. Smooth forcing uses the approved normalized moving
center/direction, physical Gaussian standard deviation 0.1 times minimum extent,
fixed t=0 center/direction for steady regime and reservation amplitude. Gaussian
exponent is `-r²/(2σ²)`. Broadband acceleration is 0.025 times smooth amplitude;
its explicit wrapping integer mixer uses seed, frame, axis and x-fastest face
index. No ambient RNG or file order participates.

Each frame must satisfy the unchanged full original-RHS residual (including cell 0)
`||b-Ap||2 <= max(1e-12,1e-8*||b||2)`, predicted divergence<=1e-7/s and actual
corrected-f32 divergence<=1e-5/s. Pressure cell 0 is zero; wall-normal face values
are exactly zero. Net recomputed f64 volume flux satisfies the approved absolute/
relative gate. Courant<=0.5. No mean shift, failed-case deletion, tolerance adjustment
or reference fallback exists. SGS-PCG independently qualifies frame 0 of each
trajectory and reports pressure discrepancy; hand/dyadic anisotropic and analytic
cosine tests check public native and independent operator semantics. A scientific
failure stops the preset, retains stderr/partial payloads and requires parent review.

The producer emits a bounded length-prefixed JSON header then descriptor-ordered
raw payloads for initial state, each frame and completion. It owns no filesystem
publication. The caller writes every payload exclusively into a newly claimed
canonical direct child of `runs/global-pressure-v1-capture/`; symlinks, traversal,
existing directories and overwrite are refused. An owned process group and
600-second total generation watchdog bound blocked reads, EOF and process waits;
timeout kills/reaps that group. Independent verification has a separate 600-second
ceiling. Partial output has no `manifest.json`; a candidate is diagnostic only.
After a fresh verifier process passes, exclusive publication creates the success
manifest and links the verification report hash.

`rheon-global-pressure-v1` uses closed top-level/provenance/trajectory/frame/array
objects (unknown keys fail). JSON duplicate keys, nonfinite values, type-confused
booleans and malformed dimensions fail. Geometry/reservation containers and all
declared numeric/string primitives are validated recursively; native diagnostics
require finite numbers and true integer iteration counts. Geometry hashes cover
the actual validated geometry object. Required fields are visible in producer/
verifier code. Every array descriptor owns path, `shape_xyz`, dtype, units,
`x-fastest` layout, exact byte count and raw SHA256. Payloads are little endian:
RHS/pressure `<f8`, MAC velocity/tracer `<f4`. No compressed-container hash
substitutes for a raw-array hash. Shapes/bytes are checked before allocation.
Units: RHS m³/s², pressure Pa, velocity m/s, tracer dimensionless. Frame index 0
forces at 0 s and accepts state at dt; subsequent times use exact integer multiples.
Geometry hashes use sorted compact JSON of counts, extents, origin, density,
sealed-MAC convention and SI units. Manifest records actual row/trajectory mapping.

Initial zero pressure/u/v/w and Gaussian tracer are stored once per trajectory.
Each frame stores RHS, pressure, preprojection u/v/w, corrected u/v/w and advected
tracer. `previous_pressure` and `previous_state` reference actual initial/previous
accepted payload paths; no duplicated history arrays or inferred NPZ ordering.
The independent verifier checks those lineage references exactly. It scatters
interior-face incidence contributions for RHS and full finite-volume operator,
then independently reconstructs the mandated f64 expression and single f32 rounding
of each interior correction. This includes gauge-cell physics and checks stored
corrected witnesses bitwise. Its direct physical divergence uses cell differences
and f64 `math.fsum`; it reports max/RMS, net flux, Courant and tracer integral/
extrema. It does not import or call producer physics. It calls only the separate
native preparation verifier for source identity and binary `--identity` for build
identity. Rehashed corruptions exercise physics independently of byte-integrity
rejections, including signed-zero corrections, which are compared as packed f32
bytes. Whole-history fingerprints include validated geometry, initial payloads
and ordered frame payload hashes, excluding trajectory IDs, seeds, splits and
filenames; equal fingerprints across train/validation fail. Fingerprints normalize
physical geometry numbers (including signed origin zero), while the raw geometry
hash retains the actual validated serialization. Every nonfixture
trajectory must contain exactly nonzero RHS and a nontrivial accepted state
(velocity or tracer change), without a new numerical threshold. Shared initial
zero/Gaussian payloads remain tagged diagnostic groups. Literal all-axis 3D
anisotropic fixtures and exact-rational rounding fixtures isolate predicted and
actual divergence failure boundaries. These sampled fixtures prove their named boundaries, not exhaustive
correctness or engineering flow fidelity.

At 16³ the raw per-frame payload is 186,368 bytes; initial state is 101,376 bytes per
trajectory. Train 12×24 needs54,890,496 raw bytes; validation 4×24 needs18,296,832;
combined 16×24 needs73,187,328 (69.80 MiB), plus bounded manifest/report metadata.
Corrected MAC witnesses add independent rounded-divergence/history evidence and
are the only per-frame additions to the approved core fields. The local cap is
128 MiB raw; no default 64 MiB managed-output claim is made for combined capture.
Future managed generation must use separate train/validation shards and account
for metadata/transient copies under the actual declared runtime/output budget.
Actual wall time, payload/metadata bytes and Linux RSS are measured during the
qualified run; prospective estimates are not measured acceptance.
