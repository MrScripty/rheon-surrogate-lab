# Early API/capture contract

Status: implementation contract under qualification,2026-10-14. No numerical
qualification inferred from this document. Parent/external validation can use this
boundary while independent source work proceeds.

Native base2209012bc5f902693d0f695ec1c2d854126af317 supports public sealed
PressureOperator/PressureWorkspace APIs. Lab-owned patch adds
`solve_rhs_with_initial_pressure(operator,rhs,initial_pressure,dt,settings,cancelled)`.
Shape/finiteness/exactcell0gauge are checked; nonsealed/free-surface returns explicit
unsupported-domain outcome. It computes full original`b-Ap`, uses original`||b||`
for relative stopping and resets Krylov scratch. Existing solve_rhs/solve_velocity
remain zero-start. Both Jacobi/SGS and cancellation/failure reuse need actual tests.

Dependency is an exact content-hashed patch over a clean pinned public source,
prepared in an exclusive task-owned directory with source/tree/patch receipt.
New probe declares Gitbase2209012 with explicit validated local Cargo override;
no arbitrary mutable sibling, remote fork or upstream/Tuldok modification.

Proposed capture orchestration: `capture.py --preset smoke --split train|validation|
development --binary ABS --prepared-source ABS --output NEW_LAB_RUN`.
Optional `--trajectory ID` selects only canonical smoke development entries.
A TRAIN-only `--fixture-frames`1..24 is a diagnostic partial fixture, never complete
preset evidence. Full/test requests are rejected before output creation.

Per-frame arrays: float64littleendian rhs/pressure; float32littleendian pre_u/v/w,
corrected_u/v/w, tracer. Counts/layout/spacing/rho/dt/units and SHA256/bytes are explicit;
x varies fastest. Initial zero pressure/velocity+Gaussian tracer stored once;
previous state/pressure references the prior accepted frame. Each16³ frame186368raw
bytes includes corrected-face witnesses needed for independent f32 correction and
divergence checks. No container size is inferred without measurement: raw frame arithmetic is384×186368=71565312bytes plus16initial
states and metadata. Training input packaging will select needed fields into
bounded shards separately, rather than duplicating prior arrays.

Step order is fixed in config/source: advect velocity from own accepted state;
advect tracer from previous velocity/tracer; apply declared acceleration; assemble
original RHS; pressure solve; independently gate residual/predicted/actualdivergence
and walls; commit accepted corrected faces/tracer. Native frame stream and Python
publisher retain their separate owners. No teacher/model input exists in slice1.

Independent stdlib verifier assembles interior-face incidence contributions, RHS,
full pressure action and exact stagedf32 correction; it does not import/call producer
physics. Hand/dyadic anisotropic fixtures provide an independent sign/unit/wall
oracle. Corruption tests target distinct hash/layout/gauge/residual/divergence/
lineage/provenance/completion boundaries. Publisher emits success manifest only
after a fresh verifier qualifies bytes; failed/partial output is not success.
