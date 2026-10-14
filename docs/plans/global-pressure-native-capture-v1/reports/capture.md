# Actual development reference qualification

Source was committed before capture at
`56b2507b763fe603b0206cd23592d488a58b30d0`. A clean qualified wrapper build bound
its actual Cargo graph to the prepared native dependency, source closure,
configuration, reservation and Rust/Cargo1.92.0. Binary SHA256:
`466c6c01bbe2c082095d25a064b92e285d8689dd828af2777e22503df0dd4c37`.
The compact [receipt](capture.json) retains all source/build/capture identities.

Actual smoke DEVELOPMENT capture:12 train and4 validation trajectories,24 frames
each,384 frames total. Counts16³; g0–g3 each supply4 trajectories;12 steady and
4 moving forcing regimes. No test/full reservation was generated. All16 complete
histories have unique fingerprints,24 nonzero-RHS frames and24 nontrivial accepted
states each. Six repeated raw-payload groups contain expected shared initial
states and first-frame tracer fields (the first tracer advection uses zero previous
velocity); this permitted field sharing is explicit and is not a duplicate whole
trajectory. No repeated group consists exclusively of noninitial fields.

| Actual metric | Result | Unchanged gate / scope |
| --- | --- | --- |
| Maximum full original-RHS residual L2 | 1.312727488650643e-10 | Every frame <=max(1e-12,1e-8*rhs_L2), including gauge cell |
| Maximum relative residual | 9.997275658030901e-9 | <=1e-8 |
| Maximum predicted divergence | 1.4742342204252277e-9/s | <=1e-7/s |
| Maximum corrected f32 divergence | 2.980232238769531e-7/s | <=1e-5/s; corrected witnesses checked bitwise |
| Maximum actual divergence RMS | 1.4932158064182864e-8/s | Reported diagnostic |
| Maximum absolute net flux | 1.3905686955514645e-19m³/s | Approved absolute/relative gate; zero wall-normal faces |
| Maximum Courant | 0.11476040681203206 | <=0.5 |
| Jacobi iterations | 141–178 | <=3000; no fallback or dropped case |
| SGS frame-zero checks | 16;52–65 iterations | Native gated diagnostics; no stored SGS witness arrays |
| Maximum Jacobi/SGS pressure L2 difference | 4.278118694159782e-9Pa | Reported diagnostic, not an imposed equivalence tolerance |

Tracer remains finite with observed range approximately3.3112191697e-37 to
0.1988556236. Final/initial relative integral change ranges from-0.0016744586354
to+0.00098306626005. Native semi-Lagrangian tracer transport is not exactly mass
conservative; per-trajectory drift is retained as the future learned/native
comparison baseline. No new tracer-conservation threshold or exact-conservation
claim was introduced. This24-frame development rollout does not qualify the
reserved96/600-frame test rollout or engineering flow fidelity.

Generation took7.804405109s including16 producer starts, native steps and raw
publication, measured after admission/source/binary identity checks. Fresh
candidate verification took19.021815930s including source checks and independent
array reads/physics. Independent numerical reviewer repeated full verification
in18.6827465s and confirmed coverage, source identities, all physics metrics and
leakage checks. Raw payload73,187,328bytes (69.797MiB); all capture files including
candidate, manifest, verifier report and stderr75,695,918bytes. Caller Linux
RUSAGE high-water17,768KiB; pre-verification child high-water17,512KiB. These are
process high-water observations, not isolated native allocation or a managed
runtime memory qualification. Host AMD EPYC9V74,4CPU-equivalent quota,16GiB memory
limit; serial no-default-feature native binary. The build reused dependency
compilation caches and took1.0455s; no cold-build performance claim.

Local raw cap128MiB and separate600s generation/verification budgets were met.
Combined capture exceeds the existing default64MiB Tuldok output contract;
future managed generation must separate train/validation and use bounded flat
shards with metadata/transient-copy admission. No managed-run result is claimed.
No inference, model training, fallback timing or speedup is measured in this slice.

A fresh two-frame anisotropic TRAIN fixture passed at the committed source, then
a separate same-runtime repeat produced23 bitwise-equal raw payload files. This
is sampled repeatability, not full384-frame duplication or cross-platform proof.
Focused corruption cases isolate hashes, metadata types, raw physics, gauge-row,
packed f32 signed-zero, history/split identity and process/publication boundaries.

Full local artifacts remain ignored at
`runs/global-pressure-v1-capture/committed-development-complete` and
`runs/global-pressure-v1-build/committed-56b2507`. Manifest SHA256:
`dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d`.
Fresh verification receipt SHA256:
`b71ad37567f360df809c94c9307ccc33c47d6d3530219a669069dfaadc40170d`.
Only compact provenance and scalar/trajectory summaries are committed. Rebuild
and capture through the documented wrapper to reproduce locally; the absolute
workspace locations in original receipts describe that actual execution.
