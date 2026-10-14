# Native API qualification

Qualified source: Rheon2209012bc5f902693d0f695ec1c2d854126af317 plus the lab-owned
patch SHA256b851b38d2d03280220bc4648cd4d721b2e796786fb18d8188dc335336fe2a998.
Prepared746file tree SHA2565a5de4635d3970c9b99e620d9fff2d9a23c55307a9e335f7473e7f95a0c7a4ab.
No upstream/Rheon-main or Tuldok file changed. Apache license and full prepared
source inventory are retained beside the patch; no redistributed compiled binary.

Actual checks on Linux x86_64 Rust/Cargo1.92.0 serial/no-default-feature source:
8newwarm-start,10existingpressure,5operator and11Python preparation tests passed.
New cases exercise both Jacobi/SGS, exact/zero/poor/invalid guesses, fullgauge-row and
original-b tolerances, both old entrypoints, cancellation/failure and workspace reuse.
Affected fmt and Clippy(all-targets,-Dwarnings) passed; preparedsource receipt verifies.
Helper identity JSON rejects duplicate/nonfinite fields; exact integer schema and
canonical typed receipt comparison reject bool/float aliases, missing/unknown and
wrong-type fields on a valid source closure. See [command/result receipt](native.json) and the [API](../../../../experiments/global-pressure-v1/dependency/README.md).

The optional entire upstream no-default suite was stopped after570seconds when
unrelated long material scenes exceeded the quick supporting-check budget. Owned
Cargo/testchild processes were terminated and confirmed absent. Exit143 is an
incomplete budget stop, not a successful suite or reported assertion failure.
The affected pressure/operator/API checks were explicitly completed separately.
No parallel-feature runtime, source-wide full-suite success, model quality or
performance gain is claimed. Independent numerical/standards source review remains
part of separate N5 acceptance; actual development trajectories close N2–N4.
