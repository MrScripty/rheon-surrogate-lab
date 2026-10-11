# Dependency provenance

Author: Puma. New project license: not selected by the owner. No license field or
license grant is invented. This source repository does not redistribute compiled
binaries or vendored dependency source. A future binary distribution must carry
its applicable third-party license/notice materials; reassess at that boundary.

| Dependency | Pinned source | Purpose | Upstream terms |
| --- | --- | --- | --- |
| Rheon | MrScripty/Rheon, 34235d33c886aa05b5bdf95d920b863d1fb2ad6f | Public numerical operator/PCG reference, default features disabled | Apache-2.0, upstream LICENSE |
| anyhow | crates.io 1.0.100 | Contextual CLI/data errors | MIT OR Apache-2.0, crate license files |
| serde | crates.io 1.0.228 | Typed manifest serialization | MIT OR Apache-2.0, crate license files |
| serde_json | crates.io 1.0.145 | JSON manifest, float roundtrip | MIT OR Apache-2.0, crate license files |
| sha2 | crates.io 0.10.9 | SHA-256 corruption checks | MIT OR Apache-2.0, crate license files |

Cargo.lock identifies the complete resolved graph and registry checksums. Source
and license files are fetched by Cargo from those upstream packages. Dependency
licenses do not choose a license for this project's original source. Standards
were consulted from MrScripty/Coding-Standards commit
188beda1fa477d21d576c233dd8c7f4c4c267d23; this repo's prose is original, not a
vendored standards package. Rheon remains unmodified.
