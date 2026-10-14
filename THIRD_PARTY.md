# Dependency provenance

Author: Puma. Project license: Apache-2.0, supplied by the owner in the public
repository's initial commit `8cb010dd060317bc160694e1e1a8d3bf11cac253` and retained
unchanged as [LICENSE](LICENSE). This source repository does not redistribute compiled binaries or a vendored
solver. The isolated review patch contains limited upstream source context;
its upstream Apache-2.0 license is retained in patches/RHEON-LICENSE. A future binary distribution must carry
its applicable third-party license/notice materials; reassess at that boundary.

| Dependency | Pinned source | Purpose | Upstream terms |
| --- | --- | --- | --- |
| Rheon | MrScripty/Rheon, 34235d33c886aa05b5bdf95d920b863d1fb2ad6f | Public numerical operator/PCG reference, default features disabled | Apache-2.0, upstream LICENSE |
| anyhow | crates.io 1.0.100 | Contextual CLI/data errors | MIT OR Apache-2.0, crate license files |
| serde | crates.io 1.0.228 | Typed manifest serialization | MIT OR Apache-2.0, crate license files |
| serde_json | crates.io 1.0.145 | JSON manifest, float roundtrip | MIT OR Apache-2.0, crate license files |
| sha2 | crates.io 0.10.9 | SHA-256 corruption checks | MIT OR Apache-2.0, crate license files |
| NumPy (optional) | PyPI 2.3.5 | CPU CNN tensor arithmetic; no downloaded weights | BSD-3-Clause core; installed LICENSE.txt also describes bundled components |

Cargo.lock identifies the complete resolved graph and registry checksums. Source
and license files are fetched by Cargo from those upstream packages. Dependency
licenses do not choose a license for this project's original source. Standards
were consulted from MrScripty/Coding-Standards commit
188beda1fa477d21d576c233dd8c7f4c4c267d23; this repo's prose is original, not a
vendored standards package. The root dependency and original Rheon checkout remain
unmodified. patches/ contains a separately identified local experimental change;
its upstream source context remains subject to Rheon's Apache-2.0 terms. No project
license grant is inferred from that patch or dependency; the root LICENSE is the
owner's separate project choice. Frozen Cargo metadata remains unchanged from
the evaluated source; no package-registry publication is part of this release.

The optional Tuldok pressure project uses Rheon trajectory revision
`2209012bc5f902693d0f695ec1c2d854126af317` (Apache-2.0), with parallel kernels,
through a separate pinned Git Cargo manifest. No solver source is copied.
CPU PyTorch2.8.0+cpu is optional, with upstream BSD-style license and bundled
component notices retained by its installed distribution. Its pinned runtime
closure is in tuldok/requirements-cpu.lock; installation is explicit and private,
not distributed in this Git change. NumPy2.3.5 retains its original license role.
Tuldok's generic APIs/readers are used from an explicitly selected source checkout;
no Tuldok code is copied or relicensed into this repository. Assembly copies the
lab license/notice files into the local project; any future native/runtime binary
distribution still requires its complete dependency notices at that boundary.
