# Isolated local Rheon patch

Author: Puma. This is a reviewable experiment, not an upstream release. The lab's
Cargo.toml and Cargo.lock still pin unmodified Rheon at
34235d33c886aa05b5bdf95d920b863d1fb2ad6f. Only the optional standalone probe links
a sibling checkout. Existing Rheon main/checkouts must not be altered.

From the lab root, create a **new, absent** sibling directory:

```sh
git clone --no-checkout https://github.com/MrScripty/Rheon.git ../rheon-warmstart
git -C ../rheon-warmstart checkout -b experiment/initial-pressure 34235d33c886aa05b5bdf95d920b863d1fb2ad6f
git -C ../rheon-warmstart am "$PWD/patches/0001-sealed-initial-pressure.patch"
cargo test --manifest-path ../rheon-warmstart/Cargo.toml --no-default-features --locked
cargo fmt --manifest-path ../rheon-warmstart/Cargo.toml --all --check
cargo clippy --manifest-path ../rheon-warmstart/Cargo.toml --no-default-features --all-targets --locked -- -D warnings
```

Do not reuse an occupied path. `git am` requires a local Git committer identity;
it can produce a different commit ID while preserving the reviewed source tree.
manifest.json records the original local commit, expected tree, patch hash and
actual checks. The driver requires that reviewed tree and a clean checkout; the
native binary records its build commits and dirty flags. Rebuild after applying
or committing changes. No upstream push/PR/merge is part of these instructions.

The new solve_rhs_with_initial_pressure validates finite values, exact length and
pressure[0]==0, starts from full b-Ap, uses the original b in relative tolerances,
and clears four Krylov buffers. It supports sealed gauge-fixed domains only.
Existing solve_rhs and solve_velocity keep zero-start behavior. Actual f32
corrected divergence remains a caller acceptance gate.

309 tests passed with default features disabled, including eight new tests for
both preconditioners. Independent source review confirmed the numerical contracts.
This does not validate desktop/PNG features or establish acceleration/general CFD
accuracy. The upstream Apache-2.0 license is retained as RHEON-LICENSE for patch
source context; it does not select a license for the lab's original source.
