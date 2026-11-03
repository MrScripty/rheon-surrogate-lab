# Cubic physical cells for variable domains

Status: complete. Acceptance: accepted for cubic-cell capture setup; rectangular
GroupNorm training remains unsupported and was not run.
Owner explicitly requested 16 cells per metre by default, shared h = 0.0625 m;
per-axis cell counts control physical extent. For 1.5 × 1 × 0.75 m derive
24 × 16 × 12 cells. Never stretch a fixed 16³ grid. Nonintegral counts fail
with an actionable error.

This numerical configuration slice follows separate UI portability commit be6e67c
on feat/live-rheon-tuldok-project. Write set: this plan, new
experiments/cubic-pressure-v1, probes/cubic-pressure-v1, live-project bundle and
observer documentation integration, live observation plan
completion, and repo project/README. Keep frozen pressure/capture/train/model
sources, reservations and results byte unchanged. Rheon solver arithmetic and
physical tolerances stay unchanged; only native capture geometry/count allocation
and independent verification follow derived per-axis counts.

Qualification: derive default/aspect/scale grids; reject fractional, invalid,
over-budget counts. Build from selected qualified Rheon source and retained patch,
locked toolchain. Qualify a real one-frame fixture first, then full selected fresh
development parents through the independent finite-volume/MAC oracle. Observe
actual cubic-cell frames in egui and status in existing Tuldok project.

The unchanged GN model accepts only 16³/32³ cubes. New rectangular counts must be
rejected before optimizer/training; do not claim that the current model/data reader
supports them, change spacing to rescue shape, or introduce an unreviewed model
change. Captures and compatibility evidence are separate from trained results.

Use previously adopted Coding Standards 188beda for implementation, contracts,
numerical oracles, dependency/build, GUI observation and atomic commits. No GPU,
final-suite data, push/merge or cleanup of retained historical/dirty worktrees.

Accepted evidence is in completion.md and acceptance.json. Source branches are
retained locally for user review and continued integration; no push or merge.
The unchanged original solver/model and historical results remain the baseline.
