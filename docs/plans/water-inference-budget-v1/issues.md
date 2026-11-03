# Findings

- Larger grids above 65,536 cells are unsupported by the current model/scene
  contract. Owner: future capacity qualification; this tranche measures through
  40-cubed and does not change that limit.
- Current UI deployment remains a separate seven-channel contract. Owner:
  future Rheon integration; laboratory one-channel policies do not replace it.

- Combined single-pass learning rate .001 caused non-finite gradient-norm
  rejection after at least 2,400 updates. Owner: this experiment. Retain failure
  and compare a separately configured .0001 schedule; do not relax finite checks.
