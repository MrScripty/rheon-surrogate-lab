# Small-grid low-mode training screen

Status: Diagnosis accepted; preparing matched CUDA projects. Branch: `experiment/pressure-low-mode-tail-20261017`.
Base: `81fa34929d4d24808069daf78d0c14d27939729c`.

Jeremy authorized continuing the recorded sequence. C53 remains the baseline;
C59 improved typical errors but failed the four-pass gate. The canonical experiment
configuration is `experiments/pressure-low-mode-tail-v1/config.json`.

1. Diagnose C53 and C59 on 240 newly manufactured single-mode cases each, spanning
   eight small odd/thin grids, six spatial modes and five signed amplitude scales.
   Independently derived Neumann eigenvalue labels must agree with the captured
   finite-volume operator to relative defect <=1e-10. Only original training
   amplitudes/scalars are read. Consumed-validation shapes inform coverage; their
   recorded pressure targets never enter training.
2. Freeze an equally exposed eight-grid supplemental bank using the existing
   analytical generator, 32 patterns and four signed amplitude multipliers. Retain
   the original broad primary bank and real, direction and current-policy frames.
   Compare matched C53-initialized 1,500-update CUDA arms: C60 existing four-pass
   physical/eight-pass teacher objective; C61 additionally emphasizes final low
   spatial-mode pressure error. Same seeds, order, Adam, schedules and banks.
   Preserve earlier-pass feedback and the frozen eight-pass teacher in both.
3. Use existing visible Tuldok projects/catalog and Pumas runtime. Qualify CPU
   recurrence, independent labels, finite nonzero autograd, unchanged initialization
   and frozen teacher before admission. Require real CUDA training and fresh CPU
   zero-update restore with verified artifact parity. No private training fallback.
4. Measure C53, C59 and candidates at 1/2/4/8 passes on the same consumed 128
   snapshots. Isolate timing from concurrent CPU work. Require all 128 four-pass
   equation checks <=1e-6 and all 128 eight-pass checks; report earlier-pass
   regressions. A qualified candidate must not increase C53's worst four-pass
   residual. Only a passing candidate merits independent scenes or longer training.

Neural computation remains float32; residual/reconstruction retains its existing
float64 contract. Cubic voxel spacing, water physics, model architecture, input
channels and default deployment pass count remain unchanged. This screen does not
promote a model. Original untouched holdouts remain unopened. Earlier fresh scenes
are consumed regression data.

Write scope: new `experiments/pressure-low-mode-tail-v1/` and
`docs/plans/pressure-low-mode-tail-v1/`; narrowly reusable preparation/collection
seams in the prior experiment owner if needed, with independent workflow commits.
Existing dirty chart edits and unrelated untracked work are outside this slice.
Published history and previous experiment captures are immutable.

Record config/source/input/weight identities, commands, device, actual updates,
trainer versus workflow elapsed time, CPU timing, peak memory, Run custody and
failed gates. CPU workers use one thread each on distinct cores, at most three;
reserve 12 GiB host memory. Teacher cache <=256 MiB, bank <=768 MiB, CUDA wall cap
1,800 seconds per arm. No large arrays, weights or credentials enter Git.

Diagnosis: both C53 and C59 fail all 240 manufactured cases at four passes. Worst
independent label defect is 1.761e-14. Changing amplitude from 1e-4 to 100 times
train correction RMS barely changes relative errors, consistent with the current
homogeneous network. This population supports low-mode contraction/shape coverage,
not an amplitude noise-floor explanation. At eight passes C53 passes 230/240 and
C59 220/240; this new manufactured population is distinct from the consumed 128
snapshot gate. Record it separately and require no reduction in its pass count.

Frozen objective: C61 adds weight 0.1 of `log1p(low-mode squared pressure error /
max(low-mode target energy, 0.01 * full target energy) / 1e-12)` at the final
correction. Retain modes 0..3 on each axis, exclude constant gauge. Both arms keep
physical/equation loss, early feedback 0.05 and frozen C53 eight-pass teacher 0.1.
This is an auxiliary pressure objective; equation acceptance stays 1e-6 relative
to full RHS. Training equation feedback divides by initial correction residual.

Next slice: qualify exact project sources, independent analytical labels, bank
memory/exposure and both objectives before two managed CUDA admissions.
