# Small-grid low-mode training screen

Status: Accepted as a completed experiment with failed model qualification.
Both CUDA fits, fresh CPU restores, artifact parity, isolated pass sweeps and
manufactured regression diagnostics completed. C53 remains the experiment baseline.
Branch: `experiment/pressure-low-mode-tail-20261017`.
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
   residual. Earlier-pass guard: neither the one-pass nor two-pass worst real
   request residual may exceed C53. Freeze these guards before CPU evaluation.
   Only a passing candidate merits independent scenes or longer training.

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
The unchanged ID-reservation and report-link functions in `four_pass_study.py`
also reserve C60/C61 and expose this report; stage only these narrow edits, leaving
the preexisting additions to that file outside this experiment's commits.

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

Preparation and zero-update CPU objective qualifications passed for both arms.
Both use identical 1,024-example bank hashes and balanced exposure; worst
independent bank operator defect is 1.404e-14. C60 training Run is
`b6ab6203750543a18ad61cd4de588cd4`; C61 is
`71e22ac78d16420bb89ebb8b0dc695e0`. Both finished successfully and drained.

Results: C60/C61 qualify 9/128 real requests at four passes versus C53's 7/128.
C61 median residual is 8.01043e-6 versus C53's 9.67990e-6, but its worst residual
is 1.28924e-4 versus 9.78553e-5. Both new models qualify 128/128 at eight passes;
their manufactured eight-pass count regresses to 200/240 from C53's 230/240.
Earlier-pass and worst-case guards fail. No longer fit, independent scene test,
untouched holdout access or model promotion follows this failed screen.

Actual CUDA trainer times: C60 322.132 s; C61 327.903 s. Complete managed workflows
take 1767.535 s and 1687.030 s respectively. Isolated four-model CPU evaluation plus
postfit manufactured diagnosis takes 58.290 s. Each managed controller reuses nine
whole-hash-verified artifacts; raw owner authorization and custody checks remain.

Postfit exposure analysis reproduces the exact retained supplemental sampler
order: 131/1500 draws are single fundamental-axis modes, 296/1500 have exclusively
low-block components, 378/1500 have any low component. Equal grid exposure did not
mean equal spectral exposure. This is an observed coverage limit, not a proved
cause of regression. C60/C61 isolate the extra objective; historical C59 also
differs in sampling and bank coverage.

Next slice: fit one fundamental mode on the hardest small grid (13x15x16) in a
bounded CUDA diagnostic, with earlier/eight-pass regression guards. If it can fit,
test a frequency-balanced low-mode curriculum rather than another broad extension
of these fits. If it cannot, separate optimization/objective interference from
representation limits before changing the correction architecture. Workflow
profiling remains a separate Tuldok task; see `issues.md`.

Retained authority: `selection-policy.json`, `decision.json`, `results.json`,
`method.md`, `bank-exposure.json`, `verification.json` and offline `index.html`.

Branch retention: private experiment branch retained for Jeremy's review and
continued model work. Jeremy owns integration into the ongoing lab branch; no
automatic rebase or merge. Retire after accepted integration or explicit owner
disposition. Historical experiment captures remain immutable.
