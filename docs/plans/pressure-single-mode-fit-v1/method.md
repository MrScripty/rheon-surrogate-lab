# Single fundamental-mode fitting test

This is a capacity and optimization diagnostic. C62 starts from C53 and optimizes
one newly manufactured pressure field, without competing training examples or
teacher pressure. It is not a new CFD simulation or a generalization test.

The grid contains 13×15×16 cells, each 0.0625 m wide. Its physical size is
0.8125×0.9375×1 m. The pressure target is the fundamental X Neumann cosine mode,
with its pressure gauge fixed at cell zero. Independent eigenvalues construct the
RHS; the implemented operator must agree within 1e-10. The original train-only
correction RMS sets amplitude (2.8535734640782584 Pa). Water density, timestep and
normalization also come from the existing training states. No validation label
constructs the fitting case.

The architecture, channels, preprocessing, float32 neural output and float64
residual/reconstruction remain unchanged. Four differentiable shared corrections
are optimized with fresh Adam, clipping at L2 norm 1, batch size one and no
weight decay. The loss is the existing analytical pressure/gradient/equation/
amplification loss, plus the final log equation residual and 0.05 times the mean
earlier-pass log equation residual. The target in those logs remains 1e-6.
Teacher weight is zero. The inherited configuration still describes other banks,
but the captured trainer explicitly bypasses their construction and losses.
Original real frames are used for scalar calibration and inherited diagnostics
only. Retained actual optimizer counters must confirm 2,000 examples, 8,000
network forwards and 24,960,000 voxel-forwards (3,120 cells per pass).

Learning rate begins at 1e-4 and changes to 1e-5, 1e-6, 1e-7 and 1e-8 at updates
1,000, 1,500, 1,750 and 1,900 respectively. A separate fitted-case measurement
records 1/2/4/8-pass residuals every 100 updates. These diagnostic forwards are
excluded from optimizer work. Inherited real-frame traces are labelled separately.

CPU autograd qualification makes no optimizer updates. CUDA training uses the
existing Tuldok catalog and Pumas Torch runtime, with a 1,800-second trainer cap.
Fresh managed CPU restore must preserve the published weights, normalization,
history and source identity exactly. Subsequent CPU measurements hide CUDA and
use one thread per worker. Timings on the 128 consumed real requests are isolated
between checkpoints; afterward at most three independent diagnosis workers use
separate cores, with a 12 GiB host reserve plus 2 GiB per worker.

Capacity success means restored CPU four- and eight-pass fitted-case residuals
are both at most 1e-6, with one/two-pass residuals no worse than C53. The 128 real
requests and 240 manufactured cases are regression guards, not untouched holdouts.
Specialist regressions prevent promotion even if the capacity gate passes.
Only capacity success authorizes the conditional frequency-curriculum arm.

The first prospective Project was never admitted or executed. Preparation review
found that the inherited history conditional would omit the optimizer counters
when companion banks were disabled. Project 02 captures that correction and is
the sole qualified/executed Project. Both prospective captures are retained under
`runs/pressure-single-mode-fit-v1/`; none was overwritten.

The commands and selected runtime environment are retained in the controller and
CPU execution receipts. Preparation uses `prepare.py --config config.json --arm
single-mode-fit --output <fresh-project>`. CPU `qualify.py --project <project>
--output <receipt>` precedes `run.py --config <frozen-arm-config> --assembly
<assembly> --qualification <receipt> --session-file <private-session> --output
<fresh-controller>`. The session contents, weights and complete field arrays
remain outside Git. `evaluate.py` consumes the completed controller receipt and
the actual Project identity through its retained assembly.

## Conditional C63 curriculum

C62 passed its restored CPU gate. C63 starts again from C53, rather than from the
specialist. Its complete training configuration matches C60 except the Project
ID, and its objective matches except the arm name. Numerical source is identical
except the supplemental-bank generator and its captured curriculum descriptor.
Primary analytical-bank generation delegates directly to the unchanged owner.
The 1,500-update seeds, learning-rate schedule, teacher weight 0.1, early weight
0.05, clipping, original real/direction/current examples, grid order, four signed
amplitudes and resident tensor budget are preserved.

Each supplemental grid gets 32 patterns: the three fundamental axes repeated four
times each; 12 explicit other modes with indices at most three; and eight modes
with at least one index four or higher, including grid-dependent near-Nyquist
axes/corner. Repetition supplies explicit sampling weight. The canonical private
sampling order is unchanged. Actual exposure is 579 fundamental, 542 other low
and 379 high examples, versus C60's 131 fundamental draws. The original 128 real
requests and 240 manufactured cases remain consumed regression diagnostics.
These manufactured targets are not new fluid trajectories or holdouts.

Independent CPU qualification verifies every supplemental label against the
operator, unchanged original sources/inputs/normalization/objective, exact outer
sampling order and equal residency, plus finite nonzero gradients without updates.
The capacity stage remains immutable in `capacity-results.json` and
`capacity-decision.json`; final aggregate results may add the curriculum stage.
