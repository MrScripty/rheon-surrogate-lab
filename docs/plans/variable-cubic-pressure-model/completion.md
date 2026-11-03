# Variable-grid model and training acceptance

The new `cubic-groupnorm-v2` model accepts positive per-axis counts, including
odd, rectangular and singleton dimensions, within the eager CPU budget of
65,536 cells per sample and batches of 1–4. It returns one pressure value per
original cell. There is no 16³/32³ whitelist. Capture and serialized-input budgets
are separate, explicit limits; this is not unlimited-memory shape admission.

Physical preprocessing, the full operator and loss enforce one shared positive
cell side length. Training packets derive counts from physical lengths and cells
per metre and reject unequal spacing before numerical decoding. Coordinates
represent actual cell centres in metres. Pooling adapts to each axis, with no
physical field stretching, padding or resampling. Seeded weights and the existing
16³ hidden-feature forward remain exactly equal in regression checks; the
GroupNorm blocks remain non-affine, with 372,377 parameters.

The actual cohort independently qualified one-frame fixtures before each full
capture, then all 144 frames: four complete training parents (96 frames), two
held-out validation parents (48 frames), at 8 and 16 cells/metre. Their grids are
8³, 9×7×5, 12×8×6, 16³, 17×13×9 and 24×16×12. Independent MAC checks cover every
reference frame; native input residuals also agree with the independent full
face-incidence operator. Each batch averages per-frame gradients while keeping
each frame's own shape. Normalization and sampling exclude validation parents.

Managed training in the user's **existing** Tuldok catalog completed 200 Adam
updates. Run f94e2a345cb14cdf8672365dbe8d615f and independent zero-update restore
80e35ae9571045898d4344b5c4a42f2c both completed with drained custody and 18 artifacts
each. Sixteen shared artifacts match byte for byte, including predictions whose
numerical contents were recomputed exactly by the restore. All 36 artifacts were
downloaded and hash checked again after a clean service/catalog reopen.
The runtime is the actual captured Python/NumPy 2.3.5/Torch 2.8.0+cpu image, with
one Torch and interop thread. Training took 7.987 s inside its managed child and
peaked at 466,759,680 observed RSS bytes; these are not speedup measurements.

The final restored weights also produced finite, correctly shaped, gauge-fixed
pressure on unseen 11×9×7 inputs at 12 cells/metre and 257×2×1 thin inputs at
16 cells/metre. The native and Torch cubic-grid residuals differ by at most
4.337e-19. These are synthetic input/geometry checks with unchanged weights and
zero optimizer updates, not simulated trajectories or pressure-accuracy claims.

**Pressure accuracy remains unqualified.** Combined held-out loss decreased from
2.6347228761 to 0.1197461126, but raw learned pressure passed **0/12** native
snapshot checks. All 12 learned warm attempts exhausted their 32-step allowance
and fell back to the temporal reference start. Corrected results qualified 12/12;
learned policy used 1,691 iterations versus 1,307 for the full-budget temporal
baseline, including 384 wasted warm iterations. Shape/training support succeeded;
this cohort demonstrates neither an accurate raw learned solve nor acceleration.
There was no post-evaluation tuning, autonomous rollout or final-suite claim.

The actual repo project is 69346e58296d46fbb8bd0e01b975d24a. Normal execution project
db50aa2223f04e9bb93ea2bd615534f9 exposes both managed Runs. All six complete reference
trajectories are draft scientific-field Data records in that same existing
catalog; original f32/f64 field bytes are preserved and no review/release was
admitted. Rheon's actual egui viewer was inspected with the correct domain,
counts and failed learned-pressure label. Recorded diagnostic frames are retained
for native scrubbing and explicitly labelled as replay. The user's original
simulation and recordings were not changed or interrupted.

Nine focused model/training checks and six native checks pass, with Rustfmt and
Clippy warnings rejected. Frozen model, capture, reservation and result sources
remain byte unchanged. Implementation source commit is 0534661 on the private
feat/variable-cubic-pressure-model branch, separate from preceding UI portability
and cubic-grid setup commits. Source/runtime/data hashes are measured; historical
hash constants were not replaced with new experiment hashes.

Retained prefixes: initial test-fixture errors used a NumPy `fill_` typo and
compared spacing metadata source labels instead of scale values; these were fixed
before source qualification. Runtime prefix runtime-1 failed private-directory
checks before publishing a snapshot or admitting a Run. Runtime-2 used umask 0077
and succeeded; the same idle Tuldok server was restored after each attempt.
Native build-1 used initial protocol names; build-2 and the material build use
the explicit variable-grid protocol. Earlier prefixes remain retained. Numerical
material and the successful managed cohort were not silently retried or relabelled.

Full evidence: acceptance.json and ignored runs/variable-pressure-v1-workspace-1,
runs/variable-pressure-v1-material-1, and runs/variable-pressure-runtime-2. Local
branches and application/runtime worktrees remain available for review and
continued user work. Nothing was pushed or merged. The next model concern is
pressure accuracy and useful correction cost, assessed separately from shape support.
