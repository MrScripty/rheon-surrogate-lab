# Method and evidence boundaries

The control is the configured one-channel, linear, bias-free residual network,
not its untransformed ten-channel ancestor. Config transforms remove active
GroupNorm and SiLU. Both fits start with identical surviving seeded tensors;
the candidate adds only a zero float32 matrix of shape 64 by 64. The original
global-average branch remains unchanged. Existing evidence that gauge removal
largely cancels its spatially constant contribution does not imply bitwise
equivalence when removing it in float32.

For input q, restrict fine-cell averages into four equally sized spatial bins
per axis using exact cell/bin overlap fractions. This is defined even when a
dimension has fewer than four cells. A dense learned matrix maps the 64 coarse
values to 64 outputs. Prolong through piecewise-linear interpolation between
coarse cell centres, with clamped endpoints. Subtract the prolonged field's mean
and multiply by (longest grid cell count / 16)^2. Add this field to the original
raw network output before existing float64 pressure reconstruction/gauge removal.
There is no pressure equation inverse in the added inference path. Spatial
restriction/prolongation are linear transformations, not fluid simulations.

The coarse bins use normalized domain coordinates implicitly, and the length
factor is an explicit architectural hypothesis. It does not by itself encode
every aspect-ratio dependence of a sealed pressure operator. The unchanged CNN
still sees the full grid. Generalization to other shapes and benefit from this
specific scale must be measured; no universal accuracy claim follows from shape
acceptance or global connectivity.

Input remains q=(rho/h)*native_residual divided by train-only Laplacian-RHS RMS.
Previous pressure participates in physical residual calculation/reconstruction,
not as a second network feature. h is a scalar cubic spacing. Counts change the
number of cells; world dimensions are counts*h. Recorded force effects reach the
pressure solve through the assembled preprojection residual. No force-widget,
tracer, boundary mask or new material input is introduced here.

The deployment retains uniform water density 1000 kg/m³, dynamic viscosity
0.001 Pa s, timestep 1/60 s, zero origin, sealed full domains and cubic spacing.
Its existing counts/spacing/cell-budget checks are unchanged. The numerical model
can accept positive dimensions within its declared cell budget; the deployment
has stricter supported limits. Shape acceptance alone does not qualify physics.

The existing v11 configurations, bank02 and all streams/loss weights stay fixed.
Each arm runs 5,000 CUDA optimizer updates, 50,000 network forwards and
425,703,026 forward cells. This matches network-call/cell work, not FLOPs: the
candidate adds coarse arithmetic and 4,096 parameters. Actual downloaded history,
checkpoint tensors, stream histories, closure and five exact CUDA-to-CPU artifact
pairs are checked after execution. No optimizer runs on CPU.

The assembler captures an explicit two-file variant plus the original model as
base_model.py. Source witness and current Tuldok declarations own those bytes.
The inference packager copies that same coarse.py and emits a closed v2 numerical
mapping with an explicit coarse boolean. The new reader has no v1 fallback;
historical packages keep their own immutable readers. Pretraining conversion
uses synthetic nonzero coarse coefficients plus retained control weights to
exercise the branch, independently of learned accuracy. Exact raw and two-call
pressure match on three grids; invalid mappings reach their intended rejection.

Fresh managed CPU restore precedes the exact first-frame fixture. Each full
development campaign has 74 scenes and 233 independent same-RHS pressure checks.
Paired AI/native trajectory pressure deviation is a separate measurement and
owns the population pressure gates. Keep original 58 / late-amended 8 /
minimum-8 failed 8 populations separate, and apply every retained comparator's
pressure, mean-call and max-call gate. Physical qualification uses the existing
32-call ceiling, 15-step native warm-up, previous-pressure history, residual
1e-6 relative / 1e-12 absolute, predicted divergence 1e-7 and mean-free pressure
bound 1e-6. No fallback/refinement or relaxed targets. Failure is retained, not
automatically retried. Unused holdouts are conditional on completed eligibility.

Stationary transfer uses consumed grids, spatial cosine modes and three pressure
amplitudes, with 1/4/8/12 calls: 252 rows and 756 forwards per model. Independent
analytic RHS eigenvalues, known signal energies and Parseval checks calibrate
the fields. Shared train-RHS diagnostics use 144 exactly matched pairs with
independent label/operator checks. Spatial mode indices are not temporal Hz;
these diagnostics do not qualify accumulated trajectory drift or a causal noise
floor. Absolute energies, signed low-mode coefficient errors and all tails are
retained. Unsigned high-band error energy cannot establish amplification.

Broad pools use 16 one-thread workers on cores 5–20, 2 GiB accounting each plus
8 GiB available reserve, sequentially; they may overlap the next CUDA fit.
Standalone raw CPU timings and paired worker timings run serially after pools
drain, two reversed-order repetitions on core 22. Raw timings exclude physical
input/reconstruction, residual/stopping, IPC and native work. Paired worker
timings include an additional same-request native diagnostic solve, capture,
correction, IPC and logging. Neither scope is standalone end-to-end AI solver
speed, and native branch time must not be subtracted as an estimate.

Optimizer, trainer, Study queue/handoff, CPU campaign and overall measurement
clocks remain separate; overlapping durations are not summed as wall time.
Original Tuldok catalog and Rheon desktop stay open. Their recorded pressure
preview is an observation aid, not a live velocity/tracer animation. Only source,
methods, bounded receipts, hashes, clocks and summary tables enter Git.
