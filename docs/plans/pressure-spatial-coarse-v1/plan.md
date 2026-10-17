# Spatial coarse correction

Status: matched managed CUDA Studies submitted; CPU qualification follows fresh restore. Owner/integration owner: root.
Authority: user's “continue with the next recommendation,” adopting the preceding
next-proposal.md. Fixed base fe00afb79f8cc3cef09d5a552ca700526037f0e5. New local
branch experiment/spatial-coarse-correction-20261017; no push or integration.

Test one learned spatial correction alongside the retained current-0.1 recipe.
The actual constructed control is linear, bias-free, without active GroupNorm or
SiLU: 370,016 parameters. The preceding proposal incorrectly inferred nonlinear
blocks from frozen source before configuration transforms. Preserve historical
records and record this correction here. The operator is linear within float32
roundoff; this is not a claim of exact numerical homogeneity at every amplitude.

The candidate adds a zero-initialized 64-to-64 matrix between four-bin-per-axis
cell-average restriction and linear coarse-centre interpolation. Scale output by
(longest grid count / 16)^2 and remove its mean. This supplies a nonconstant
global response without an analytic/native inverse at inference. It adds 4,096
parameters. Leave the original global-mean branch in place: removal is a separate
roundoff intervention, unnecessary for this controlled addition.

The model variant owns this architecture only. Existing captured base model owns
preprocessing, cubic scalar h, residual recurrence, physical operator and loss;
the shared assembler captures an explicit two-file model overlay before source
witness and Tuldok policy construction. No alternate catalog, service, runtime,
simulator, trainer or numerical acceptance owner is introduced. Source identity
and architecture witness identify the variant; configurations remain the existing
closed v11 recipe. Removing the coarse module deletes the experimental addition;
all baseline behavior remains at its existing owner. No framework registry or
general plugin mechanism is needed.

First verify constant/volume identities, nonconstant far-field response,
autodiff, zero initialization/RNG preservation, exact retained surviving tensors,
consumed rectangular/degenerate shapes, CPU cost and CUDA deterministic gradients.
Use already consumed fields only. Reject before full fits if prototype contracts
fail or measured median standalone CPU forward overhead exceeds 25% on any
consumed 16/24/32/40 cube. Record all timing repetitions and source/input hashes.
Do not reject a zero-initialized path merely for no untrained accuracy gain.

If admitted, freeze a fresh control and candidate, each 5,000 CUDA updates,
50,000 actual network forwards and 425,703,026 forward cells, same seeds, bank02,
normalization, original band feedback 0.03/current feedback 0.1, data and schedule.
Shared original Tuldok catalog owns Studies/Runs. Fresh exact CPU restoration and
first-frame qualification precede spatial diagnostics and the full 74-case/
233-independent-check campaigns. Apply all original population pressure, mean-
and max-call gates unchanged. Only an eligible frozen package may use the eight
unused whole scenes. No automatic promotion or extra sweep.

CPU work: 16 one-thread workers, cores 5–20, 2 GiB accounting/worker, 8 GiB
available reserve; broad pools sequential. Serial timing after drainage. CUDA
training and CPU inference use float32; existing physical arithmetic float64.
No native fallback/refinement. Original Rheon desktop and Tuldok catalog/UI
preserved; existing preview paths show recorded pressure, not live tracer flow.

Write set: this plan, experiments/pressure-spatial-coarse-v1, the narrow shared
assembler/package options, and the existing dependency-migration inference
packager/adapter numerical mapping. Review found that training capture alone
could not deploy the new tensor; qualify a conversion before training. Newly
generated packages use a closed v2 numerical mapping with explicit coarse flag;
there is no new-reader fallback to v1. Retained historical packages contain
their own unchanged reader bytes. No Tuldok/Rheon changes, historical rewrites,
weights/datasets/runtime/credentials in Git. Reuse the existing lab worktree; no
new registration. Retain branch for user review/continuation with immutable
commits until later directed integration or retirement. CPU clocks, CUDA clocks,
Study queue/handoff, resources, failures and reviews remain separately scoped.
Standards MCP snapshot:v1:7f7e294c-f67e-4194-bd5c-2dde99a7cb62 selected 22 policies,
zero unresolved facts. User continuation authorizes this reversible sequence.

Development decision: implement the bounded module and measure it. The uncertain
CPU cost can reject this design before ~1 hour of managed training/evaluation.
Exact accuracy gain requires a learned candidate; fixed-call diagnostic evidence
alone cannot qualify autonomous trajectories. If measurements reject the design,
retain the negative result and prepare a new proposal without further training.
