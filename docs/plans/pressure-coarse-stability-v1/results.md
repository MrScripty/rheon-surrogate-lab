# Clipping integration and capped CUDA stability results

The overflow-only clipping repair is integrated and qualified. One managed CUDA
screen completed **1500 optimizer updates and1500 original CUDA finite-state
checks**. It repaired the native float32 norm overflow only at1398 and1399.
All13 shared logged losses through1300 match the preceding unrepaired diagnostic
exactly. No longer fit or pressure-accuracy evaluation was run.

This is stability evidence for the original5000-update configuration's first1500
updates. It is not completed training, a deployable final model, or improved
pressure accuracy. The next stage remains a separately recorded full fit and
CPU restore/physics qualification.

## Method and exact scope

Fixed base84525d986cdaba1b5252802e07dc6b43dee9b701; branch
experiment/coarse-clipping-stability-20261017. Qualified source freeze
5059dbca3f043c9e36421ccb2917d31242adb8ce.

The canonical train.one_update imports gradient_clipping.clip_l2_ and calls it
where native clipping previously ran. It calls native Torch first with unchanged
max_norm1 and error_if_nonfinite=True. Normal finite/no-op paths retain the native
operation and exact gradient bits, including absent gradients. Only the specific
native nonfinite L2 norm error may recover when all present gradients are
colocated finite float32 and the max-scaled float32 final norm is representable.
NaN/Inf components, unrepresentable total and unrelated errors retain rejection
before modifying gradients. Model/Adam dtype remains float32; original float64
physical tensors and detached diagnostic reductions are unchanged.

General Project and throughput assemblers both capture the new dependency. The
historical trainer is retained as baseline_train.py in the ignored new capture;
verification permits exactly the clipping import/call edits and new qualified
helper, not any other numerical change. Existing historical captures/results
remain unchanged. No compatibility aliases or package migrations were added.

The actual new capture has36 recipe sources and9input references. Configuration,
model/loss/physics/preprocessing, inputs, normalization, seeded initial tensors,
learning-rate schedule and five full5000 sampling arrays are retained exactly.
The separate entry cap raises DiagnosticLimit before update1501, after1500
original CUDA gradient/Adam checks. It never takes the completed-fit/CPU restore
path and cannot publish an eligible final pressure model.

## Qualification

[Final integration03](reports/integration-qualification-03.json) passed in
3.305426675s internally (4.019907291s command).
Two original vs two repaired/observed updates on a consumed real frame with five
feedback streams preserve losses, weights, gradients, Adam and CPU/CUDA RNG
exactly. The integrated observer caps before a third update after two successful
original CUDA checks.

Backward hooks then supply explicit fixtures at the real one_update clipping
seam. The two observed per-parameter norm scalars reproduce native rejection
before Adam and complete one finite repaired Adam step with independently checked
unit norm/direction. These are scalar-derived fixtures, not recovered original
model gradient tensors. NaN, Inf and unrepresentable totals preserve identical
native messages and unchanged weights/gradients with zero Adam steps. An
unrelated RuntimeError preserves exception identity. Four ordinary qualification
optimizer steps plus one repaired fixture step are separate from the1500 managed
fitting updates; no simulation advances.

[Arithmetic02](reports/arithmetic-qualification-02.json) passed on CPU and CUDA in
1.609823823s internally: per device five ordinary cases, three
missing/empty-gradient cases, three representable overflow cases and three
negative cases. Zero optimizer/forward/simulation calls. General source-package
v2/v10 isolated import regression passed (one test, two subtests;2.371s test time).

Preliminary integration01 imported the shared base model ahead of the coarse
capture, so it is limited preliminary evidence and was not admitted. Integration02
used the correct model but precedes the final missing-gradient repair. Only03
and arithmetic02 qualify the final sources. The first isolated-import fixture
failed because its temporary directory was shallower than captured policy allows;
the corrected nested fixture02 passes. Retained preliminary receipts/logs remain
linked in the ledger; preliminary outer command timings were overwritten by the
wrapper and are unavailable. The final command journals are retained exactly.

## Managed result

Original Tuldok catalog a7da4a8de53541a2a0e2aad08005bb71,
Project a629e335f28f471688fbe0cd1144b164,
Study f0df577b7c07486cba52ec597c48f99c, Run e8f26974fa8a4705ba4d63cf7179a3ab.
The Project is visible at http://127.0.0.1:8091/?workspace=models as
“water-coarse-candidate clipping stability (cap1500)” (UI spacing may differ).

| Update | Repaired pre-clip float32 norm | Applied clip coefficient |
| --- | --- | --- |
| 1398 | 3.04450657558e+19 | 3.28460464236e-20 |
| 1399 | 1.73892531415e+20 | 5.75067836412e-21 |

Both rejected ordinary norms were infinity despite finite gradient components.
First repair1398 reproduces the preceding diagnostic's exact stopping update;
its float32 repaired total agrees with the earlier diagnostic float64 reference
3.044506589355366e19. All13 logged losses100..1300 are exact; this is sampled
logged-prefix evidence, not a stored per-update tensor comparison of1397steps.
No additional repairs occurred through1500. The logged loss at1500 was
3.28596717256; logged checkpoints use different training
samples, so their values do not demonstrate a monotonic improvement or physical
accuracy. Large finite losses/gradients remain a concern for longer fitting.

Tuldok records **failed/drained, exit1**, because the explicit cap intentionally
raises before complete training. The classified result is
1500_updates_finite_with_clipping_repair, not numerical failure. Custody proves drainage,
ECHILD and spawn closure. Study is closed, artifacts published0, CPU restore0,
new simulations0, holdouts0, promotions0. The controller's exit0 means terminal
readback and closure succeeded. [Analysis](reports/analysis.json) binds the actual
admission, Project, material, spec, all36 staged sources and custody streams.

## Time, environment and resources

| Measurement | Seconds | Meaning |
| --- | --- | --- |
| Assembly | 1.304406756 | Source/declaration capture; no optimization |
| Capture verification | 3.119601319 | CPU input/initialization/full-stream construction |
| Instrumented fitting loop | 173.953927779 | Initial-tensor event through cap; includes probes and periodic diagnostics |
| Managed Run | 224.220248 | Start through drained terminal state |
| Managed command | 387.191292793 | Project/model/Study admission, queue, execution, readback and closure |
| Custody collection | 13.249415875 | Read-only source/material/receipt/log joins |
| Analysis command | 0.032860639 | Verified log classification |

These clocks overlap and must not be summed. Optimizer-only time is unavailable.
This instrumented screen is not a throughput comparison with prior Runs.

Pumas-managed Python3.14.7, Torch2.14.0+cu132, NumPy2.5.3;
NVIDIA GeForce RTX 5090 Laptop GPU. Runtime manifest SHA256
b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682.
Qualification/CPU construction use CPU23 affinity and one library thread;
managed training resources/environment are retained in the immutable recipe and
private actual execution record. The managed controller uses app-env Python3.12
and the original Owner SDK/session; no standalone trainer or alternate catalog.

10 sparse host samples: minimum available RAM
53609373696bytes
(49.93GiB), maximum observed GPU memory
1926MiB and utilization
57%. These are machine-wide observations,
not peaks, model-only memory or a GPU throughput qualification. No OOM occurred
in the observed task. [Completion](reports/completion.json) verifies owned
controller/custodian absence and original Tuldok/Rheon processes alive. Bounded
one-shot activity updates used the recorded qualified control pressure preview;
no background publisher remains and no new AI result is displayed as simulation.

Exact commands and clocks are in reports/*command*.json, capture-clocks.json and
collection-clocks.json. Local orchestration sources are hashed in
[command-sources.json](reports/command-sources.json), retained under ignored
runs/pressure-coarse-stability-v1/command-sources. No weights, datasets, private
internal execution/environment record or credentials belong in Git.

See [execution ledger](execution-ledger.md), [review](review.md), and
[next proposal](next-proposal.md). Longer training has not started.
