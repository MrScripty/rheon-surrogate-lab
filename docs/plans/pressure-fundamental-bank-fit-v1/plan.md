# Shared fundamental-mode fitting capacity

Status: Implementing. Jeremy authorized continuing the accepted C62/C63 next
proposal. Canonical settings: `experiments/pressure-fundamental-bank-fit-v1/config.json`.
Branch `experiment/pressure-fundamental-bank-20261017` is private, based on
`b3e64e726cef931e4f91c80d1fcc8f1ebbae1c27`. Jeremy owns integration after review;
retain for experiment review, no automatic merge/rebase/synchronization, retire
after integration or explicit disposition. C53 remains baseline.

C62 fitted one fundamental mode but regressed elsewhere. C63 gave mixed training
more low-mode exposure, improving those modes against C60 without satisfying C53
worst-case guards. This study tests shared fitting capacity without competing
examples, before another mixed training run or architecture change.

1. C64 fits 24 manufactured cases: the eight recorded hard grids and the three
   fundamental axes. Original train-only correction RMS sets amplitude; original
   water scalars and normalization are retained. Cubic spacing is one scalar h.
   The network, physical operator, precision and four-pass objective match C62;
   no teacher, real-frame, primary, direction or current companion loss is used.
2. Every case gets exactly 200 optimizer exposures, balanced with a private RNG.
   C64 makes 4,800 CUDA float32 updates, below the existing 5,000-update admission
   limit. Fresh Adam, clipping norm 1, no weight decay. Each case gets 100 exposures
   at LR 1e-4, 50 at 1e-5, 25 at 1e-6, 15 at 1e-7 and 10 at 1e-8. Change rates
   after full sampling cycles. Early equation feedback remains 0.05, teacher zero.
   This does not claim a fully matched comparison against historical C62's budget.
3. Before admission, qualify independent eigenvalue labels/operator defect <=1e-10,
   cubic grids, exact initialization and default recurrence, finite nonzero CPU
   autograd, exact sampling/exposure/residency, and no optimizer updates on CPU.
   Publish a final composite receipt only after every check. Use the existing
   visible Tuldok catalog and Pumas runtime, journalled CUDA training and fresh
   CPU restore; exact published artifact parity is required.
4. On restored CPU weights measure every fitted case at 1/2/4/8 passes. The capacity
   gate requires every case <=1e-6 at four/eight and no worse one/two-pass residual
   than C53 on each fitted case. Record per-case learning traces every ten complete
   sampling cycles. Measure 128 consumed real requests and existing 240 manufactured
   cases separately as regression guards. A specialist is not automatically deployed.
5. If C64 fails this gate, qualify and run C65 (three axes, first grid) and C66
   (X fundamental, eight grids), each from C53. Keep 200 exposures and the same LR
   allocation per case: 600 and 1,600 updates respectively. These isolate supported
   subsets; shared parameter optimization still differs and no architectural
   impossibility is inferred from a bounded fit. The existing scheduler owns one
   CUDA slot and may overlap independent CPU restores/readback. No automatic replay.
6. If C64 succeeds, do not run the smaller fits. Propose a separate retention test
   when real-frame feedback returns. If it fails, use the subset results to choose
   the next diagnosis. No longer fit, new CFD scenes, original untouched holdouts,
   physical-threshold relaxation or model promotion is authorized by a failed screen.

Write set: new experiment and matching docs directories; an explicit qualification
key parameter in the shared controller owner and fitted-evaluator parameter in
the shared CPU owner, with focused verification;
narrow C64-C66 reservations and report-link edits in the existing chart helper.
Preserve all unrelated dirty work, historical captures, weights and results. Do not
commit weights, complete simulation fields or credentials. Commit workflow changes
separately from model/experiment changes, following the cached coding-standards
MCP snapshot `snapshot:v1:ffcfc3f8-b52a-434e-b57f-d259a0355711`.

Record exact configurations, code/input/weight identities, actual examples and
network/cell work, trainer/workflow/CPU elapsed times, peak memory, Run custody,
failed attempts and CPU numerical results. Independent CPU workers use one thread
and distinct cores, max three; preserve 12 GiB host reserve plus 2 GiB per worker.
Fitting-bank resident tensors <=8 MiB; each CUDA trainer wall cap 1,800 seconds.

Next slice: C64 completed CUDA training, fresh CPU restore, readback and CPU
qualification. Four-pass fitting passed 15/24; eight-pass passed 24/24; early-pass
case guards failed. Immutable evidence: `combined-fundamentals/capacity-decision.json`.
Qualify and execute C65/C66 through the existing scheduler, then compare their
restored CPU gates. C53 remains baseline; C64 real-request worst errors regressed.
