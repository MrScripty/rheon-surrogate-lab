# Execution ledger

2026-10-15: User authorized the prepared sequence and durable method/result
retention. Clean base 18159af; new experiment branch created. Existing managed
trainer and source capture path inspected. Previous no-GroupNorm fitted residual
0.0003982221868309573, validation median 0.0828159014298824; 0/128 qualified.
This is a fitting improvement, not improved held-out accuracy. No new runs yet.

Initial focused test: 4/5 passed; superposition's per-cell tolerance rejected
5/315 near-cancelling float32 values (largest absolute discrepancy 2.15e-7).
Use a correction-vector norm-relative 3e-5 comparison for this algebraic
arithmetic check; unchanged 1e-6 physical acceptance is evaluated independently.
The operator, preprocessing and prepared gradient checks already passed.

Focused checks: all five residual tests and seven historical conditioning tests
pass. First registration returned Tuldok HTTP 409 because an immutable project
ID already referred to the prior snapshot. Assign a new snapshot ID, retaining
the old project and the same catalog/root. No Run was submitted by that failed
registration. The initial observer used system Python without NumPy; restart
with the existing laboratory virtual environment. No source/model change.

Source 725b100 captured physical-residual smoke and full nonlinear/linear trials.
CUDA smoke Run 455586b9f3c24a06ba0b50df1d932c12 completed two actual updates;
float32 params/gradients/Adam moments on RTX 5090 Laptop GPU, deterministic,
AMP/TF32 off. Fresh CPU restore remains pending. Distinct project snapshot
54e60e2d5f8a4267a002d8d11e5e2666 registered in the original catalog.

CPU diagnostics commit fc2b160: independent spectral regression passes; old
no-GroupNorm validation audit completed on 128 states, 23 one-thread workers
(22 observed cores), 7.59s, no CUDA. Summary retained in reports. Full analytic
control is not AI success; eight raw learned proposals do not converge well.

Physical-residual smoke accepted: train 455586b9f3c24a06ba0b50df1d932c12 and
CPU restore 87786368c8794a8eb827a7f35a9db458 completed/drained, reopened and
whole artifacts redownloaded with parity. CPU has no optimizer or CUDA context,
exact fresh prediction inventory. The full nonlinear 10,000-update input ablation
is now executing through the same catalog. Exactly next scientific slice remains
its native evaluation, followed by the matched linear architecture comparison.
