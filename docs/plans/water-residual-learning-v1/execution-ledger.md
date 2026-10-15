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
