# Detached correction training results

Nine fits from three starting checkpoints. C53 remains the retained reference. No model was promoted.

| Start | Arm | Fitted four pass | Worst fitted residual | Real four pass | Real eight pass | Capacity gate |
| --- | --- | --- | --- | --- | --- | --- |
| C53 | U | 5/8 | 2.29e-05 | 3/128 | 128/128 | fail |
| C53 | A | 6/8 | 3.21e-05 | 1/128 | 128/128 | fail |
| C53 | D | 5/8 | 2.61e-05 | 2/128 | 128/128 | fail |
| C44 | U | 6/8 | 3.71e-05 | 0/128 | 126/128 | fail |
| C44 | A | 5/8 | 2.23e-05 | 0/128 | 122/128 | fail |
| C44 | D | 5/8 | 4.55e-05 | 2/128 | 121/128 | fail |
| C37 | U | 4/8 | 0.000112 | 0/128 | 119/128 | fail |
| C37 | A | 4/8 | 0.000197 | 0/128 | 121/128 | fail |
| C37 | D | 4/8 | 0.000102 | 0/128 | 120/128 | fail |

U is the existing fitting objective; A adds attached intermediate supervision; D detaches that auxiliary history.

Capacity success is a specialist fitting result. The JSON report retains separate real and manufactured regression checks, paired comparisons, and source identities.

[Full results and qualification gates](results.json)
