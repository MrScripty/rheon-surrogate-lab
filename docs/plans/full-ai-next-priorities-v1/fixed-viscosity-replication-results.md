# Fixed neural viscosity: two additional seeds per selected recipe

All six additional-seed managed CUDA fits, exact CPU restores and development evaluations complete. The same 243 learned viscosity coefficients are frozen from initialization. Every model supervises 36,000 physical states; different optimizer counts and internal work are reported, with no equal-FLOP claim.

| Model | Grid group | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- | --- | --- |
| b1-fixed-viscosity-s2 | small | single_block | True | 0.0001747069 / 0.001385881 / 0.0001327859 / 6.516288e-06 | True |
| b1-fixed-viscosity-s2 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 9.849898e-07 | True |
| b1-fixed-viscosity-s2 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 3.69971e-05 | True |
| b1-fixed-viscosity-s2 | small | rollout_240 | True | 0.00109509 / 0.003322732 / 0.000392928 / 0.001138108 | True |
| b1-fixed-viscosity-s2 | 32cubed | single_block | True | 0.0006200604 / 0.0006414155 / 0.0003365973 / 5.124505e-07 | True |
| b1-fixed-viscosity-s2 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.528397e-07 | True |
| b1-fixed-viscosity-s2 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.525291e-06 | True |
| b1-fixed-viscosity-s2 | 32cubed | rollout_240 | True | 0.00026237 / 0.0007214918 / 0.0003365973 / 0.0001313096 | True |
| b2-fixed-viscosity-s2 | small | single_block | True | 0.0001522235 / 0.001478714 / 0.0001327859 / 1.650444e-05 | True |
| b2-fixed-viscosity-s2 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 9.557369e-07 | True |
| b2-fixed-viscosity-s2 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 3.45332e-05 | True |
| b2-fixed-viscosity-s2 | small | rollout_240 | True | 0.001369267 / 0.003559932 / 0.000413557 / 0.001348292 | True |
| b2-fixed-viscosity-s2 | 32cubed | single_block | True | 0.0006200534 / 0.0006413776 / 0.0003365973 / 5.484645e-07 | True |
| b2-fixed-viscosity-s2 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491195e-07 | True |
| b2-fixed-viscosity-s2 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 7.005119e-07 | True |
| b2-fixed-viscosity-s2 | 32cubed | rollout_240 | True | 0.00026237 / 0.0008098702 / 0.0003365973 / 7.643323e-05 | True |
| b4-fixed-viscosity-s2 | small | single_block | True | 0.0002261412 / 0.001838726 / 0.0001327859 / 3.703655e-05 | True |
| b4-fixed-viscosity-s2 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 1.003466e-06 | True |
| b4-fixed-viscosity-s2 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.167665e-05 | True |
| b4-fixed-viscosity-s2 | small | rollout_240 | True | 0.00143067 / 0.003328111 / 0.0003881491 / 0.001199506 | True |
| b4-fixed-viscosity-s2 | 32cubed | single_block | True | 0.0006200659 / 0.0006414428 / 0.0003365973 / 7.527608e-07 | True |
| b4-fixed-viscosity-s2 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.490703e-07 | True |
| b4-fixed-viscosity-s2 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 5.643344e-07 | True |
| b4-fixed-viscosity-s2 | 32cubed | rollout_240 | True | 0.00026237 / 0.0006791875 / 0.0003365973 / 6.525862e-05 | True |
| b1-fixed-viscosity-s3 | small | single_block | True | 0.0001323936 / 0.001548464 / 0.0001327859 / 8.70718e-06 | True |
| b1-fixed-viscosity-s3 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.7103e-07 | True |
| b1-fixed-viscosity-s3 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 3.859309e-05 | True |
| b1-fixed-viscosity-s3 | small | rollout_240 | True | 0.0009323496 / 0.004348945 / 0.0003935702 / 0.001354668 | False |
| b1-fixed-viscosity-s3 | 32cubed | single_block | True | 0.000620061 / 0.0006410043 / 0.0003365973 / 3.534387e-07 | True |
| b1-fixed-viscosity-s3 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491106e-07 | True |
| b1-fixed-viscosity-s3 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 7.812492e-07 | True |
| b1-fixed-viscosity-s3 | 32cubed | rollout_240 | True | 0.00026237 / 0.0007573958 / 0.0003365973 / 0.0001040247 | True |
| b2-fixed-viscosity-s3 | small | single_block | True | 0.0001781903 / 0.001832139 / 0.0001327859 / 1.769348e-05 | True |
| b2-fixed-viscosity-s3 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.066875e-07 | True |
| b2-fixed-viscosity-s3 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 4.342643e-05 | True |
| b2-fixed-viscosity-s3 | small | rollout_240 | True | 0.0008139454 / 0.003945515 / 0.0003963127 / 0.001125643 | True |
| b2-fixed-viscosity-s3 | 32cubed | single_block | True | 0.000620061 / 0.000640357 / 0.0003365973 / 9.66027e-07 | True |
| b2-fixed-viscosity-s3 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.490208e-07 | True |
| b2-fixed-viscosity-s3 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 9.961902e-07 | True |
| b2-fixed-viscosity-s3 | 32cubed | rollout_240 | True | 0.00026237 / 0.000552703 / 0.0003365973 / 0.0001400897 | True |
| b4-fixed-viscosity-s3 | small | single_block | True | 0.0002286392 / 0.001860153 / 0.0001327859 / 4.05695e-05 | True |
| b4-fixed-viscosity-s3 | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 7.801023e-07 | True |
| b4-fixed-viscosity-s3 | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.275376e-05 | True |
| b4-fixed-viscosity-s3 | small | rollout_240 | True | 0.0007165861 / 0.003020889 / 0.0003336428 / 0.001826008 | True |
| b4-fixed-viscosity-s3 | 32cubed | single_block | True | 0.000620057 / 0.0006406946 / 0.0003365973 / 3.159311e-06 | True |
| b4-fixed-viscosity-s3 | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.489263e-07 | True |
| b4-fixed-viscosity-s3 | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.331382e-06 | True |
| b4-fixed-viscosity-s3 | 32cubed | rollout_240 | True | 0.00026237 / 0.0008639509 / 0.0003365973 / 0.0002638107 | True |

Every horizon and all 240 own-history steps use the original equation/field and near-zero criteria. These are accuracy and custody results; actual replica CPU speed and qualified execution variants, reserved trajectories and sustained application/event costs remain separate.

[Exact work, controls, raw evidence and hashes](fixed-viscosity-family-results.json) · [Registered family](fixed-viscosity-family-plan.md)
