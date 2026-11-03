# Fixed neural viscosity: four full-state fits

All four ordinary managed CUDA fits, exact CPU restores and development evaluations complete. The same 243 learned viscosity coefficients are frozen from initialization. Every model supervises 36,000 physical states; different optimizer counts and internal work are reported, with no equal-FLOP claim.

| Model | Grid group | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |
| --- | --- | --- | --- | --- | --- |
| b1-fixed-viscosity | small | single_block | True | 0.0003310321 / 0.001430861 / 0.0001327859 / 1.749072e-05 | True |
| b1-fixed-viscosity | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.279604e-07 | True |
| b1-fixed-viscosity | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.470556e-05 | True |
| b1-fixed-viscosity | small | rollout_240 | True | 0.001539633 / 0.003721229 / 0.0005810633 / 0.002930948 | True |
| b1-fixed-viscosity | 32cubed | single_block | True | 0.0006200722 / 0.0006405184 / 0.0003365973 / 3.252186e-07 | True |
| b1-fixed-viscosity | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491299e-07 | True |
| b1-fixed-viscosity | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 5.127321e-07 | True |
| b1-fixed-viscosity | 32cubed | rollout_240 | True | 0.00026237 / 0.0006733132 / 0.0003365973 / 0.0001420479 | True |
| b2-fixed-viscosity | small | single_block | True | 0.0002400306 / 0.001685576 / 0.0001327859 / 2.015908e-05 | True |
| b2-fixed-viscosity | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.838229e-07 | True |
| b2-fixed-viscosity | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.87977e-05 | True |
| b2-fixed-viscosity | small | rollout_240 | True | 0.0007612313 / 0.003396646 / 0.0003784483 / 0.001276272 | True |
| b2-fixed-viscosity | 32cubed | single_block | True | 0.0006200678 / 0.0006412402 / 0.0003365973 / 7.229575e-07 | True |
| b2-fixed-viscosity | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.49181e-07 | True |
| b2-fixed-viscosity | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 6.989596e-07 | True |
| b2-fixed-viscosity | 32cubed | rollout_240 | True | 0.00026237 / 0.0008830882 / 0.0003365973 / 9.070661e-05 | True |
| b3-fixed-viscosity | small | single_block | True | 0.0002010863 / 0.001818919 / 0.0001327859 / 3.45316e-05 | True |
| b3-fixed-viscosity | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 1.040963e-06 | True |
| b3-fixed-viscosity | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.966534e-05 | True |
| b3-fixed-viscosity | small | rollout_240 | True | 0.001222302 / 0.003531768 / 0.0003703838 / 0.001574887 | True |
| b3-fixed-viscosity | 32cubed | single_block | True | 0.000620064 / 0.0006409607 / 0.0003365973 / 1.178118e-06 | True |
| b3-fixed-viscosity | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.491619e-07 | True |
| b3-fixed-viscosity | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 6.044687e-07 | True |
| b3-fixed-viscosity | 32cubed | rollout_240 | True | 0.00026237 / 0.0008343505 / 0.0003365973 / 0.0001159064 | True |
| b4-fixed-viscosity | small | single_block | True | 0.0001992496 / 0.001988543 / 0.0001327859 / 4.519954e-05 | True |
| b4-fixed-viscosity | small | rollout_12 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 8.443739e-07 | True |
| b4-fixed-viscosity | small | rollout_48 | True | 0.000101005 / 0.000290779 / 0.0001327859 / 5.493862e-05 | True |
| b4-fixed-viscosity | small | rollout_240 | True | 0.0006908277 / 0.003367723 / 0.0003478145 / 0.001861791 | True |
| b4-fixed-viscosity | 32cubed | single_block | True | 0.0006200872 / 0.0006409538 / 0.0003365973 / 2.579902e-06 | True |
| b4-fixed-viscosity | 32cubed | rollout_12 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.490002e-07 | True |
| b4-fixed-viscosity | 32cubed | rollout_48 | True | 0.00026237 / 0.0003978269 / 0.0003365973 / 1.01797e-06 | True |
| b4-fixed-viscosity | 32cubed | rollout_240 | True | 0.00026237 / 0.000732322 / 0.0003365973 / 0.0001980088 | True |

Every horizon and all 240 own-history steps use the original equation/field and near-zero criteria. These are accuracy and custody results; actual CPU speed, qualified execution variants, replicated seeds, reserved trajectories and sustained application/event costs remain separate.

[Exact work, controls, raw evidence and hashes](fixed-viscosity-family-results.json) · [Registered family](fixed-viscosity-family-plan.md)
