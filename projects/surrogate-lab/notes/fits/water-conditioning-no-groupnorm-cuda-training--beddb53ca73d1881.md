# water-conditioning-no-groupnorm · CUDA training

Retrospective source review · fit-beddb53ca73d1881

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. One call proposes pressure; it does not advance a simulation. GroupNorm is replaced by Identity; its affine-free form had no trainable parameters. The rho/dt channels use the fixed physical units described in the purpose; residual/output amplitude is not normalized per sample.

**Purpose.** Remove parameter-free GroupNorm from the scalar-conditioned network while preserving its trainable tensor inventory.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}], seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. The equal-weight objective combines normalized squared pressure-field and FV equation errors. Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. The final configured update is selected; no optimizer-resume state is saved.

**Observed outcome.** The last sampled training loss is 7.945501438495157e-08; captured validation objective 0.020027807479606485 initially and 0.01099678950640433 finally. On the common 128 native pressure requests, worst mean-free pressure relative L2 is 0.20646775842106907 and equation relative L2 is 0.29309019363418942. These common errors differ from the training objective. This review adds no timing measurement or drift result.

**Lesson.** Removing GroupNorm substantially improves common pressure error relative to C14 and lowers the sampled training loss, with the same 371,729 trainable parameters. The execution strategy also changes to resident inputs and combined finite checks, so training throughput is not attributable solely to GroupNorm. It remains a one-frame fitting experiment, without broad training or own-history qualification.

**Reproduction.** Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "beddb53ca73d18816b88cdd740e08b779a58d1275d6b53c8cefc0a747e86f27d"
  },
  "history": {
    "path": "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/history.json",
    "sha256": "d7bcd3705c7ee0cbef9f27858171025a79011d2b38676a5b85c4e568146f00a7"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/config.json",
      "sha256": "a3ddd92f9873ec1e737931f6509ebfefa0313d231af7c3337d543a531572911d"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/model.py",
      "sha256": "2c3f185503c880a6ab2dc88ba453fa93bd84fe7bad77cbc3a109d9c08b63fa25"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/policy.py",
      "sha256": "32d6fdeff5af083c51035972ac447164c0089b60e7bbe506f0c8cb2e9ab2bd7e"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/qualification.json",
      "sha256": "3b97aacc295d425047f1179401b7ca866fe45a0f04022f321204c47a03d7ba36"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/train.py",
      "sha256": "8ed26467bc7f42d1f0e09a5eadba15797e6f950dfa1322eea91d0a9a4e5f6731"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/no-groupnorm/project/source-witness.json",
    "sha256": "dd3f4cb325f1ac400a00cc06793cc2b199366e0c49269181b9fb7dd988e5ca09"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-conditioning-v1/no-groupnorm/project/inputs/validation_splits.json",
      "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5",
      "slot": "validation_splits",
      "bytes": 372865
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-beddb53ca73d1881",
    "kind": "training_attempt",
    "name": "water-conditioning-no-groupnorm · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-conditioned-cubic-no-normalization-v1",
      "input_channels": 7,
      "encoder_widths": [
        8,
        16,
        24,
        32
      ],
      "bottleneck_width": 48,
      "decoder_widths": [
        32,
        24,
        16,
        8
      ],
      "kernel": [
        3,
        3,
        3
      ],
      "activation": "silu",
      "dtype": "float32",
      "pressure_dtype": "float64",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "beddb53ca73d18816b88cdd740e08b779a58d1275d6b53c8cefc0a747e86f27d",
    "receipt_paths": [
      "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/no-groupnorm/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/71ea5e29ef68466d9bf7ed4a55f17b30/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-71ea5e29ef68466d9bf7ed4a55f17b30/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "71ea5e29ef68466d9bf7ed4a55f17b30",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-no-groupnorm · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "a3ddd92f9873ec1e737931f6509ebfefa0313d231af7c3337d543a531572911d",
      "inputs": {
        "dataset": {
          "bytes": 2275660,
          "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"
        },
        "dataset_extra": {
          "bytes": 2278650,
          "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"
        },
        "reserved_test_dataset": {
          "bytes": 2561142,
          "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"
        },
        "reserved_test_dataset_extra": {
          "bytes": 2564086,
          "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"
        },
        "reserved_test_splits": {
          "bytes": 339677,
          "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"
        },
        "splits": {
          "bytes": 374928,
          "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
        },
        "validation_dataset": {
          "bytes": 2209622,
          "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"
        },
        "validation_dataset_extra": {
          "bytes": 2212566,
          "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"
        },
        "validation_splits": {
          "bytes": 372865,
          "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"
        }
      },
      "normalization_sha256": "97714ab1aa6d0ebf69026925620254f7c2b0a4b8e083e1a387323f877c938378",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8",
      "source_witness_sha256": "dd3f4cb325f1ac400a00cc06793cc2b199366e0c49269181b9fb7dd988e5ca09"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "1446d88254579135ce01f0f0c67a720cd0e338d5cfce8e7aad3f2f7b77314975",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "cb46452ce23afff84984383c5345ce4aafffe32b6a891dc2ae1954f2b7dc165e",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "d6e3df9d999154a986bee3029f3bc55987820a69af2ed25356374897593a14b0",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "e975d2b3fb382d84b351cdef8866b067ba495860e15da51bf7e8ac5bbfccb347",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "d846fc5ad32655ae718e795730d8128749282415a2545ff4194fc88f205259f2",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "7482b858c6098ade1fc12e4c2cc505cf666e6e30a298e34e277a07a49efbdd7e",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-6000.npz",
        "sha256": "01e5714ec1019719e2b476c20e7dfb8721af3acf163e178db50c2afce9ac6d00",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "dc5fae494015711a2c8aef95f59492798963d31c4856cca56f9a22fd4921fa73",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "d6e3df9d999154a986bee3029f3bc55987820a69af2ed25356374897593a14b0",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2229993472,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/14"
    ],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": null
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/no-groupnorm/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "beddb53ca73d18816b88cdd740e08b779a58d1275d6b53c8cefc0a747e86f27d",
    "bytes": 9472
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C15": {
        "name": "no-groupnorm",
        "weights_sha256": "d6e3df9d999154a986bee3029f3bc55987820a69af2ed25356374897593a14b0",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.2930901936341894,
          "pressure_mean_free_relative_l2": 0.20646775842106907,
          "pressure_absolute_rms": 22.410963071717937,
          "pressure_absolute_max": 76.48626299083449
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0050",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              5
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.28897286715808873,
              "pressure_mean_free_relative_l2": 0.20646775842106907,
              "pressure_absolute_rms": 22.410963071717937,
              "pressure_absolute_max": 76.48626299083449
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.2930901936341894,
              "pressure_mean_free_relative_l2": 0.20505665584882568,
              "pressure_absolute_rms": 7.36503776412368,
              "pressure_absolute_max": 19.28428184956212
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C15/results.json",
          "sha256": "091340bc51a44ec26bedfd119a35b22aa228cdc1355b5702947992f0fe637d20"
        },
        "CNN_operation_AST_preserved": null,
        "CPU_neural_forward": false
      }
    },
    "qualified_latency": false
  },
  "interpretation": {
    "card_status": "retrospective_evidence_draft",
    "architecture_description": "Recorded fields; source semantics still require review.",
    "purpose_and_lesson": "No causal lesson or completion claim is inferred from a training receipt.",
    "reproduction": "Source/data/checkpoint identities retained. Clean reproduction remains unverified.",
    "historical_original_Run_observations": "References only; no native Run, Trial, ancestry or current execution authority is created."
  }
}
```

## Exact original receipt bytes

```json
{"architecture":{"activation":"silu","amplitude":"historical fixed training RMS","block_normalization":{"name":"Identity","position":"replace only parameter-free GroupNorm"},"bottleneck_width":48,"conditioning":"scalar-units","convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","scalar_feature_units":{"dt_s":0.016666666666666666,"rho_kg_m3":1000.0},"schema":"rheon-conditioned-cubic-no-normalization-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"a3ddd92f9873ec1e737931f6509ebfefa0313d231af7c3337d543a531572911d","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"97714ab1aa6d0ebf69026925620254f7c2b0a4b8e083e1a387323f877c938378","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"dd3f4cb325f1ac400a00cc06793cc2b199366e0c49269181b9fb7dd988e5ca09"},"elapsed_s":129.90740199701395,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.653132920864664e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":14499328,"peak_reserved_bytes":33554432,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"1446d88254579135ce01f0f0c67a720cd0e338d5cfce8e7aad3f2f7b77314975"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"cb46452ce23afff84984383c5345ce4aafffe32b6a891dc2ae1954f2b7dc165e"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d6e3df9d999154a986bee3029f3bc55987820a69af2ed25356374897593a14b0"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"e975d2b3fb382d84b351cdef8866b067ba495860e15da51bf7e8ac5bbfccb347"},"checkpoint-4000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d846fc5ad32655ae718e795730d8128749282415a2545ff4194fc88f205259f2"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"7482b858c6098ade1fc12e4c2cc505cf666e6e30a298e34e277a07a49efbdd7e"},"checkpoint-6000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"01e5714ec1019719e2b476c20e7dfb8721af3acf163e178db50c2afce9ac6d00"},"checkpoint-8000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"dc5fae494015711a2c8aef95f59492798963d31c4856cca56f9a22fd4921fa73"},"history.json":{"bytes":73846,"role":"training_diagnostics","sha256":"d7bcd3705c7ee0cbef9f27858171025a79011d2b38676a5b85c4e568146f00a7"},"normalization.json":{"bytes":1186,"role":"preprocessing","sha256":"97714ab1aa6d0ebf69026925620254f7c2b0a4b8e083e1a387323f877c938378"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"a1b5166eb14e1bf399b7c9393f5c58e0af463372dde77c01926bb67005050400"},"profile.json":{"bytes":123,"role":"training_diagnostics","sha256":"9b7e7aeae3d2ee41e4476663f3b360f1ad1021d627aa19ec34aa969bfe5f1f4b"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":838,"role":"result_summary","sha256":"663a65dfdd87a9d1266d511f4d5435c0240a6224138ef7fcd11c54e515fa2784"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d6e3df9d999154a986bee3029f3bc55987820a69af2ed25356374897593a14b0"}},"per_case_inference_s":[0.013422966003417969,0.008559879992390051,0.00247021799441427,0.003815180010860786,0.002577687002485618,0.0015493599930778146,0.003150579985231161,0.0022566840052604675,0.002421201003016904,0.0025207849976141006,0.002728283987380564,0.0053775440028402954,0.0020196050172671676],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"94d8c098359a00a914d4134fecc656dcb4a01a199df6426687eb7e284b4bfc44","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"837e6a3b0756dbbcf96f210fc78309e6d8b71c032f7d6514292312a269a91a56","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"5127ba5f5084cc688da7065b6233fdaef94e94dc117457d962e998972f621f37","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"aa27d99235c9a1b72e1ae9b91d382381e749d0810743465205fba5ef03f56fb3","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"d476aa111fbfeebbc148daa5098d97a6172cd9f2bcc060352749add29f67fafd","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"d9e931ffe50533d95ecd2e64d9322925756c52ac3060a957f96c16384f2dcd7f","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"cbd3a3a230171d89b74eb0f65e9c6cb2d326039ce4b5847ccb6d84fd38a98b42","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"2dfa74ef5d1b720b600a9c4452a8107e050acb6440817101fa1254d1a558daf2","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"d5bc00bdea4eac1244488519621adcfe3416783c568d12b5f01de0202331ad8a","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"595386cba1fe6111560e4bb99ed7d19aa4e627d7f0ba2aeb87dfc6907eeed1e3","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"9c8e05d720c5731d5124b4eff32a376cb639a53209ac4dffaa5d338d371729c9","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"aa1005678bbf2009e4ed37e2fb143810af1381758c66d9b02037b36680e5be83","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"cca09871a05b2c39277e1b24955d74c6964aec086f74b1ac344fb0aac76d8ace","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2229993472,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
