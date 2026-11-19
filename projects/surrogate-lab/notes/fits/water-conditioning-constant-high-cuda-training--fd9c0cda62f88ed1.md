# water-conditioning-constant-high · CUDA training

Retrospective source review · fit-fd9c0cda62f88ed1

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. One call proposes pressure; it does not advance a simulation. Affine-free four-group GroupNorm remains in each convolutional block.

**Purpose.** Test a constant 0.001 learning rate on the same single-frame fitting control.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.001, schedule [], seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. The equal-weight objective combines normalized squared pressure-field and FV equation errors. Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. The final configured update is selected; no optimizer-resume state is saved.

**Observed outcome.** The last sampled training loss is 8.7058237643506163e-07; captured validation objective 17.70540289055894 initially and 0.011801414615625866 finally. On the common 128 native pressure requests, worst mean-free pressure relative L2 is 0.20990593032860855 and equation relative L2 is 0.27333002869364859. These common errors differ from the training objective. This review adds no timing measurement or drift result.

**Lesson.** The higher constant rate improves common worst pressure error versus C07, while captured average validation loss is higher. Keep average training objectives and worst field-error comparisons separate.

**Reproduction.** Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/receipt.json",
    "sha256": "fd9c0cda62f88ed10ff7fa310e8b04a5c08bf853326924dc25c84ef06054b178"
  },
  "history": {
    "path": "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/history.json",
    "sha256": "0a21dc2dfe1fa1c55a6fb5fd6cc95ec2e2cbc88ecd58c8e860f596f50e345c31"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/constant-high/project/config.json",
      "sha256": "33e8f4d5e7c662c888966fa6851bf8c654e5b6e20513e76399a0d5ac893d6870"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/qualification.json",
      "sha256": "8c4df8e16d8f682ed0075ba55f8f566c60708e75964a1afcb059827ce7cf2179"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/constant-high/project/source-witness.json",
    "sha256": "dc26c5eff2ea2750fde84b6a82e087ae71aefa16d2969e79047817ecc11d254d"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-conditioning-v1/constant-high/project/inputs/validation_splits.json",
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
    "item_id": "fit-fd9c0cda62f88ed1",
    "kind": "training_attempt",
    "name": "water-conditioning-constant-high · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-cubic-groupnorm-model-v3",
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
    "receipt_sha256": "fd9c0cda62f88ed10ff7fa310e8b04a5c08bf853326924dc25c84ef06054b178",
    "receipt_paths": [
      "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/constant-high/managed-train/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/8f5d33757ec34f97bc0e0fec56c76f1f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-8f5d33757ec34f97bc0e0fec56c76f1f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "8f5d33757ec34f97bc0e0fec56c76f1f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-constant-high · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "33e8f4d5e7c662c888966fa6851bf8c654e5b6e20513e76399a0d5ac893d6870",
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
      "normalization_sha256": "37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8",
      "source_witness_sha256": "dc26c5eff2ea2750fde84b6a82e087ae71aefa16d2969e79047817ecc11d254d"
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
        "sha256": "d94c5e9416afa0d352004c712b6b2bde762d49fb6e21c8de2365124adf6eaa88",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "6942b4e47f7a3e6c4f2b25d145f0412cf4ef35a61b816577c6c251051a5ded06",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "52622ea6d24d66a76da66a6aaaee4b3d7083a408213d535af97b64247d5a8e29",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "49b976bd497c4a4019db015d711406b44aefb1ce72161cf24985b9cc8a410f6f",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "5aabd71de062234b618c05098b2756859eb3d6ee9a33a3bd616727e98a27d027",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "3d86cb2a9a7eb7e87a435d17bbe7937e9f4435724a5a80302a5e1ae491055748",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-6000.npz",
        "sha256": "38c54e2efa5c23d773206ab06ee444d8f425c701640f5c9bb191a223a995c589",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "30ba4af3f7a6d1569553f94be96768ad55f3de2ca3b229650e01a42f878a09ea",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "52622ea6d24d66a76da66a6aaaee4b3d7083a408213d535af97b64247d5a8e29",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2269118464,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/11"
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
      "receipt": "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/receipt.json",
    "sha256": "fd9c0cda62f88ed10ff7fa310e8b04a5c08bf853326924dc25c84ef06054b178",
    "bytes": 9217
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C12": {
        "name": "constant-high",
        "weights_sha256": "52622ea6d24d66a76da66a6aaaee4b3d7083a408213d535af97b64247d5a8e29",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.2733300286936486,
          "pressure_mean_free_relative_l2": 0.20990593032860855,
          "pressure_absolute_rms": 20.982468906709602,
          "pressure_absolute_max": 69.21278395446106
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
            "trajectory_id": "baseline--water-validation-0019",
            "native_frame": 15,
            "counts_xyz": [
              11,
              5,
              15
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.23553311879692979,
              "pressure_mean_free_relative_l2": 0.17829694600566137,
              "pressure_absolute_rms": 20.982468906709602,
              "pressure_absolute_max": 69.21278395446106
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.2733300286936486,
              "pressure_mean_free_relative_l2": 0.20990593032860855,
              "pressure_absolute_rms": 7.442595058296448,
              "pressure_absolute_max": 19.5972496529145
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C12/results.json",
          "sha256": "8f130f5f4088150127bfe7d767309fa5d5a82d29e3992680b71c88a178f80896"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"33e8f4d5e7c662c888966fa6851bf8c654e5b6e20513e76399a0d5ac893d6870","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"dc26c5eff2ea2750fde84b6a82e087ae71aefa16d2969e79047817ecc11d254d"},"elapsed_s":205.43424977798713,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":5.038119378042438e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8774656,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d94c5e9416afa0d352004c712b6b2bde762d49fb6e21c8de2365124adf6eaa88"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"6942b4e47f7a3e6c4f2b25d145f0412cf4ef35a61b816577c6c251051a5ded06"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"52622ea6d24d66a76da66a6aaaee4b3d7083a408213d535af97b64247d5a8e29"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"49b976bd497c4a4019db015d711406b44aefb1ce72161cf24985b9cc8a410f6f"},"checkpoint-4000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"5aabd71de062234b618c05098b2756859eb3d6ee9a33a3bd616727e98a27d027"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"3d86cb2a9a7eb7e87a435d17bbe7937e9f4435724a5a80302a5e1ae491055748"},"checkpoint-6000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"38c54e2efa5c23d773206ab06ee444d8f425c701640f5c9bb191a223a995c589"},"checkpoint-8000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"30ba4af3f7a6d1569553f94be96768ad55f3de2ca3b229650e01a42f878a09ea"},"history.json":{"bytes":62613,"role":"training_diagnostics","sha256":"0a21dc2dfe1fa1c55a6fb5fd6cc95ec2e2cbc88ecd58c8e860f596f50e345c31"},"normalization.json":{"bytes":1173,"role":"preprocessing","sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"c26761cb927918bb2a52c78eee12dce0dd016696b5e2e790dcc9448060a428fd"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":836,"role":"result_summary","sha256":"328d9d21657fd547a13a7679ff9da7c0fb36e1b07ad73a0536f8d4653644acdd"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"52622ea6d24d66a76da66a6aaaee4b3d7083a408213d535af97b64247d5a8e29"}},"per_case_inference_s":[0.012753437971696258,0.008341196982655674,0.002307065005879849,0.003438976011238992,0.0023975989897735417,0.001528131018858403,0.0031175240001175553,0.002266083989525214,0.0024527949863113463,0.002572743018390611,0.002751185995293781,0.005709554010536522,0.0021678679913748056],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"2c704d8bfd688efa262840a6bd60909ab1144121467a35fe556d958adc69e5b3","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"c7c1d71fe92f999ccd10ca93e592cc6e58363a811b0c1a6f067bbd3c5d1df58d","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"3781ab67b4f78c6ec1b881b60c0afe57ad2ee7b201f7edad07871646a26ba9f9","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"b76511eeea88702998fd95da363220cf952d14e711c331b63579417618c10a06","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"6a91a48ca05024d25c9d74a9b868fba95cb22a2550fde20cbc1eb80d50eb33f3","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"35c62edb71f2a9bd2295f693b4c1f76740715ca1e4e80d75f60117ff3a106f1c","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"06779f39f71c3985e55b876c345875cbb719011da1cfc926c674336db9d2a797","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"a06a61cfedb68d1754bad037a9f7083ce49f1733bf72ac6032706ae82d174263","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"167faa884f869b085a0ee5496f1982d9185a72b5b6ee63270400aaa2ad657555","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"f1f254e60b681553e3bff8ff731fd590b7e35f62b6b1eca28796ad7e469fe1ce","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"8766ab7442c64b17d6a2cc2d22dd4da07e9e6df6481f525057278321e4cbffd7","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"a801d1f165b8a344a44a2583e89508ed32f39ee8d5331643253f5b5e2d41b50a","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"ecaf71f901de474350c1fec8b4cc8a79854b530702b9b0ff28d567b115c6b1d6","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2269118464,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
