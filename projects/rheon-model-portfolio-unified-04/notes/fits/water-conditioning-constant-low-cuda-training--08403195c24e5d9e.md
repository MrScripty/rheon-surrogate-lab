# water-conditioning-constant-low · CUDA training

Retrospective source review · fit-08403195c24e5d9e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. One call proposes pressure; it does not advance a simulation. Affine-free four-group GroupNorm remains in each convolutional block.

**Purpose.** Test a constant 0.0001 learning rate on the single-frame fitting control instead of its original decaying schedule.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.0001, schedule [], seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. The equal-weight objective combines normalized squared pressure-field and FV equation errors. Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. The final configured update is selected; no optimizer-resume state is saved.

**Observed outcome.** The last sampled training loss is 9.0146226011443676e-07; captured validation objective 17.70540289055894 initially and 0.0098684145045975745 finally. On the common 128 native pressure requests, worst mean-free pressure relative L2 is 0.26771369940635648 and equation relative L2 is 0.28487942035817893. These common errors differ from the training objective. This review adds no timing measurement or drift result.

**Lesson.** A lower constant learning rate fits the one frame well but has worse common worst pressure/equation errors than C07. Lower training loss is not evidence of better scene generalization.

**Reproduction.** Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/receipt.json",
    "sha256": "08403195c24e5d9e198f964e69e2ea6192041ed445b9b6a4086b7e5baa26d788"
  },
  "history": {
    "path": "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/history.json",
    "sha256": "09399038642b85bfc7db1a4609463b7d126968b2ad0e165673502884c4ca698e"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/constant-low/project/config.json",
      "sha256": "bcc8ffc8c97b883145f5eed242008f06259270f935915ca776a5ee57eee3fe59"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/qualification.json",
      "sha256": "8c4df8e16d8f682ed0075ba55f8f566c60708e75964a1afcb059827ce7cf2179"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/constant-low/project/source-witness.json",
    "sha256": "13c4e9504cb42c7bc2f12cb4f3e0f5bd51c2aa3b71d8cfcb55952700b10ce9b0"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-conditioning-v1/constant-low/project/inputs/validation_splits.json",
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
    "item_id": "fit-08403195c24e5d9e",
    "kind": "training_attempt",
    "name": "water-conditioning-constant-low · CUDA training",
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
    "receipt_sha256": "08403195c24e5d9e198f964e69e2ea6192041ed445b9b6a4086b7e5baa26d788",
    "receipt_paths": [
      "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/constant-low/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/9387f101da0b4df5844658bf0080a3ab/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-9387f101da0b4df5844658bf0080a3ab/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "9387f101da0b4df5844658bf0080a3ab",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-constant-low · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "bcc8ffc8c97b883145f5eed242008f06259270f935915ca776a5ee57eee3fe59",
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
      "source_witness_sha256": "13c4e9504cb42c7bc2f12cb4f3e0f5bd51c2aa3b71d8cfcb55952700b10ce9b0"
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
        "sha256": "f437606b8bbe2f4769352ea9c5d140d8e420d9d50841778effc1ac5542906dac",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "53989c4c74bbccc879894fbfa955569a06f59b2c3274197adfa03b7ff4c655b2",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "9fd330482b0405393ee73b8e29d8a40e60139b3018d190e8d1f8c8a8c7f8ada5",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "22c0ddb2bd5d86897a41ba82a550ed0cb4a61ddb629e9f1846e7f8ec4dc7ed82",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "a3c2f47d650a5931e77eb0152aed1472dd8f41b0634b2f6630459a19fcf416cb",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "c9d6df1fcaeaf2637cf9ba5156f153586f87652486c4d982033da15e309892c6",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-6000.npz",
        "sha256": "4a9c0fadd6a586cb57a415b324e4a00a9e38a1ee32e8374706c82d1e2e6e8f9f",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "91107f1585b4e1d10236dc24b4c5738d89047530c8711946ec859f9861ff1257",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "9fd330482b0405393ee73b8e29d8a40e60139b3018d190e8d1f8c8a8c7f8ada5",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2268901376,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/10"
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
      "receipt": "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/constant-low/managed/stage-00/artifacts/receipt.json",
    "sha256": "08403195c24e5d9e198f964e69e2ea6192041ed445b9b6a4086b7e5baa26d788",
    "bytes": 9218
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C11": {
        "name": "constant-low",
        "weights_sha256": "9fd330482b0405393ee73b8e29d8a40e60139b3018d190e8d1f8c8a8c7f8ada5",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.2848794203581789,
          "pressure_mean_free_relative_l2": 0.2677136994063565,
          "pressure_absolute_rms": 18.66941101659411,
          "pressure_absolute_max": 63.95784507656832
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
            "trajectory_id": "baseline--water-validation-0021",
            "native_frame": 15,
            "counts_xyz": [
              10,
              5,
              5
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.23588460837081418,
              "pressure_mean_free_relative_l2": 0.1738427199838613,
              "pressure_absolute_rms": 18.66941101659411,
              "pressure_absolute_max": 63.95784507656832
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.2848794203581789,
              "pressure_mean_free_relative_l2": 0.2677136994063565,
              "pressure_absolute_rms": 6.744384106349791,
              "pressure_absolute_max": 17.86151611503383
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C11/results.json",
          "sha256": "880c60668546cd9bc5fff5ac65bf3d72da14a21c88443d977b4fa3bd95764ec1"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"bcc8ffc8c97b883145f5eed242008f06259270f935915ca776a5ee57eee3fe59","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"13c4e9504cb42c7bc2f12cb4f3e0f5bd51c2aa3b71d8cfcb55952700b10ce9b0"},"elapsed_s":234.93686278598034,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00012398496906662615,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8774656,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"f437606b8bbe2f4769352ea9c5d140d8e420d9d50841778effc1ac5542906dac"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"53989c4c74bbccc879894fbfa955569a06f59b2c3274197adfa03b7ff4c655b2"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9fd330482b0405393ee73b8e29d8a40e60139b3018d190e8d1f8c8a8c7f8ada5"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"22c0ddb2bd5d86897a41ba82a550ed0cb4a61ddb629e9f1846e7f8ec4dc7ed82"},"checkpoint-4000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"a3c2f47d650a5931e77eb0152aed1472dd8f41b0634b2f6630459a19fcf416cb"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"c9d6df1fcaeaf2637cf9ba5156f153586f87652486c4d982033da15e309892c6"},"checkpoint-6000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"4a9c0fadd6a586cb57a415b324e4a00a9e38a1ee32e8374706c82d1e2e6e8f9f"},"checkpoint-8000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"91107f1585b4e1d10236dc24b4c5738d89047530c8711946ec859f9861ff1257"},"history.json":{"bytes":62592,"role":"training_diagnostics","sha256":"09399038642b85bfc7db1a4609463b7d126968b2ad0e165673502884c4ca698e"},"normalization.json":{"bytes":1173,"role":"preprocessing","sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"5aa06e97970157138c734b5f51492d0b4371ed1f89aa14e82530895ede8e5003"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":836,"role":"result_summary","sha256":"b3c4ece73eb2cffdb44d2a21b580ec00e80c6b8ce42592884413129bc8bca4be"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9fd330482b0405393ee73b8e29d8a40e60139b3018d190e8d1f8c8a8c7f8ada5"}},"per_case_inference_s":[0.015750902995932847,0.010448528017150238,0.00296388499555178,0.004779578011948615,0.003310740983579308,0.0021415009978227317,0.0041181840060744435,0.003153675002977252,0.003211061004549265,0.0032292340183630586,0.0033942130103241652,0.007029862026683986,0.0026444779941812158],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"d054845a091769234e16404b0b49fac5e3356e603f1d469dd89b481bfa474548","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"c21a60b3c6a992389c55bb4316ee221e5fbf55496c83a9ca5b47d19c579884d9","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"283001c2a870573ead34943e48307edac6d95275338f65453a6dcc8eaa2949aa","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"5a714d5fa3b196550eccae51ad224e9c6030ef0f3e19c4971d2459f08020d586","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"92ecdf537f47bfd8cc44791fa3c7d25ffd6a99b854c4a8c21167286c65a24d72","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"fbb5bc9328df8cf8b2794ea29f17b3062a6f9160c9eab44b68905ee229a8d822","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"285afaed20fdb4ea727a7808707e5cb95af645a1469977fa60c1036a142c54b6","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"f9da649d514251e533d1a2a414d13f290ce83196cb4319e93972141025ffacae","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"e538903f80393f479ac95e175f4ae4a843089ad645f3f91425eb7afbf9b6f6bc","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"84044e9af01e1d5f22553397e634a1b5c0626327b889edc0ae4840cd92395d10","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"a0a0dd4a00cce20f307132896a3bec97e2a4c90c848fadaa376603e288f36ebb","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"0fa5268e750d5cdd71191f3d8d385a3adfd356834c3a806ff47152a20bc5194f","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"fb85cfc97b79099ff4a41a0af088ee705643444608427aedd1af5ed8a4db28b4","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2268901376,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
