# water-conditioning-delayed-decay · CUDA training

Retrospective source review · fit-dfce8332d409f00c

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. One call proposes pressure; it does not advance a simulation. Affine-free four-group GroupNorm remains in each convolutional block.

**Purpose.** Delay learning-rate reductions to updates 5,000 and 8,000 on the same single-frame control.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 5000}, {'learning_rate': 1e-05, 'update': 8000}], seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. The equal-weight objective combines normalized squared pressure-field and FV equation errors. Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. The final configured update is selected; no optimizer-resume state is saved.

**Observed outcome.** The last sampled training loss is 2.9030706803434157e-06; captured validation objective 17.70540289055894 initially and 0.011228825159044347 finally. On the common 128 native pressure requests, worst mean-free pressure relative L2 is 0.20531130574028847 and equation relative L2 is 0.27453207015540015. These common errors differ from the training objective. This review adds no timing measurement or drift result.

**Lesson.** Delaying decay gives a small common worst pressure improvement over the constant-high candidate, despite a larger final sampled training loss. Neither schedule resolves the large error on held-out scenes.

**Reproduction.** Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/receipt.json",
    "sha256": "dfce8332d409f00cb5bce7a6bfa751834ff8907c5f544c9d324cd838dd5e45a0"
  },
  "history": {
    "path": "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/history.json",
    "sha256": "abc3cc4a6023b92575ddf668bba26375252a648bde71787a81922fc339117f9e"
  },
  "sources": [
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/config.json",
      "sha256": "414c1acff672fadbb894b1e3c40640cf5ce764abfd772ff143cdbd60c9d4d368"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/qualification.json",
      "sha256": "8c4df8e16d8f682ed0075ba55f8f566c60708e75964a1afcb059827ce7cf2179"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-conditioning-v1/delayed-decay/project/source-witness.json",
    "sha256": "0940d4724b92c88f75bf85167a3d7a17b16d0a0d91f03be88a34230e43fc996c"
  },
  "training_population": {
    "path": "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-conditioning-v1/delayed-decay/project/inputs/validation_splits.json",
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
    "item_id": "fit-dfce8332d409f00c",
    "kind": "training_attempt",
    "name": "water-conditioning-delayed-decay · CUDA training",
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
    "receipt_sha256": "dfce8332d409f00cb5bce7a6bfa751834ff8907c5f544c9d324cd838dd5e45a0",
    "receipt_paths": [
      "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/receipt.json",
      "runs/water-conditioning-v1/delayed-decay/managed-train/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/9cb46d2205e74afd802bfb23f91e1882/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-9cb46d2205e74afd802bfb23f91e1882/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "9cb46d2205e74afd802bfb23f91e1882",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-conditioning-delayed-decay · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "414c1acff672fadbb894b1e3c40640cf5ce764abfd772ff143cdbd60c9d4d368",
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
      "source_witness_sha256": "0940d4724b92c88f75bf85167a3d7a17b16d0a0d91f03be88a34230e43fc996c"
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
        "sha256": "7fb4db31fae2a2490fd4f108ff295fd4c440ba412789f1979a4085783c219b6e",
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
        "sha256": "f0e7c022daa18b8467e8c53e5e239f9d71017b0c88f15b33cea50877488c0f1d",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "e75657a4eba8f8047d2e59f846ab978e6b326f132b88b82a29c89735a8148dc4",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "7fb4db31fae2a2490fd4f108ff295fd4c440ba412789f1979a4085783c219b6e",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2269429760,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/12"
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
      "receipt": "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/receipt.json",
    "sha256": "dfce8332d409f00cb5bce7a6bfa751834ff8907c5f544c9d324cd838dd5e45a0",
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
      "C13": {
        "name": "delayed-decay",
        "weights_sha256": "7fb4db31fae2a2490fd4f108ff295fd4c440ba412789f1979a4085783c219b6e",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.27453207015540015,
          "pressure_mean_free_relative_l2": 0.20531130574028847,
          "pressure_absolute_rms": 19.969573766724604,
          "pressure_absolute_max": 64.68986915228238
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
              "common_equation_relative_l2": 0.22598917078257863,
              "pressure_mean_free_relative_l2": 0.17678063409631972,
              "pressure_absolute_rms": 19.969573766724604,
              "pressure_absolute_max": 64.68986915228238
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.27453207015540015,
              "pressure_mean_free_relative_l2": 0.20531130574028847,
              "pressure_absolute_rms": 7.432041128780606,
              "pressure_absolute_max": 19.50810731213767
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C13/results.json",
          "sha256": "ee6415ff175cba60b0dbb2a68d7e7c27403ff5aed36d5036cf5da50e3f3505f2"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"414c1acff672fadbb894b1e3c40640cf5ce764abfd772ff143cdbd60c9d4d368","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"0940d4724b92c88f75bf85167a3d7a17b16d0a0d91f03be88a34230e43fc996c"},"elapsed_s":205.74501480499748,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":7.87206152779163e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8774656,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"d94c5e9416afa0d352004c712b6b2bde762d49fb6e21c8de2365124adf6eaa88"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"6942b4e47f7a3e6c4f2b25d145f0412cf4ef35a61b816577c6c251051a5ded06"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"7fb4db31fae2a2490fd4f108ff295fd4c440ba412789f1979a4085783c219b6e"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"49b976bd497c4a4019db015d711406b44aefb1ce72161cf24985b9cc8a410f6f"},"checkpoint-4000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"5aabd71de062234b618c05098b2756859eb3d6ee9a33a3bd616727e98a27d027"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"3d86cb2a9a7eb7e87a435d17bbe7937e9f4435724a5a80302a5e1ae491055748"},"checkpoint-6000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"f0e7c022daa18b8467e8c53e5e239f9d71017b0c88f15b33cea50877488c0f1d"},"checkpoint-8000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"e75657a4eba8f8047d2e59f846ab978e6b326f132b88b82a29c89735a8148dc4"},"history.json":{"bytes":62672,"role":"training_diagnostics","sha256":"abc3cc4a6023b92575ddf668bba26375252a648bde71787a81922fc339117f9e"},"normalization.json":{"bytes":1173,"role":"preprocessing","sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"09d02f53c28f366f561101420b551c00abff3ddff29c1380056c609590676ff1"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":836,"role":"result_summary","sha256":"cb0b64f0c20f730a22619f3f51b39fca738183dc9fefca5f2282e17d5d12decc"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"7fb4db31fae2a2490fd4f108ff295fd4c440ba412789f1979a4085783c219b6e"}},"per_case_inference_s":[0.013484880997566506,0.008656546007841825,0.002520746988011524,0.0037078190071042627,0.0025198890070896596,0.0015862250002101064,0.0031732259958516806,0.0023020539956633,0.0025377949932590127,0.0026178639964200556,0.0027040389832109213,0.005305343016516417,0.002056534984149039],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"73774c2ec2e3e2ed5fba07471d6dbb93b0e0235627fb05994b8dd1db63801bfc","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"19bb141243c7776f27b068279f939a63ab3d7df1233e249cea0fc3f2b50ca382","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"8804d23265fd8fcdbda2d77d9e87dbee56e90f7d6ba3f2d947db281a97991a3c","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"1473acf235c5033ae81006902d29263a2f2b5f0f17ec4cb49d6a3b644c11ce45","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"2f13e23dd37d246ac85fa609efe4bd4264fde3a2316ae8cf5121dc9a751d17cb","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"cc537c00b594a02e966fbdb83c044da18dee92980a7cd1511b39d01ed80e1223","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"1b0f7f16a281b9f3e55f06e993206f7541757f436820c83e14f57ea343205efa","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"b24af4b68aa6354cb2cffef4d7164897b54ed1b64114d171aaa7a40a8419244d","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"f5b74dddac800f322d9971105949271ad72772554ab8bef0baf3b2fd6b616c79","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"cead06202f44531f16f0ead95af3c7d266f471e146a2573fa8ee64f3c84fbad8","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"564b35b2aaf4509cfc48c91c198ece0e25226fa21ba7cbde1ffbb8194a83d0aa","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"4e88f57b2d445cf433d2434a5bcc9aff021d594d71a47cd3035c774aa0cfb69b","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"6fb419577326f7081aa97e7f460bf43d686cfe53abd6d91802420cc3fa634541","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2269429760,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
