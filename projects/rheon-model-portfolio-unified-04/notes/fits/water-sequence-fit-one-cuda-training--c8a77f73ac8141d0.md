# water-sequence-fit-one · CUDA training

Retrospective source review · fit-c8a77f73ac8141d0

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.

**Purpose.** Test whether the unchanged network can memorize one pressure frame, separating basic fitting capacity from broad generalization.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. Learning-rate schedule: [{'learning_rate': 0.0001, 'update': 1000}, {'learning_rate': 1e-05, 'update': 6000}]. Gradient norm is clipped at 1.0. 1 training trajectories/1 saved pressure frames; 64 validation trajectories/128 frames. Actual training densities, in scenes: {'8.0': 1}. Native training snapshot indices present: [15]. The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.

**Observed outcome.** The final sampled training loss is 1.5849312775382874e-06; the retained validation objective changes from 17.70540289055894 to 0.0098622718278007261. The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 0.23065380535642638 and equation relative L2 0.27569872995962674. These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.

**Lesson.** A final sampled training loss near 1.6e-6 on one frame coexists with materially larger held-out loss and common pressure error. This is a narrow fitting-capacity control; its training population has only 8 cells/metre on one 16x16x16 scene. The two-density validation population must not be described as two-density training.

**Reproduction.** Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/receipt.json",
    "sha256": "c8a77f73ac8141d001f9c1d154bc5373eb2854a9853c44be755a3de13a6ad860"
  },
  "history": {
    "path": "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/history.json",
    "sha256": "e962e851c6b4f7e76b1b8fa0a774793e8980680a10d67326a7656994ec6b173a"
  },
  "sources": [
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/config.json",
      "sha256": "b65e90e3daff2530f2b51005a14035230532940e53a8663cdc5877c46c0db28a"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/qualification.json",
      "sha256": "d98357e8a1e288772f782c87217f46ea886812f6c80c9a37f398a4135cd95fce"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/test_inference.py",
      "sha256": "43bc71643e362670c960c184b6b1a378f93aba7a6408a2763a960059592a1b56"
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/train.py",
      "sha256": "75a6e4281f70646ef0e66fc3b93163b573df05b66e54006ca6630c0051338be5"
    }
  ],
  "source_witness": {
    "path": "runs/water-learning-sequence-v1/fit-one/project/source-witness.json",
    "sha256": "f7b249fef2f222f322ceb74026c7eaee33f680a22fe276240704c56d4cd40bcf"
  },
  "training_population": {
    "path": "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset",
      "bytes": 2275660
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra",
      "bytes": 2278650
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset",
      "bytes": 2561142
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra",
      "bytes": 2564086
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits",
      "bytes": 339677
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits",
      "bytes": 374928
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset",
      "bytes": 2209622
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra",
      "bytes": 2212566
    },
    {
      "path": "runs/water-learning-sequence-v1/fit-one/project/inputs/validation_splits.json",
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
    "item_id": "fit-c8a77f73ac8141d0",
    "kind": "training_attempt",
    "name": "water-sequence-fit-one · CUDA training",
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
    "receipt_sha256": "c8a77f73ac8141d001f9c1d154bc5373eb2854a9853c44be755a3de13a6ad860",
    "receipt_paths": [
      "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/receipt.json",
      "runs/water-learning-sequence-v1/fit-one/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/bc1ea77166d444dc8f791cc532a49541/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-bc1ea77166d444dc8f791cc532a49541/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "bc1ea77166d444dc8f791cc532a49541",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-sequence-fit-one · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "b65e90e3daff2530f2b51005a14035230532940e53a8663cdc5877c46c0db28a",
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
      "source_witness_sha256": "f7b249fef2f222f322ceb74026c7eaee33f680a22fe276240704c56d4cd40bcf"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "685d617f49e23cc78f6aab8e5567d8ec74d8a80262f2beb8de2c38e029be3223",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "4c08199eaaa6424b75a0fc407a648c3fb9b8abada4bd15dae8c72829cb573138",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "685d617f49e23cc78f6aab8e5567d8ec74d8a80262f2beb8de2c38e029be3223",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2269437952,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/6"
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
      "receipt": "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-learning-sequence-v1/fit-one/managed/stage-00/artifacts/receipt.json",
    "sha256": "c8a77f73ac8141d001f9c1d154bc5373eb2854a9853c44be755a3de13a6ad860",
    "bytes": 8402
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C07": {
        "name": "fit-one",
        "weights_sha256": "685d617f49e23cc78f6aab8e5567d8ec74d8a80262f2beb8de2c38e029be3223",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.27569872995962674,
          "pressure_mean_free_relative_l2": 0.23065380535642638,
          "pressure_absolute_rms": 21.87373791397685,
          "pressure_absolute_max": 74.49976888221772
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
            "trajectory_id": "baseline--water-validation-0041",
            "native_frame": 63,
            "counts_xyz": [
              5,
              16,
              5
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.22705582950612624,
              "pressure_mean_free_relative_l2": 0.16834636163167666,
              "pressure_absolute_rms": 21.87373791397685,
              "pressure_absolute_max": 74.49976888221772
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.27569872995962674,
              "pressure_mean_free_relative_l2": 0.23065380535642638,
              "pressure_absolute_rms": 7.432682929177471,
              "pressure_absolute_max": 19.28714316168115
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C07/results.json",
          "sha256": "4e93cd10d5d0b0ed2d1fa90a37f044db7bbab9e741d744f7570958eab53b84ef"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"b65e90e3daff2530f2b51005a14035230532940e53a8663cdc5877c46c0db28a","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b9ce164d30e4101b009fe4be765a070593cfbdd48f897853de159a8c177fabe8","source_witness_sha256":"f7b249fef2f222f322ceb74026c7eaee33f680a22fe276240704c56d4cd40bcf"},"elapsed_s":238.92699992199778,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00010548562447709742,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":8774656,"peak_reserved_bytes":10485760,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-10000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"685d617f49e23cc78f6aab8e5567d8ec74d8a80262f2beb8de2c38e029be3223"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"4c08199eaaa6424b75a0fc407a648c3fb9b8abada4bd15dae8c72829cb573138"},"history.json":{"bytes":27318,"role":"training_diagnostics","sha256":"e962e851c6b4f7e76b1b8fa0a774793e8980680a10d67326a7656994ec6b173a"},"normalization.json":{"bytes":1173,"role":"preprocessing","sha256":"37965c625cb8c3efb06bfd3e5b2d2e422d62a979a386c2e3b6bb164cda09da8e"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"f774aa445daf0d813b04555795db81fbf83de0883328b7e34b550e4d4bf6becf"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":836,"role":"result_summary","sha256":"7346dd2920fd849e0486d7208d921ad10a8b261ae6f4e9470a76e33bee8f46b0"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"685d617f49e23cc78f6aab8e5567d8ec74d8a80262f2beb8de2c38e029be3223"}},"per_case_inference_s":[0.0136111679894384,0.009652082982938737,0.002755279012490064,0.003969535988289863,0.002579418011009693,0.0015981720061972737,0.003288796986453235,0.0024322250101249665,0.002713763009523973,0.002785372023936361,0.0029402239888440818,0.006033808982465416,0.0023223749885801226],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"be730af077b7b6b3d928cd02b1dcf3d94937e9fcc5272870ff3d7eabf27437ca","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"47d6e97aad272fc0e11bbd3ea998ae7bbb6501af9e53a16d5418cc3498bd277c","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"17e5dbeded643a548fb635d36d081d199d0bce0961f2fc6df512b76cefff2aae","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"183e67817edf4a6518001f54f3d1c23d585121d3be6df3fb7807ebb3366e4b9f","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"7f371e6033916495f5f202e6abca47f3acf0ed375721c392f3c62dd1c8424511","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"426941b48ddcf00509bd7b12c79a5ad536872e9810f1aa23af043ed8cd3e8423","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"967534868d4fc6d7e3d80a68f7e806eb43cbe513c1675402d905509d33a12ba9","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"ac7b7decdeb33637549fdea95415139a223ebdc55b5fe4f57d7882e0eada2b04","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"4eb1642f67f6d36b5899d5d1fbad6ae2d580a6c1b2bb68d72f9712462c8b48dd","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"64a519e7069554e9c35fa859898d303996b08210318bc1ed9a2d19a38633392e","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"fc8641ccfbed32ef95d7d60228d37484eb1ee2cdf84bd83ba581e02ae7575f3b","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"605ffea14b9d9a429780afe3d7ec525ca1629ce7c86962c58ee2cfac7be2d940","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"cd67ba3b36c60fb835098d9ec8a9eed7e24147c146589d61e71da84a503cc637","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2269437952,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
