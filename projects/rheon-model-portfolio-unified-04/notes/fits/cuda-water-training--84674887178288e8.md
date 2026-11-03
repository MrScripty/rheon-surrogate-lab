# CUDA water training

Retrospective source review · fit-84674887178288e8

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.

**Purpose.** Establish a CUDA-trained pressure baseline using 256 simulated training scenes and two saved frames per scene.

**Experiment.** 2,000 whole-network Adam updates, batch 2, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. Learning-rate schedule: [{'learning_rate': 0.0001, 'update': 1000}]. Gradient norm is clipped at 1.0. 256 training trajectories/512 saved pressure frames; 64 validation trajectories/128 frames. Actual training densities, in scenes: {'8.0': 128, '16.0': 128}. Native training snapshot indices present: [15, 63]. The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.

**Observed outcome.** The final sampled training loss is 0.0012450788431066216; the retained validation objective changes from 55.894181807199708 to 0.0035414697685510134. The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 0.12359800365869276 and equation relative L2 0.19846745626812354. These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.

**Lesson.** This is already a 256-scene physical population, distinct from the much smaller early variable-grid dataset. Two snapshots per trajectory remain sparse temporal supervision. Common worst pressure/equation errors are substantial despite the wider scene count; do not infer larger-domain or timestep generalization from parameter sharing alone.

**Reproduction.** Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/sealed-water-v1-3/managed/stage-00/artifacts/receipt.json",
    "sha256": "84674887178288e8eb6e48d3e6bf152113f24958357453c59c3a9029bd51fd7c"
  },
  "history": {
    "path": "runs/sealed-water-v1-3/managed/stage-00/artifacts/history.json",
    "sha256": "cd2d24c6fd274039055593eee1ecb0d8ff288e3d2eca03120a2f7c98b5aaecdc"
  },
  "sources": [
    {
      "path": "runs/sealed-water-v1-3/project/config.json",
      "sha256": "640c1e8f52512f8f127fe8d0967fc9e0538a2cfb0bb98929e13afc232eb6be79"
    },
    {
      "path": "runs/sealed-water-v1-3/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/sealed-water-v1-3/project/data.py",
      "sha256": "eef8719bf4c17685f564529c4ecf0a2c7c4f3b8834818ab4c8397343bcc178fe"
    },
    {
      "path": "runs/sealed-water-v1-3/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/sealed-water-v1-3/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/sealed-water-v1-3/project/model.py",
      "sha256": "d42453ac9060275c7fc6676eab0e23eee7188cf2604a82714bfd4570104d7c0c"
    },
    {
      "path": "runs/sealed-water-v1-3/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/sealed-water-v1-3/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/sealed-water-v1-3/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/sealed-water-v1-3/project/policy.py",
      "sha256": "5e6fb00db31f558df0bf3665422c7afd4a6ed14e8608e9514576fb86632858cd"
    },
    {
      "path": "runs/sealed-water-v1-3/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/sealed-water-v1-3/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/sealed-water-v1-3/project/qualification.json",
      "sha256": "7169fb713d2627bfc0d5999091263a4deede29f31369ac3c7eb937ed6138e735"
    },
    {
      "path": "runs/sealed-water-v1-3/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/sealed-water-v1-3/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/sealed-water-v1-3/project/test_inference.py",
      "sha256": "f8810a25e6f97816a05948e965dd55b13cfee8a4ff6fb3d6ecee8a8f3084317c"
    },
    {
      "path": "runs/sealed-water-v1-3/project/train.py",
      "sha256": "ec0a28b88bafa3c69ca9d7ca2fd1db9bc838d6a1142bc01dce2d286359de933d"
    }
  ],
  "source_witness": {
    "path": "runs/sealed-water-v1-3/project/source-witness.json",
    "sha256": "582d1f2c77cc2a37bf06035a868867b7118cd5933418a93b390bb92961fd8c56"
  },
  "training_population": {
    "path": "runs/sealed-water-v1-3/managed/stage-00/artifacts/splits.json",
    "sha256": "cb17fca6863ee22164c556ecd9f557a1241dcabcbfa619964a15922e7c9c1233"
  },
  "training_inputs": [
    {
      "path": "runs/sealed-water-v1-3/project/inputs/dataset.npz",
      "sha256": "bb0102174364117ea1065aef47039867e4f432e1edf620d421f8a780c42723fa",
      "slot": "dataset",
      "bytes": 24326678
    },
    {
      "path": "runs/sealed-water-v1-3/project/inputs/splits.json",
      "sha256": "cb17fca6863ee22164c556ecd9f557a1241dcabcbfa619964a15922e7c9c1233",
      "slot": "splits",
      "bytes": 930868
    },
    {
      "path": "runs/sealed-water-v1-3/project/inputs/test_dataset.npz",
      "sha256": "abe99b4461349e7e27211ebe9305d4fe8c33110ecdf4e1e442204da76b30a103",
      "slot": "test_dataset",
      "bytes": 5709078
    },
    {
      "path": "runs/sealed-water-v1-3/project/inputs/test_splits.json",
      "sha256": "06e17c09ada7c03deb02c6bd2a73e9a7893e5852dd4e2afb9197e7ace86b2c59",
      "slot": "test_splits",
      "bytes": 364879
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-84674887178288e8",
    "kind": "training_attempt",
    "name": "CUDA water training",
    "task": "pressure_proposal",
    "schema": "rheon-sealed-water-training-v1",
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
    "receipt_sha256": "84674887178288e8eb6e48d3e6bf152113f24958357453c59c3a9029bd51fd7c",
    "receipt_paths": [
      "runs/live-ai-comparison/deployment-1/receipt.json",
      "runs/live-ai-comparison/deployment-2/receipt.json",
      "runs/live-ai-comparison/deployment-3/receipt.json",
      "runs/sealed-water-v1-3/managed/stage-00/artifacts/receipt.json",
      "runs/sealed-water-v1-3/managed/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/archive/models/aa82/receipt.json",
      "runs/dependency-migration-v1/archive/models/aa82-final/receipt.json",
      "runs/dependency-migration-v1/archive/models/aa82-qualified/receipt.json",
      "runs/dependency-migration-v1/archive/models/aa82-runtime-current/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/fa641802967e4d339ad2d0d82dd54b7c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-fa641802967e4d339ad2d0d82dd54b7c/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "fa641802967e4d339ad2d0d82dd54b7c",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "CUDA water training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "640c1e8f52512f8f127fe8d0967fc9e0538a2cfb0bb98929e13afc232eb6be79",
      "inputs": {
        "dataset": {
          "bytes": 24326678,
          "sha256": "bb0102174364117ea1065aef47039867e4f432e1edf620d421f8a780c42723fa"
        },
        "splits": {
          "bytes": 930868,
          "sha256": "cb17fca6863ee22164c556ecd9f557a1241dcabcbfa619964a15922e7c9c1233"
        },
        "test_dataset": {
          "bytes": 5709078,
          "sha256": "abe99b4461349e7e27211ebe9305d4fe8c33110ecdf4e1e442204da76b30a103"
        },
        "test_splits": {
          "bytes": 364879,
          "sha256": "06e17c09ada7c03deb02c6bd2a73e9a7893e5852dd4e2afb9197e7ace86b2c59"
        }
      },
      "normalization_sha256": "bee9be00a94f0241bc8baf6df2499314918b0a9e708b3f995bea5046c40a3296",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "1c45a0e242852eceb5e395c5bd12b2248bc1cc3460583dce41d9c72296a01812",
      "source_witness_sha256": "582d1f2c77cc2a37bf06035a868867b7118cd5933418a93b390bb92961fd8c56"
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
        "sha256": "e4c1478ed3e852be13f0a6936918d98e12d98482052589635e08958b90a11258",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "c091b7957201f501b3b85220153af3942f5247a3c7850f533aaec55c4ebdce82",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "c1f9b7fd573c2646e1c727ecfe742180f8a1a9bdf5d9340aca57ed935ebf68eb",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2467135488,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/5"
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
      "receipt": "runs/live-ai-comparison/deployment-1/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/live-ai-comparison/deployment-1/receipt.json",
    "sha256": "84674887178288e8eb6e48d3e6bf152113f24958357453c59c3a9029bd51fd7c",
    "bytes": 11179
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C06": {
        "name": "Original 256-simulation water cohort",
        "weights_sha256": "aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.19846745626812354,
          "pressure_mean_free_relative_l2": 0.12359800365869276,
          "pressure_absolute_rms": 6.5614570729474355,
          "pressure_absolute_max": 31.074239195223562
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0050",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              5
            ],
            "cell_size_m": 0.125
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0060",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              15
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.19846745626812354,
              "pressure_mean_free_relative_l2": 0.12359800365869276,
              "pressure_absolute_rms": 6.5614570729474355,
              "pressure_absolute_max": 31.074239195223562
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.15966888385441433,
              "pressure_mean_free_relative_l2": 0.10620839353259053,
              "pressure_absolute_rms": 2.0738381499036045,
              "pressure_absolute_max": 5.060226888319766
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C06/results.json",
          "sha256": "37a163d474792e044c32db36e5b65a3afb90bd57287aac0ee847b1345aab9586"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[15,10,14],"frame":1,"native_frame":63,"parent":23,"split":"train","trajectory_id":"water-train-0023"},{"cell_size_m":0.125,"counts_xyz":[5,14,8],"frame":0,"native_frame":15,"parent":46,"split":"train","trajectory_id":"water-train-0046"},{"cell_size_m":0.0625,"counts_xyz":[10,15,16],"frame":1,"native_frame":63,"parent":69,"split":"train","trajectory_id":"water-train-0069"},{"cell_size_m":0.125,"counts_xyz":[14,14,9],"frame":0,"native_frame":15,"parent":92,"split":"train","trajectory_id":"water-train-0092"},{"cell_size_m":0.0625,"counts_xyz":[9,11,16],"frame":1,"native_frame":63,"parent":115,"split":"train","trajectory_id":"water-train-0115"},{"cell_size_m":0.0625,"counts_xyz":[13,16,16],"frame":0,"native_frame":15,"parent":139,"split":"train","trajectory_id":"water-train-0139"},{"cell_size_m":0.125,"counts_xyz":[7,10,12],"frame":1,"native_frame":63,"parent":162,"split":"train","trajectory_id":"water-train-0162"},{"cell_size_m":0.0625,"counts_xyz":[14,14,6],"frame":0,"native_frame":15,"parent":185,"split":"train","trajectory_id":"water-train-0185"},{"cell_size_m":0.125,"counts_xyz":[16,7,6],"frame":1,"native_frame":63,"parent":208,"split":"train","trajectory_id":"water-train-0208"},{"cell_size_m":0.0625,"counts_xyz":[12,7,7],"frame":0,"native_frame":15,"parent":231,"split":"train","trajectory_id":"water-train-0231"},{"cell_size_m":0.0625,"counts_xyz":[6,16,13],"frame":1,"native_frame":63,"parent":255,"split":"train","trajectory_id":"water-train-0255"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":256,"split":"validation","trajectory_id":"water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":261,"split":"validation","trajectory_id":"water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":267,"split":"validation","trajectory_id":"water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":273,"split":"validation","trajectory_id":"water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":278,"split":"validation","trajectory_id":"water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":284,"split":"validation","trajectory_id":"water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":290,"split":"validation","trajectory_id":"water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":296,"split":"validation","trajectory_id":"water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":301,"split":"validation","trajectory_id":"water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":307,"split":"validation","trajectory_id":"water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":313,"split":"validation","trajectory_id":"water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":319,"split":"validation","trajectory_id":"water-validation-0063"}],"closure":{"experiment_sha256":"640c1e8f52512f8f127fe8d0967fc9e0538a2cfb0bb98929e13afc232eb6be79","inputs":{"dataset":{"bytes":24326678,"sha256":"bb0102174364117ea1065aef47039867e4f432e1edf620d421f8a780c42723fa"},"splits":{"bytes":930868,"sha256":"cb17fca6863ee22164c556ecd9f557a1241dcabcbfa619964a15922e7c9c1233"},"test_dataset":{"bytes":5709078,"sha256":"abe99b4461349e7e27211ebe9305d4fe8c33110ecdf4e1e442204da76b30a103"},"test_splits":{"bytes":364879,"sha256":"06e17c09ada7c03deb02c6bd2a73e9a7893e5852dd4e2afb9197e7ace86b2c59"}},"normalization_sha256":"bee9be00a94f0241bc8baf6df2499314918b0a9e708b3f995bea5046c40a3296","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"1c45a0e242852eceb5e395c5bd12b2248bc1cc3460583dce41d9c72296a01812","source_witness_sha256":"582d1f2c77cc2a37bf06035a868867b7118cd5933418a93b390bb92961fd8c56"},"elapsed_s":80.57971252000425,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00036504737081344274,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":32808448,"peak_reserved_bytes":58720256,"tf32":false},"mode":"train","optimizer_updates":2000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"e4c1478ed3e852be13f0a6936918d98e12d98482052589635e08958b90a11258"},"checkpoint-1000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"c091b7957201f501b3b85220153af3942f5247a3c7850f533aaec55c4ebdce82"},"checkpoint-2000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d"},"checkpoint-500.npz":{"bytes":1497594,"role":"checkpoint","sha256":"c1f9b7fd573c2646e1c727ecfe742180f8a1a9bdf5d9340aca57ed935ebf68eb"},"history.json":{"bytes":29534,"role":"training_diagnostics","sha256":"cd2d24c6fd274039055593eee1ecb0d8ff288e3d2eca03120a2f7c98b5aaecdc"},"normalization.json":{"bytes":89838,"role":"preprocessing","sha256":"bee9be00a94f0241bc8baf6df2499314918b0a9e708b3f995bea5046c40a3296"},"predictions.npz":{"bytes":298222,"role":"predictions","sha256":"03cda13659fae8bb697c05e1f4fe7f50f5c035d3d9c00ef99aefaab8971be932"},"splits.json":{"bytes":930868,"role":"training_splits","sha256":"cb17fca6863ee22164c556ecd9f557a1241dcabcbfa619964a15922e7c9c1233"},"summary.json":{"bytes":3316,"role":"result_summary","sha256":"c4c99b381b37b4ac1f0768eb02697716e636c4d95697a3606a0789d8a07d964f"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d"}},"per_case_inference_s":[0.013377299008425325,0.004614055011188611,0.0022333980014082044,0.00515764401643537,0.003760904015507549,0.003924681979697198,0.007330716995056719,0.002992869005538523,0.0028023049817420542,0.002067371999146417,0.0021546619827859104,0.0036227529926691204,0.00782923397491686,0.0027234969893470407,0.0039498900005128235,0.002731200016569346,0.0017094409849960357,0.0034380560100544244,0.0025179549993481487,0.002640127990162,0.0028974320157431066,0.0029793309804517776,0.005810925998957828,0.0023059619998093694],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"9af4715d737da4404093f8b80cf709ce9f676d364dd71148c160cd5873148361","shape":[16,16,16]},"case_001":{"bytes":16800,"dtype":"<f8","sha256":"313f9f87b232832a115c6975e7600a75021364a5b4428aa044fde473a44bc483","shape":[14,10,15]},"case_002":{"bytes":4480,"dtype":"<f8","sha256":"4857bde91f61a450d1823a363f638f6ac1b2f193aa33e4ddd82cefb1894c4707","shape":[8,14,5]},"case_003":{"bytes":19200,"dtype":"<f8","sha256":"d709615e36688609b47640a17310193a6122bf682fb5f23d3d93782a9e19eef5","shape":[16,15,10]},"case_004":{"bytes":14112,"dtype":"<f8","sha256":"e7ed6ccd1f86e2fc29e9c340b415d34b2cce399830849a8198c9003f41937973","shape":[9,14,14]},"case_005":{"bytes":12672,"dtype":"<f8","sha256":"c5e35d0696c1fea38450c6502761aea79d8576015cfab06470f47adf6bdcd832","shape":[16,11,9]},"case_006":{"bytes":26624,"dtype":"<f8","sha256":"90356a35ba06fdf79cdeb5f9c76e6160a8d2c598afb8dd0c3d2c65d83dade8d9","shape":[16,16,13]},"case_007":{"bytes":6720,"dtype":"<f8","sha256":"10c207ce5d5e065b069684d325eae9a42e9e87b7abd25e1d00356a7f8ab2d5f7","shape":[12,10,7]},"case_008":{"bytes":9408,"dtype":"<f8","sha256":"2d639814196ddc3f61e53ccc366491a2b7f45d3b5cc0bb25842f8404e7a61e9a","shape":[6,14,14]},"case_009":{"bytes":5376,"dtype":"<f8","sha256":"1c9ac97c2c6f931c0b0520d6a222f10ad1c3b8171738e35c2546acf95a3ac8fd","shape":[6,7,16]},"case_010":{"bytes":4704,"dtype":"<f8","sha256":"3ee4e02f00b1c739d1f508459bfd8c685ef267e160687b905dad20576df60d4c","shape":[7,7,12]},"case_011":{"bytes":9984,"dtype":"<f8","sha256":"cfce03699fe5f76cd9bb420c312dcf11f8a2c6ddadc39ab1e1e0a39bbb78c74e","shape":[13,16,6]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"36c3fc2a8441f92469ef4f5e44ded1e1f41a5986d362b5d0b5fea16a119cdce5","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"4d190f434449d49d0b8fc78f6d4e242aa721383b5c0bec030cc4a32d2dff1953","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"dbb64ebfb2ccbb05dca712d35e637ef8a860b0b968663afd5f012d5558584c32","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"b8a0d005f3cbc34b8d2f718d58dbfb505dc4626551cf1e3c0bb8a56833af96e6","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"919eef2f4c2e5f63302781549ca726ef0a2e53f0d3c6d334361de69c5ed7630c","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"02d18304856ce8d60740b64b5df1cbf061f66e11c82f9853798b663a568a6774","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"c87d9c1481ebc4782b09b5007de419563a9e4182fe3916816f471e8a1f3a8a72","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"ffc1aad6d473bbc25f7f9c391faf207c0fee8f072650a9739c7299b0a16e262c","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"fd0a2ebb4f2222a888c7cb48cc7070d051584f4b3e3b1d07027b2241887d5ec5","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"c8e223829200f89ec1459113890c2f0ae25f2d520d56a85a8f5185f131c04ac0","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"409dda8f5cee2c7d2fb631be1217d9378593ae430d88a86417b9f5bc8b636738","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"8db103d6efeadd4b161dd81e58ba34712a7402116ce2f39c601f7ac54d37d6ef","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2467135488,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-sealed-water-training-v1","status":"complete"}
```
