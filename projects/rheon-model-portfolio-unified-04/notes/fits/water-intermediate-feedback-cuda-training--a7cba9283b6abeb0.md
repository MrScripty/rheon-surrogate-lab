# water-intermediate-feedback · CUDA training

Retrospective source review · fit-a7cba9283b6abeb0

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel 370,016-parameter linear pressure-correction network with 19 weight tensors, no GroupNorm/SiLU/biases despite inherited cubic-groupnorm-v3 naming. F64 equation residual is physically scaled by rho/h and train-only q RMS before F32 convolutions. Two shared-weight corrections recompute the F64 operator residual; previous-pressure RMS scales cell-zero-gauged corrections before F64 addition. Physical dt is validated but not a separate CNN channel. The sealed uniform cubic-voxel Neumann policy returns pressure only, with no obstacle/free-surface mask or physical simulation advancement.

**Purpose.** Train the first of two pressure corrections for residual contraction in addition to the final canonical objective, without changing deployed correction count.

**Experiment.** 5,000 whole-network Adam updates, batch 1, initial rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 3000}, {'learning_rate': 1e-05, 'update': 4500}], model seed 20261015, physical sample seed 2026102501, clipping 1.0. The same 16 physical trajectories / 32 frames at 8/16 cells per metre supply training; axes are at most 16 cells and physical dt is 1/60 s. Validation uses 64 trajectories / 128 frames. Manufactured feedback has 1024 fields on 8 grids, 5,000 additional proposals at weight 0.1, train-cohort correction-RMS amplitude, and balanced grid exposures [625, 625, 625, 625, 625, 625, 625, 625]. Base manufactured grids include 16/24/32/40-cubed and 24x16x12 at 16 cells/metre, plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre. Training adds 0.1 times the mean relative squared nonfinal equation residual to both actual and manufactured final losses; with two corrections this supervises the first correction. Manufactured examples are pressure-operator fields, not new fluid trajectories.

**Observed outcome.** Last logged training objective 0.00025619310417730138; retained final validation objective 1.2019895957721969e-05. Common 128-request worst mean-free pressure relative L2 0.014911085013361117; equation relative L2 0.0083245410792506108. Complete GPU pressure-call median of case medians 4.4307987500000001 ms; worst case median 4.8439755 ms. The interval includes CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization. It excludes setup/scoring and is not a physical-step or drift measurement.

**Lesson.** Intermediate contraction improves both common worst metrics over C33 at the same physical/manufactured sample budget and two-correction deployment. The larger logged training loss includes an extra term; final validation remains canonical. This does not establish complete-step or rollout improvement.

**Reproduction.** Use the original captured typed recipe/config/runtime, source witness, input slots, physical split/order, train-only normalization and seeded initial inventory. Recreate the manufactured fields from explicit shapes/densities, train-cohort correction RMS and private RNG seed, preserving proposal counts and objective. The historical trainer includes CPU deployment/parity forwards; qualify a GPU-only reproducer before rerunning under current policy. Clean retraining/optimizer resume/new-Rheon regeneration are not established here.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "a7cba9283b6abeb0be7c9565e0790f9799ea978bd3a272e796a6af11e9808ff3"
  },
  "history": {
    "path": "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/history.json",
    "sha256": "741bd9bea90fe555cf60a17b59572eba262f7a48a8d75fc4d7e1ab16165f31da"
  },
  "sources": [
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/analytical.py",
      "sha256": "2fdca0e2cb2cd4ed2301c032915c6bdd3a17fcf9763e8e6861ac436cd24a3283"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/config.json",
      "sha256": "9d8ab90a97b7a92d276310145cf1f7bd30327813519c391632fe45923cdffecf"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/model.py",
      "sha256": "0a4d8a638684056a090a4361590a6165fc03745cc9e1de62930491457f15d8c3"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/policy.py",
      "sha256": "f873dd33ef6dc2b33f1e3c59e2df3d2415d51bdfa2628c0029ed0182134ea2b1"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/train.py",
      "sha256": "f3a1f62aa67a580a03c2379cdb66e918b1ab338299125b881cf157efa46d30de"
    }
  ],
  "source_witness": {
    "path": "runs/water-coverage-feedback-v1/feedback-only/project/source-witness.json",
    "sha256": "607c93e74615b3f976ea80bc1676da06e41629b7d48ef39994fc7daa6cbc065d"
  },
  "training_population": {
    "path": "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-coverage-feedback-v1/feedback-only/project/inputs/validation_splits.json",
      "sha256": "9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5",
      "slot": "validation_splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a7cba9283b6abeb0",
    "kind": "training_attempt",
    "name": "water-intermediate-feedback · CUDA training",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-physical-residual-pressure-v1",
      "input_channels": 1,
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
      "correction": {
        "activation": "linear",
        "steps": 2
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "a7cba9283b6abeb0be7c9565e0790f9799ea978bd3a272e796a6af11e9808ff3",
    "receipt_paths": [
      "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-coverage-feedback-v1/feedback-only/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/c35ccab4c98a4ebd80409a41969a060e/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-c35ccab4c98a4ebd80409a41969a060e/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "c35ccab4c98a4ebd80409a41969a060e",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-intermediate-feedback · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "9d8ab90a97b7a92d276310145cf1f7bd30327813519c391632fe45923cdffecf",
      "inputs": {
        "dataset": {
          "bytes": 2820310,
          "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"
        },
        "dataset_extra": {
          "bytes": 2823990,
          "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"
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
          "bytes": 411192,
          "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
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
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423",
      "source_witness_sha256": "607c93e74615b3f976ea80bc1676da06e41629b7d48ef39994fc7daa6cbc065d"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "b6fcb0d1d5b41c1f2aae87a9750415295194d809789bf2be43afe9db13da3b90",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "c3785b1d5d37203626b65f061881e7f0f4abc008e8bce4f72b53066be6f1df9a",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "bbd97024a25157f1744d464a60a29c244d9dd7edcbefafc0a850a3bb5e84052a",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "1d93caf928e588ef528dd2c882e715c23b8f863c3bc8f76c162ed7f26456acd5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "df5cdde324a3297fe6c7aa1e399a2307662be667d26f1fbd3803ceb48ae23e57",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2407333888,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/39"
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
      "receipt": "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-coverage-feedback-v1/feedback-only/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "a7cba9283b6abeb0be7c9565e0790f9799ea978bd3a272e796a6af11e9808ff3",
    "bytes": 13529
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C40": {
        "name": "feedback-only",
        "weights_sha256": "74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00832454107925061,
          "pressure_mean_free_relative_l2": 0.014911085013361117,
          "pressure_absolute_rms": 0.5282980631347886,
          "pressure_absolute_max": 1.2903494715357344
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 15,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0041",
            "native_frame": 15,
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
              "common_equation_relative_l2": 0.004807088654644239,
              "pressure_mean_free_relative_l2": 0.009701749369888386,
              "pressure_absolute_rms": 0.5282980631347886,
              "pressure_absolute_max": 1.2903494715357344
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00832454107925061,
              "pressure_mean_free_relative_l2": 0.014911085013361117,
              "pressure_absolute_rms": 0.22850307685160806,
              "pressure_absolute_max": 0.36656843691347446
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C40/results.json",
          "sha256": "5cea0ab57e5bf91516d0dd66b0a9c2622a23cb60a3ea5d77f87b2c1199433790"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"9d8ab90a97b7a92d276310145cf1f7bd30327813519c391632fe45923cdffecf","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"607c93e74615b3f976ea80bc1676da06e41629b7d48ef39994fc7daa6cbc065d"},"elapsed_s":229.57262233601068,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.3876408235423696e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597156864,"peak_reserved_bytes":620756992,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"b6fcb0d1d5b41c1f2aae87a9750415295194d809789bf2be43afe9db13da3b90"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c3785b1d5d37203626b65f061881e7f0f4abc008e8bce4f72b53066be6f1df9a"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bbd97024a25157f1744d464a60a29c244d9dd7edcbefafc0a850a3bb5e84052a"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1d93caf928e588ef528dd2c882e715c23b8f863c3bc8f76c162ed7f26456acd5"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"df5cdde324a3297fe6c7aa1e399a2307662be667d26f1fbd3803ceb48ae23e57"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631"},"history.json":{"bytes":144645,"role":"training_diagnostics","sha256":"741bd9bea90fe555cf60a17b59572eba262f7a48a8d75fc4d7e1ab16165f31da"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"3b77247a8075bf220f1f69a1ebc436eb56181dbcbc5ee8b4cf7a6789e05d91de"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":983,"role":"result_summary","sha256":"af8fa02cc5a18fde71d65f1732065a6f2c86613d8caadf8e08de67e1ef525899"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"74a4c71e6e5624e5cece90e665a9dffaa0be03431511caf5990c33f1e950d631"}},"per_case_inference_s":[0.018339139001909643,0.0035447209957055748,0.00492467702133581,0.003971387981437147,0.0034911539987660944,0.0032953420013654977,0.004200398019747809,0.00437961399438791,0.008737337018828839,0.0049506089999340475,0.005771094001829624,0.0051444220007397234,0.010598444001516327,0.0038381670019589365,0.005895865004276857,0.00404661800712347,0.0024241759965661913,0.005065070989076048,0.0039379780064336956,0.004388485016534105,0.004783793003298342,0.005042061995482072,0.00958732797880657,0.0037363349983934313],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"92506fe8a332f7e7dda601b7630c1da6be02f9cb496db86348d39028dbba8555","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"151149c05a84d76a4751517e8706bf43cdae940396412b4643e0e9c8c5ee6a42","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"bd128b1472cbc1cf1f0ab10053c60f3501600326113af28001c03eeafd19eae7","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"64c1d6056da2f4fa4aeb73008dd6d1a0ddb35b13abfa3dae57086ccec5a39dba","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"3cc2adf22d327a488fd137cdfa9bea7d1c58ff6ab6105731093c7abc136b96bc","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"793ef283926bd1a58f8bfcacb94f5991ada9d871fe42d6e51950cd98029891f1","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"e6a95cf123479c1fc992d86e0f21faf78f3433747f164b01e5d6f5e446695665","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"2e84a7de47b6cd5f63ae0b8713ce61af5b1f8a8a30a7ef72942a59425c155e99","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"75b2a059e227be1005a7f9f4f20b729e0a107ec0f62ae23c2ce539428cd666d1","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"39fe747870860d581636e949043e43e6f1f31176d486a002a6193fed9ccbf6f7","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"f60f3d222591d356de51d679d1715bcad72a67a3a9135f75806f0608231ff5c7","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"f6873ad2f0355a74d43d8975bfef379e2800bb04db5084ffc496bae98e678551","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"5a3b9a78265ede87b15cd5e8b58cf9c55a9a353e271c28084d385eaf69fd54cb","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"8dddfb3b70d65cf347c81939e79cf5b756b0ddad824eeeb8db5742447d1aa785","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"5e044f452b4cd4d499c311d96e0ffb5264ac880263cedf5666ae9324f09c7a2c","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"a6a217ea781237a91d9231369508bfc53d6ce1bf4018ab64732e2dab6bd21e4c","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"a524c8b4e874cc179e183c4f4fa5af6d063cfe71c79ac65623b432ca5d36b873","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"dc3a6297ae3599a911c057166359db8c6786270a8662e4c98e461a5fca2c34be","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"6038c7880715beba102c22cb6cf44f636e52849996e3ec282a55c084338266f7","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"1de5c9dbc6ca2ffec9d91d1c1e0f5fac15b89c9a44d7b88bae2c98d2d9e5e706","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"771580224babee278128169ac7d879e9defc873907cda0a80184b99226979865","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"9980cb67c31efe77c642453af784a940724e4a86e586f771597369ff3c9d91b1","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"1caa92f8f8d2ded745ccf057946917cd1f662e4ba0f4c3ba544547a44d9cb978","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"3fb7fd0add02b53b09724eb27a9c0ed5f43eace0cd1e982bdbd601ce9fa67063","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2407333888,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
