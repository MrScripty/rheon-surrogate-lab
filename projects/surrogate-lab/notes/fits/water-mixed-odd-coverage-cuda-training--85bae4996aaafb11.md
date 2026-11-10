# water-mixed-odd-coverage · CUDA training

Retrospective source review · fit-85bae4996aaafb11

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel 370,016-parameter linear pressure-correction network with 19 weight tensors, no GroupNorm/SiLU/biases despite inherited cubic-groupnorm-v3 naming. F64 equation residual is physically scaled by rho/h and train-only q RMS before F32 convolutions. Two shared-weight corrections recompute the F64 operator residual; previous-pressure RMS scales cell-zero-gauged corrections before F64 addition. Physical dt is validated but not a separate CNN channel. The sealed uniform cubic-voxel Neumann policy returns pressure only, with no obstacle/free-surface mask or physical simulation advancement.

**Purpose.** Add three mixed odd/even rectangular manufactured grids at 16 cells/metre to the 5,000-update two-correction cohort policy.

**Experiment.** 5,000 whole-network Adam updates, batch 1, initial rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 3000}, {'learning_rate': 1e-05, 'update': 4500}], model seed 20261015, physical sample seed 2026102501, clipping 1.0. The same 16 physical trajectories / 32 frames at 8/16 cells per metre supply training; axes are at most 16 cells and physical dt is 1/60 s. Validation uses 64 trajectories / 128 frames. Manufactured feedback has 1408 fields on 11 grids, 5,000 additional proposals at weight 0.1, train-cohort correction-RMS amplitude, and balanced grid exposures [455, 454, 454, 454, 454, 455, 455, 455, 455, 455, 454]. Base manufactured grids include 16/24/32/40-cubed and 24x16x12 at 16 cells/metre, plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre. Three additions are 16x23x25 and its cyclic axis permutations at 16 cells/metre. Manufactured examples are pressure-operator fields, not new fluid trajectories.

**Observed outcome.** Last logged training objective 0.00027482889145092091; retained final validation objective 6.7348271860916069e-05. Common 128-request worst mean-free pressure relative L2 0.022446335167471035; equation relative L2 0.032289647710059405. Complete GPU pressure-call median of case medians 4.3929597500000002 ms; worst case median 4.8879539999999997 ms. The interval includes CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization. It excludes setup/scoring and is not a physical-step or drift measurement.

**Lesson.** Adding manufactured shapes does not improve these common worst errors over C33. Eleven grids redistribute 5,000 proposals into 454/455 exposures each instead of 625; this is not an equal-exposure pure shape comparison. Grid capacity 16 is a validation bound, not 16 executed grids or new CFD scenes.

**Reproduction.** Use the original captured typed recipe/config/runtime, source witness, input slots, physical split/order, train-only normalization and seeded initial inventory. Recreate the manufactured fields from explicit shapes/densities, train-cohort correction RMS and private RNG seed, preserving proposal counts and objective. The historical trainer includes CPU deployment/parity forwards; qualify a GPU-only reproducer before rerunning under current policy. Clean retraining/optimizer resume/new-Rheon regeneration are not established here.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "85bae4996aaafb119dbf14e249462885d51fae26ee15371d7e764e399fbb54d2"
  },
  "history": {
    "path": "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/history.json",
    "sha256": "97ae744518dff65c3e38e57e821e865595bc3fa662cf59abedd458f4d6959403"
  },
  "sources": [
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/analytical.py",
      "sha256": "2fdca0e2cb2cd4ed2301c032915c6bdd3a17fcf9763e8e6861ac436cd24a3283"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/config.json",
      "sha256": "cef881b71d4f55ad2fa10e2ff54ee59a7f43ddd732b53046d1032dcef338b67d"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/model.py",
      "sha256": "0a4d8a638684056a090a4361590a6165fc03745cc9e1de62930491457f15d8c3"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/policy.py",
      "sha256": "f873dd33ef6dc2b33f1e3c59e2df3d2415d51bdfa2628c0029ed0182134ea2b1"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/train.py",
      "sha256": "f3a1f62aa67a580a03c2379cdb66e918b1ab338299125b881cf157efa46d30de"
    }
  ],
  "source_witness": {
    "path": "runs/water-coverage-feedback-v1/coverage-only/project/source-witness.json",
    "sha256": "05c17bd76ff4e52549b4c5652fbbef89062db84f1a6c872f7c4a9a89c3fb7603"
  },
  "training_population": {
    "path": "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-coverage-feedback-v1/coverage-only/project/inputs/validation_splits.json",
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
    "item_id": "fit-85bae4996aaafb11",
    "kind": "training_attempt",
    "name": "water-mixed-odd-coverage · CUDA training",
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
    "receipt_sha256": "85bae4996aaafb119dbf14e249462885d51fae26ee15371d7e764e399fbb54d2",
    "receipt_paths": [
      "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-coverage-feedback-v1/coverage-only/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/f0e6db6cf3dd4128a63ce7d34381fa8b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-f0e6db6cf3dd4128a63ce7d34381fa8b/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f0e6db6cf3dd4128a63ce7d34381fa8b",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-mixed-odd-coverage · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "cef881b71d4f55ad2fa10e2ff54ee59a7f43ddd732b53046d1032dcef338b67d",
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
      "source_witness_sha256": "05c17bd76ff4e52549b4c5652fbbef89062db84f1a6c872f7c4a9a89c3fb7603"
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
        "sha256": "9dee4e2335822c62292aa291354e171eaca1a4a52fac4685e7984f9e798333ea",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "c9bef5f7078c4cfce58f3ab1d85ae252d930fbeb5712cf8278f0a4f277906817",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "1b01ce773d9245a7da024a30f3996749e414fd9c49247e00f1a707bc6ffc3ab3",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "60a9eda5936cf8c4e26f38bb9be3d65b341295885c96f15c293fad883a1056be",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "acf4b3a15154b6ad426ebe4b38e466ff1a835cbfc174fa6d5c09b291d3876ee5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2401714176,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/38"
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
      "receipt": "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-coverage-feedback-v1/coverage-only/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "85bae4996aaafb119dbf14e249462885d51fae26ee15371d7e764e399fbb54d2",
    "bytes": 13480
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C39": {
        "name": "coverage-only",
        "weights_sha256": "e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.032289647710059405,
          "pressure_mean_free_relative_l2": 0.022446335167471035,
          "pressure_absolute_rms": 0.8727402393429665,
          "pressure_absolute_max": 3.6358124231189906
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
              "common_equation_relative_l2": 0.0221952673837598,
              "pressure_mean_free_relative_l2": 0.017163390527326575,
              "pressure_absolute_rms": 0.8727402393429665,
              "pressure_absolute_max": 3.6358124231189906
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.032289647710059405,
              "pressure_mean_free_relative_l2": 0.022446335167471035,
              "pressure_absolute_rms": 0.27793080280301774,
              "pressure_absolute_max": 0.9924760124648975
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C39/results.json",
          "sha256": "616e96e6af0774664c40f5a4bc49ede649ea4a1c77e72ac0dcb46c029dc99292"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]},{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"cef881b71d4f55ad2fa10e2ff54ee59a7f43ddd732b53046d1032dcef338b67d","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"05c17bd76ff4e52549b4c5652fbbef89062db84f1a6c872f7c4a9a89c3fb7603"},"elapsed_s":214.38434672402218,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00011142323845092506,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":710992896,"peak_reserved_bytes":734003200,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"9dee4e2335822c62292aa291354e171eaca1a4a52fac4685e7984f9e798333ea"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c9bef5f7078c4cfce58f3ab1d85ae252d930fbeb5712cf8278f0a4f277906817"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1b01ce773d9245a7da024a30f3996749e414fd9c49247e00f1a707bc6ffc3ab3"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"60a9eda5936cf8c4e26f38bb9be3d65b341295885c96f15c293fad883a1056be"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"acf4b3a15154b6ad426ebe4b38e466ff1a835cbfc174fa6d5c09b291d3876ee5"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f"},"history.json":{"bytes":186431,"role":"training_diagnostics","sha256":"97ae744518dff65c3e38e57e821e865595bc3fa662cf59abedd458f4d6959403"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"24330eb425a312a965c03ccfb051ee877a7d023ec44b647d9e6a730b562da245"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"abb097cedf11b186a52fa39ba67af7641d13443caac30e5f1b00953eeb36a708"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e53a043be8c7c77c3ccca0bcaac719312ef3d034b5cd3469333e530ac37ab91f"}},"per_case_inference_s":[0.02012273602304049,0.003712441015522927,0.004781365016242489,0.003857635980239138,0.0034620279911905527,0.003031467000255361,0.003456047998042777,0.003788058995269239,0.0076367360015865415,0.0040468869847245514,0.00512916999286972,0.004537080007139593,0.011131156992632896,0.004332701995735988,0.006695552991004661,0.004567661002511159,0.002881563996197656,0.005775461002485827,0.004098471021279693,0.004391841997858137,0.00462817499646917,0.004630603012628853,0.008473647991195321,0.0034125959791708738],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"e4f72a2b43a49cd0980956aaa020abcea60725eb1fc0173d5f02966b4b43c478","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"d5cbd45481a90897b76453f0e1b9303ad95aa2167eeba0c1caa59de821357f90","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"71b780d30f7d388e8f8b5c46042ecb0ad4b9a581a779af0cf8b275bb43edfc94","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"5b6d4f9cab262d4303605e73b3c89298a386c873a158d5ab245704f8477c4f61","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"996ce182642ca0bff1124d6e717c8a9cf7558ee895ac683f19f2eae277ad5abd","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"28f7356f70a914c32637e6371bcad58bdda2c7845abdf23a4a56ccfa2a832455","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"83dd06f388b73420a046b1d70cc428050fbd2d02295e4457c57adee329166481","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"18d5285c0aaad15d39ea5497b6504a4eec162dfa3b9335de723ae5e8430233c1","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"02b2c1d48004b67d89435bdc70fe7ea662363646c44c1e70a58db1014c303214","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"3f8510268d81685728e35767d5fac069d39c1438e8d37439c481505f1fe5b667","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"abba6b27e01b312e4c6e4dcb05b1eca3b079af67f1a7624373208140d52ad23a","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"c5d8cbf3f47e95c0bd13b01cdf87f97ddb606b459a92126d798d2c6babcd5fb1","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"325eb8842a5199acb964a5df46f02b418ba62adb012f5ccde822e5a899ca34d0","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"82bc5807fce63a3fdaeb484d4dc0f6c4ec5ab3bfee686c561897c9261facad62","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"65f2c613a159bba5d6b134967c139da34986a21d7e4cdcca6471b33cc96d2cac","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"67dc46faf8840c9b8fbc7398bb600a8f69bba32e02133cedbf0cb5592ab2b17e","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"f9178023402683f20aacc4bd010a054781455d4e930fae6c93a6a113e5add58e","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"a0c3e72371337789be5de71aa3b1ecf720a422f55670aa3bbb637a73324f13bf","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"b92948d721a62e60740718a501f17675b0d04e2dddc73619d2adbd4939fb1944","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"36ae128b7c4e702ac8c569b5cc6fedf930f90a1c3b44d5bb2f0d7d42193374c2","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"26bf878295242829570a9bbf91d421891cc831f8fcc109b45ec0e18bcf6bd8ef","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"bc168a9b87f89b2b9c01983763eb5aa6c4c4290a7be096817e2b41a7e7e44f6e","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"053ba62ee3afa2af3eec844bbad0a40c6249c027c2ab40e3ee7f8ad6e0c25931","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"77f765f46e190293f65688ad9d41edd40ceb27a306193f040df9a34fdc20b882","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2401714176,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
