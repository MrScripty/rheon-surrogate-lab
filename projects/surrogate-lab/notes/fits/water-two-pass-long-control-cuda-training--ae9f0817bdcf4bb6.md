# water-two-pass-long-control · CUDA training

Retrospective source review · fit-ae9f0817bdcf4bb6

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel 370,016-parameter linear pressure-correction network with 19 weight tensors, no GroupNorm/SiLU/biases despite inherited cubic-groupnorm-v3 naming. F64 equation residual is physically scaled by rho/h and train-only q RMS before F32 convolutions. Two shared-weight corrections recompute the F64 operator residual; previous-pressure RMS scales cell-zero-gauged corrections before F64 addition. Physical dt is validated but not a separate CNN channel. The sealed uniform cubic-voxel Neumann policy returns pressure only, with no obstacle/free-surface mask or physical simulation advancement.

**Purpose.** Extend the two-correction cohort policy to 10,000 optimizer updates, moving learning-rate drops later while keeping eight manufactured grids.

**Experiment.** 10,000 whole-network Adam updates, batch 1, initial rate 0.001, schedule [{'learning_rate': 0.0001, 'update': 6000}, {'learning_rate': 1e-05, 'update': 9000}], model seed 20261015, physical sample seed 2026102501, clipping 1.0. The same 16 physical trajectories / 32 frames at 8/16 cells per metre supply training; axes are at most 16 cells and physical dt is 1/60 s. Validation uses 64 trajectories / 128 frames. Manufactured feedback has 1024 fields on 8 grids, 10,000 additional proposals at weight 0.1, train-cohort correction-RMS amplitude, and balanced grid exposures [1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250]. Base manufactured grids include 16/24/32/40-cubed and 24x16x12 at 16 cells/metre, plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre. Manufactured examples are pressure-operator fields, not new fluid trajectories.

**Observed outcome.** Last logged training objective 3.6252521952526786e-06; retained final validation objective 1.0148227875528465e-05. Common 128-request worst mean-free pressure relative L2 0.0098655587432741661; equation relative L2 0.0096752632632077975. Complete GPU pressure-call median of case medians 4.3939354999999995 ms; worst case median 5.0676744999999999 ms. The interval includes CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization. It excludes setup/scoring and is not a physical-step or drift measurement.

**Lesson.** Both common worst errors improve over C33, but training budget and learning-rate schedule both change. More optimizer work does not imply more inference corrections or extra physical trajectories.

**Reproduction.** Use the original captured typed recipe/config/runtime, source witness, input slots, physical split/order, train-only normalization and seeded initial inventory. Recreate the manufactured fields from explicit shapes/densities, train-cohort correction RMS and private RNG seed, preserving proposal counts and objective. The historical trainer includes CPU deployment/parity forwards; qualify a GPU-only reproducer before rerunning under current policy. Clean retraining/optimizer resume/new-Rheon regeneration are not established here.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "ae9f0817bdcf4bb680c8a2272b64b336a09bbeb5154639c681eb38b2d14c4613"
  },
  "history": {
    "path": "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/history.json",
    "sha256": "f4f8f337e5227fdda60cd4d9094478f08e7023cc4106fac77390b0fb04cbbd3a"
  },
  "sources": [
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/config.json",
      "sha256": "9fd5d1f5e9813605ed7db14ebca0d170d6d0c14a6666a598fa4fe19c5a7de522"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/qualification.json",
      "sha256": "f18f046e62f96816bc23152256b736ae56e9f018de613b123402961915e39045"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-two-pass-long-v1/two-pass-10000/project/source-witness.json",
    "sha256": "4ed8fa3b78ebad31356fe4404a42a5f04de7c6a2f3a6c954a64d1e55eeac7569"
  },
  "training_population": {
    "path": "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-two-pass-long-v1/two-pass-10000/project/inputs/validation_splits.json",
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
    "item_id": "fit-ae9f0817bdcf4bb6",
    "kind": "training_attempt",
    "name": "water-two-pass-long-control · CUDA training",
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
    "receipt_sha256": "ae9f0817bdcf4bb680c8a2272b64b336a09bbeb5154639c681eb38b2d14c4613",
    "receipt_paths": [
      "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-two-pass-long-v1/two-pass-10000/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/816e6c5815e549a8aa73c26152f5df7b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-816e6c5815e549a8aa73c26152f5df7b/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "816e6c5815e549a8aa73c26152f5df7b",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-two-pass-long-control · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "9fd5d1f5e9813605ed7db14ebca0d170d6d0c14a6666a598fa4fe19c5a7de522",
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
      "sample_order_sha256": "b17001b4396a76c214341720fc66246ff8623e2a11adac984bfe68e73afcdf66",
      "source_witness_sha256": "4ed8fa3b78ebad31356fe4404a42a5f04de7c6a2f3a6c954a64d1e55eeac7569"
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
        "sha256": "ee906c0d31006e70cdd870c21648ca92e5ef191f67cc37d8de93e7df24753036",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "a1b50f438935f826634d39e661d7638f3f57f7d7b58490c1f622996c1a784914",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "3e415f0e35cb2b2cfe6bfd0dd18d5f3dd544e7490bd10bae748dba68a7d2734d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "df78485649606ae94e1fca8ef85914122fe4759f285e72eb08d0629c1059fe9b",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-6000.npz",
        "sha256": "572f74bdd58835d65fa8881e960c01eb294bcce8d74b823c22d201ab1f847ad5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-8000.npz",
        "sha256": "02ef52bfc06463b83ae9b235a52619c41c477c9872e099d1b141f818f5e2b2d1",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2405531648,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/37"
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
      "receipt": "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-two-pass-long-v1/two-pass-10000/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "ae9f0817bdcf4bb680c8a2272b64b336a09bbeb5154639c681eb38b2d14c4613",
    "bytes": 13460
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C38": {
        "name": "two-pass-10000",
        "weights_sha256": "4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.009675263263207797,
          "pressure_mean_free_relative_l2": 0.009865558743274166,
          "pressure_absolute_rms": 0.4133786230605646,
          "pressure_absolute_max": 1.1797363450615224
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
              "common_equation_relative_l2": 0.009202589470666222,
              "pressure_mean_free_relative_l2": 0.0075913505569297,
              "pressure_absolute_rms": 0.4133786230605646,
              "pressure_absolute_max": 1.1797363450615224
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.009675263263207797,
              "pressure_mean_free_relative_l2": 0.009865558743274166,
              "pressure_absolute_rms": 0.09690249625669495,
              "pressure_absolute_max": 0.2922319422588688
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C38/results.json",
          "sha256": "ecb8a3cbc739f865e8ae4f11f23d2c1e59c379009c84b7f155a93fcb9f057424"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"9fd5d1f5e9813605ed7db14ebca0d170d6d0c14a6666a598fa4fe19c5a7de522","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"b17001b4396a76c214341720fc66246ff8623e2a11adac984bfe68e73afcdf66","source_witness_sha256":"4ed8fa3b78ebad31356fe4404a42a5f04de7c6a2f3a6c954a64d1e55eeac7569"},"elapsed_s":410.20289667500765,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.7575689412296924e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597156864,"peak_reserved_bytes":618659840,"tf32":false},"mode":"train","optimizer_updates":10000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ee906c0d31006e70cdd870c21648ca92e5ef191f67cc37d8de93e7df24753036"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a1b50f438935f826634d39e661d7638f3f57f7d7b58490c1f622996c1a784914"},"checkpoint-10000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3e415f0e35cb2b2cfe6bfd0dd18d5f3dd544e7490bd10bae748dba68a7d2734d"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"df78485649606ae94e1fca8ef85914122fe4759f285e72eb08d0629c1059fe9b"},"checkpoint-6000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"572f74bdd58835d65fa8881e960c01eb294bcce8d74b823c22d201ab1f847ad5"},"checkpoint-8000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"02ef52bfc06463b83ae9b235a52619c41c477c9872e099d1b141f818f5e2b2d1"},"history.json":{"bytes":157788,"role":"training_diagnostics","sha256":"f4f8f337e5227fdda60cd4d9094478f08e7023cc4106fac77390b0fb04cbbd3a"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"b8d1baff1dc83b401b35559f15ad0cacee7f47788216e64f8c5daa9a49bd8c20"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":985,"role":"result_summary","sha256":"8b759d82b5e5cb76e31b71db7864763ea5c4abe58d81740a764fff7006a22f17"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"4d43f7866a291aceafa31755609e720349f7f633e8b94c20169f3375e7308cf2"}},"per_case_inference_s":[0.018879245995776728,0.0035949300217907876,0.004613133991369978,0.0035343780182302,0.0031195089977700263,0.002916494006058201,0.003691585996421054,0.0038234289968386292,0.008067173999734223,0.004967577988281846,0.006564562005223706,0.005532827985007316,0.011393174005206674,0.004119467019336298,0.006325446011032909,0.004163681995123625,0.0026850939902942628,0.005522867984836921,0.004102846985915676,0.004330736002884805,0.004564267990645021,0.004798219015356153,0.009172427002340555,0.003583826997783035],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"2a0a9a3dd491c49be8b9d4679442a735104101af7d90748bafa75c3cc9504c02","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"cdf4d7c90b7c2568a85a70b59f571729033f036ec3eb0ae113d91c940b9df008","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"137414da166133da8bc633a7f9058743e972a7ea507ff52173cee2193846004b","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"ab63c1555961c5b16d59764189c96e386f1eb2e1f56edb78c47ec276d7ec2788","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"cd9be925825ab02ff7144af01b236307b8fa3c0742f459714d837b24b2b52964","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"56b54f9f457fc6485fdf79ee92752907ddeacacc4d9055094adc5eba420ffa81","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"75787267c9bdd108503a0ba2a4f1e67b0196556c0efe4612e5f9312755b52b9b","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"b13b7d26f2917d0919352a3e7f4706df9ac9686b4cee321c65adcaf600d42463","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"50aaf91f2b2489820f0cf0f9497ceecd1f7e1bb1be03bf160f34ab384274975b","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"3603248ecee218c0a82510dd0b25ec282e2ef3c670421413a1e17905dce546f1","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"b8a7581bcac367000985ee3b04c3d2fdb3c681875772ddec33f0d9c66740ce30","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"0857dbb8e7064fd4fe7135244605c29e97ca33ddc52734451fb5f2b3d2e3202e","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"9ab9c7506ffce95b509f60ab9fef5ccaf1a176cf92ae2c35a20f45cec1a5d342","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"af4805d77f6e872eece8f0940d7696888b58145414df62a68226b3325829fc76","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"efa29beb2532df516eff09686a67d0d5c68d9305afd1af88fdf9003ef337a176","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"be2e40a9e0eb083c8758991295da0c46163dc411c837454f881e968cb6cb0f46","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"0d4aeb4c43691722f06bb6deee5ca8998ef474d8620526acccda4e963ffcf0cf","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"4858d70762eb48084e4c161caec6c41b1cbaf994d505a9fa62f795fa697416b2","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"0127208a13b297f9ba5ce131ae5cd1671ab41801b7d82608b6d4791f997d20b5","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"d3c5a8ec37bb44496c8c56ae95cb9a8b8112fce20ef297d47e1453cfb609f0e5","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"8a7dacf5eda0c60896263e49cad2110b88fe3b5cffb59d974309bb0a13b9f2f5","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"8fd0e628c8c7c0900d84d9d99f1a9b805b9d61498bde5d46c4826aff577c184b","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"6f399426cccf7cb48bb9eb19fba5b191abf216fc1e7ee0f6818161ab8111aaf6","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"4f1258ec676abffb54bdbc4de206ecc2ddc0351ccc9d68dd99b813361e6ca962","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2405531648,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
