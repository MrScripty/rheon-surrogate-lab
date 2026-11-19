# water-shape-one-control · CUDA training

Retrospective source review · fit-d32713e7541281a3

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 1 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Test a single-correction control with the eight-grid rectangular-feedback cohort policy.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 1024 pressure-operator fields across 8 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre; plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 0.00060624028885060389; retained final validation objective 0.0004623835212151127. Common 128-request worst mean-free pressure relative L2 0.081132259733648196; equation relative L2 0.066551390540204952. Complete GPU pressure-call timing: median of case medians 2.0550252499999999 ms; worst case median 2.4030300000000002 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** The single-correction rectangular-bank control remains much less accurate than C33 on both common worst metrics. It improves over the corresponding five-grid single-correction control, while changing the allocation of manufactured exposures.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "d32713e7541281a362fa8ad94db0636a943f5642d840f6b779d71b97528a498f"
  },
  "history": {
    "path": "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/history.json",
    "sha256": "fdee14871e123bd6d7de70d61eb269e1a0daf0043156d4eb9b7048f543a42f01"
  },
  "sources": [
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/config.json",
      "sha256": "cc0c9eeb60b30753bede36cb79c660a22e3d8bd10cc8ae148db1c506cdddd7e9"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-shape-repeatability-v1/shape-one-control/project/source-witness.json",
    "sha256": "a7f150cd1a11a0ba313fa0b01301ea2c3cd57bf89302ec5ce338e7791a63d422"
  },
  "training_population": {
    "path": "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-shape-repeatability-v1/shape-one-control/project/inputs/validation_splits.json",
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
    "item_id": "fit-d32713e7541281a3",
    "kind": "training_attempt",
    "name": "water-shape-one-control · CUDA training",
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
        "steps": 1
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "d32713e7541281a362fa8ad94db0636a943f5642d840f6b779d71b97528a498f",
    "receipt_paths": [
      "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-shape-repeatability-v1/shape-one-control/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/d45cf84cd7084b469dfb67f8be65ea94/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-d45cf84cd7084b469dfb67f8be65ea94/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "d45cf84cd7084b469dfb67f8be65ea94",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-shape-one-control · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "cc0c9eeb60b30753bede36cb79c660a22e3d8bd10cc8ae148db1c506cdddd7e9",
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
      "source_witness_sha256": "a7f150cd1a11a0ba313fa0b01301ea2c3cd57bf89302ec5ce338e7791a63d422"
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
        "sha256": "6a1be0cd74474af3952e96793bae5709858c759e08d349b9a3a9b31a7eaab818",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "0bd8b04a3a74a415944f02969d4e12ba96d553dbeb2224125efe83deb77cb563",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "70f46bef91094f5c5066ae76479732e893cbb32b49af56644929882e9f9e8fec",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "833fdea59184bc7f897d8a9c5ffe1b0bc5c5230f5a624abbf2d46739e1cc26cd",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "2358b2629e4fd4773ef27cc018383c4517489461546325fceb5f3241979323e7",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "19f68d70bdbb28a2f78ce2ad0d3fa99554bcee5c15ebf89a4e471c2edcc9ea84",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "19f68d70bdbb28a2f78ce2ad0d3fa99554bcee5c15ebf89a4e471c2edcc9ea84",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2346065920,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/35"
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
      "receipt": "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-one-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "d32713e7541281a362fa8ad94db0636a943f5642d840f6b779d71b97528a498f",
    "bytes": 13327
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C36": {
        "name": "shape-one-control",
        "weights_sha256": "19f68d70bdbb28a2f78ce2ad0d3fa99554bcee5c15ebf89a4e471c2edcc9ea84",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.06655139054020495,
          "pressure_mean_free_relative_l2": 0.0811322597336482,
          "pressure_absolute_rms": 2.2522522484041225,
          "pressure_absolute_max": 7.058207413603089
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
              "common_equation_relative_l2": 0.040028703609157945,
              "pressure_mean_free_relative_l2": 0.04519374753168933,
              "pressure_absolute_rms": 2.2522522484041225,
              "pressure_absolute_max": 7.058207413603089
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.06655139054020495,
              "pressure_mean_free_relative_l2": 0.0811322597336482,
              "pressure_absolute_rms": 0.8899600975217629,
              "pressure_absolute_max": 2.1825024098310406
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C36/results.json",
          "sha256": "0ebd2573ecd7a589c322f777c2c6edfe18d1a75085e2f1e453550e7cda82b93f"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"cc0c9eeb60b30753bede36cb79c660a22e3d8bd10cc8ae148db1c506cdddd7e9","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"a7f150cd1a11a0ba313fa0b01301ea2c3cd57bf89302ec5ce338e7791a63d422"},"elapsed_s":112.883865041018,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.00010081150139740203,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":575062016,"peak_reserved_bytes":593494016,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"6a1be0cd74474af3952e96793bae5709858c759e08d349b9a3a9b31a7eaab818"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0bd8b04a3a74a415944f02969d4e12ba96d553dbeb2224125efe83deb77cb563"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"70f46bef91094f5c5066ae76479732e893cbb32b49af56644929882e9f9e8fec"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"833fdea59184bc7f897d8a9c5ffe1b0bc5c5230f5a624abbf2d46739e1cc26cd"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2358b2629e4fd4773ef27cc018383c4517489461546325fceb5f3241979323e7"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"19f68d70bdbb28a2f78ce2ad0d3fa99554bcee5c15ebf89a4e471c2edcc9ea84"},"history.json":{"bytes":144548,"role":"training_diagnostics","sha256":"fdee14871e123bd6d7de70d61eb269e1a0daf0043156d4eb9b7048f543a42f01"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"6d903187a4b6de6cf240aabd37bab44f02c2b443174e6213480a6fb8b83ba1bd"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":983,"role":"result_summary","sha256":"6198dfa792bee5f74c3b786759f461e333bd722d6d0d030f7a0eb9b4e49f2571"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"19f68d70bdbb28a2f78ce2ad0d3fa99554bcee5c15ebf89a4e471c2edcc9ea84"}},"per_case_inference_s":[0.012278236012207344,0.0018712410237640142,0.0023513099877163768,0.0018352999759372324,0.0016238370153587312,0.0015356910007540137,0.001965650008060038,0.0020177790138404816,0.004957791999913752,0.0021693389862775803,0.0024259490019176155,0.002242753020254895,0.005988945020362735,0.001959359011380002,0.003087886987486854,0.002046472014626488,0.0012746520224027336,0.002555356011725962,0.001654273015446961,0.001976748026208952,0.002173532993765548,0.0023549399920739233,0.00467046900303103,0.0017741549818310887],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"633bc79f07133571c8733f9e7e8ebff792fe4f012468c818ebaf208866f705fd","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"85df30ece320f47517cdc1e7ac0aacd00408d9e9999e00b607700a663e875529","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"a8b3e017ef124b7946fa8afe88aba9e65d7dd6c6a2d3d62231021af2968ee8ff","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"8d76d25d94a0bc1408b2bd1f99c5e9008e043e0cce355d5735df3c8c06062e85","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"7de8a23442280f5db69082259b0039541487176cf7a0a9ae14ae6fa423940d65","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"6175ac50955875646d6c40e8f977639e0e55fb7fad0dce9840c04e7f40c7f876","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"02fa998416c49e9409cadf9e58528f1fa5f8d0372d0a1f1cbe0aa47dc33e0e87","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"070684de095316d81f590d12c011388e09ffde62e03e6b9b0a39b3132e81ab76","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"cf9d1c512a48190a8523fd605dd927d66730b0221461dcfb510834f87d59bd8f","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"84c7365f6f2496e60d763554768ba3b9d50de9bd5075ffe35e1ff5b94a4d5b45","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"91d035e86296d474284edc90c8f44730263db4b5b6abf6fcadb3c1aad1a99004","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"7cae7c30340de5548da0a76538b7b6cd6ea5086b816249ecc6314e3bae1c762e","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"f82befdfe649eb62faa2094c05663c7eb22f65b2eb2f67174c0ffd1fe6f3fae2","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"4281d53bd7fc274be35e4da601f98e34e7f5a229ca414865b269141da62d51b1","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"2a99d68504150955c5d5734c7d2b097e69464d17e8351b39cde26aefc7b98c1a","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"cc17382e7b44e06a701e7483d10ad63e9f77a96333565d2d9ecef6172b018b0f","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"6ebd2655ff78a6f7047ba128af8c512a2aa64e2461aef9cad521f1b6a8c28f98","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"699567140fc659f570fbe91dd4c61667d98d0d621d68e6039202a8c7af23cd50","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"63e28572799dd46b844e03ac2186cc54162c01d1bb914facebaafa987d3c1881","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"8296cf6568cc009a499610fc8149928e60ad91d44c0f29058fa3bd324f791ed6","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"440805a8b533ba8028a95040e157b154335f7ea0dc8143a528e21a41065fa88a","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"a3d1c56cdeac7098fe4582a1d86470754732befb97d04a4d31d41937b58c545e","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"92faf1951eafef7307b041e499a373d039ec552d401b8e1c366aded509eb4de5","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"b321ff58561bba266702e02c962a0208a09a2db8beeb5b6d02d5c3b6b65af316","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2346065920,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
