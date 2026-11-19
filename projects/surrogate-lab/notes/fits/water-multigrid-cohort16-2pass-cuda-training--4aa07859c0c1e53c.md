# water-multigrid-cohort16-2pass · CUDA training

Retrospective source review · fit-4aa07859c0c1e53c

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Replace one-frame fitting with a 16-trajectory physical cohort and use its cell-weighted correction RMS for two-correction manufactured feedback.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 9.2866338626065192e-05; retained final validation objective 2.4947966171120651e-05. Common 128-request worst mean-free pressure relative L2 0.016669687744681173; equation relative L2 0.019291235379702617. Complete GPU pressure-call timing: median of case medians 4.4623945000000003 ms; worst case median 4.9730635000000003 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** Broader physical fitting and train-cohort amplitude calibration improve both common errors over C27. Both factors change, so the effect cannot be assigned to scene count alone. The 16 physical trajectories remain small grids; the larger shapes are manufactured operator fields.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "4aa07859c0c1e53c2b17faf8b755a5266489c572272c8ece24533f6b831245de"
  },
  "history": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/history.json",
    "sha256": "c577d321448fa4325eba3410a0d832f0ea00c365c756f881ce7ac7412a4981ca"
  },
  "sources": [
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/config.json",
      "sha256": "0a3c865063977763959e3786f6f76d0c51b27e7067ed1cc8f3efa57c4f697d50"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/source-witness.json",
    "sha256": "b17d45ba93a79a1f965fd4d771fb9b0b39d856f2897935715143baa32f16cd3f"
  },
  "training_population": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/project/inputs/validation_splits.json",
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
    "item_id": "fit-4aa07859c0c1e53c",
    "kind": "training_attempt",
    "name": "water-multigrid-cohort16-2pass · CUDA training",
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
    "receipt_sha256": "4aa07859c0c1e53c2b17faf8b755a5266489c572272c8ece24533f6b831245de",
    "receipt_paths": [
      "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/8a4f5c778e73408bb784375ae9a94689/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-8a4f5c778e73408bb784375ae9a94689/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "8a4f5c778e73408bb784375ae9a94689",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-multigrid-cohort16-2pass · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "0a3c865063977763959e3786f6f76d0c51b27e7067ed1cc8f3efa57c4f697d50",
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
      "source_witness_sha256": "b17d45ba93a79a1f965fd4d771fb9b0b39d856f2897935715143baa32f16cd3f"
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
        "sha256": "afa54aa4119847ec738f6c50f742c2e6d153ff1d0f6b7670791a03e5dbab29bf",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "8bb28895d8f06597124245996be417c63972b365754f5b342d13184c4208c645",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "4c70adccb45f1b5baf50bbb5ca5240b11a1cacf3f9593af92f04eca66ac59d22",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "8eeefe888d0dd4b6add6310c8449a5bfa4554b94a3a32ae582be9429d1e32c56",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "cf570c98e1df3628b99076f25bef13c9fc265f29034d9a73e24593877e76e51c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2405244928,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/27",
      "legacy-models/3"
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
      "receipt": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-cohort16-2pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "4aa07859c0c1e53c2b17faf8b755a5266489c572272c8ece24533f6b831245de",
    "bytes": 13186
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C28": {
        "name": "multigrid-cohort16-2pass",
        "weights_sha256": "0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.019291235379702617,
          "pressure_mean_free_relative_l2": 0.016669687744681173,
          "pressure_absolute_rms": 0.5109315078296004,
          "pressure_absolute_max": 1.6381512612590106
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
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 15,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.013618196809443303,
              "pressure_mean_free_relative_l2": 0.009689385169077209,
              "pressure_absolute_rms": 0.5109315078296004,
              "pressure_absolute_max": 1.6381512612590106
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.019291235379702617,
              "pressure_mean_free_relative_l2": 0.016669687744681173,
              "pressure_absolute_rms": 0.2821880262548056,
              "pressure_absolute_max": 0.7469658742792479
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C28/results.json",
          "sha256": "92ff4149f79acb5559a76caa78154a6de718ec9e8dbe429cd3ef6c800492dbc6"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"0a3c865063977763959e3786f6f76d0c51b27e7067ed1cc8f3efa57c4f697d50","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"b17d45ba93a79a1f965fd4d771fb9b0b39d856f2897935715143baa32f16cd3f"},"elapsed_s":219.80613070799154,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":8.025126106758762e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":555410432,"peak_reserved_bytes":578813952,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"afa54aa4119847ec738f6c50f742c2e6d153ff1d0f6b7670791a03e5dbab29bf"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8bb28895d8f06597124245996be417c63972b365754f5b342d13184c4208c645"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"4c70adccb45f1b5baf50bbb5ca5240b11a1cacf3f9593af92f04eca66ac59d22"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8eeefe888d0dd4b6add6310c8449a5bfa4554b94a3a32ae582be9429d1e32c56"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"cf570c98e1df3628b99076f25bef13c9fc265f29034d9a73e24593877e76e51c"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6"},"history.json":{"bytes":103288,"role":"training_diagnostics","sha256":"c577d321448fa4325eba3410a0d832f0ea00c365c756f881ce7ac7412a4981ca"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"d5284014e9973bf5babd3c095200d75f0b065818031b4e31473180f88f38e3f9"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"904cf7c17f21710533d5a5141e939af15c1368ea9c956e768d215909ba0153d9"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0de683cccc3fb878c3310f098676d08bcda519defe3c13275b768fd4931bb4d6"}},"per_case_inference_s":[0.019722858007298782,0.0038548349984921515,0.0049033779941964895,0.0038262539892457426,0.003391281992662698,0.003221018996555358,0.0039759399951435626,0.004279399989172816,0.008431241993093863,0.004212542000459507,0.005021732009481639,0.00442359599401243,0.010127185989404097,0.003950982994865626,0.006337364990031347,0.004406182997627184,0.00278648899984546,0.005845077976118773,0.0042213119741063565,0.004390725021949038,0.004629533999832347,0.004843727016123012,0.00963650201447308,0.0037386420008260757],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"ede563af5c4a11a9a1109f532f8d3ef4d3e71ac779e984acbc6449839667d138","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"0ad9d10f793ad96c30a42dffb035d2bd941b7f3ab1a36a1b77b5ec7c77012c3d","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"26f74465df1a552d6e5b941f3fa400c22ad2c9a27caf7cfbfd94116f7299b03c","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"2a103c91aafb6cf9a20f24e12cc1e94f2c788e0ca2284883785478a320ff6cbb","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"38db2f057c045f8235d4225edd18d0458275625fdcbcb22a6b04cc774b3bfbe1","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"f98e952aad70d6501af861b925fabb592988404a1dbfe93af1318b74cddac6ba","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"bee1bc2a1bce7b5d6b4a085512df8c9169630f6c591b043643f755f7d3151644","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"e622390c9871c421df3f4d11a864a5c6c49135188b9ac505241a6c62a3005809","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"b3c1921922dbb5dab30af76912d09e27fc726c3195258d1ae453af703234c3c7","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"9abb42820bff55d9b35f878f6ae0814d7b86c02db62f2f842a5b1d6a0c678a9d","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"624ace39bec879bb0c522327cfb05b337078d666e79eae544da3e36f6ae3d0cb","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"1f3ac77d7085d3debeb0475e58bcededfff0f91c65da9ffef1754e7a6b20fcd3","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"169f248909ca929dad427d38257573542df64bb9115c595fc85f2dc2d43bfe0c","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"9525ce4d3e5f0f0a80c6280db3031f426715ddb213572fe15e82f3ae035c62e3","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"9fea4093b02a036f3dc652621588b4340fdf75e0f0091c4ad63d804bf913bdae","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"50f14d530e0db0ff1c6dc3684934ac3a0567b5909bc9c3ec72d0497340164651","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"19b8cdf3a7ab86a66d672cb5ce6898ae9363f8b9d9bb83a57420d24c1d3f337a","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"8a07d4cf9392dab1eeeab66c28e2f2107bc55c855ac68afd644a48a9f9c43a1d","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"301c9595d6a800a71998b81b8597a52cf74d888738b47aa43cd07ae6993f9dbb","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"df6a7c4f149eea28bb2ddc77f4ebd78c67d0431967da4d32ca0da1ed75bd9099","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"60d69a7ee0d8a15594be9f2b591a5bb5a9c8670b6425cc96f3ef96e548b338de","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"f1d32691dfee1cd0b65c9a71f955bcf1d3b062cffbf96b2d4181f0ff3d820d51","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"2b98bd2c3e074d01fb5b14600dec4298ac61da1236bb03f256a857297b78e04a","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"e7625b645abf9c913e0f2c278476cb2f54a07f20e537fd8b7186f814181e2483","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2405244928,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
