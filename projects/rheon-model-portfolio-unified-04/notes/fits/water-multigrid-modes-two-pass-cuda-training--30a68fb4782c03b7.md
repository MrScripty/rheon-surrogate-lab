# water-multigrid-modes-two-pass · CUDA training

Retrospective source review · fit-30a68fb4782c03b7

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Test two shared-weight pressure corrections with the same single physical frame and five-grid manufactured bank as C26.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 1 trajectories / 1 frames, 8 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.1793165157463439 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from the single physical train correction. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 3.1274505750482102e-05; retained final validation objective 0.00012563808089289592. Common 128-request worst mean-free pressure relative L2 0.042275540022721977; equation relative L2 0.047725731109623415. Complete GPU pressure-call timing: median of case medians 4.3989867500000006 ms; worst case median 4.8079450000000001 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** Two corrections improve both common worst errors over C26 with matching initialization, physical samples and manufactured draws. These corrections consume work inside a pressure call and produce no additional simulation frames.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "30a68fb4782c03b706f0d74a7ae9be90fabbc1c8008e2414f7564f3cf2edcd3a"
  },
  "history": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/history.json",
    "sha256": "437a271a7a844fde7a08439f602b01a7a45641034a8881e04b63ebae73e77ff2"
  },
  "sources": [
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/analytical.py",
      "sha256": "8443a6234a5b9ea1217f7fb07cbc4bdc7afb36d158f0193c988a36cf1e39a883"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/config.json",
      "sha256": "ea8a73863388685002cc90871ce891fa0164e0c15aa5cbf1113d0c9eef7da776"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/model.py",
      "sha256": "c35b1bfa491fb966ba062243c25659c3cd02ea75031a50ac87afdb5290258be2"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/policy.py",
      "sha256": "faccd15f908436b77e8d8ee07dcc260d59f31bc7e52dde57c03fa820c5a89618"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/qualification.json",
      "sha256": "450e41f89d91d56b8de4d24ad8d24c3d31d44757d8cb6bd51517955d393643c4"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/train.py",
      "sha256": "baaacbebd73377c51299adde2b872aa61dd153a96b4d20b9c569229b044b5b99"
    }
  ],
  "source_witness": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/source-witness.json",
    "sha256": "ade85033a95dcba69e52e05bf4fecb8f583cdb8ea603913dd6c51a7aecf0a059"
  },
  "training_population": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/project/inputs/validation_splits.json",
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
    "item_id": "fit-30a68fb4782c03b7",
    "kind": "training_attempt",
    "name": "water-multigrid-modes-two-pass · CUDA training",
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
    "receipt_sha256": "30a68fb4782c03b706f0d74a7ae9be90fabbc1c8008e2414f7564f3cf2edcd3a",
    "receipt_paths": [
      "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/ba3c39d443fd4cd387d8ae74a3440a95/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-ba3c39d443fd4cd387d8ae74a3440a95/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ba3c39d443fd4cd387d8ae74a3440a95",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-multigrid-modes-two-pass · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "ea8a73863388685002cc90871ce891fa0164e0c15aa5cbf1113d0c9eef7da776",
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
      "normalization_sha256": "73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0",
      "runtime_manifest_sha256": "a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d",
      "sample_order_sha256": "f8c784aa6b57396e7c5e094c34d079d8252473e46e2f60593a921dbebf941fcc",
      "source_witness_sha256": "ade85033a95dcba69e52e05bf4fecb8f583cdb8ea603913dd6c51a7aecf0a059"
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
        "sha256": "db1806368a28e2d0aed594d40d00e99ba97cff34dddd1a879316e6ef84af6ff1",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "f580ecaccdc7aec44aee9e75e634a68e9cfd955de3ef0a313eb74ea1f121722a",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "a45292ee9c18727305ffd2f70d1797a37879a213709c0da90f471ea561d7675f",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "674a9832a133107d09779cc451857e0a18a5af7663405071f2f8d1bc63790b83",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "894b571f287446bc14ba941f4b4658e1438ede37cae6fa2601877cec2a7aac75",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "a119e93560638208573e8e914410e56cc86a37c079e968550b6bf10eb304161d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "a119e93560638208573e8e914410e56cc86a37c079e968550b6bf10eb304161d",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2376929280,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/26",
      "legacy-models/2"
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
      "receipt": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-two-pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "30a68fb4782c03b706f0d74a7ae9be90fabbc1c8008e2414f7564f3cf2edcd3a",
    "bytes": 9825
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C27": {
        "name": "multigrid-modes-two-pass",
        "weights_sha256": "a119e93560638208573e8e914410e56cc86a37c079e968550b6bf10eb304161d",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.047725731109623415,
          "pressure_mean_free_relative_l2": 0.04227554002272198,
          "pressure_absolute_rms": 1.2669414802932935,
          "pressure_absolute_max": 4.571195621314755
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
            "native_frame": 63,
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
              "common_equation_relative_l2": 0.023483012000975686,
              "pressure_mean_free_relative_l2": 0.03240971268064653,
              "pressure_absolute_rms": 1.2669414802932935,
              "pressure_absolute_max": 4.571195621314755
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.047725731109623415,
              "pressure_mean_free_relative_l2": 0.04227554002272198,
              "pressure_absolute_rms": 0.7582042071140616,
              "pressure_absolute_max": 2.00544694577655
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C27/results.json",
          "sha256": "d2882f0ffc14630bd52b63e71de4a1effbabf2880c7e792e013293f6b25b9b37"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"ea8a73863388685002cc90871ce891fa0164e0c15aa5cbf1113d0c9eef7da776","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"f8c784aa6b57396e7c5e094c34d079d8252473e46e2f60593a921dbebf941fcc","source_witness_sha256":"ade85033a95dcba69e52e05bf4fecb8f583cdb8ea603913dd6c51a7aecf0a059"},"elapsed_s":217.33233081299113,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.5657956945888145e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":554265088,"peak_reserved_bytes":576716800,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"db1806368a28e2d0aed594d40d00e99ba97cff34dddd1a879316e6ef84af6ff1"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"f580ecaccdc7aec44aee9e75e634a68e9cfd955de3ef0a313eb74ea1f121722a"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a45292ee9c18727305ffd2f70d1797a37879a213709c0da90f471ea561d7675f"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"674a9832a133107d09779cc451857e0a18a5af7663405071f2f8d1bc63790b83"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"894b571f287446bc14ba941f4b4658e1438ede37cae6fa2601877cec2a7aac75"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a119e93560638208573e8e914410e56cc86a37c079e968550b6bf10eb304161d"},"history.json":{"bytes":102869,"role":"training_diagnostics","sha256":"437a271a7a844fde7a08439f602b01a7a45641034a8881e04b63ebae73e77ff2"},"normalization.json":{"bytes":1384,"role":"preprocessing","sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"7d74f7e225c5e82faad3e4c26679cce68fec3fe372f1c292b635467814763584"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":837,"role":"result_summary","sha256":"cb1152cc806d0d4bc3996c5bcb9219dbbfabfd6737ebc4912239fb04bd504ffa"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a119e93560638208573e8e914410e56cc86a37c079e968550b6bf10eb304161d"}},"per_case_inference_s":[0.020489881018875167,0.011160891997860745,0.004436140996403992,0.006792530009988695,0.00442871599807404,0.002791827020701021,0.0058211470022797585,0.004139554017456248,0.004283532995032147,0.004665987013140693,0.0048729490081314,0.009647244994994253,0.003856136987451464],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"5c2a3352c0fc86ae81ae6e4407a4f2fc23bf9ac4b4757db3a788d6ab71d7fa9c","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"ae4fe5084653d2be99f2edc55eb13692ab73d17775294c56dab097f60c472ddb","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"0e73b9f02bf645da7f4f63a2dfe798ff9279fbef9e36ebb9d6f43027a3601855","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"9806406dfa4f198943b04d63978dd476b1c81a35d1c88f8e7635b362b665518e","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"9056b900468515358be3dcecbdd9d9970a4fbcf53e51d4061732e914056e6fb9","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"8376b7e59705d16a120d47dc74084853b43abfcc21da2302acfb92fea58e54fc","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"a585ba2caafa940d914868c9a4a9d196c2c8a7d6a79425225358c728118ce69b","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"d0723d8eab05d642a8c2ccaac7872fd6d89a22f4d4c62860740ecbaed7de3803","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"e25f2d3ebc5ccf6a6ef006693c381ac7614b33e0a3403799319a2aee813b857b","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"8dddf560452bb0384e78c9fdc936fe6bb3b46d3952c6b9fcecffe1c10ccd5913","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"f8741d8046d0216801cd9e037f69851760d5e0e4caa21bbd02804e2fa75469b7","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"59134754af5de68223bd10bfce8874b3ca01a1e03698c5ef7426f993edb6776c","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"e938b288b6f48234671a2fb471ee8e9925087e10a1d538f319361d84d490b795","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2376929280,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
