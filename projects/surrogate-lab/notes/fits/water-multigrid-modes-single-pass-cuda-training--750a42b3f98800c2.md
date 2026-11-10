# water-multigrid-modes-single-pass · CUDA training

Retrospective source review · fit-750a42b3f98800c2

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 1 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Expand manufactured Neumann feedback to five domain shapes, including 24/32/40-cubed grids, while retaining one physical training frame and one correction.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 1 trajectories / 1 frames, 8 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.1793165157463439 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from the single physical train correction. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 0.00039810615477439541; retained final validation objective 0.00091557174761755218. Common 128-request worst mean-free pressure relative L2 0.15028938587399235; equation relative L2 0.094272124194715284. Complete GPU pressure-call timing: median of case medians 2.0983607499999999 ms; worst case median 2.3677605000000002 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** Manufactured 32/40-cubed operator fields are real additional training work, but are not large-grid CFD scenes. A one-correction fit still has substantial common error; larger synthetic shapes alone do not establish accuracy at larger simulation domains.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "750a42b3f98800c2f4b00848051566ed77f78aed7fdd7e93f37b8812fef738ef"
  },
  "history": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/history.json",
    "sha256": "fc4cd38a42f10de1bfe893cb699755fcf7e96e3158a0d1d2a308ce358bfb3e4e"
  },
  "sources": [
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/analytical.py",
      "sha256": "8443a6234a5b9ea1217f7fb07cbc4bdc7afb36d158f0193c988a36cf1e39a883"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/config.json",
      "sha256": "6c8cfc29504c6246a7e8c43a94837f45a352b84da3e0253d52815c7dea9647de"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/model.py",
      "sha256": "c35b1bfa491fb966ba062243c25659c3cd02ea75031a50ac87afdb5290258be2"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/policy.py",
      "sha256": "faccd15f908436b77e8d8ee07dcc260d59f31bc7e52dde57c03fa820c5a89618"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/qualification.json",
      "sha256": "450e41f89d91d56b8de4d24ad8d24c3d31d44757d8cb6bd51517955d393643c4"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/train.py",
      "sha256": "baaacbebd73377c51299adde2b872aa61dd153a96b4d20b9c569229b044b5b99"
    }
  ],
  "source_witness": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/source-witness.json",
    "sha256": "382fdd2ec7b02147f0a05b500b867c0d6d5de5fdcf65c665172b1b6a67155686"
  },
  "training_population": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/splits.json",
    "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"
  },
  "training_inputs": [
    {
      "bytes": 2275660,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/dataset.npz",
      "sha256": "6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1",
      "slot": "dataset"
    },
    {
      "bytes": 2278650,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/dataset_extra.npz",
      "sha256": "4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 374928,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/splits.json",
      "sha256": "80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/project/inputs/validation_splits.json",
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
    "item_id": "fit-750a42b3f98800c2",
    "kind": "training_attempt",
    "name": "water-multigrid-modes-single-pass · CUDA training",
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
    "receipt_sha256": "750a42b3f98800c2f4b00848051566ed77f78aed7fdd7e93f37b8812fef738ef",
    "receipt_paths": [
      "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/f66927a8b70a47a0873f3730b55c6f7c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-f66927a8b70a47a0873f3730b55c6f7c/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f66927a8b70a47a0873f3730b55c6f7c",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-multigrid-modes-single-pass · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "6c8cfc29504c6246a7e8c43a94837f45a352b84da3e0253d52815c7dea9647de",
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
      "source_witness_sha256": "382fdd2ec7b02147f0a05b500b867c0d6d5de5fdcf65c665172b1b6a67155686"
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
        "sha256": "1e09e0fafd52e34508a416ce2aa4763d717b156d844a41f1cbf9acf180fa83b8",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "21ceb5e49b98be627058692ce3371306a02c909e939e1cc98c8a4a8de8cbd8e7",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "94d01940b1e4b2c432855cc33ffc3a0c5e6c062471e26d6283aa7955cf30be77",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "83ee3e51c34aaded5eca36b881331194954df161669757edbd49b651ff40dd96",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "baa891caa8bc0a7f2a8493c5fa6e42a8f85d5b9d3c37862ccc3478745ed13cd0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "455b74f95f27b0ed9a5c18c3ddef7f2979f5bc0143fd3285ad9d264950de1ab1",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "455b74f95f27b0ed9a5c18c3ddef7f2979f5bc0143fd3285ad9d264950de1ab1",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2326540288,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/25",
      "legacy-models/1"
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
      "receipt": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-multigrid-learning-v1/multigrid-modes-single-pass/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "750a42b3f98800c2f4b00848051566ed77f78aed7fdd7e93f37b8812fef738ef",
    "bytes": 9831
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C26": {
        "name": "multigrid-modes-single-pass",
        "weights_sha256": "455b74f95f27b0ed9a5c18c3ddef7f2979f5bc0143fd3285ad9d264950de1ab1",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.09427212419471528,
          "pressure_mean_free_relative_l2": 0.15028938587399235,
          "pressure_absolute_rms": 2.736004883299286,
          "pressure_absolute_max": 9.441361438587608
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
              "common_equation_relative_l2": 0.06612495197706734,
              "pressure_mean_free_relative_l2": 0.15028938587399235,
              "pressure_absolute_rms": 2.736004883299286,
              "pressure_absolute_max": 9.441361438587608
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.09427212419471528,
              "pressure_mean_free_relative_l2": 0.09611396513885408,
              "pressure_absolute_rms": 1.7237866788106209,
              "pressure_absolute_max": 4.377110098101912
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C26/results.json",
          "sha256": "019a5eb7263a125a2987a6653eccca01ed2e0ad5c742131285cd972a4d51a3d3"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":1,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":6,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":12,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":18,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":23,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":29,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":35,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":41,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":46,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":52,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":58,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":64,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"6c8cfc29504c6246a7e8c43a94837f45a352b84da3e0253d52815c7dea9647de","inputs":{"dataset":{"bytes":2275660,"sha256":"6b1be98660eb38a45fa3323490675ae37a1c6d8bd9a8ee9b1f5e3c300bff33e1"},"dataset_extra":{"bytes":2278650,"sha256":"4c23766d38ee8fc178c105e6bc6c426bd3adf5fb82296042c5c8d7d28e8739e3"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":374928,"sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"f8c784aa6b57396e7c5e094c34d079d8252473e46e2f60593a921dbebf941fcc","source_witness_sha256":"382fdd2ec7b02147f0a05b500b867c0d6d5de5fdcf65c665172b1b6a67155686"},"elapsed_s":116.62150632898556,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.0002896918642250057,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":532170240,"peak_reserved_bytes":551550976,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1e09e0fafd52e34508a416ce2aa4763d717b156d844a41f1cbf9acf180fa83b8"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21ceb5e49b98be627058692ce3371306a02c909e939e1cc98c8a4a8de8cbd8e7"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"94d01940b1e4b2c432855cc33ffc3a0c5e6c062471e26d6283aa7955cf30be77"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"83ee3e51c34aaded5eca36b881331194954df161669757edbd49b651ff40dd96"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"baa891caa8bc0a7f2a8493c5fa6e42a8f85d5b9d3c37862ccc3478745ed13cd0"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"455b74f95f27b0ed9a5c18c3ddef7f2979f5bc0143fd3285ad9d264950de1ab1"},"history.json":{"bytes":102797,"role":"training_diagnostics","sha256":"fc4cd38a42f10de1bfe893cb699755fcf7e96e3158a0d1d2a308ce358bfb3e4e"},"normalization.json":{"bytes":1384,"role":"preprocessing","sha256":"73d1cb4a0d9e37c5dc10b40328d10e93d881151310c6c330978eece155c38dd0"},"predictions.npz":{"bytes":165414,"role":"predictions","sha256":"cd8dca7d066d72bf02cff115a537a9f5ec5aeb57dcf4eddee85eeeb672c70717"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":374928,"role":"training_splits","sha256":"80f345e733ac0fa419fda1249fa714fcb25c1ac7ef2ffe3f09038f33ce313feb"},"summary.json":{"bytes":838,"role":"result_summary","sha256":"e6d9f61eab121eee87ce8b49850a998bdfea5eb5c7fda917235c69d687024ee9"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"455b74f95f27b0ed9a5c18c3ddef7f2979f5bc0143fd3285ad9d264950de1ab1"}},"per_case_inference_s":[0.012906222022138536,0.008570601989049464,0.0022002399782650173,0.0033485469757579267,0.0021626349771395326,0.0012637609906960279,0.002416194009128958,0.001903799013234675,0.002109765977365896,0.002177805989049375,0.002353325020521879,0.004715532006230205,0.0017825139802880585],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"c0a08c4116134d09012c14e55f83f44b89cc131edb668fc5a0567f2ec0297fda","shape":[16,16,16]},"case_001":{"bytes":32768,"dtype":"<f8","sha256":"336e90681435af49ee1c7cfb165d5b82d8b2198a08ed4a6fc9b3aa3b742704e1","shape":[16,16,16]},"case_002":{"bytes":5760,"dtype":"<f8","sha256":"82edd4ddd5a95a6a3d532bb91e8f2217aaee4c0cb2632f2c656cdbfc07f9a657","shape":[9,16,5]},"case_003":{"bytes":13312,"dtype":"<f8","sha256":"9eea3d09009182f97e8e19414ff660e44c4b8b664ae906641f3f85d602d3d34f","shape":[13,16,8]},"case_004":{"bytes":8736,"dtype":"<f8","sha256":"2ae789f5494f0c90182eec449fc9886c271804f8060e19c483f13e6ca2f93648","shape":[7,13,12]},"case_005":{"bytes":3840,"dtype":"<f8","sha256":"46a5b204aac15b102f51282e6e6af295da2977c665929df4833fecfe0961c140","shape":[5,6,16]},"case_006":{"bytes":11440,"dtype":"<f8","sha256":"20ddf08b6a5000e6885bf5547de311b9e228ca368478c7ef975365a353ac6013","shape":[11,10,13]},"case_007":{"bytes":5200,"dtype":"<f8","sha256":"43eda9f005921dc7b1f864a49a62b2944f113d205300f5e849021477ad29573b","shape":[13,5,10]},"case_008":{"bytes":7680,"dtype":"<f8","sha256":"727dae406ff00152ca3127cfde124389123ddbafc07a8b8f9dca7b02033e4b3f","shape":[12,8,10]},"case_009":{"bytes":5120,"dtype":"<f8","sha256":"29327e6a077e37e12c7d6891c1bd55cf41273c41ae53799cf9f998dd6eecc1b5","shape":[16,8,5]},"case_010":{"bytes":6552,"dtype":"<f8","sha256":"13c556a0146d30b08fcb6d18466b5868671ffb69cc24af98f74a7c09a171c4cb","shape":[13,9,7]},"case_011":{"bytes":24960,"dtype":"<f8","sha256":"c2093c466d5de61c854c43fe38b47bdc1306f7e8c9dfc0298376c04bc2c28dc1","shape":[16,15,13]},"case_012":{"bytes":4032,"dtype":"<f8","sha256":"0709cbcf03ff7e8e7b0719a4895b3d2b2fd2878898f6ad3cd006542facfb6499","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2326540288,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
