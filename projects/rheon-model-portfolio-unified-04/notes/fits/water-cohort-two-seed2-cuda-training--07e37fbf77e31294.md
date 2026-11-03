# water-cohort-two-seed2 · CUDA training

Retrospective source review · fit-07e37fbf77e31294

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Repeat the C28 two-correction policy with different model-initialization and physical-frame sampling seeds.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 2026101602, physical sampling seed 2026102602, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 5.0377486416360528e-05; retained final validation objective 2.7539929102069733e-05. Common 128-request worst mean-free pressure relative L2 0.018699412090631935; equation relative L2 0.024295938855013393. Complete GPU pressure-call timing: median of case medians 4.3786387500000004 ms; worst case median 4.8561379999999996 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** A second seed pair gives different common worst errors on the same physical/manufactured populations. Repeating architecture labels is not repeating the fitted state; initialization and physical sample order both change.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "07e37fbf77e312944dd44c651826ec4c046b2e333819550d20fa01ff42c6a4fe"
  },
  "history": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/history.json",
    "sha256": "c0b4f374f4ed14759fbc4692217e338888876005a5986459b2203e29ca6b4f4b"
  },
  "sources": [
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/config.json",
      "sha256": "ff3758538ef4b7860b39dd2c4f294416c269383f13159beebcf5e814b3272eb3"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/source-witness.json",
    "sha256": "b70c34bbd37013c43a16b701f2a5a602ec45098bb7b0838b2de63589ff00ec67"
  },
  "training_population": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/project/inputs/validation_splits.json",
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
    "item_id": "fit-07e37fbf77e31294",
    "kind": "training_attempt",
    "name": "water-cohort-two-seed2 · CUDA training",
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
    "receipt_sha256": "07e37fbf77e312944dd44c651826ec4c046b2e333819550d20fa01ff42c6a4fe",
    "receipt_paths": [
      "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/91ccb8b6276642a5a82b99e2a83b1ea1/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-91ccb8b6276642a5a82b99e2a83b1ea1/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "91ccb8b6276642a5a82b99e2a83b1ea1",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-cohort-two-seed2 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "ff3758538ef4b7860b39dd2c4f294416c269383f13159beebcf5e814b3272eb3",
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
      "sample_order_sha256": "57d175ba480f72cf506502a0b857393bffa590e0a034f2c68e89cf3fda2dd0f7",
      "source_witness_sha256": "b70c34bbd37013c43a16b701f2a5a602ec45098bb7b0838b2de63589ff00ec67"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "a4d30066abd6c8cfe818302ce4e7e48e1f2e00ca2723f244d124b8a795353832",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "f6e723e706ca76b8d56a274100aa8c835f9edb393ae968030fbf9294461b899f",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "12f71b184b109c403dc3480a35b80f67c0c0f3db5a262e4a4a0aa0785f62f17e",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "bc2d7e805c650bc8583119491014479c0ff9f87009cc5f8e631c53f57e42ff1c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "31669e1ce030c93153ed29389503df95f65f0a753bfa34b4edd88aab615e81a5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "8d53ba9502566d67f51aa4c5ec172afd605b7cabd3213bac85c62fb53581b05e",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "2237d0deac73eddd31073ff596b5d30a1ddf38fa52918350ab1a01a7afaca510",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "2237d0deac73eddd31073ff596b5d30a1ddf38fa52918350ab1a01a7afaca510",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2405224448,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/28"
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
      "receipt": "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed2/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "07e37fbf77e312944dd44c651826ec4c046b2e333819550d20fa01ff42c6a4fe",
    "bytes": 13189
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C29": {
        "name": "cohort-two-seed2",
        "weights_sha256": "2237d0deac73eddd31073ff596b5d30a1ddf38fa52918350ab1a01a7afaca510",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.024295938855013393,
          "pressure_mean_free_relative_l2": 0.018699412090631935,
          "pressure_absolute_rms": 0.9331388598049948,
          "pressure_absolute_max": 1.9865049494067655
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
              "common_equation_relative_l2": 0.013261919941635775,
              "pressure_mean_free_relative_l2": 0.018279939459826266,
              "pressure_absolute_rms": 0.9331388598049948,
              "pressure_absolute_max": 1.9865049494067655
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.024295938855013393,
              "pressure_mean_free_relative_l2": 0.018699412090631935,
              "pressure_absolute_rms": 0.33537059278382725,
              "pressure_absolute_max": 0.7932794415672277
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C29/results.json",
          "sha256": "b05f5763febcaee59013e6c3147647a5952456f0a19a89ad4adb3874702ecd52"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"ff3758538ef4b7860b39dd2c4f294416c269383f13159beebcf5e814b3272eb3","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"57d175ba480f72cf506502a0b857393bffa590e0a034f2c68e89cf3fda2dd0f7","source_witness_sha256":"b70c34bbd37013c43a16b701f2a5a602ec45098bb7b0838b2de63589ff00ec67"},"elapsed_s":224.89313821698306,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":5.0074133923772024e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":555410432,"peak_reserved_bytes":578813952,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a4d30066abd6c8cfe818302ce4e7e48e1f2e00ca2723f244d124b8a795353832"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"f6e723e706ca76b8d56a274100aa8c835f9edb393ae968030fbf9294461b899f"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"12f71b184b109c403dc3480a35b80f67c0c0f3db5a262e4a4a0aa0785f62f17e"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bc2d7e805c650bc8583119491014479c0ff9f87009cc5f8e631c53f57e42ff1c"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"31669e1ce030c93153ed29389503df95f65f0a753bfa34b4edd88aab615e81a5"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8d53ba9502566d67f51aa4c5ec172afd605b7cabd3213bac85c62fb53581b05e"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2237d0deac73eddd31073ff596b5d30a1ddf38fa52918350ab1a01a7afaca510"},"history.json":{"bytes":103274,"role":"training_diagnostics","sha256":"c0b4f374f4ed14759fbc4692217e338888876005a5986459b2203e29ca6b4f4b"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"cc13d08020013585771e67e1e5b49301503066c5d22c8e86dd81e07b8ed33447"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":983,"role":"result_summary","sha256":"cfde5d75ae1909ff9403e64b516ed26efed24854afafd575678dc0efede9eabb"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2237d0deac73eddd31073ff596b5d30a1ddf38fa52918350ab1a01a7afaca510"}},"per_case_inference_s":[0.020314793015131727,0.0038061829982325435,0.004804158001206815,0.0038181639974936843,0.0031572529987897724,0.0030204729991964996,0.003763248008908704,0.003681169997435063,0.007945633988128975,0.004434358997968957,0.0054258659947663546,0.005129019991727546,0.011320727993734181,0.004358960024546832,0.006905329995788634,0.004495211003813893,0.0027687730034813285,0.005739518004702404,0.004163820005487651,0.004556588013656437,0.004671478003729135,0.0049694439803715795,0.00930398900527507,0.00344680598936975],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"d78d91e9dab0cc8d69702a641baa9de4f4183907a6dd2f6b3e11955154e3cd00","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"18e708d438b513df0fe057a1d1b966bd2c64c45eb04d1d780ce8df21e8e00a5c","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"336f40b8e6da008b764d51577d752f8ca6f1068caf67d82fc9a7c4ed58744d97","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"30a125f5dd5eec297f1d61b9a2bad7b7decc39f556c1f2250f9295c1bea9e1b8","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"a516acea6b529a774ba767f1c80eebdfec14b552999abeddffd95fb10a426f71","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"dbf8030915266171998ffa6507cf0123c7c3b16a56be059de845a4214253f88f","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"f0308c3bd89405fabb2013bb3bb88eabb5b057133491780f0220e092139768dc","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"ca61d21fdb962b60f3b94824cfc09269c140d6a3ebd4862acddb04ee4606cca8","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"ee158329f99e37f99b6a1d22ce8f348d84104697acf01c02e2facf8a1b61f785","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"09d51c2dc0df5d11ec078d9b8257c25a31260c0b36005dc1e161aa0b797b50f7","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"f54ba6a5e2ebf360258786308b811617779e02e5a12f6c0f5ae6a3086889f546","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"8effe46222d195b94ca99e36c5143f973e896d5f994a4e74423d3d9ab2ad7744","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"e5d153bb1616d92d12d9aaf66e9d41d22a38e9b1c833b3e8225fcec008403889","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"1c8082b0796748d9e714597502cd1961f8f9b70149c9d31f82ea0c918ae1aa0b","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"8fd21d07466aa2e764fdf6389ac228482280d8a87bd7a00545b8798ab304b269","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"002e4a4c40d36d26d2f7ff819c6690e76945b344e2b47e0181587ffa2e2afb10","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"62392735f75deca88467450855a78c20a902a325f4bd57cfbcfa7d241cdc2287","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"0a4651b9ec81467bef2ae895e7d0ec0343307fae41b050f692ff2d8240867d65","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"ee905a86689a4bb3c8cc21124b9555d65410211e85b7457824f8444e809e8ea8","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"b06e3350db16c30c08b3fb944faee20f2530a031eddaa6e3b15bbfc4028c200b","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"e24e15c04d82b2e4ada5c3d0d550287b0c094832cce40ca10ef3afd393201086","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"345debcb34e92193170b375587bf381d706395d3127394caa476b358d850cade","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"37ffd170c754b900ce2d424cc7beba33c00182928cb93a37f4c9615f76599212","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"dca83c84b4b51ec36959fcda388be5c6627b7a820e1e87661f3009a2242184c1","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2405224448,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
