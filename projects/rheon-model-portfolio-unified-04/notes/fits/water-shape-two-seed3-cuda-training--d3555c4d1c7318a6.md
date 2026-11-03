# water-shape-two-seed3 · CUDA training

Retrospective source review · fit-d3555c4d1c7318a6

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Repeat the eight-grid rectangular-feedback two-correction policy with a third initialization/sampling seed pair.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 2026101703, physical sampling seed 2026102703, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 1024 pressure-operator fields across 8 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre; plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 7.0378396670782819e-05; retained final validation objective 3.5824852325357639e-05. Common 128-request worst mean-free pressure relative L2 0.018309087994618282; equation relative L2 0.020026484696711606. Complete GPU pressure-call timing: median of case medians 4.3682522500000003 ms; worst case median 4.7692680000000003 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** A third rectangular-bank repeat retains the same physical cohort and feedback draw policy but changes initialized weights and physical draws. Compare all repeats and both error metrics instead of promoting one training loss.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "d3555c4d1c7318a64223b17b635feaae07f490bf5c0d967589d6d73608754ba4"
  },
  "history": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/history.json",
    "sha256": "b5822aeffec14e72888dd112b38a260e3b4b0ffd637c2375f9c32d574ba4f6eb"
  },
  "sources": [
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/config.json",
      "sha256": "06785163d08d0e1cf6529824a63e08fcc0a0f418fdfe3e5ff7e6408adb6bbec9"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/source-witness.json",
    "sha256": "3d128fa554df7076f9b7ace4b9e35068c32f7d96977ec4731e300fe66498649f"
  },
  "training_population": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed3/project/inputs/validation_splits.json",
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
    "item_id": "fit-d3555c4d1c7318a6",
    "kind": "training_attempt",
    "name": "water-shape-two-seed3 · CUDA training",
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
    "receipt_sha256": "d3555c4d1c7318a64223b17b635feaae07f490bf5c0d967589d6d73608754ba4",
    "receipt_paths": [
      "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/fdfb01aebeba4c4bad31962d815baa97/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-fdfb01aebeba4c4bad31962d815baa97/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "fdfb01aebeba4c4bad31962d815baa97",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-shape-two-seed3 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "06785163d08d0e1cf6529824a63e08fcc0a0f418fdfe3e5ff7e6408adb6bbec9",
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
      "sample_order_sha256": "8507ebb1c345a2b246d1d4c12e5066c287df8fdba65d53bd9c7b6a1a6d7534e7",
      "source_witness_sha256": "3d128fa554df7076f9b7ace4b9e35068c32f7d96977ec4731e300fe66498649f"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "4139d39c27541135496f6ba8ecb5550f0ae7ec20a8ee0eb71060ea27c1a8e183",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "9e96b265da3fe9a693a3f6fd482b5a67520763267219ea91a30dbe591c8dfe34",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "bfd7f936ae04ffb49d7aedd65224cb6fa649a7a3da2456b7de4ce33ed609ac95",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "546718f25aacb9366f1dac16c2ba23813421087e2c14f1c414ec92d67849b281",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "c5f0af7d16f83d8a9e268963d13b2002ba9c12041b90fd0bcf96ac17d055bccb",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "fa691c9299e3baf09d2e4991c605a49983956dedf574fe05dc96a3962e71f3a1",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "f37933d4faabc57964e7de49ebff5ffd74a4d73613d542cdc89508f08deade6c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "f37933d4faabc57964e7de49ebff5ffd74a4d73613d542cdc89508f08deade6c",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2404966400,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/34"
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
      "receipt": "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed3/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "d3555c4d1c7318a64223b17b635feaae07f490bf5c0d967589d6d73608754ba4",
    "bytes": 13319
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C35": {
        "name": "shape-two-seed3",
        "weights_sha256": "f37933d4faabc57964e7de49ebff5ffd74a4d73613d542cdc89508f08deade6c",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.020026484696711606,
          "pressure_mean_free_relative_l2": 0.018309087994618282,
          "pressure_absolute_rms": 0.8000291784015059,
          "pressure_absolute_max": 1.9312772259142754
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
            "trajectory_id": "baseline--water-validation-0032",
            "native_frame": 63,
            "counts_xyz": [
              15,
              6,
              8
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.020026484696711606,
              "pressure_mean_free_relative_l2": 0.018309087994618282,
              "pressure_absolute_rms": 0.8000291784015059,
              "pressure_absolute_max": 1.9312772259142754
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.016427789455512716,
              "pressure_mean_free_relative_l2": 0.012958816964309632,
              "pressure_absolute_rms": 0.22202873580097263,
              "pressure_absolute_max": 0.4725784693015964
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C35/results.json",
          "sha256": "debce41767f7e9a076acb959f772b0b07790d00568f0184720b3cb17cacd9935"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"06785163d08d0e1cf6529824a63e08fcc0a0f418fdfe3e5ff7e6408adb6bbec9","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"8507ebb1c345a2b246d1d4c12e5066c287df8fdba65d53bd9c7b6a1a6d7534e7","source_witness_sha256":"3d128fa554df7076f9b7ace4b9e35068c32f7d96977ec4731e300fe66498649f"},"elapsed_s":210.55719507302274,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.77528164992691e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597156864,"peak_reserved_bytes":618659840,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"4139d39c27541135496f6ba8ecb5550f0ae7ec20a8ee0eb71060ea27c1a8e183"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"9e96b265da3fe9a693a3f6fd482b5a67520763267219ea91a30dbe591c8dfe34"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bfd7f936ae04ffb49d7aedd65224cb6fa649a7a3da2456b7de4ce33ed609ac95"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"546718f25aacb9366f1dac16c2ba23813421087e2c14f1c414ec92d67849b281"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c5f0af7d16f83d8a9e268963d13b2002ba9c12041b90fd0bcf96ac17d055bccb"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"fa691c9299e3baf09d2e4991c605a49983956dedf574fe05dc96a3962e71f3a1"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"f37933d4faabc57964e7de49ebff5ffd74a4d73613d542cdc89508f08deade6c"},"history.json":{"bytes":144618,"role":"training_diagnostics","sha256":"b5822aeffec14e72888dd112b38a260e3b4b0ffd637c2375f9c32d574ba4f6eb"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"623d153621613493067c077087186fecda52b18508394a154319ebfb943abd30"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"4a27dc5b65c9430517dac2095f46c767c034ed9b6f80ba1f5b1a4cabdb6aea76"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"f37933d4faabc57964e7de49ebff5ffd74a4d73613d542cdc89508f08deade6c"}},"per_case_inference_s":[0.01751338699250482,0.0038911209849175066,0.004893506004009396,0.003893232991686091,0.003408308984944597,0.0031675439968239516,0.003990135999629274,0.004171662003500387,0.008525442972313613,0.0046703369880560786,0.005423419992439449,0.004666013992391527,0.01064345400664024,0.003869071020744741,0.005665266013238579,0.003952204977395013,0.0026363530196249485,0.005542470025829971,0.004132811009185389,0.004340950021287426,0.004653677984606475,0.004907397989882156,0.009511993994237855,0.003667015000246465],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"0605cd7731b7b7f114b25b3d64c2bae878cd83039aa1157dce6b112640a082e6","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"eb8e3e0480f7388935a54451ae4d950cbaf2509da09f8c6f93e8c0ab0b3733a5","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"d66c6d787fd1a3907b7019c8111b1f46c37474f5fa9f7b0fce271dbf506f661b","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"590930123d05e734a8fe3fb516d8d22899a3ac2feefed91d75ec0229c13ce779","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"525283f40567ffad2d9f34a43ca117c0be75bbe65c296b9bec97c83ee27b15de","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"de2d056bbdbd84f371241f6ce5db7bf4529ea83d6bafe397d934b6f42501b31f","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"020b6b971e4971a6a6cb98afee88e4674fce636b0c05e11875c82d1dfdeb7cb1","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"b2036de41f4c5ebd4ca696255b1efb287b80717beea5224857452879accd3bdf","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"88defa394c1589c9e7df3fb33b1b6d85e8b6fd305da1532eca10134920e1d879","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"afc0e23ad7fc2ea647d9f9879acc4d6ebda2947712db7d3f44e2c6b59b8cb622","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"a5b130098e269be990c2f3bce326585aef4680ecf28a06766b3bc67ed8b0fc24","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"68d0a65db0652ac11caab71d36213ab4f40a5df814ff34a4353fb67f8d8a50e8","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"b499ba814115035fc4f88cae232c4750e7474477445338e6520b9b93b19196cb","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"36e92ba113501b534c4acb31730650d433f01d9ef946c7403fd2ed6cc5fc072b","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"78a02c8dd80a1a8c6fa158cacc067d0d1813810647828844c8b988419ffe2446","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"1b33c171be32cf1488d6963f549a47fb1711b9a2bdf2fe37a20a4e08c589ee76","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"f356887397b16021e180c1fc3869f1aa5ceaac85d698aaf6d687c7bd372e9301","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"78047e4ced52153ace408b75a65085087c2a5a1ba2bba8f928b3bf0f46228289","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"9ec1ae790640b5ff6dee54ac8eda36dfb5f706f46aaeb17a9a4c6ac512540b6e","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"8b480c0d8ab58c20fed880f7ae0fab42dc4bc22dc20fc1a61c2d40945b5d5eff","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"0867e101571a22aa45364ac9d6822a05b68803a867bbd9599fd8cc26e09a81f3","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"bc401063dac9410d19bf7a3d70fe8547aed6956589cc90099a069c62a169bde2","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"e644b44209f5f41c9fda58bcbadcb7d345c06af067e4943ac8a0ccd84cd28c80","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"a6d0f2e54caf26e31959cdc6b5f918ae6cd190117e97cab0f2acefe6d8bdd12b","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2404966400,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
