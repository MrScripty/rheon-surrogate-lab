# water-cohort-two-shape-feedback · CUDA training

Retrospective source review · fit-07e98b58c8c3aafb

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Add three irregular rectangular manufactured grids at 8 cells/metre to the two-correction cohort policy.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 1024 pressure-operator fields across 8 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre; plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 9.6215007932710342e-05; retained final validation objective 3.952976322036374e-05. Common 128-request worst mean-free pressure relative L2 0.019278994075802049; equation relative L2 0.017396274118909996. Complete GPU pressure-call timing: median of case medians 4.3543099999999999 ms; worst case median 4.8017655000000001 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** The rectangular bank improves common worst equation error over C28 but worsens common worst pressure error. Five-grid exposures fall from 1,000 to 625 each when eight grids share the same 5,000 proposals; this is not an equal-exposure pure geometry comparison.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "07e98b58c8c3aafbe14429d620deeb06c1bdd9baa64ff29892ff1ab55bdb4b8e"
  },
  "history": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/history.json",
    "sha256": "122e5bcbf192cf0b2b1bdbad277157ab42891d4d8f38ec3c211537a21d9dc411"
  },
  "sources": [
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/config.json",
      "sha256": "856f9a823bd9722d9013246b45ae0960126506b7a036dfcea0439d4e2da807f1"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/source-witness.json",
    "sha256": "37573632a07f209057825f32f1d5ee57f5621c555140d5d7372feb0300bd7db9"
  },
  "training_population": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/project/inputs/validation_splits.json",
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
    "item_id": "fit-07e98b58c8c3aafb",
    "kind": "training_attempt",
    "name": "water-cohort-two-shape-feedback · CUDA training",
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
    "receipt_sha256": "07e98b58c8c3aafbe14429d620deeb06c1bdd9baa64ff29892ff1ab55bdb4b8e",
    "receipt_paths": [
      "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/76090fe64e344e708b6080cfe9e18d20/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-76090fe64e344e708b6080cfe9e18d20/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "76090fe64e344e708b6080cfe9e18d20",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-cohort-two-shape-feedback · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "856f9a823bd9722d9013246b45ae0960126506b7a036dfcea0439d4e2da807f1",
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
      "source_witness_sha256": "37573632a07f209057825f32f1d5ee57f5621c555140d5d7372feb0300bd7db9"
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
        "sha256": "e1f78c8a12b3a907a8ea7afacb75cc561a2b24decd61ab321079f014760b799c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "44affc7aacf87651775892a4ecdea748dbc9fca2fc117071f8ab97c4ad9e70cd",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "03427eab51c3375d1d716221c9435cd49d5b777ae8e829ddce2f0166d9f346fe",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "972bafb3cfd7baace898561fe46bf5b3de30715502d6522b3552e04325f50b36",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "005fa29b9f65bb3b3e162165eed039a0518ddc2dfb69520f91886af690d73879",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2406137856,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/32"
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
      "receipt": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-shape-feedback/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "07e98b58c8c3aafbe14429d620deeb06c1bdd9baa64ff29892ff1ab55bdb4b8e",
    "bytes": 13321
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C33": {
        "name": "cohort-two-shape-feedback",
        "weights_sha256": "17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.017396274118909996,
          "pressure_mean_free_relative_l2": 0.01927899407580205,
          "pressure_absolute_rms": 0.6086666756983703,
          "pressure_absolute_max": 1.6679477146121588
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
              "common_equation_relative_l2": 0.017396274118909996,
              "pressure_mean_free_relative_l2": 0.011177651309923696,
              "pressure_absolute_rms": 0.6086666756983703,
              "pressure_absolute_max": 1.6679477146121588
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.017069053727536625,
              "pressure_mean_free_relative_l2": 0.01927899407580205,
              "pressure_absolute_rms": 0.16516792896096508,
              "pressure_absolute_max": 0.4620891361231146
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C33/results.json",
          "sha256": "dca33dd325ed0aa057cf1ced7d14dbee69a9d2460326f6ceff212bf9d4db5ce0"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"856f9a823bd9722d9013246b45ae0960126506b7a036dfcea0439d4e2da807f1","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"37573632a07f209057825f32f1d5ee57f5621c555140d5d7372feb0300bd7db9"},"elapsed_s":207.9876125799783,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.9897006836799846e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597156864,"peak_reserved_bytes":618659840,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e1f78c8a12b3a907a8ea7afacb75cc561a2b24decd61ab321079f014760b799c"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"44affc7aacf87651775892a4ecdea748dbc9fca2fc117071f8ab97c4ad9e70cd"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"03427eab51c3375d1d716221c9435cd49d5b777ae8e829ddce2f0166d9f346fe"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"972bafb3cfd7baace898561fe46bf5b3de30715502d6522b3552e04325f50b36"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"005fa29b9f65bb3b3e162165eed039a0518ddc2dfb69520f91886af690d73879"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8"},"history.json":{"bytes":144633,"role":"training_diagnostics","sha256":"122e5bcbf192cf0b2b1bdbad277157ab42891d4d8f38ec3c211537a21d9dc411"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"7dd72b79842f2df2b0eedcb3e51143e6eefd83e0dabea5397570bc466c76573f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"d1da9aca222d23cacb5ca67b7c8d8a862446893a5df6c3bcc33ea20d27c9189d"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"17b423b78f82468a3e879af64dc2f6ecf61c94500a4bb154a08ebf527768e2e8"}},"per_case_inference_s":[0.018979652028065175,0.00360675499541685,0.004676638985984027,0.0037586260004900396,0.0035126819857396185,0.003705677983816713,0.0050095779879484326,0.005189694988075644,0.01003562897676602,0.005020418000640348,0.005750361015088856,0.0051513760117813945,0.011364451987901703,0.004276776016922668,0.006351609976263717,0.004326236987253651,0.002700952987652272,0.005555495008593425,0.0039121549925766885,0.004165437014307827,0.004382951010484248,0.004475499998079613,0.008948312024585903,0.003740257991012186],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"0dca544a6c4abca83a30af0a05be82697c01db49ffc92ddbb7917c3aaf85f2a2","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"8767b38c59e956068b6d0dd7489b451997da3afcfb675104c23b3ba335f8ef47","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"49bbfd88370821ae7325733c5f8cc219aa91d1045ea19f7c19df3715dd365921","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"d2c2a589006498b533229b20d6d8eec523c28a51fddc415fb29f63555f97c51f","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"298f04b38643e80326c05954c9bbbfb2034354dcdc7e3271647badf9e180de2e","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"69849b5d2600059d2547510710dcc7fd2a36da0986e74f743150cdfc126b67ce","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"bca1c82863d899fa5440253ff67c870357dbace6ef1119325cde0f2f229f9a8f","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"02237394f2d1b82a0a648e7facc408a4c20b1d1ebe40fde029b16e8d14ea5808","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"8f24217c371773983f0f3d19df9d6d5ac9cfa2820b8054f6c4f6763b58c8a953","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"63d71a161a218bd4f301da1553fc6c0d96319dcef7b7e86c7a8f5279bd248f6e","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"2a84a11b98b2c63abb7b52e7cf4ca1967bc61e73b0416b78985a0744f912ebee","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"ea9400d2598c386e43b7b13a8a764d393fc610970296eb89868742476a7d027b","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"8801b143fb8657977458f838c4af9e6d4e886efc6b15fbda1a40435e6bfde4f7","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"ef00c4df5e88d73e73442ff4daf24b00a4a458a66c0f858cd778c090813182f9","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"ec4a5b1bd2a63ab6e6b38409870eaf68c9ccbba005daba20d295562d0e0cfd2d","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"86e949ce35ba2b5256011d810d58fd0de0a37167c2836c839e72b87075ab94e3","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"22693f2c2ea3e2162b2a6a094a8e8d8dd6da1f8a96b2c489792a0feda69a6b26","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"839e6e4635cf1f043b18f46b41178f913845b4dbea9d53844e72b34c21c1b3bb","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"6cd749c69c002ff3f653c3ddbe5662f5b9a7aa4ddd827bb1f097f71d45fd1181","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"3dcf800cdb2ef3e6bd88b8bf8e348d7d3bbf11297058862237da294e543ba5f4","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"6c835851259144bd3d2927defe9e078e98b6ea929143917d29409eb50db1c478","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"14d1ea48217918285154538f14fc9c9c200117140a3b81025e4a0fda382c56df","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"9b46c5b01a00c1cf4c6f66dd53e76e05f53b66b1ef0c7ad1defec62dfdfd5c3e","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"f98654740c1030638b00588c63840709982d943da08bd68d30befb7c5b4c92aa","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2406137856,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
