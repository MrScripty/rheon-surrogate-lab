# water-cohort-one-control · CUDA training

Retrospective source review · fit-c95f68354ff66fb6

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 1 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Measure a single-correction control on the same 16-trajectory cohort and five-grid manufactured bank as C28.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 0.00045721926384189535; retained final validation objective 0.00060627189414283586. Common 128-request worst mean-free pressure relative L2 0.093818011751618552; equation relative L2 0.073253374167723276. Complete GPU pressure-call timing: median of case medians 2.0736179999999997 ms; worst case median 2.3905884999999998 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** The single-correction cohort control is faster work in principle but materially less accurate than C28 on both common worst metrics. Use measured complete GPU latency for the actual speed tradeoff.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "c95f68354ff66fb6dd50a066ae3317f48cfb58782561e013722f7eb58be2ae18"
  },
  "history": {
    "path": "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/history.json",
    "sha256": "33731d0634ed8fc36266f580fb1776d130db9b61ef9a6d3592b15b9d87bc0755"
  },
  "sources": [
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/config.json",
      "sha256": "26e2c0e637049dd21f5f31c8c91c72e877db4ca0e5cf272fe67eb7026ef1e5dc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/source-witness.json",
    "sha256": "4fa90867043626dfa4472f91fe78ba4eebaca70eaf20e31a7d3fd75ac92c2c6b"
  },
  "training_population": {
    "path": "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-recurrence-generalization-v1/cohort-one-control/project/inputs/validation_splits.json",
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
    "item_id": "fit-c95f68354ff66fb6",
    "kind": "training_attempt",
    "name": "water-cohort-one-control · CUDA training",
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
    "receipt_sha256": "c95f68354ff66fb6dd50a066ae3317f48cfb58782561e013722f7eb58be2ae18",
    "receipt_paths": [
      "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-one-control/managed-recovered/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-one-control/managed-recovered/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/b33ced71168941c09d96979cd30d9621/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-b33ced71168941c09d96979cd30d9621/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b33ced71168941c09d96979cd30d9621",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-cohort-one-control · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "26e2c0e637049dd21f5f31c8c91c72e877db4ca0e5cf272fe67eb7026ef1e5dc",
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
      "source_witness_sha256": "4fa90867043626dfa4472f91fe78ba4eebaca70eaf20e31a7d3fd75ac92c2c6b"
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
        "sha256": "0389f155c293c2d4599b6c40f7872c34070cfd148d971f0634af7b6895ec661d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "a8cdd295076119cc02c0ffb7c961b9a37a06bfd7eb70e51a481b95720d6ec638",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "0429a4929b259227d20e1d1cc13cf874b0f79b87986395b14cf02d1d059ac2f6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "94af5146050db8c4cdeff6f3d154be277322c0f39327c1d0112d2403ccdc43b0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "cc2bb0b98f0421cb633f36c29e142b7381be5aa8fbc6263bdd217378f6187ac1",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "e4317096e0dccf647ad57013813100e310e22cb5587783abb5ec24d5c76ed7d6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "e4317096e0dccf647ad57013813100e310e22cb5587783abb5ec24d5c76ed7d6",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2349293568,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/30"
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
      "receipt": "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-one-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "c95f68354ff66fb6dd50a066ae3317f48cfb58782561e013722f7eb58be2ae18",
    "bytes": 13188
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C31": {
        "name": "cohort-one-control",
        "weights_sha256": "e4317096e0dccf647ad57013813100e310e22cb5587783abb5ec24d5c76ed7d6",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.07325337416772328,
          "pressure_mean_free_relative_l2": 0.09381801175161855,
          "pressure_absolute_rms": 2.6617947565627365,
          "pressure_absolute_max": 7.70542710088445
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
            "trajectory_id": "baseline--water-validation-0054",
            "native_frame": 15,
            "counts_xyz": [
              15,
              12,
              5
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.038341132834404594,
              "pressure_mean_free_relative_l2": 0.09381801175161855,
              "pressure_absolute_rms": 2.6617947565627365,
              "pressure_absolute_max": 7.70542710088445
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.07325337416772328,
              "pressure_mean_free_relative_l2": 0.06323729657249952,
              "pressure_absolute_rms": 1.1341495408933542,
              "pressure_absolute_max": 3.026428606024891
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C31/results.json",
          "sha256": "83b6f0bf8ccee9a98c770c49c74bf9ebda11783958572c72c5dd97828fdc51a3"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":1},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"26e2c0e637049dd21f5f31c8c91c72e877db4ca0e5cf272fe67eb7026ef1e5dc","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"4fa90867043626dfa4472f91fe78ba4eebaca70eaf20e31a7d3fd75ac92c2c6b"},"elapsed_s":121.9595476819959,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":9.550563294169478e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":533315584,"peak_reserved_bytes":553648128,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0389f155c293c2d4599b6c40f7872c34070cfd148d971f0634af7b6895ec661d"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"a8cdd295076119cc02c0ffb7c961b9a37a06bfd7eb70e51a481b95720d6ec638"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0429a4929b259227d20e1d1cc13cf874b0f79b87986395b14cf02d1d059ac2f6"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"94af5146050db8c4cdeff6f3d154be277322c0f39327c1d0112d2403ccdc43b0"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"cc2bb0b98f0421cb633f36c29e142b7381be5aa8fbc6263bdd217378f6187ac1"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e4317096e0dccf647ad57013813100e310e22cb5587783abb5ec24d5c76ed7d6"},"history.json":{"bytes":103198,"role":"training_diagnostics","sha256":"33731d0634ed8fc36266f580fb1776d130db9b61ef9a6d3592b15b9d87bc0755"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"6624641632166f29e135fc088285427bafb4ddf3cfb972f2cccc5655b907c771"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":983,"role":"result_summary","sha256":"04aaad26d02beacb62b05596aa32f194bc14abf868cc64c26814d602a8eb1e92"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e4317096e0dccf647ad57013813100e310e22cb5587783abb5ec24d5c76ed7d6"}},"per_case_inference_s":[0.013272782001877204,0.0022951090068090707,0.0028260309773031622,0.002197458001319319,0.001915191998705268,0.00176868902053684,0.002296715014381334,0.002401411999017,0.006041148997610435,0.002740555995842442,0.0032470359874423593,0.002795501990476623,0.008111287985229865,0.002630531002068892,0.003759995015570894,0.0024148100055754185,0.0014645449991803616,0.0031137079931795597,0.0023078850063029677,0.002491599996574223,0.0025917690072674304,0.002640978986164555,0.0053652529895771295,0.0020733480050694197],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"d90c1904f42c631b2599290b76d7a3d6ead6dfbc634f4647c99b0142397af82e","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"878b312d0241bac100472be95101e9616ac31b059bf05b48f73f1e18177fb2c5","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"c0099bc61369d295cc8da8ad7bf7b67cc641ae56a72f85ed57d84b1a2d55c85d","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"951dbaecd3c254bf6f96b28dc20c83a84a0c89929f3face1d57356a6772545ec","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"89a4dbde19f1bb563555e0a3819047f8460bce8eaf62e4a7ce77bde7a118fad1","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"01fe2725308db0835ba3605dcdea0009c2f15873818f7f8df9212009022d4322","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"d5657e8674ce30e0ea5fc1c17abe1bb366b3e784cfadf61f6fc1ce53071e3eef","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"852ae7086d65154d5d80ca9c81f9f4ed185e20d934e1fa6b716b2504621ae358","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"830a0c4150d68062087bb0573876bbc749b8690b8e9e6ffd8e46a21f214149ca","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"40f7237deeb757532b0bca322fbe7bcbfab0d5f5670037a7fe366615b891df6b","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"360a967fe98d64477c088b4e58c83350ae3e6e1ce3b5e60f15c4eafd6359961b","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"8fb2e4417df039792d70b48a094859d896dd738c01f009ed4c3011732bcd5855","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"24b8a1b00806d17bd98ebe699a1cc9f12557453c77d8da1943a4386af0d37374","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"860907a9099c1f8168c3c5a1c08f6c70500dd9094c6dae6e577f0937a68d8975","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"a13bbf211490eb95833ac0d8c35f18cc746688b551bf30fd7ade859e0b986ce7","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"299063febb89c8c6eaa14832e5af941b788ddda963ebb432522a47fd9a97fb8c","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"483b72a7682dd413d0f3a30ea1679797683f3da30ca9445f70f8db58c031ec45","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"ba58491bb3ec49434e523fa65ba40d4f5ae3a1fc5d5f1dc6cbaa92a529dc099c","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"ef8823bd8fbbce2c51bd643cf29bb22a6faac9137253565e127b3d760e7d1444","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"bc86252a595d81b44c9009caa7247e32c17455cf9dbdd3e7d3bed73d8020ccc0","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"2620fa833cae29409698aea1af78fe40b9c27a0c4847cb019f3cb86f82ec5810","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"819455aae9d82a91e720ca01a0f498c80500cdbeb2cf52a0c7111fa3dc1ad070","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"9d68705986c943f27104ab71e170513242a2eba6b54063f32b9bae7f85b887fc","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"b4ab215b0fad342ccdafc79fafb3b2d913e5bfd9f249190c96f1aab7d0f88e38","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2349293568,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
