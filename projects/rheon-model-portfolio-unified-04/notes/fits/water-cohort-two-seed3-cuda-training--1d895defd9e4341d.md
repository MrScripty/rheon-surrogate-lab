# water-cohort-two-seed3 · CUDA training

Retrospective source review · fit-1d895defd9e4341d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Add a third initialization/sampling repeat of the C28 two-correction policy.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 2026101603, physical sampling seed 2026102603, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 9.2451328095100323e-05; retained final validation objective 4.2136017554081287e-05. Common 128-request worst mean-free pressure relative L2 0.02584243166306651; equation relative L2 0.019836451743569135. Complete GPU pressure-call timing: median of case medians 4.3756135 ms; worst case median 4.8004850000000001 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** The third repeat confirms meaningful seed variation, especially in worst pressure error. Three fits do not establish an uncertainty distribution or unseen-scene qualification.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "1d895defd9e4341d3b02591726e372c1a1a90c5fc79d987031a476f7cc8dffe4"
  },
  "history": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/history.json",
    "sha256": "3a98936b81fcf28054d2fd46c4e69cd30fcaa31674964c71851006d85bbfba2c"
  },
  "sources": [
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/config.json",
      "sha256": "da5bb300b0e7aa278f668b2d279891d5a167c4545f8be2c8bc7c56852bbb91d9"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/source-witness.json",
    "sha256": "6318164f1a3bc75376c6e4627db4b24c7e5b02abcbed15458f8c55cfcd392e40"
  },
  "training_population": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/project/inputs/validation_splits.json",
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
    "item_id": "fit-1d895defd9e4341d",
    "kind": "training_attempt",
    "name": "water-cohort-two-seed3 · CUDA training",
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
    "receipt_sha256": "1d895defd9e4341d3b02591726e372c1a1a90c5fc79d987031a476f7cc8dffe4",
    "receipt_paths": [
      "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/98333ac8f88c47619dc3996c946663d3/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-98333ac8f88c47619dc3996c946663d3/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "98333ac8f88c47619dc3996c946663d3",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-cohort-two-seed3 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "da5bb300b0e7aa278f668b2d279891d5a167c4545f8be2c8bc7c56852bbb91d9",
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
      "sample_order_sha256": "aa754040a04c5f111080e00829b9e80884baa7be61a34412b794231d7382271a",
      "source_witness_sha256": "6318164f1a3bc75376c6e4627db4b24c7e5b02abcbed15458f8c55cfcd392e40"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "0b8a114a1daca6acd2be4de29f02fe85d7f9c73d519dcbdcaabbe505565d5740",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "35e4ecbdf0268e0dd7e59c22b3b2d7e890ed3dd3a881768e70bb9349ec33f2cc",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "2462a752a1e315cdf32322a39525ad52224ab8ea92e0a6babb78269bebb7d5d9",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "29a3671b9c7c04d68886ef225191d4ee4b2aac2e104a2ae2994e2a2e954c2800",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "7b5d73470498d6d5a42ca5686916f344317940093f82c77eee1393cd5e2469aa",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "7d57f98d825791fd1f268385db25c7b93b3d09d5c5662e592b91dc96428f1dc4",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "d4d07898fcca7e7294f57e7277e0f1d57187d691cf853b7a68e2dc59878eeeca",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "d4d07898fcca7e7294f57e7277e0f1d57187d691cf853b7a68e2dc59878eeeca",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2407202816,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/29"
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
      "receipt": "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-two-seed3/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "1d895defd9e4341d3b02591726e372c1a1a90c5fc79d987031a476f7cc8dffe4",
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
      "C30": {
        "name": "cohort-two-seed3",
        "weights_sha256": "d4d07898fcca7e7294f57e7277e0f1d57187d691cf853b7a68e2dc59878eeeca",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.019836451743569135,
          "pressure_mean_free_relative_l2": 0.02584243166306651,
          "pressure_absolute_rms": 1.441594351429657,
          "pressure_absolute_max": 3.144748967071366
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
              "common_equation_relative_l2": 0.01920502639395232,
              "pressure_mean_free_relative_l2": 0.009723788326379473,
              "pressure_absolute_rms": 1.441594351429657,
              "pressure_absolute_max": 3.144748967071366
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.019836451743569135,
              "pressure_mean_free_relative_l2": 0.02584243166306651,
              "pressure_absolute_rms": 0.1988624161076343,
              "pressure_absolute_max": 0.5134805477942328
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C30/results.json",
          "sha256": "77042f68d60903ab0f6cf8b8975db9b035c2d2bd0320004ebd559b97998e26c6"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"da5bb300b0e7aa278f668b2d279891d5a167c4545f8be2c8bc7c56852bbb91d9","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"aa754040a04c5f111080e00829b9e80884baa7be61a34412b794231d7382271a","source_witness_sha256":"6318164f1a3bc75376c6e4627db4b24c7e5b02abcbed15458f8c55cfcd392e40"},"elapsed_s":279.0188851359999,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.9233773310034508e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":555410432,"peak_reserved_bytes":578813952,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0b8a114a1daca6acd2be4de29f02fe85d7f9c73d519dcbdcaabbe505565d5740"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"35e4ecbdf0268e0dd7e59c22b3b2d7e890ed3dd3a881768e70bb9349ec33f2cc"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2462a752a1e315cdf32322a39525ad52224ab8ea92e0a6babb78269bebb7d5d9"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"29a3671b9c7c04d68886ef225191d4ee4b2aac2e104a2ae2994e2a2e954c2800"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"7b5d73470498d6d5a42ca5686916f344317940093f82c77eee1393cd5e2469aa"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"7d57f98d825791fd1f268385db25c7b93b3d09d5c5662e592b91dc96428f1dc4"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"d4d07898fcca7e7294f57e7277e0f1d57187d691cf853b7a68e2dc59878eeeca"},"history.json":{"bytes":103291,"role":"training_diagnostics","sha256":"3a98936b81fcf28054d2fd46c4e69cd30fcaa31674964c71851006d85bbfba2c"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"4939c1f78dc6d28d652d5b6d9efaa34cff0e525f5caf4edfb796f36437968005"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"c6381f93c7e1086f5eb17b12f80b31e323cb8ca3a2cf6a339563ef1b8862411f"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"d4d07898fcca7e7294f57e7277e0f1d57187d691cf853b7a68e2dc59878eeeca"}},"per_case_inference_s":[0.020718919986393303,0.003691940975841135,0.004863423004280776,0.004270232020644471,0.0036081639991607517,0.0034448099904693663,0.0042986960033886135,0.004414431983605027,0.009307783999247476,0.0052647090051323175,0.006077750993426889,0.005517387995496392,0.012352292018476874,0.004357489000540227,0.007172554993303493,0.005258355988189578,0.0032230389770120382,0.006474207999417558,0.004687918000854552,0.004975406016455963,0.005335828987881541,0.005813745025079697,0.01114703100756742,0.00414058001479134],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"e04172d23970358ca2227fb592c8d7a82c6f49f428447d5d650fe831b4b78094","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"a38b2acaf9eed6abe81860d19999eef8c99860b92f6c7b2b9818a65398ecf910","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"228f663a90ce2e4842d75fd3814594e2b1584749b627bee41e5f743fb5c202c3","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"59f09bab9ee88c719a7f8c2095ee48408a56c16c108ed5332ff271c7b1af9939","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"0dfc5ea0432058da56f188de1fa3202f790f5b166fcd1a4a06396f86c860bb97","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"0c99279f57b3b57cbcf6b4b6162c34cdf7acda775d8f6e58e46005d82ef7d480","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"ad44386ac35d8eea17a6886d8d079245b972df616aa1c8acbf8b7127e8e02301","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"25a104e185f9bf32c77c4f163653ddbbe071e4a8308f28614df567d7fbb223b6","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"9cb2c157295f465a4c3f79de9b4fe95756e0d6f4e6df80390400a175c4596a48","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"4f9ee81855904bb9cce6b79051bbda0cb27a7df5f4a29cf8f4f132b8d64e6179","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"2e4b5ac1d57cada7bddd46241e1c5fe176fe59dbc2b5f676f1fe600881877b4b","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"dcd782ed3ad00696f30b869d9a54840aa2b99ff7c67e7503c2e4bfc137be0ae0","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"04553fef09aa26572eb80f18a3400d5b0b9d01bf34d265403b6aef913b74515a","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"b55246bcbf964d53b8e7ef409a962b30ca0a029686f1b1df58b891f7b6deba7b","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"fade54777d632b08380e9b2e98704c2220b2eaea97a8836b0dc40b77f36ced8a","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"672ae4ec766927ff457134978f8d55c5bed58e61d3271e1abc7c3b26926c4f32","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"267d7236f3c6642a50d1207a35e9e0aebef453502fee4cf9df1bbdc87b77dc6f","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"515538896dd67deb4bf4fcaab733c7e5df817ee14f7bf3e1440a543b8f034c84","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"aaa326eb2c771600eaa5bd539cf0b39f8591d14e3865e3ddc58a6ae4bb6d8cce","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"1c7bc4afadfe885252b6e47caa3cdc3572989379145d381f2f3c6a12a089d2eb","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"b9921e1a5a1e6cb0e763e5a68a8065702628d4924633acb27397caffcdcc9075","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"9d6be15535e7328a73bb0376fad4b6a1406d51101546190c1a4629c9c7a81f57","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"bc86d0816e35e7c425b98c35edf3e404ce57d1c7b16f41218b382469e9ffc5f9","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"27a13dfdaa8b3af81a80ddbe80dff848877b90edf7e0e85728ea69059c181193","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2407202816,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
