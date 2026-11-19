# water-shape-two-seed2 · CUDA training

Retrospective source review · fit-4f94810fb5b11953

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 2 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Repeat the eight-grid rectangular-feedback two-correction policy with a second initialization/sampling seed pair.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 2026101702, physical sampling seed 2026102702, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 1024 pressure-operator fields across 8 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre; plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 4.514596798996393e-05; retained final validation objective 2.4885702583652996e-05. Common 128-request worst mean-free pressure relative L2 0.0134615900145253; equation relative L2 0.015272181054025221. Complete GPU pressure-call timing: median of case medians 4.3842792500000005 ms; worst case median 4.8065705000000003 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** The second rectangular-bank seed pair improves both common worst errors relative to C33. Its range overlaps the five-grid repeat results, so a single best seed does not prove the bank universally superior.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "4f94810fb5b119533b39be1a3835603afb2d820416a60637a90143adc8c1e4ba"
  },
  "history": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/history.json",
    "sha256": "c452ce35f9bbf5db71bcdf2d594b4a69ec46fbff92e24de22d89baaadc4f03c2"
  },
  "sources": [
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/config.json",
      "sha256": "06379a081028cd657b34237ac3c236e91f4fc92c105803824435a91ef9d52047"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/source-witness.json",
    "sha256": "3f1491d7581bc0e13c8d67fb3862a20b1be643748d8265be4d5857be19b276ee"
  },
  "training_population": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-shape-repeatability-v1/shape-two-seed2/project/inputs/validation_splits.json",
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
    "item_id": "fit-4f94810fb5b11953",
    "kind": "training_attempt",
    "name": "water-shape-two-seed2 · CUDA training",
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
    "receipt_sha256": "4f94810fb5b119533b39be1a3835603afb2d820416a60637a90143adc8c1e4ba",
    "receipt_paths": [
      "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/32781836090d474ab5b34b1e12e79cc2/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-32781836090d474ab5b34b1e12e79cc2/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "32781836090d474ab5b34b1e12e79cc2",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-shape-two-seed2 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "06379a081028cd657b34237ac3c236e91f4fc92c105803824435a91ef9d52047",
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
      "sample_order_sha256": "32d1256c5ec3597e2d911ff7d9e36be4759c1e14e4ca394e8efcac520133fd7b",
      "source_witness_sha256": "3f1491d7581bc0e13c8d67fb3862a20b1be643748d8265be4d5857be19b276ee"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "c4457a1cc950cd199b80f694113b02abbf980529a3e10c0203b10ebfaa012b12",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "2f0936683e803c9e3fd5291f012ae0e09dc38cc7277bbc9c47f0bfd5195d088c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "e458d78f492e4bc88a26548afa3e255f11cb7ba91f86b10145070dc191f5f4d0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "7a55774e1f1213e6a4c5ca013c8431d22ddaaa8270b5c49f0780ec10479f53d0",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "33c561aab416886a0fb0d43740a9996e6856f91e85cf1e6929bfad03d333cc44",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "16efa57e21cd8d0f80a63f124d41c88537e7f42b9df29e99a79ad3b603d5b117",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "6f2bfa40cfb6994ffe32d23d1e7f72cb61df31cc2d8576722bb0e6501ed2bd84",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "6f2bfa40cfb6994ffe32d23d1e7f72cb61df31cc2d8576722bb0e6501ed2bd84",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2405007360,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/33"
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
      "receipt": "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-shape-repeatability-v1/shape-two-seed2/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "4f94810fb5b119533b39be1a3835603afb2d820416a60637a90143adc8c1e4ba",
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
      "C34": {
        "name": "shape-two-seed2",
        "weights_sha256": "6f2bfa40cfb6994ffe32d23d1e7f72cb61df31cc2d8576722bb0e6501ed2bd84",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.015272181054025221,
          "pressure_mean_free_relative_l2": 0.0134615900145253,
          "pressure_absolute_rms": 0.6169821533929905,
          "pressure_absolute_max": 1.387804453029276
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
            "trajectory_id": "baseline--water-validation-0003",
            "native_frame": 63,
            "counts_xyz": [
              16,
              6,
              12
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.015272181054025221,
              "pressure_mean_free_relative_l2": 0.012701328721544327,
              "pressure_absolute_rms": 0.6169821533929905,
              "pressure_absolute_max": 1.387804453029276
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0130979516392747,
              "pressure_mean_free_relative_l2": 0.0134615900145253,
              "pressure_absolute_rms": 0.1513033097145439,
              "pressure_absolute_max": 0.3505941792372554
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C34/results.json",
          "sha256": "3f77e286db3bba91dbe4fb52a398992aeb3a29535762c04ef127324f985234de"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"06379a081028cd657b34237ac3c236e91f4fc92c105803824435a91ef9d52047","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"32d1256c5ec3597e2d911ff7d9e36be4759c1e14e4ca394e8efcac520133fd7b","source_witness_sha256":"3f1491d7581bc0e13c8d67fb3862a20b1be643748d8265be4d5857be19b276ee"},"elapsed_s":211.52131192499655,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.2549941036231758e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597156864,"peak_reserved_bytes":618659840,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c4457a1cc950cd199b80f694113b02abbf980529a3e10c0203b10ebfaa012b12"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2f0936683e803c9e3fd5291f012ae0e09dc38cc7277bbc9c47f0bfd5195d088c"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"e458d78f492e4bc88a26548afa3e255f11cb7ba91f86b10145070dc191f5f4d0"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"7a55774e1f1213e6a4c5ca013c8431d22ddaaa8270b5c49f0780ec10479f53d0"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"33c561aab416886a0fb0d43740a9996e6856f91e85cf1e6929bfad03d333cc44"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"16efa57e21cd8d0f80a63f124d41c88537e7f42b9df29e99a79ad3b603d5b117"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"6f2bfa40cfb6994ffe32d23d1e7f72cb61df31cc2d8576722bb0e6501ed2bd84"},"history.json":{"bytes":144650,"role":"training_diagnostics","sha256":"c452ce35f9bbf5db71bcdf2d594b4a69ec46fbff92e24de22d89baaadc4f03c2"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"d90c9112b6588618a5b09087b5ce051251f9f845984fa56594dcd74d696b7ffc"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":983,"role":"result_summary","sha256":"f36810c092d7281ed51682ad9786f7c5f22069350ce89de14f9a3d3f241a8442"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"6f2bfa40cfb6994ffe32d23d1e7f72cb61df31cc2d8576722bb0e6501ed2bd84"}},"per_case_inference_s":[0.020508374000201002,0.0038277959974948317,0.00492631399538368,0.003930442006094381,0.00345278499298729,0.0031765660096425563,0.003842689999146387,0.0039932309882715344,0.008052791003137827,0.0044302949972916394,0.005377452005632222,0.004662909981561825,0.010291325976140797,0.004206457990221679,0.006161814002553001,0.004224674019496888,0.0027054210077039897,0.00575879798270762,0.004240918002324179,0.004511073988396674,0.004836379026528448,0.005045237019658089,0.00987515298766084,0.0041222230065613985],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"3a68894759beb9c617cf206a779253fec45818af7b5aaca2d9f9a6c970eb90b5","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"5ac8b2a6fabf82ac81b5c3917f92ef46aa8fa9152924ce58ce76f6d30c478ee0","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"21531fcbf7e2c6a5de71f2949bd604b1ecfe2a951b60a0d48a0596cfb3b0d80d","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"d2e6aebe23ba7fabb26db70e302e21f5512c12a863de3eac0264ff15258c1842","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"de2b0679c1eb619a6dc62ae73978409e53102d71a69160d101b1f672c21262cd","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"49814332eb0b22f2ff5e4a41571bed3f8e501a72ba15e367b0f8662f0cb7381e","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"e30de4fa6cb014c0bd2c7b11eb639ec89fdba1c6b1be6275993c1a3e8fdfbe13","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"fb390145b951fd66f81eedd3019de7678e5ba11e3b3197ee30eb5bd446650f7d","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"eef6cb9e78e01d272120f2fd771afef2da0b74158137651f7cfcd8b2e023cf5f","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"e1cf26d62dc40926e2fd31ed43f7f82c7b9bd1960eb778a161d013e766956078","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"1531a5ac6cf64339e8b7b4b8d469e9d549034bdb2b8090ff2a8156b01d333e45","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"00dbdc3c74142909ac38d9d510d10d34ef7803f5ee42e6842ef6522100762913","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"2894ca4131af77f44b234ba06cc74bd43989e5f14752f4761d09fc257ecddd71","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"11f0d6d63b27cbc5c49e334f70ece4ad394fa4f3bffa2670334a94fd6f4dd1f4","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"827f705ea489f2c611dcac80edd67bb861d6dec4ac004594c4814240a0468831","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"c826f504faddf14a6570146bf7330c063475b18a5fa64db6f69945ae1800732d","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"d535f299ac4d474d1df2802c25e75f75142e71f3b6a76138149f12d8e786d1a1","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"9223171b54e6123d89cae75460478b1096b874e270af1f622f4daa86ba3b6a13","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"e1e7223154b230b8d93bade5822d804a31eb362e3f3e5cadc63f73fa698df074","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"47b95cf9bf70c2c6aaa3d052a27b77b333ad2cb46a94b1afce18e90a10f5bcec","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"f3d950c01e94a84d8731b7108fa6d658047d1eaa18c00fb9175f3578ae10df1e","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"dcddd9f56022e5243f8e9544a44d68ec34fc154198a943e12ffd4470aaa33350","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"6751ac59fcca2ecd4dbf3eb9eacd981aca61cc5aeab619bffe365b82f2d2e19e","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"6f37028bd9affaaa0dc87f23eafd3d96c4705deb0835d4146eb5d714aacaf481","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2405007360,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
