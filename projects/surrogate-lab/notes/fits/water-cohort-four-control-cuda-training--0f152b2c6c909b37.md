# water-cohort-four-control · CUDA training

Retrospective source review · fit-0f152b2c6c909b37

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. The configured 4 shared-weight correction(s) recompute the F64 finite-volume residual between passes; outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.

**Purpose.** Measure a four-correction control on the same 16-trajectory cohort and five-grid manufactured bank as C28.

**Experiment.** 5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, model seed 20261015, physical sampling seed 2026102501, gradient clipping 1.0. Physical fitting uses 16 trajectories / 32 frames, 8 and 16 cells/metre, at most 16 cells on each axis, dt=1/60 s; validation uses 64 trajectories / 128 frames at 8/16 cells/metre. Manufactured feedback adds 640 pressure-operator fields across 5 grids (16/24/32/40-cubed and 24x16x12 at 16 cells/metre). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS 2.8535734640782584 Pa. Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. Amplitude comes from all physical train cells; normalization remains physical-train-only. Manufactured examples are not additional physical simulation trajectories.

**Observed outcome.** Last logged training objective 2.0151184263517961e-05; retained final validation objective 3.0659969938024104e-06. Common 128-request worst mean-free pressure relative L2 0.005566015064467765; equation relative L2 0.0064500190127297404. Complete GPU pressure-call timing: median of case medians 8.9891927500000008 ms; worst case median 10.088708 ms. Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.

**Lesson.** Four corrections improve both common worst errors relative to the matched one/two-correction cohort controls. More internal pressure work is not more temporal advancement; complete-step drift remains unmeasured by these pressure cases.

**Reproduction.** Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "0f152b2c6c909b37b778d57906076cdca994ca46216425e0844e838301ef302c"
  },
  "history": {
    "path": "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/history.json",
    "sha256": "42f2c899978a5029cb0f6d7d4c09595a223957c776fa8880fffc40875e388523"
  },
  "sources": [
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/analytical.py",
      "sha256": "b4b9bb8ad37f51bf0e7190f881588c5503c5d80d85b9c110d06fdaa42068ffcf"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/config.json",
      "sha256": "6eb7f273d96b297b507a14212229b161e5f15096f0baa2997c193b2fcf49d0b0"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/model.py",
      "sha256": "0886a7301877b92a0e62c1e6d759193262ec28adf40b0954b182e6f782fdb702"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/physical_residual.py",
      "sha256": "59ce14ab0d73024b6d728eab0d3bb4bf316bd80cab15066cf06e732f4cb87c03"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/policy.py",
      "sha256": "6fac4ba260eb6c627722b68bf9cde7c7b934797cacc244204b6fad6afad347cc"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/train.py",
      "sha256": "5001b5123a6d468be8ae401c0bf54af60dd4289c577c8f930b812b2e978e493b"
    }
  ],
  "source_witness": {
    "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/source-witness.json",
    "sha256": "5cb8c8f27a8ec65f3ce9a065b384a0d3a548542c3b99b4c26c03de87c12434ec"
  },
  "training_population": {
    "path": "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-recurrence-generalization-v1/cohort-four-control/project/inputs/validation_splits.json",
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
    "item_id": "fit-0f152b2c6c909b37",
    "kind": "training_attempt",
    "name": "water-cohort-four-control · CUDA training",
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
        "steps": 4
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "0f152b2c6c909b37b778d57906076cdca994ca46216425e0844e838301ef302c",
    "receipt_paths": [
      "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/90410f42f39147fdb9fa22c789d1e06f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-90410f42f39147fdb9fa22c789d1e06f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "90410f42f39147fdb9fa22c789d1e06f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-cohort-four-control · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "6eb7f273d96b297b507a14212229b161e5f15096f0baa2997c193b2fcf49d0b0",
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
      "source_witness_sha256": "5cb8c8f27a8ec65f3ce9a065b384a0d3a548542c3b99b4c26c03de87c12434ec"
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
        "sha256": "1a7c9923a792b70f864d98fd0500a2fad8b6857207e701a7f97156ae306c1ad5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "ffb581104d46f47f8bcc5c1770993fc6f354fff9b4ffb2fe2b07b89aa012fabe",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "766ea632222990a315e2c32b96b451c8fe7ef999193ac3be93af8c8bb00b8317",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "127d1b70af3ae2d518bf6e937a0bcc638fa75b2e350ddf4961ac969970eb8f68",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "33f46d019961a3d9c692fe195fced37a600efd2f524eed27c3a49a6c3b57d8a6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "db1da22408d8c6bac948d2798cb7e28871fcefaca6458f0e979c3d5486abee40",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "db1da22408d8c6bac948d2798cb7e28871fcefaca6458f0e979c3d5486abee40",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2406322176,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/31"
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
      "receipt": "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-recurrence-generalization-v1/cohort-four-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "0f152b2c6c909b37b778d57906076cdca994ca46216425e0844e838301ef302c",
    "bytes": 13179
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C32": {
        "name": "cohort-four-control",
        "weights_sha256": "db1da22408d8c6bac948d2798cb7e28871fcefaca6458f0e979c3d5486abee40",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00645001901272974,
          "pressure_mean_free_relative_l2": 0.005566015064467765,
          "pressure_absolute_rms": 0.15125650470771787,
          "pressure_absolute_max": 0.8808374817688358
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
              "common_equation_relative_l2": 0.00645001901272974,
              "pressure_mean_free_relative_l2": 0.0040797343347342176,
              "pressure_absolute_rms": 0.15125650470771787,
              "pressure_absolute_max": 0.8808374817688358
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.006171862119027482,
              "pressure_mean_free_relative_l2": 0.005566015064467765,
              "pressure_absolute_rms": 0.06836986370584185,
              "pressure_absolute_max": 0.17080684608713242
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C32/results.json",
          "sha256": "620f0206b74f92c1df77e0384149f82bb0d45d2a7e6ed30d3661c3632fb45129"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":4},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"6eb7f273d96b297b507a14212229b161e5f15096f0baa2997c193b2fcf49d0b0","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"5cb8c8f27a8ec65f3ce9a065b384a0d3a548542c3b99b4c26c03de87c12434ec"},"elapsed_s":407.7026758219872,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.568961897646659e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":594532352,"peak_reserved_bytes":629145600,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1a7c9923a792b70f864d98fd0500a2fad8b6857207e701a7f97156ae306c1ad5"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ffb581104d46f47f8bcc5c1770993fc6f354fff9b4ffb2fe2b07b89aa012fabe"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"766ea632222990a315e2c32b96b451c8fe7ef999193ac3be93af8c8bb00b8317"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"127d1b70af3ae2d518bf6e937a0bcc638fa75b2e350ddf4961ac969970eb8f68"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"33f46d019961a3d9c692fe195fced37a600efd2f524eed27c3a49a6c3b57d8a6"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"db1da22408d8c6bac948d2798cb7e28871fcefaca6458f0e979c3d5486abee40"},"history.json":{"bytes":103332,"role":"training_diagnostics","sha256":"42f2c899978a5029cb0f6d7d4c09595a223957c776fa8880fffc40875e388523"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"ad462f5485e4f47fe50ca45c98cef929f3107d3c2d9287a48d123c27b3e123c9"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"d2f715df0116823fcebc48573c3f7608bef606d59f838d6d567f766cb49d517a"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"db1da22408d8c6bac948d2798cb7e28871fcefaca6458f0e979c3d5486abee40"}},"per_case_inference_s":[0.029807837010594085,0.006914169003721327,0.00916915200650692,0.007993159990292042,0.007033249014057219,0.006560451001860201,0.008234396984335035,0.008807968988548964,0.017933438008185476,0.009745807998115197,0.011365204001776874,0.010200847988016903,0.02260301701608114,0.008981813007267192,0.014173628995195031,0.009058483992703259,0.00578754601883702,0.011715079977875575,0.008347638009581715,0.008965065004304051,0.009429929981706664,0.009995093976613134,0.019295212987344712,0.008084667002549395],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"89b7b1c6bf5f4ce3d11600b3d7d3a7ba9e9409a228e39c3b615e71f779a68067","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"6f1913ba94e34aa8b77c1abe2cc38c79fed0cebde054fdb52cddb080705a8c6a","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"6e6c642045a0ed6b66834916524121cc72d9e972e526ab6d3e45a17cb32ba69b","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"fd98886852518480400d783daf844554d8c97f2968c58737c91069d7375fced1","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"c9ab943feea5532c97585303720a5cb866a3e48295b4f133429f6a8903356913","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"c64ed3a108de50412c0f4564df15758da3f801cb669fc2bdda5d12cbe79a7a7a","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"a5831e36a39e18de83095ee5f789596eab85d88ebf183de5cafb6882ccf6ad86","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"36f6b1292755a6958d509ee889e88867868dfcb58b24301cb00864a40b643fea","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"a91b4538cf50fce588119f6a190ab0e2a3bb95215b2e19afb5861b20a3ef5045","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"38e62c326798e7fac9c3c21b23406e48105a84fbfde4cb8798799a4a5b923fe7","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"1dd534de67c7d48b21dec0865f8ba9b5fddd89dc8603638a20e0e104e6352ae5","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"00c29f29e28dffc580a1cbf882810012f8a79fbdd4aa935a9b488618e9086245","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"0bd08025c4ab20d2341ab308809d9e99fd0b6f24ef7b1c0cb51c5e5b199cfbe5","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"221878a4f21440621fa5a2e4cfada341c3373ee5ed08fc554c12ed22e5ee4354","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"9d02649d912b5ad65e9bc725e4d2509ec2821030df1f88e6b1c29ff57cd1c338","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"de3c56cfad93dd496ca36224dc407ce4e682aeb21e5fc673ae18f0800c2e4606","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"62bb2a8774a1421fdb5ebe5852631c6fc03f47bbb2d0ecf8b3a811a459713f3f","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"79f489bd1dc7f078ba8db52a645f57d6f8e44b95bb214e3eb8b8b698783211a5","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"1f5f555d7742d2c31a9ad4c0ab7d7e6ccfaaa736e7ce4c486e5492d2f3832124","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"68f549afaa6d911bca1339779706dde84b16dd30f0c6f56608dce34db1b6b94b","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"8d0e30747801f25dc9996a7715ac09c0ee45fd242a2d494c7ce243c8ff3e6695","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"016e1b298a2d66d8e70681209ea4edcdd97bb0f9c9026762f426c71c2bb00171","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"9f789ab4dd6a7f34aff88249dabb1e36ca4df9d80325416e966475ad516dffbf","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"a748adb9331015cc028c827e07d69c870d52a01f78fbd8b17bed309146ad068e","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2406322176,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
