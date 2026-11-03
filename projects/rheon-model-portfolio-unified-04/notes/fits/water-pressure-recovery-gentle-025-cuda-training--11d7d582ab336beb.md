# water-pressure-recovery-gentle-025 · CUDA training

Retrospective source review · fit-11d7d582ab336beb

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only 370,016-parameter multiscale 3D CNN with19weight tensors, one physically scaled residual input q=(rho/h)r and train-only RMS normalization. Executed transforms remove GroupNorm, SiLU and convolution biases despite inherited model labels. Two shared-weight F32 corrections recompute the F64 sealed Neumann residual, apply train previous-pressure RMS and cell-zero gauge, then accumulate in F64. dt is validated at the call boundary, not an independent neural input channel. Training-only captured directions and first-correction losses do not add deployed passes or advance velocity/tracer.

**Purpose.** Test first-correction pressure feedback at weight0.25 using the expanded late-direction bank and unchanged two-correction inference.

**Experiment.** 5,000Adam updates, batch1, initial rate0.001 with drops at3000/4500; model seed20261015 and physical sampling seed2026102501. The same16physical training trajectories/32frames at8/16cells per metre and dt1/60s, with axes at most16cells, supply fitting; validation uses64trajectories/128frames. Each update also uses one of1,024primary manufactured fields, one of768supplemental rectangular fields and one of192captured direction cases, each stream weighted0.1. Direction seed2026102701 balances21groups over5,000extra proposals; 32cases derive from existing real frames and160from manufactured fields. The192-case bank adds48manufactured directions on8x12x37 and12x8x45 and cyclic permutations, at16cells/metre. The first-correction objective uses canonical physical loss for real data and the selected analytical loss for auxiliary streams. Existing final loss and0.1intermediate residual contraction remain; additional first-correction loss weight is0.25. These are residual-stage directions, not new CFD scenes or simulation timesteps.

**Observed outcome.** Last logged weighted training objective 0.0010100263907912323; retained final canonical validation objective 6.6791841116420112e-06. On128common consumed requests, worst mean-free pressure relative L2 0.0097925119844010679; equation relative L2 0.0043981398559624673. Complete GPU pressure-call median of case medians 4.4080517500000003ms, worst case median 4.8520285000000003ms. Timing includes CPU validation/packing, H2D, all pressure work, independent CPU output/D2H/synchronization; setup/scoring excluded. No physical-step or trajectory-drift result follows.

**Lesson.** Weight0.25 improves worst equation error over C45 but slightly raises worst pressure error. C47 at0.1 has lower values of both common worst errors. More first-correction supervision is not monotonically better.

**Reproduction.** Use the captured typed train recipe/config/runtime, full source witness, physical inputs/order, normalization and initial tensors. Preserve primary/supplemental banks and exact frozen C42 direction-source weights, eight-correction extraction, independent label generation, paired manifest/payload, selected early/late stage, balanced private order and first-correction objective. This audit checks retained arrays against the sealed FV equation without rerunning extraction or training. Historical recipes contain CPU neural forwards; qualify a GPU-only execution route before reproduction. Clean retraining/new-Rheon regeneration remains unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "11d7d582ab336beb120fb879a030d4870ac8d05f92b37b5b8e873a12a13d9f32"
  },
  "history": {
    "path": "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/history.json",
    "sha256": "664f383b2948325442e7a15df3f8e1856a188a9326315296d0c9aac0bf2611e2"
  },
  "sources": [
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/config.json",
      "sha256": "1539c89fd99f8f4f2c3b0c0e3d9ec3f948d31f9187c637c5fb0e63c85c424733"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/model.py",
      "sha256": "0bce2799411bb8da0e2dec5d82638f63959d9849924265b4c95fc239fcd57e99"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/policy.py",
      "sha256": "60aafd2c399fecf7bc6852b7e5ec386617741415cdd424e6aaeb29226a4ce7da"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/train.py",
      "sha256": "62c7dd0ad8bcc1b4e707800d6d329c318962d8e4abce973bf722e832f5f791b8"
    }
  ],
  "source_witness": {
    "path": "runs/water-pressure-recovery-v1/gentle-025/project/source-witness.json",
    "sha256": "0dfd283438f5893ff8416a1747094f5a986d7e9caa3177248bc56f27aea1be3d"
  },
  "training_population": {
    "path": "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 513808,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/direction_manifest.json",
      "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06",
      "slot": "direction_manifest"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-pressure-recovery-v1/gentle-025/project/inputs/validation_splits.json",
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
    "item_id": "fit-11d7d582ab336beb",
    "kind": "training_attempt",
    "name": "water-pressure-recovery-gentle-025 · CUDA training",
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
    "receipt_sha256": "11d7d582ab336beb120fb879a030d4870ac8d05f92b37b5b8e873a12a13d9f32",
    "receipt_paths": [
      "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-pressure-recovery-v1/gentle-025/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/5e270feb1e8648998b733158bc165c4b/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-5e270feb1e8648998b733158bc165c4b/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "5e270feb1e8648998b733158bc165c4b",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-pressure-recovery-gentle-025 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "1539c89fd99f8f4f2c3b0c0e3d9ec3f948d31f9187c637c5fb0e63c85c424733",
      "inputs": {
        "dataset": {
          "bytes": 2820310,
          "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"
        },
        "dataset_extra": {
          "bytes": 2823990,
          "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"
        },
        "direction_dataset": {
          "bytes": 24476086,
          "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"
        },
        "direction_manifest": {
          "bytes": 513808,
          "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"
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
      "source_witness_sha256": "0dfd283438f5893ff8416a1747094f5a986d7e9caa3177248bc56f27aea1be3d"
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
        "sha256": "56ac42e4248729fb0c432316cf5d18af4ea1ef346419c9dcbcca874f294de336",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "bd6c1c39129723c365bbd338c755686cdab0c0e3e1d6f3570b8b4c574550cde6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "9605cbdf61d41563679d68bd97aea5b07d848991dadcafc5a37f218e1d7a3948",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "ca06869c73e8b2b518135f94203d0e8dd7b038498fc3b3278535eda18d97ad45",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "fb46f71864cbfddd8adf83c313cf876ff403d03e640037bf983106dcb8250477",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "5e3f0f5c15c45b6dac414bb8567504a7bc2700ffb8d740b3bb75c9aa45babe84",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "5e3f0f5c15c45b6dac414bb8567504a7bc2700ffb8d740b3bb75c9aa45babe84",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2426331136,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/47"
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
      "receipt": "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/gentle-025/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "11d7d582ab336beb120fb879a030d4870ac8d05f92b37b5b8e873a12a13d9f32",
    "bytes": 14486
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C48": {
        "name": "gentle-025",
        "weights_sha256": "5e3f0f5c15c45b6dac414bb8567504a7bc2700ffb8d740b3bb75c9aa45babe84",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004398139855962467,
          "pressure_mean_free_relative_l2": 0.009792511984401068,
          "pressure_absolute_rms": 0.5049818826929969,
          "pressure_absolute_max": 0.9854351735604803
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0025",
            "native_frame": 63,
            "counts_xyz": [
              9,
              11,
              11
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0004",
            "native_frame": 15,
            "counts_xyz": [
              13,
              12,
              8
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004239850296606561,
              "pressure_mean_free_relative_l2": 0.009792511984401068,
              "pressure_absolute_rms": 0.5049818826929969,
              "pressure_absolute_max": 0.9854351735604803
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004398139855962467,
              "pressure_mean_free_relative_l2": 0.006590720351301718,
              "pressure_absolute_rms": 0.12724556520472155,
              "pressure_absolute_max": 0.2825560976141043
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C48/results.json",
          "sha256": "f1befdb7764617e4f95e274ca6de7461d46fa2dd7fee7b0a07fd523754255482"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.25,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"1539c89fd99f8f4f2c3b0c0e3d9ec3f948d31f9187c637c5fb0e63c85c424733","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":513808,"sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"0dfd283438f5893ff8416a1747094f5a986d7e9caa3177248bc56f27aea1be3d"},"elapsed_s":498.4793953920016,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.132774142817652e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":770565632,"peak_reserved_bytes":822083584,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"56ac42e4248729fb0c432316cf5d18af4ea1ef346419c9dcbcca874f294de336"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"bd6c1c39129723c365bbd338c755686cdab0c0e3e1d6f3570b8b4c574550cde6"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"9605cbdf61d41563679d68bd97aea5b07d848991dadcafc5a37f218e1d7a3948"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ca06869c73e8b2b518135f94203d0e8dd7b038498fc3b3278535eda18d97ad45"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"fb46f71864cbfddd8adf83c313cf876ff403d03e640037bf983106dcb8250477"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"5e3f0f5c15c45b6dac414bb8567504a7bc2700ffb8d740b3bb75c9aa45babe84"},"history.json":{"bytes":230286,"role":"training_diagnostics","sha256":"664f383b2948325442e7a15df3f8e1856a188a9326315296d0c9aac0bf2611e2"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"829b464903103ea9fc2eeb23258772b42d6170c0947f56ebeb9fac476ba1342a"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"61ae805edae72c05f2e46d752cb62a6370cff33573333f3211117b7aa5e86ae4"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"5e3f0f5c15c45b6dac414bb8567504a7bc2700ffb8d740b3bb75c9aa45babe84"}},"per_case_inference_s":[0.02012569599901326,0.0033377880026819184,0.004124385995964985,0.0032094320049509406,0.0027695460012182593,0.0026365439989604056,0.003331558997160755,0.0035338219968252815,0.008109530004730914,0.004675379997934215,0.005749586001911666,0.005121211004734505,0.010982000996591523,0.004257033993781079,0.0065127909983857535,0.004172540000581648,0.002440999000100419,0.004738575000374112,0.003541224999935366,0.003739471998414956,0.004068848000315484,0.004291377998015378,0.00870849700004328,0.0035410979980952106],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"3ed5c171d1e69442153906f3130d9ba9ed8f0b3684a9031227c0d4d5d7a36675","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"c155f1799f0f314e0fca35eddb5bdacf64edfed1a91a763ce26f38a5a92de403","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"6f2494bda96e97f0b716662a29f2440cb67b7a1a6585db65cf152ac78e5284ef","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"9cf79e0c360be4072400e5b7717974fba6579871b0fdd0ccabd7616f0a4fed78","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"731be67ea691e7ff470a582978d0e3ed5a744dc389894d65c18b378daf5d701c","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"1299f01473d0f9a7cccec0fdd8a764591c74ecb4654ee9f971c753fd2644deed","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"31cc2599c2f887c55514e529b281f4a37f589665f6accc41692318009b4081b4","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"5795b8730fd96658314e51db42477483b079d88f5d967067e0867f2c64fd5b48","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"43136b41451a1205de7ae672933568186e7fd43418723ef4065ac4f6b1eb3e19","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"8d641de8493cd846a3cf91fbff76fe7b6d0bf31b61f1b6732e100b41beb9775a","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"603d87b17f39fab6027bd77b7bdbf459fa5ed2fbe38fa254e2f566c783ae3a4c","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"7298c7de8aed99cc8f0770aed0aa2abe3f9b115a183b2080681b158f31d24d27","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"15ecfebbcd9d3dabaaff01f447812e9b397b0129720954bab960941eb6ffa0b5","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"4f62af40419e213f48b64a421e8ae251f6ea0caf107bbe55f7ad506e52afe2d7","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"154117e872009107b1a02eef324455b6e2511735b737feee4e28e6147787d12e","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"0e79000fc8c21a86bc799bcb0345882928737fec3f84ef0eda24d85dc7ef2f21","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"3e7ce5f298db863d77a04ca14cb8e70e06c790136d8cd395c8ee62616a733006","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"714c610852e10ad58e1229c58dc2de62c6d295ba0222917e2a20f51dd2414592","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"24e2c0a078fe87d9dc3ec96fb190d71a08675e7fb3cf3cd327931e6ffe3508fd","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"6bb395ed81cd6c4ae133d33da0b76993ca8657dd287c211cf44c18577e50a4bd","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"7c1dfb1ff207aca2563a5727ad6023c1ab70ebfd5033cd4c48072fd3c890935d","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"86821b0b1cd98f0beb8ddf24c9dd32a1e6e00563c5d0894c6ec4578b42c3bcac","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"ac320e72135b398f0399757b3805b6397acac31c2724cb2c76a3ab178cb41bd8","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"8cd16b410d1b509338d9f43a1537b90b7de90244df4159120072e8dc4fa90962","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2426331136,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
