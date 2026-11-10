# water-early-pressure-first-pressure · CUDA training

Retrospective source review · fit-1747cff30c82c8df

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only 370,016-parameter multiscale 3D CNN with19weight tensors, one physically scaled residual input q=(rho/h)r and train-only RMS normalization. Executed transforms remove GroupNorm, SiLU and convolution biases despite inherited model labels. Two shared-weight F32 corrections recompute the F64 sealed Neumann residual, apply train previous-pressure RMS and cell-zero gauge, then accumulate in F64. dt is validated at the call boundary, not an independent neural input channel. Training-only captured directions and first-correction losses do not add deployed passes or advance velocity/tracer.

**Purpose.** Test first-correction pressure feedback at weight1 using the expanded late-direction bank and unchanged two-correction inference.

**Experiment.** 5,000Adam updates, batch1, initial rate0.001 with drops at3000/4500; model seed20261015 and physical sampling seed2026102501. The same16physical training trajectories/32frames at8/16cells per metre and dt1/60s, with axes at most16cells, supply fitting; validation uses64trajectories/128frames. Each update also uses one of1,024primary manufactured fields, one of768supplemental rectangular fields and one of192captured direction cases, each stream weighted0.1. Direction seed2026102701 balances21groups over5,000extra proposals; 32cases derive from existing real frames and160from manufactured fields. The192-case bank adds48manufactured directions on8x12x37 and12x8x45 and cyclic permutations, at16cells/metre. The first-correction objective uses canonical physical loss for real data and the selected analytical loss for auxiliary streams. Existing final loss and0.1intermediate residual contraction remain; additional first-correction loss weight is1. These are residual-stage directions, not new CFD scenes or simulation timesteps.

**Observed outcome.** Last logged weighted training objective 0.0020036664511149018; retained final canonical validation objective 6.2177295500790058e-06. On128common consumed requests, worst mean-free pressure relative L2 0.013275492344397441; equation relative L2 0.0040508704613554519. Complete GPU pressure-call median of case medians 4.3413504999999999ms, worst case median 4.7265104999999998ms. Timing includes CPU validation/packing, H2D, all pressure work, independent CPU output/D2H/synchronization; setup/scoring excluded. No physical-step or trajectory-drift result follows.

**Lesson.** Relative to C45, a weight1.0 first-correction objective lowers worst equation error but raises worst pressure error. A lower residual tail does not guarantee a lower pressure tail, and this training term does not add inference passes.

**Reproduction.** Use the captured typed train recipe/config/runtime, full source witness, physical inputs/order, normalization and initial tensors. Preserve primary/supplemental banks and exact frozen C42 direction-source weights, eight-correction extraction, independent label generation, paired manifest/payload, selected early/late stage, balanced private order and first-correction objective. This audit checks retained arrays against the sealed FV equation without rerunning extraction or training. Historical recipes contain CPU neural forwards; qualify a GPU-only execution route before reproduction. Clean retraining/new-Rheon regeneration remains unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "1747cff30c82c8dfd8436e5e106d81fb1dcaf7abd8422ba4a022b3c729346391"
  },
  "history": {
    "path": "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/history.json",
    "sha256": "76bdd13d9c2e3db7786f95593e1facca5d1ce1a69ab944a3c34e453b0ac82771"
  },
  "sources": [
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/config.json",
      "sha256": "ec3afb1cf8f059708f1e2fc3d7af560cbbe3bd9c1593f7864dff067fe2d8313e"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/model.py",
      "sha256": "0bce2799411bb8da0e2dec5d82638f63959d9849924265b4c95fc239fcd57e99"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/policy.py",
      "sha256": "60aafd2c399fecf7bc6852b7e5ec386617741415cdd424e6aaeb29226a4ce7da"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-early-pressure-v1/first-pressure/project/train.py",
      "sha256": "62c7dd0ad8bcc1b4e707800d6d329c318962d8e4abce973bf722e832f5f791b8"
    }
  ],
  "source_witness": {
    "path": "runs/water-early-pressure-v1/first-pressure/project/source-witness.json",
    "sha256": "4091e34279e59afe7c08a806c7b8f31b6178c03fa7803cb1ac6cbbe690a3f149"
  },
  "training_population": {
    "path": "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 513808,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/direction_manifest.json",
      "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06",
      "slot": "direction_manifest"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-early-pressure-v1/first-pressure/project/inputs/validation_splits.json",
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
    "item_id": "fit-1747cff30c82c8df",
    "kind": "training_attempt",
    "name": "water-early-pressure-first-pressure · CUDA training",
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
    "receipt_sha256": "1747cff30c82c8dfd8436e5e106d81fb1dcaf7abd8422ba4a022b3c729346391",
    "receipt_paths": [
      "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-early-pressure-v1/first-pressure/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/dcef4b7521954f089542b704dcaeb3ce/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-dcef4b7521954f089542b704dcaeb3ce/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "dcef4b7521954f089542b704dcaeb3ce",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-early-pressure-first-pressure · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "ec3afb1cf8f059708f1e2fc3d7af560cbbe3bd9c1593f7864dff067fe2d8313e",
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
      "source_witness_sha256": "4091e34279e59afe7c08a806c7b8f31b6178c03fa7803cb1ac6cbbe690a3f149"
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
        "sha256": "1ff4a5782c4e86a93feb06701cbd63ec6997519f4dd2c3e7d15150063f42d47c",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "dc16ff9b1ab1e0ea0b408839f8032b808f3607653a3cf9bd91ac71cc020f2a07",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "d8b1bfda4de137ab3f4cb04db8dd4588290823d23d11f8d9f3274838e251bf19",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "caa86038db63948b9f3f664cdae6612244c89905a0d0af3138b021fba2f63dc5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "8d1b49d2655e105f127b161997044b3720b86edd6d0b1eb74361b544d2437485",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "6cbe0e245ff8491c4afdff526e254261ce2132f63174653e2772bbad31f120e7",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "6cbe0e245ff8491c4afdff526e254261ce2132f63174653e2772bbad31f120e7",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2425950208,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/45"
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
      "receipt": "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-early-pressure-v1/first-pressure/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "1747cff30c82c8dfd8436e5e106d81fb1dcaf7abd8422ba4a022b3c729346391",
    "bytes": 14483
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C46": {
        "name": "first-pressure",
        "weights_sha256": "6cbe0e245ff8491c4afdff526e254261ce2132f63174653e2772bbad31f120e7",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004050870461355452,
          "pressure_mean_free_relative_l2": 0.013275492344397441,
          "pressure_absolute_rms": 0.8560009261459465,
          "pressure_absolute_max": 1.6161490161506729
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
            "trajectory_id": "baseline--water-validation-0060",
            "native_frame": 15,
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
              "common_equation_relative_l2": 0.003265024623170688,
              "pressure_mean_free_relative_l2": 0.013275492344397441,
              "pressure_absolute_rms": 0.8560009261459465,
              "pressure_absolute_max": 1.6161490161506729
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004050870461355452,
              "pressure_mean_free_relative_l2": 0.008199141663764068,
              "pressure_absolute_rms": 0.11644636135175634,
              "pressure_absolute_max": 0.3124960196333394
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C46/results.json",
          "sha256": "b9222fc34648be7bb8fa9e58d236fe8f38d409a9690a7907613447d86b8bd65f"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":1.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"ec3afb1cf8f059708f1e2fc3d7af560cbbe3bd9c1593f7864dff067fe2d8313e","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":513808,"sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"4091e34279e59afe7c08a806c7b8f31b6178c03fa7803cb1ac6cbbe690a3f149"},"elapsed_s":458.7430112639995,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":6.0022637399015366e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":770565632,"peak_reserved_bytes":822083584,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"1ff4a5782c4e86a93feb06701cbd63ec6997519f4dd2c3e7d15150063f42d47c"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"dc16ff9b1ab1e0ea0b408839f8032b808f3607653a3cf9bd91ac71cc020f2a07"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"d8b1bfda4de137ab3f4cb04db8dd4588290823d23d11f8d9f3274838e251bf19"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"caa86038db63948b9f3f664cdae6612244c89905a0d0af3138b021fba2f63dc5"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"8d1b49d2655e105f127b161997044b3720b86edd6d0b1eb74361b544d2437485"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"6cbe0e245ff8491c4afdff526e254261ce2132f63174653e2772bbad31f120e7"},"history.json":{"bytes":230247,"role":"training_diagnostics","sha256":"76bdd13d9c2e3db7786f95593e1facca5d1ce1a69ab944a3c34e453b0ac82771"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"66676236017654c5d7183bfa9a24335ab326f2a1a097ee1fd418ffbcb78e795f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"1717507217546126c621e0b4544310bc4c6bbcb48f4c0acd2e155f063af169c3"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"6cbe0e245ff8491c4afdff526e254261ce2132f63174653e2772bbad31f120e7"}},"per_case_inference_s":[0.018254783000884345,0.003321151998534333,0.004848316002608044,0.004011855999124236,0.0036606020003091544,0.003324919001897797,0.004240509999362985,0.004422988000442274,0.008899700998881599,0.004851464000239503,0.005952676998276729,0.005382087001635227,0.010915278002357809,0.003875551999954041,0.00589722600125242,0.0041907949998858385,0.002629960999911418,0.005412346999946749,0.004385772997920867,0.004581819001032272,0.004945781998685561,0.005204599001444876,0.009792316999664763,0.0038346920009644236],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"e8a641e87739f73c29ed543e1cb49df433837525e62d9cc00bbc05666b5cb15b","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"16250c4f9a4db5d235215a6d415da677a8bff5f9279ccbdfc9cdab4ac99002c1","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"ef2df39119b06469d2ee12bd7a7c22a21dc7994f3b541424eee457131974e42e","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"645c78b608b841a1e1fc194bfeb2fdda99bd8a313af5fa849fea0000de1381f7","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"9f9d809aa8b3cc4d14ad4bd4841a4315bf445eb12e96ee0a65c5744f9074a4e4","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"bd7076d1f52f998ca21f057ac53bec879be49efba687e9e3c3226f598501b348","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"d2e28170c5ef89bf038cdd26f22ae8f36de36f977237246239edfa6441bfec8e","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"63470a50aa534206f962e6ce9bcac6a47ab2e63c8ac086b9726e526db86f1762","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"a2a529045e12b79566e57506cc21e4a8894fad8f52df2c464bd3f1f5b7846b9a","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"77e0db4bdcacc7a6978a2c4ae95227ea9615a26893d255fafccf1af5e7cbf04f","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"7b979bdb0340afeaa9222cb1acd9aaefe5a003c563107b45cac25fabd9c341fc","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"ed6155c2ede6291657aebb8e40c60049d5c0fd6e87c7f5af80f09bda4fd729b7","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"d122e74f8007b2da130137293df4d8a849c927adddf47a0f3d66fc7bb8c3263f","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"a1d0e09f0abb9dccd9755384b7222cc30dabf38814178e21c18fb56d37fd31e5","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"3fc1539ef39658aad604d01c2f86bae6bbb5a2d0f47617cb7652ff88285b94d5","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"58938de57319f5ca3ce6f1e47b3d284ca6fa32e5ed1b951894b2652b0dc29403","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"d7d1315c9ab813043d1a5fbc0c3c0fd7f8a1838489d239731aee302fef34f545","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"826d025e7bd3db00a6c06aec9f967c7a2429bce2fed2a9a4b9b797228cbeb548","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"9546f01b198ca887c8bb72af34a1354f8876f848ac0fe05715a14f7adb77a969","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"9b5a7c3bc341f4637160ab6dc2ae45bdef52e6ed9b4208c26771156e6e3ae7f2","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"c69ce91e35946d85b7fb1ffcb577d08e3bc7c838596d9e0d5196029891ee54e6","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"a1792c9490557c679afddd1709eba4926e3fa5510aaf4d26b727e7c2cf7bfc8e","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"829ba4c7a3065d952f6081c9a8accaea0b68ac2555f65f596af90e6c5ad6dd52","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"102725b315e9cb503d0504561ee0a8507114da8ff13d0edcfece040b8ae26fc1","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2425950208,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
