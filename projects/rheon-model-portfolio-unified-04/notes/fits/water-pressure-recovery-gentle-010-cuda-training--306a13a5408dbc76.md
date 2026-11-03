# water-pressure-recovery-gentle-010 · CUDA training

Retrospective source review · fit-306a13a5408dbc76

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only 370,016-parameter multiscale 3D CNN with19weight tensors, one physically scaled residual input q=(rho/h)r and train-only RMS normalization. Executed transforms remove GroupNorm, SiLU and convolution biases despite inherited model labels. Two shared-weight F32 corrections recompute the F64 sealed Neumann residual, apply train previous-pressure RMS and cell-zero gauge, then accumulate in F64. dt is validated at the call boundary, not an independent neural input channel. Training-only captured directions and first-correction losses do not add deployed passes or advance velocity/tracer.

**Purpose.** Test first-correction pressure feedback at weight0.1 using the expanded late-direction bank and unchanged two-correction inference.

**Experiment.** 5,000Adam updates, batch1, initial rate0.001 with drops at3000/4500; model seed20261015 and physical sampling seed2026102501. The same16physical training trajectories/32frames at8/16cells per metre and dt1/60s, with axes at most16cells, supply fitting; validation uses64trajectories/128frames. Each update also uses one of1,024primary manufactured fields, one of768supplemental rectangular fields and one of192captured direction cases, each stream weighted0.1. Direction seed2026102701 balances21groups over5,000extra proposals; 32cases derive from existing real frames and160from manufactured fields. The192-case bank adds48manufactured directions on8x12x37 and12x8x45 and cyclic permutations, at16cells/metre. The first-correction objective uses canonical physical loss for real data and the selected analytical loss for auxiliary streams. Existing final loss and0.1intermediate residual contraction remain; additional first-correction loss weight is0.1. These are residual-stage directions, not new CFD scenes or simulation timesteps.

**Observed outcome.** Last logged weighted training objective 0.00080621303367485486; retained final canonical validation objective 4.6953178976916208e-06. On128common consumed requests, worst mean-free pressure relative L2 0.0075183026587037549; equation relative L2 0.0029634213762156961. Complete GPU pressure-call median of case medians 4.3307690000000001ms, worst case median 5.1623095000000001ms. Timing includes CPU validation/packing, H2D, all pressure work, independent CPU output/D2H/synchronization; setup/scoring excluded. No physical-step or trajectory-drift result follows.

**Lesson.** A weight0.1 first-correction objective improves both common worst errors over C45 and C46. This favors a gentler training weight on the retained diagnostic population; independent seeds, held-out domains and rollouts remain unqualified.

**Reproduction.** Use the captured typed train recipe/config/runtime, full source witness, physical inputs/order, normalization and initial tensors. Preserve primary/supplemental banks and exact frozen C42 direction-source weights, eight-correction extraction, independent label generation, paired manifest/payload, selected early/late stage, balanced private order and first-correction objective. This audit checks retained arrays against the sealed FV equation without rerunning extraction or training. Historical recipes contain CPU neural forwards; qualify a GPU-only execution route before reproduction. Clean retraining/new-Rheon regeneration remains unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "306a13a5408dbc76e291e8c399ac3b006f70da31ec38702b17ccfd729636e8ae"
  },
  "history": {
    "path": "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/history.json",
    "sha256": "a45cc3e878e1d2289fa2f8bf177d1d777da3ffd4335d3176337a9b17ab5732aa"
  },
  "sources": [
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/config.json",
      "sha256": "f9c0e356286ed0f05fd4e601a724dca04876e5eb27b296766df1eb99038c7586"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/model.py",
      "sha256": "0bce2799411bb8da0e2dec5d82638f63959d9849924265b4c95fc239fcd57e99"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/policy.py",
      "sha256": "60aafd2c399fecf7bc6852b7e5ec386617741415cdd424e6aaeb29226a4ce7da"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/train.py",
      "sha256": "62c7dd0ad8bcc1b4e707800d6d329c318962d8e4abce973bf722e832f5f791b8"
    }
  ],
  "source_witness": {
    "path": "runs/water-pressure-recovery-v1/gentle-010/project/source-witness.json",
    "sha256": "98f5055916436f4d297d40d44a852e2dfcd3745cd88484d2d7cb3f219d95be2c"
  },
  "training_population": {
    "path": "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 513808,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/direction_manifest.json",
      "sha256": "15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06",
      "slot": "direction_manifest"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-pressure-recovery-v1/gentle-010/project/inputs/validation_splits.json",
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
    "item_id": "fit-306a13a5408dbc76",
    "kind": "training_attempt",
    "name": "water-pressure-recovery-gentle-010 · CUDA training",
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
    "receipt_sha256": "306a13a5408dbc76e291e8c399ac3b006f70da31ec38702b17ccfd729636e8ae",
    "receipt_paths": [
      "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-pressure-recovery-v1/gentle-010/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/836fc1d5741443bba6de416197950579/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-836fc1d5741443bba6de416197950579/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "836fc1d5741443bba6de416197950579",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-pressure-recovery-gentle-010 · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "f9c0e356286ed0f05fd4e601a724dca04876e5eb27b296766df1eb99038c7586",
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
      "source_witness_sha256": "98f5055916436f4d297d40d44a852e2dfcd3745cd88484d2d7cb3f219d95be2c"
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
        "sha256": "59fca19144c83267b34d5b1076774bce4e28ed778c5f074f8e0cb20febb21967",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "3e2a4c0122e1a69588832f5c12287f3905134dc97045413212a9161df9b05366",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "3f640b03015df54728627dbb5523844090dbbb112591794f09eb78d884e87bb5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "c4f58307edba6f349a1d0054d4c91ee2da923140ad43d12ae88e6c7df1ceb064",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "0a2e0c9fc2f42f4007bf82669b1b9f683d936eb38c849e59174acf3796466790",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "66df7bc5df5aeb1bc9b914054a4ffd4148002248ea319afc14364f0015aa48f3",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "66df7bc5df5aeb1bc9b914054a4ffd4148002248ea319afc14364f0015aa48f3",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2419789824,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/46"
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
      "receipt": "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-pressure-recovery-v1/gentle-010/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "306a13a5408dbc76e291e8c399ac3b006f70da31ec38702b17ccfd729636e8ae",
    "bytes": 14482
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C47": {
        "name": "gentle-010",
        "weights_sha256": "66df7bc5df5aeb1bc9b914054a4ffd4148002248ea319afc14364f0015aa48f3",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.002963421376215696,
          "pressure_mean_free_relative_l2": 0.007518302658703755,
          "pressure_absolute_rms": 0.5004077041217551,
          "pressure_absolute_max": 1.040374510382394
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0000",
            "native_frame": 63,
            "counts_xyz": [
              16,
              16,
              16
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
              "common_equation_relative_l2": 0.002963421376215696,
              "pressure_mean_free_relative_l2": 0.007228377113740479,
              "pressure_absolute_rms": 0.5004077041217551,
              "pressure_absolute_max": 1.040374510382394
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.002731512340434527,
              "pressure_mean_free_relative_l2": 0.007518302658703755,
              "pressure_absolute_rms": 0.10090603793132967,
              "pressure_absolute_max": 0.2352530840252884
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-material-GPU-accuracy-01/C47/results.json",
          "sha256": "7cb38685ee50041660f4495a28abeaac2e056b82fe2f92725269328680187c7d"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.1,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"f9c0e356286ed0f05fd4e601a724dca04876e5eb27b296766df1eb99038c7586","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":513808,"sha256":"15995c95b2f9aec9695bd53f76700f73f8c71aa1449d816f61e7618540b3ee06"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"98f5055916436f4d297d40d44a852e2dfcd3745cd88484d2d7cb3f219d95be2c"},"elapsed_s":567.8473105529993,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":6.284138004275519e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":770565632,"peak_reserved_bytes":822083584,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"59fca19144c83267b34d5b1076774bce4e28ed778c5f074f8e0cb20febb21967"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3e2a4c0122e1a69588832f5c12287f3905134dc97045413212a9161df9b05366"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3f640b03015df54728627dbb5523844090dbbb112591794f09eb78d884e87bb5"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"c4f58307edba6f349a1d0054d4c91ee2da923140ad43d12ae88e6c7df1ceb064"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0a2e0c9fc2f42f4007bf82669b1b9f683d936eb38c849e59174acf3796466790"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"66df7bc5df5aeb1bc9b914054a4ffd4148002248ea319afc14364f0015aa48f3"},"history.json":{"bytes":230272,"role":"training_diagnostics","sha256":"a45cc3e878e1d2289fa2f8bf177d1d777da3ffd4335d3176337a9b17ab5732aa"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"28640871c2d88d475d8ce78b5a0f7905ee5a99cb0286d64e5b51c53bdd07451f"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"f61f5cbc4114bc1e9bc4b42c766f46095055899e18cf0c08a642276cf66f4b3a"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"66df7bc5df5aeb1bc9b914054a4ffd4148002248ea319afc14364f0015aa48f3"}},"per_case_inference_s":[0.024273567003547214,0.0045563800013042055,0.005521028993825894,0.004343125001469161,0.003713728001457639,0.0035198870027670637,0.004442890000063926,0.00477777899504872,0.01005129700206453,0.0056326720005017705,0.006950178001716267,0.00602096899820026,0.014055760999326594,0.0051884170025005005,0.007914524001535028,0.005292908004776109,0.003335863999382127,0.006878196996694896,0.004761796997627243,0.005105554002511781,0.0055021910011419095,0.005918381997616962,0.012125401000957936,0.004637099998944905],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"da9e5544005d98aea9b156f747d108b69f6285367d5d5c2a47c22a8f86bb93ab","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"6b62f92c84c7b571688aae81d4ab06a66e3c81bae8053b1d830f8d2e4afa1868","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"60f6c2e70c6c7b6e0fd1d227b01ddc9e307af25873e5cfc19120ffece8bf8ba3","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"b8b11887541faf6c136425bcf85997475bffcfcb2a884df7a14a5c361a5b6cb0","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"5a4cb9fe9755d5b518677c3419a6aa6366c6a4303634123861463238ec1a09fd","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"4a0e6c2e391b1bf753fa02bc1b5657f38241f0b32e8b1f7e09a15dc59b88df8c","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"23555b24255c989aeedf2d6792c6ef6772b6dc44aea710bba8ea0e7ceff14602","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"a162cf55333412eac60c8bc84b53f04b75854fd0d6bcf1938e147aae68bc4f30","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"f9892d81d35dd391d83d94bee5b06468154c576a35f59b823f43ab499f341531","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"bff62c666c09be89b938fdb519f41f6aced024bf8b49f6254edf5e127398ebdf","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"0a6d0b922bd86b5a496293b4e7d6608d8a11076c440d262ed5b615ca0144f4e3","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"716458cdea4665629b67d8a8459fdf756f82d583702beaad24f9db3b06fa6083","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"bb431c6d4af6fdda7545e3fce6ce049dd5938c48d3ea7da5ad6c6b1db5c069c5","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"b748e5e996c19ec726d32c095086e32841af633caa4218616b73140e73155b3b","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"3fe67cd4904d2160147fd2d8d3502c154dd1f1114c3b2f1c2fbe46e74cbf6373","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"9afaf50b5a4746c8a60cf5a206d74661a7694eec093758c02c2f6310d4b74e05","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"96cbfc4ca2073e1253a63930c84f41734b3889dc491a715664b2ad05cd5604f7","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"82dab75eefb2f8ac59fdb9f0fbb9e2e7ac237294860673fdc0dc5ad0e3ab0d65","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"473ca389c6679230753d14eb625388ed69e576ba946881d0b467bbf9de65fda4","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"ff65e3d50a8e6335f3d50b449a8a55dcbe78ae35ff0640484d2e5f9e4880101e","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"45945cb71f5d1f73bc3cf3cb0b1532cff602e380205351648ea4714522213e4a","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"18f4e0afe3ad3b0ccca6b34ca7bc564f1959816b812e03decc56357816d5da24","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"ce22c04d2317704c8af501bf270f8ffedda1513e8df73bddbdf2997ed7a7d60e","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"93a054d8f97b7ae4a0743552f17daff1e773bffc69d724c4f8f3ce2a9635b38c","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2419789824,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
