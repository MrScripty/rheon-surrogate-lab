# water-supplement-control · CUDA training

Retrospective source review · fit-2ed95199d5282a9e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only 370,016-parameter multiscale 3D convolutional correction network with 19 weight tensors. Executed layers remove GroupNorm, SiLU and biases despite inherited cubic-groupnorm naming. The physical equation residual q=(rho/h)r is divided by train-only q RMS before F32 convolution. Two shared-weight corrections recompute the F64 residual, scale by train previous-pressure RMS, apply cell-zero gauge and add in F64. dt is validated and reflected in externally assembled fields, not a separate CNN input. This sealed uniform cubic-voxel Neumann policy advances neither velocity nor tracer and accepts no independent geometry mask.

**Purpose.** Control the extra training work with a second independently sampled proposal from the existing eight manufactured grids.

**Experiment.** Both arms start from the same seeded linear tensors and physical sample order as C40: 5,000 Adam updates, batch1, initial rate0.001 with drops at3000/4500, model seed20261015 and physical sampling seed2026102501. Physical training uses16trajectories/32frames at8/16cells per metre; validation uses64trajectories/128frames, axes at most16cells, dt1/60s. Each update combines one physical example, one primary manufactured proposal at weight0.1 and a supplemental proposal at weight0.1. All three paths use final loss plus0.1nonfinal relative equation-residual squared. The primary bank has1,024fields across eight grids,625exposures/grid; the independent supplemental stream uses seed2026101607. C41 reuses the same1,024primary tensors with625extra exposures/grid and zero additional resident tensors. These are pressure-operator fields, not new CFD trajectories or additional optimizer updates.

**Observed outcome.** Last logged training objective 0.00021361487661882746; final retained canonical validation objective 7.6318719296305125e-06. Common128-request worst mean-free pressure relative L2 0.011980935908120841; equation relative L2 0.011665529413214764. Complete GPU pressure-call median of case medians 4.4204892500000001ms and worst case median 4.7653920000000003ms. Timing includes native CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization, excluding setup/scoring; it is not a physical-step or drift measurement.

**Lesson.** Relative to C40, the extra primary stream lowers worst pressure error but raises worst equation error. More feedback work has a metric-dependent effect; the changed weighted training objective cannot be compared as the same scalar loss.

**Reproduction.** Use original typed recipe/config/runtime, exact captured source/input closures, train-only normalization and initial tensors. Recreate primary and supplemental manufactured descriptors, amplitudes, private seeds and balanced order; preserve three weighted loss paths and combined memory budget. Historical trainers include CPU parity/deployment forwards; qualify a GPU-only reproducer before rerunning. Clean retraining, exact final-checkpoint reproduction and new-Rheon generation remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "2ed95199d5282a9e28d54d0b4c0fcbcf808b897ff7068dedd9d735186d9254ac"
  },
  "history": {
    "path": "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/history.json",
    "sha256": "631139bea4e811b4ae88cdf166ea81c5c71763c258d1c1029740be4b201a75b3"
  },
  "sources": [
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/config.json",
      "sha256": "506b14192e43b0ea15c2e2ff9789af5c3403bbcef301fadcc705debf95a4c142"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/model.py",
      "sha256": "8ba4368fc0bf38b434cd8382e374d7a19da9cc8c631ee3a7b4c22f57bb110ac7"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/policy.py",
      "sha256": "62342bd622496c8f355260d3a899aaa0002dff7ea7c31761cac39287b5822a49"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/train.py",
      "sha256": "c8ce954ee2d64ff0a2350c051e596224cdf48c47b7ab2e06d63caf377dfb111d"
    }
  ],
  "source_witness": {
    "path": "runs/water-targeted-feedback-v1/supplement-control/project/source-witness.json",
    "sha256": "debc33a3d968efad257a59854dd0215af5b00117fa2eeb1ec38a213918a03161"
  },
  "training_population": {
    "path": "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-targeted-feedback-v1/supplement-control/project/inputs/validation_splits.json",
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
    "item_id": "fit-2ed95199d5282a9e",
    "kind": "training_attempt",
    "name": "water-supplement-control · CUDA training",
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
    "receipt_sha256": "2ed95199d5282a9e28d54d0b4c0fcbcf808b897ff7068dedd9d735186d9254ac",
    "receipt_paths": [
      "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-targeted-feedback-v1/supplement-control/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/b6f7ac68a12947b8bfac18120628413c/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-b6f7ac68a12947b8bfac18120628413c/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b6f7ac68a12947b8bfac18120628413c",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-supplement-control · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "506b14192e43b0ea15c2e2ff9789af5c3403bbcef301fadcc705debf95a4c142",
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
      "source_witness_sha256": "debc33a3d968efad257a59854dd0215af5b00117fa2eeb1ec38a213918a03161"
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
        "sha256": "70df77cdd1715d7080cd5f59e6dba82bd0a624774e0bf3b45dfefd0944f6279d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "01d02e06212dc57749a853c7d0dd668077558cfbb54221fbfc4b8f4a4af92e97",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "7603d3dc58c214b928629aabf80ecf09f1d8361f2e8239b2e2bd5f339c240248",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "f1216555e79f6bb076894458578521205dc5243b4cfbcba9540c6a9eddb0b615",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "98f8600032cbf5fd2d8f3755048bbe769c7bab054243d4e501086ddc6c9a7b3f",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "2e618e68871a3777c761a75691ca8ec9525717acfdafc501362562b6c8c30d19",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "2e618e68871a3777c761a75691ca8ec9525717acfdafc501362562b6c8c30d19",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2406203392,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/40"
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
      "receipt": "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-targeted-feedback-v1/supplement-control/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "2ed95199d5282a9e28d54d0b4c0fcbcf808b897ff7068dedd9d735186d9254ac",
    "bytes": 13611
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C41": {
        "name": "supplement-control",
        "weights_sha256": "2e618e68871a3777c761a75691ca8ec9525717acfdafc501362562b6c8c30d19",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.011665529413214764,
          "pressure_mean_free_relative_l2": 0.01198093590812084,
          "pressure_absolute_rms": 0.3157961461188488,
          "pressure_absolute_max": 0.7867468696157485
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
              "common_equation_relative_l2": 0.005629505485328828,
              "pressure_mean_free_relative_l2": 0.006351738073604725,
              "pressure_absolute_rms": 0.3157961461188488,
              "pressure_absolute_max": 0.7867468696157485
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.011665529413214764,
              "pressure_mean_free_relative_l2": 0.01198093590812084,
              "pressure_absolute_rms": 0.21487593075851297,
              "pressure_absolute_max": 0.4805169149260978
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C41/results.json",
          "sha256": "5d6f50502c4e4c940ab8bcdc40fb872f137510ef0a0a9ca0ced47339966584a8"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"seed":2026101607,"source":"primary","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"506b14192e43b0ea15c2e2ff9789af5c3403bbcef301fadcc705debf95a4c142","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"debc33a3d968efad257a59854dd0215af5b00117fa2eeb1ec38a213918a03161"},"elapsed_s":377.9223058060161,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":9.616886643470934e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":597157376,"peak_reserved_bytes":620756992,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"70df77cdd1715d7080cd5f59e6dba82bd0a624774e0bf3b45dfefd0944f6279d"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"01d02e06212dc57749a853c7d0dd668077558cfbb54221fbfc4b8f4a4af92e97"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"7603d3dc58c214b928629aabf80ecf09f1d8361f2e8239b2e2bd5f339c240248"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"f1216555e79f6bb076894458578521205dc5243b4cfbcba9540c6a9eddb0b615"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"98f8600032cbf5fd2d8f3755048bbe769c7bab054243d4e501086ddc6c9a7b3f"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2e618e68871a3777c761a75691ca8ec9525717acfdafc501362562b6c8c30d19"},"history.json":{"bytes":256874,"role":"training_diagnostics","sha256":"631139bea4e811b4ae88cdf166ea81c5c71763c258d1c1029740be4b201a75b3"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"2e1fc7f2b0fd32570978aa94d52b1db6a73a9ca480006e4a6f2fa51a62842624"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"3f675744adaa473fadc77962f9af4c55bd20772541f09346435f7ae4c5a020c2"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2e618e68871a3777c761a75691ca8ec9525717acfdafc501362562b6c8c30d19"}},"per_case_inference_s":[0.021544418996199965,0.0044367219961714,0.005413184000644833,0.0044397170131560415,0.003986280004028231,0.0037776220124214888,0.004732872999738902,0.0049935080169234425,0.009906357998261228,0.005382713978178799,0.006072903022868559,0.005348648992367089,0.011738477012841031,0.0046174599847290665,0.006849034980405122,0.004836143023567274,0.0030764240073040128,0.0063782670185901225,0.004586297000059858,0.004899822000879794,0.005144894006662071,0.005442179011879489,0.011569549998966977,0.004278579988749698],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"216da6c6489202e3d65a1a40ca8d7b259cf95397c3fb627f44594829be1f954a","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"9da2e136f58ef2fd5a6631113915d03856ffccb79bfba6985adaaf7bd699283b","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"0000f62ce72bcd6721f30989d7c2a9e741acd1699f307140a7a5706bd1930061","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"1f2bb395bd7262f37c23f48d44c4c4f3b50c9612be60bd15b76e3b78c2280a2f","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"46169b1253fa30e93d8fec30fd44cdcb55d0f1c2e907500e936492f1b5d0086c","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"cb8acbd4d5833f63dee60a3801271e88fab90bcb50478e5a4559b6e1f38261f7","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"1f50e4d0663c01febe3cf0629c6b9ec5ff2d91c189619a8a16042b4ba70b361d","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"e36daaef1bd4e143a47330599d89b9759ea71dcc39b5788f232e23e023c0bb1a","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"50053c3c03cb59ad1e62e3411602ef9d8c50ea413e0068a79c81003bf757697b","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"ff7bb242a9705dbd78cac52f61625f59c49fa1a46a1291833e92e1b3c6ee2cbd","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"ae9d63bebb9651e764451b3a1137c7c836806ef29b8ecedb6c40e0681e62b429","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"0198cd9b7f601ecec6a24252d7657f239b58db39be553d991c9a668d52ca8190","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"d1c74fe6e027a7c0f0f1027e37df90fd5ebe52b00b8d0610c98c43debe187141","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"cec4c8cd6be365d1d72bcf365704a0b345ca305110b8e20bf94756a1bc4327df","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"45cd4cfafcdc9a87ed75c9f73511a219ea08ebe18b96a27b0bbb92e76e956284","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"03252fcd18080c6f6d8f97897b2d89e416a45f674f20335ca1f173c24fb3f06b","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"ef357a1d05c68de3dcdb5b139282692417d0c25a94d628ac7be3407a736214f3","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"5f49b2df6cb525ac7d4d6193e9e2772f54ee615cfc38575c357060638a8cd495","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"f132b4c77886b173df46680e42eb30630206d30e1c807389171ad3042a61fa66","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"bbd4256790ee2e43d4b3b783f20c8204caf3dcbd4bb389333a8de66c5487047b","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"39ac7d2df461dd387b5bed2188231ccb5859eae9dcdb7aee35ffd5f92eb5e760","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"13d0129feef79065507450a7a5948de5f50142a5b8ff93020fdae966b4caab17","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"20facaa31cf13cb31a5f964db10ceac48df2c78c802cabec5bf88470c739c1c3","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"ee9980f7284e58491bb7b4b5603f87e85e5cb3dbb00cc798cfd8131530edf969","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2406203392,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
