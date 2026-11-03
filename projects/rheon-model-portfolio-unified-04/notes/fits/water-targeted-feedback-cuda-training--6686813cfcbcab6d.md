# water-targeted-feedback · CUDA training

Retrospective source review · fit-6686813cfcbcab6d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only 370,016-parameter multiscale 3D convolutional correction network with 19 weight tensors. Executed layers remove GroupNorm, SiLU and biases despite inherited cubic-groupnorm naming. The physical equation residual q=(rho/h)r is divided by train-only q RMS before F32 convolution. Two shared-weight corrections recompute the F64 residual, scale by train previous-pressure RMS, apply cell-zero gauge and add in F64. dt is validated and reflected in externally assembled fields, not a separate CNN input. This sealed uniform cubic-voxel Neumann policy advances neither velocity nor tracer and accepts no independent geometry mask.

**Purpose.** Spend the same extra proposal/update budget on six targeted rectangular manufactured grids instead of resampling the existing bank.

**Experiment.** Both arms start from the same seeded linear tensors and physical sample order as C40: 5,000 Adam updates, batch1, initial rate0.001 with drops at3000/4500, model seed20261015 and physical sampling seed2026102501. Physical training uses16trajectories/32frames at8/16cells per metre; validation uses64trajectories/128frames, axes at most16cells, dt1/60s. Each update combines one physical example, one primary manufactured proposal at weight0.1 and a supplemental proposal at weight0.1. All three paths use final loss plus0.1nonfinal relative equation-residual squared. The primary bank has1,024fields across eight grids,625exposures/grid; the independent supplemental stream uses seed2026101607. C42 adds768fields across16x23x25 and10x7x25 with their cyclic permutations, all at16cells/metre; supplemental exposures are833/834per grid. Combined resident tensor projection is664,041,472bytes. These are pressure-operator fields, not new CFD trajectories or additional optimizer updates.

**Observed outcome.** Last logged training objective 0.00018567321082842798; final retained canonical validation objective 8.2308915674324754e-06. Common128-request worst mean-free pressure relative L2 0.01040558795365808; equation relative L2 0.010799691742291161. Complete GPU pressure-call median of case medians 4.4291665ms and worst case median 4.9793329999999996ms. Timing includes native CPU input validation/packing, transfers, pressure work, independent CPU output and synchronization, excluding setup/scoring; it is not a physical-step or drift measurement.

**Lesson.** C42 improves both common worst pressure and equation errors over C41 at equal optimizer and proposal counts. Compared with C40, pressure improves but equation error remains worse. Geometry, stream size and per-grid exposure change together; this is diagnostic evidence for targeted feedback, not held-out generalization or extra physical-scene coverage.

**Reproduction.** Use original typed recipe/config/runtime, exact captured source/input closures, train-only normalization and initial tensors. Recreate primary and supplemental manufactured descriptors, amplitudes, private seeds and balanced order; preserve three weighted loss paths and combined memory budget. Historical trainers include CPU parity/deployment forwards; qualify a GPU-only reproducer before rerunning. Clean retraining, exact final-checkpoint reproduction and new-Rheon generation remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "6686813cfcbcab6dd19bc8bd79bc348e03896dc542b8061aa84a77dcb17afaf7"
  },
  "history": {
    "path": "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/history.json",
    "sha256": "859923c371a78be97aec432b59c678074b2e046f72307a47d2fcf2ff0d46c762"
  },
  "sources": [
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/config.json",
      "sha256": "c48bd63375b5500d4708e29b71f3b75b4938eb74d2b3c2185f5983d3fc6a3872"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/custody-validator.py",
      "sha256": "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/model.py",
      "sha256": "8ba4368fc0bf38b434cd8382e374d7a19da9cc8c631ee3a7b4c22f57bb110ac7"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/policy.py",
      "sha256": "62342bd622496c8f355260d3a899aaa0002dff7ea7c31761cac39287b5822a49"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/qualification.json",
      "sha256": "3ac648e38d5d4b4c26546298e1e135bf19ce421c4fa33a5544b5c0d1a5646827"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/train.py",
      "sha256": "c8ce954ee2d64ff0a2350c051e596224cdf48c47b7ab2e06d63caf377dfb111d"
    }
  ],
  "source_witness": {
    "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/source-witness.json",
    "sha256": "30f35107050997bbf29bd734097047e1912aa42dd63004eda5d5449524b1f0f2"
  },
  "training_population": {
    "path": "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/splits.json",
    "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"
  },
  "training_inputs": [
    {
      "bytes": 2820310,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/dataset.npz",
      "sha256": "7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7",
      "slot": "dataset"
    },
    {
      "bytes": 2823990,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/dataset_extra.npz",
      "sha256": "73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904",
      "slot": "dataset_extra"
    },
    {
      "bytes": 2561142,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/reserved_test_dataset.npz",
      "sha256": "e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9",
      "slot": "reserved_test_dataset"
    },
    {
      "bytes": 2564086,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/reserved_test_dataset_extra.npz",
      "sha256": "08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618",
      "slot": "reserved_test_dataset_extra"
    },
    {
      "bytes": 339677,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/reserved_test_splits.json",
      "sha256": "45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012",
      "slot": "reserved_test_splits"
    },
    {
      "bytes": 411192,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/splits.json",
      "sha256": "86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42",
      "slot": "splits"
    },
    {
      "bytes": 2209622,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/validation_dataset.npz",
      "sha256": "c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e",
      "slot": "validation_dataset"
    },
    {
      "bytes": 2212566,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/validation_dataset_extra.npz",
      "sha256": "a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7",
      "slot": "validation_dataset_extra"
    },
    {
      "bytes": 372865,
      "path": "runs/water-targeted-feedback-v1/targeted-feedback/project/inputs/validation_splits.json",
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
    "item_id": "fit-6686813cfcbcab6d",
    "kind": "training_attempt",
    "name": "water-targeted-feedback · CUDA training",
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
    "receipt_sha256": "6686813cfcbcab6dd19bc8bd79bc348e03896dc542b8061aa84a77dcb17afaf7",
    "receipt_paths": [
      "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/receipt.json",
      "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/035610cdbfc94ffebbdbeb2403bf8b9f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-035610cdbfc94ffebbdbeb2403bf8b9f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "035610cdbfc94ffebbdbeb2403bf8b9f",
        "epoch": "archived_schema8_catalog",
        "observed_in": [
          "archived_schema8_catalog"
        ],
        "graph_name": "water-targeted-feedback · CUDA training",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "c48bd63375b5500d4708e29b71f3b75b4938eb74d2b3c2185f5983d3fc6a3872",
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
      "source_witness_sha256": "30f35107050997bbf29bd734097047e1912aa42dd63004eda5d5449524b1f0f2"
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
        "sha256": "df38a491cd70f48b86581039e0a519c368fc36e233ed45791b3b10797f0316ac",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-1000.npz",
        "sha256": "706c3cd051acc270abb19d378d6f006b8772ace2b2a62bfe0e67d61474c500e9",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "cd79eacee738ce8819cfccea00a0eeadbd17b68dfcca22468aa09f16a5f4b65d",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-4000.npz",
        "sha256": "4037c5b55f3e1442ed46fa52929f8d9d4b4abb99e5817e0160c6cd33c8df245e",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-500.npz",
        "sha256": "028fb31a4c334f54ec9027264beb5a4da563dc69df2f63bad3e95252146ba6c6",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2415157248,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/41"
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
      "receipt": "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/water-targeted-feedback-v1/targeted-feedback/managed-full/stage-00/artifacts/receipt.json",
    "sha256": "6686813cfcbcab6dd19bc8bd79bc348e03896dc542b8061aa84a77dcb17afaf7",
    "bytes": 14049
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C42": {
        "name": "targeted-feedback",
        "weights_sha256": "5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.010799691742291161,
          "pressure_mean_free_relative_l2": 0.01040558795365808,
          "pressure_absolute_rms": 0.3401834208512267,
          "pressure_absolute_max": 0.9024655843461319
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
              "common_equation_relative_l2": 0.007043338701785576,
              "pressure_mean_free_relative_l2": 0.009690212856801852,
              "pressure_absolute_rms": 0.3401834208512267,
              "pressure_absolute_max": 0.9024655843461319
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.010799691742291161,
              "pressure_mean_free_relative_l2": 0.01040558795365808,
              "pressure_absolute_rms": 0.18662234851922713,
              "pressure_absolute_max": 0.3960037811924109
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C42/results.json",
          "sha256": "54f1ca456f12ebad7c7ee56ff30c744da396612f70527025a02f915a8e87bee1"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":16,"split":"validation","trajectory_id":"baseline--water-validation-0000"},{"cell_size_m":0.0625,"counts_xyz":[5,16,9],"frame":1,"native_frame":63,"parent":21,"split":"validation","trajectory_id":"baseline--water-validation-0005"},{"cell_size_m":0.0625,"counts_xyz":[8,16,13],"frame":0,"native_frame":15,"parent":27,"split":"validation","trajectory_id":"baseline--water-validation-0011"},{"cell_size_m":0.0625,"counts_xyz":[12,13,7],"frame":1,"native_frame":63,"parent":33,"split":"validation","trajectory_id":"baseline--water-validation-0017"},{"cell_size_m":0.125,"counts_xyz":[16,6,5],"frame":0,"native_frame":15,"parent":38,"split":"validation","trajectory_id":"baseline--water-validation-0022"},{"cell_size_m":0.125,"counts_xyz":[13,10,11],"frame":1,"native_frame":63,"parent":44,"split":"validation","trajectory_id":"baseline--water-validation-0028"},{"cell_size_m":0.125,"counts_xyz":[10,5,13],"frame":0,"native_frame":15,"parent":50,"split":"validation","trajectory_id":"baseline--water-validation-0034"},{"cell_size_m":0.125,"counts_xyz":[10,8,12],"frame":1,"native_frame":63,"parent":56,"split":"validation","trajectory_id":"baseline--water-validation-0040"},{"cell_size_m":0.0625,"counts_xyz":[5,8,16],"frame":0,"native_frame":15,"parent":61,"split":"validation","trajectory_id":"baseline--water-validation-0045"},{"cell_size_m":0.0625,"counts_xyz":[7,9,13],"frame":1,"native_frame":63,"parent":67,"split":"validation","trajectory_id":"baseline--water-validation-0051"},{"cell_size_m":0.0625,"counts_xyz":[13,15,16],"frame":0,"native_frame":15,"parent":73,"split":"validation","trajectory_id":"baseline--water-validation-0057"},{"cell_size_m":0.0625,"counts_xyz":[7,8,9],"frame":1,"native_frame":63,"parent":79,"split":"validation","trajectory_id":"baseline--water-validation-0063"}],"closure":{"experiment_sha256":"c48bd63375b5500d4708e29b71f3b75b4938eb74d2b3c2185f5983d3fc6a3872","inputs":{"dataset":{"bytes":2820310,"sha256":"7d9a7c3769f35c59796fc1a663aacec4d38944534af12d00ed7f1f5bbe16b6f7"},"dataset_extra":{"bytes":2823990,"sha256":"73c9845df83276b261b3a7c01c565f361894a8794cddbe3ce6e4d0331c6e8904"},"reserved_test_dataset":{"bytes":2561142,"sha256":"e5c8e6413db8a5f86ddf1b62a4727a8dae26be033cd3617c25bca716eca1ffb9"},"reserved_test_dataset_extra":{"bytes":2564086,"sha256":"08ef42b15ebdf52e1d840163635bc15d340cfdfd5b6d9438cba66bc3981e4618"},"reserved_test_splits":{"bytes":339677,"sha256":"45a0a51537c50de2680c47bc76dc3a6be3a5483c25e431ffd2e97cb1cb65d012"},"splits":{"bytes":411192,"sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"validation_dataset":{"bytes":2209622,"sha256":"c917515d8290c9d60094c3412db19e0f5d06e83c62cace2eb8ec2c979f1ddf3e"},"validation_dataset_extra":{"bytes":2212566,"sha256":"a881a83e6b9832e95fbaa18013f4e049341a2146991257c9b258402a6b7ea1e7"},"validation_splits":{"bytes":372865,"sha256":"9d68874f91fe903b253a6f19e193c57304bde5e5dbc280db7fea43dee04371c5"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"a9154509441d79c4e04991a43e4fcdf5ce96c0ae18c17a2b5f792afa97c1a69d","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"30f35107050997bbf29bd734097047e1912aa42dd63004eda5d5449524b1f0f2"},"elapsed_s":341.44933804901666,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.7907306187225913e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":733602816,"peak_reserved_bytes":757071872,"tf32":false},"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-100.npz":{"bytes":1485452,"role":"checkpoint","sha256":"df38a491cd70f48b86581039e0a519c368fc36e233ed45791b3b10797f0316ac"},"checkpoint-1000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"706c3cd051acc270abb19d378d6f006b8772ace2b2a62bfe0e67d61474c500e9"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"cd79eacee738ce8819cfccea00a0eeadbd17b68dfcca22468aa09f16a5f4b65d"},"checkpoint-4000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"4037c5b55f3e1442ed46fa52929f8d9d4b4abb99e5817e0160c6cd33c8df245e"},"checkpoint-500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"028fb31a4c334f54ec9027264beb5a4da563dc69df2f63bad3e95252146ba6c6"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe"},"history.json":{"bytes":228627,"role":"training_diagnostics","sha256":"859923c371a78be97aec432b59c678074b2e046f72307a47d2fcf2ff0d46c762"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":261606,"role":"predictions","sha256":"89c4ed39f81ff0133bad0362c7105031bd3ca736e0ed8dde52c8684f05ccad09"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":411192,"role":"training_splits","sha256":"86b4c8ca83aa95a145297af4cf0692bf2aa24ce6fe6e990118ed70e16b00bb42"},"summary.json":{"bytes":982,"role":"result_summary","sha256":"8b519ed2cf0c9cfaca8bf632f756c990151ee3bbfa68ea12bae73da364450d05"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"5bbdc3159fe965b4768c14501bcb678ecffc837e0ccd01a513003bcad533f3fe"}},"per_case_inference_s":[0.020324596000136808,0.0036283149966038764,0.004550710989860818,0.0035433990124147385,0.003038766997633502,0.002953122020699084,0.0035930250014644116,0.003894683002727106,0.008786968974163756,0.004952982009854168,0.006214880006154999,0.00609710201388225,0.011653099005343392,0.004579475993523374,0.0070331679889932275,0.0045976919936947525,0.0027208759856875986,0.0053878249891567975,0.003845953004201874,0.003987957985373214,0.004298187996027991,0.004744211008073762,0.010029533004853874,0.0039145850169006735],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"611b0dad84d5efe5aab7cae3046c1d0c511d5918299d7002e547db782242464c","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"0a35fba8f94e69de557fc3a92feae5fb222d73723ddde867f71055e122cb3a65","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"bbf3d25fb773a717870612d3dcd65f21f2da225a7dfc62f606fa094a933af0e7","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"5c820ae409c7e493312b69de2cbd495191ac58ae6413af6be7a60ce565b78351","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"84e13e60a5fe23ff738bec54e7e4e9279b8a538c2ae212b160b27ea2bf077e05","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"dc6c6b9d8980f35499db4f4becebebbcda717927f86dedf0152b4b099352cd29","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"4d9a529e65521b286850f60ff5a776665d9c17dc688830bb10c37b0a912a11eb","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"0873596ac4f628e2de63e0bafcfc90983cbc092e32d66b08bc924e48776e53e5","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"2dccbd4c87087d56dce308f313881049eb646e6f63731a17fcb4b1dff2d7d253","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"c101713f631934370e955b278f6b9cfde2e8b2b6f10dda34515eebf65474ea5f","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"0663e23227ef0bf9422b27b482df885a2962a6a296de6ea476f97ed03968e1f9","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"0eabde8b5300ad8c3ec2e18e2291013307037ed3ceef16dfc99a69f37cc95731","shape":[7,15,14]},"case_012":{"bytes":32768,"dtype":"<f8","sha256":"9de8957f46f3f93021cfc9aaf16c7acb1502d8ea8181aa933d6a1b61bef21dc6","shape":[16,16,16]},"case_013":{"bytes":5760,"dtype":"<f8","sha256":"c0077c58b2c93ed24660f9a104debfdf3cd904e9034c87b186a9beec80d8813f","shape":[9,16,5]},"case_014":{"bytes":13312,"dtype":"<f8","sha256":"d23805d7b2819c0d589b3360620b886b986e425dbb866649f703bf3d0e325245","shape":[13,16,8]},"case_015":{"bytes":8736,"dtype":"<f8","sha256":"32e70f4f55b4d1e74de742f6bf326f0ecef066147791230d61982b83d5741b89","shape":[7,13,12]},"case_016":{"bytes":3840,"dtype":"<f8","sha256":"5729f0308ddb30b84ee1c733ce0e776ffbd2dd590f351f6af2d968608b0b45a8","shape":[5,6,16]},"case_017":{"bytes":11440,"dtype":"<f8","sha256":"c0a2f3ff14866ecfe4aa91b0f5c3ea1f96fcfa9398948e3fb84365aae00e4dc3","shape":[11,10,13]},"case_018":{"bytes":5200,"dtype":"<f8","sha256":"4105a605ee0cd188af19ab371e275e6c58bac442ca08d1cf387452c36ce47a15","shape":[13,5,10]},"case_019":{"bytes":7680,"dtype":"<f8","sha256":"92671fd4193bba4e52ba0e7dd2d387d780b6c5b6351a05dce3aa13930861115f","shape":[12,8,10]},"case_020":{"bytes":5120,"dtype":"<f8","sha256":"88c4531e48c4c44d450a4a5a395ce48d4f4d89df2be8dc4548c0f12d15aa2ad0","shape":[16,8,5]},"case_021":{"bytes":6552,"dtype":"<f8","sha256":"69bb66c3fbe77f9cf8ee1ed97bb4b4059fdf45ef328f2e562329244fa30064bf","shape":[13,9,7]},"case_022":{"bytes":24960,"dtype":"<f8","sha256":"b063a815d15325637438eb567ab86b1d2987191ab3f1953e569786824ad3d62c","shape":[16,15,13]},"case_023":{"bytes":4032,"dtype":"<f8","sha256":"0316802a61c51978d059c4097f77823059185d3c1684fd66cd3bf0b13e9535c9","shape":[9,8,7]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2415157248,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","status":"complete"}
```
