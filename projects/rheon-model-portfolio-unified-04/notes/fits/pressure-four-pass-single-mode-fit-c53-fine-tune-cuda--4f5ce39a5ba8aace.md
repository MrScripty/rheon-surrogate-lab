# Pressure four-pass · single-mode-fit · C53 fine-tune CUDA

Retrospective source review · fit-4f5ce39a5ba8aace

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear, bias-free one-channel F32 multiscale pressure CNN warm-started from exact C53 tensors. F64 sealed cubic-grid equation assembly uses q=(rho/h)r and fixed train-only normalization. Shared neural corrections are gauged and accumulated in F64. The fitting objective unrolls four pressure corrections with final logarithmic equation loss and total earlier-stage weight 0.05; the captured default inference still executes two. These passes do not advance physical simulation time.

**Purpose.** Test fitting capacity on one axis fundamental on one rectangular grid, calibrated using original train-only scales.

**Experiment.** 2,000 CUDA Adam updates; model seed 20261020, retained physical-index seed 2026102001, initial rate 0.0001 and schedule [{'update': 1000, 'learning_rate': 1e-05}, {'update': 1500, 'learning_rate': 1e-06}, {'update': 1750, 'learning_rate': 1e-07}, {'update': 1900, 'learning_rate': 1e-08}]. Original 16 physical trajectories/32 frames calibrate fixed normalization and manufactured pressure amplitude, but no physical scene frame enters the optimizer. Actual fitting uses 1 manufactured fundamental fields and 2,000 example presentations; all inherited auxiliary banks are disabled. There are 2,000 optimizer updates and 8,000 student neural root forwards. Physical metadata remains dt=1/60 s and fluid density 1000 kg/m3; grid density means cells per metre, not fluid density. Calibration and consumed diagnostics add no independent optimizer scenes.

**Observed outcome.** Retained final fitting loss 0.20097483978835981. Common 128 consumed pressure requests use default TWO-correction inference: worst mean-free pressure relative L2 0.019326517469390975, equation relative L2 0.0078423796769587165, median case-median complete GPU pressure call 4.3745427499999998 ms. Host input/output, transfers and synchronization are included; setup/scoring are excluded. Fitting-bank 1/2/4/8-correction traces are retained consumed-case equation diagnostics, not independent pressure or full-step benchmarks. No new model execution, fresh four-correction common benchmark or rollout result is added.

**Lesson.** A physical data manifest does not establish physical optimizer exposure. The one-pattern capacity probe can fit a narrow operator direction without demonstrating scene generalization. Report calibration data, actual optimizer population, update aggregation, fitting recurrence and measured inference recurrence separately.

**Reproduction.** Retain exact typed native source/config/runtime/input receipts, C53 warm tensors, fine-tuning contract, normalization, final checkpoint and the actual optimizer owner. Recreate manufactured fundamental fields from explicit grid/mode/density definitions and original train-calibrated amplitude; reproduce private balanced cycles and gradient aggregation. For the curriculum preserve its repeated patterns and supplemental category order. Historical publication/restore uses CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, intended four-correction common evaluation, new-Rheon regeneration and long own-history drift remain unqualified. Preserve original receipts and ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-study-01/train/receipt.json",
    "sha256": "4f5ce39a5ba8aace2a7cf54b2abe524d33a97d6335cc465b5c68d3fbe0628753"
  },
  "history": {
    "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-study-01/train/history.json",
    "sha256": "4d18791f5eb44c52c3fa172d7392ec4d1ced45a8f933e0c1f0a8034314999711"
  },
  "sources": [
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/config.json",
      "sha256": "f8c83b51250237f9f48d4098000b79437cab12750aad2974a4c2ebe59434f80e"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/fine-tuning.json",
      "sha256": "561e81be550fbb9c976a69e93e0783fd5068a568b17afcf073251721fb9bc9c1"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/fine_tuning.py",
      "sha256": "6fd07a5fc88979e2ee32c9718bf327c013132524086fe5cfbfd3f030b4dcf4c2"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/original_fine_tuning.py",
      "sha256": "6e8d5e11744c0622f700bc645226d5dd96dbf831dd9bea754328c97dc70ecc12"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/single_case.py",
      "sha256": "e6f20e8ab0595ef46696a75601eb4e13f1bca248716576690112d34d89556397"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/train.py",
      "sha256": "501f9a27d9bdeb84c902c096e16251f454e2e8a7313504349b84be7a46029cca"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-project-02/source-witness.json",
    "sha256": "484061bab3beb6fc4f82a495f7f7fa7a6284f6ca077c473e91e1f4883fbb7595"
  },
  "training_population": {
    "path": "runs/pressure-single-mode-fit-v1/single-mode-fit-study-01/train/splits.json",
    "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
  },
  "training_inputs": [
    {
      "bytes": 21428086,
      "path": "runs/pressure-current-policy-v1/bank-02/late.npz",
      "sha256": "8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521",
      "slot": "current_dataset"
    },
    {
      "bytes": 168912,
      "path": "runs/pressure-current-policy-v1/bank-02/manifest.json",
      "sha256": "fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e",
      "slot": "current_manifest"
    },
    {
      "bytes": 610070,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset.npz",
      "sha256": "89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e",
      "slot": "dataset"
    },
    {
      "bytes": 610806,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/dataset_extra.npz",
      "sha256": "c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5",
      "slot": "dataset_extra"
    },
    {
      "bytes": 24476086,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_dataset.npz",
      "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6",
      "slot": "direction_dataset"
    },
    {
      "bytes": 298490,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/direction_manifest.json",
      "sha256": "ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc",
      "slot": "direction_manifest"
    },
    {
      "bytes": 1485452,
      "path": "runs/pressure-current-policy-v1/current-late-study-03/restore/weights.npz",
      "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
      "slot": "initial_weights"
    },
    {
      "bytes": 256361,
      "path": "runs/dependency-migration-v1/training-project-6/inputs/splits.json",
      "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-4f5ce39a5ba8aace",
    "kind": "training_attempt",
    "name": "Pressure four-pass · single-mode-fit · C53 fine-tune CUDA",
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
    "receipt_sha256": "4f5ce39a5ba8aace2a7cf54b2abe524d33a97d6335cc465b5c68d3fbe0628753",
    "receipt_paths": [
      "runs/pressure-single-mode-fit-v1/preliminary-readback-01/receipt.json",
      "runs/pressure-single-mode-fit-v1/single-mode-fit-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f4011890960040359b6ab83db245dd6e",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · single-mode-fit · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "f8c83b51250237f9f48d4098000b79437cab12750aad2974a4c2ebe59434f80e",
      "inputs": {
        "current_dataset": {
          "bytes": 21428086,
          "sha256": "8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"
        },
        "current_manifest": {
          "bytes": 168912,
          "sha256": "fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"
        },
        "dataset": {
          "bytes": 610070,
          "sha256": "89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"
        },
        "dataset_extra": {
          "bytes": 610806,
          "sha256": "c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"
        },
        "direction_dataset": {
          "bytes": 24476086,
          "sha256": "4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"
        },
        "direction_manifest": {
          "bytes": 298490,
          "sha256": "ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"
        },
        "initial_weights": {
          "bytes": 1485452,
          "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"
        },
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "a0a83d11e22bd65c988a0606e9c16a93313aecbc1be6c3815efb68a408e5cffa",
      "source_witness_sha256": "484061bab3beb6fc4f82a495f7f7fa7a6284f6ca077c473e91e1f4883fbb7595"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2343051264,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/61"
    ],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": false
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/pressure-single-mode-fit-v1/preliminary-readback-01/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-single-mode-fit-v1/preliminary-readback-01/receipt.json",
    "sha256": "4f5ce39a5ba8aace2a7cf54b2abe524d33a97d6335cc465b5c68d3fbe0628753",
    "bytes": 16688
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C62": {
        "name": "single-mode-fit",
        "weights_sha256": "ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.007842379676958717,
          "pressure_mean_free_relative_l2": 0.019326517469390975,
          "pressure_absolute_rms": 0.8572848480318463,
          "pressure_absolute_max": 2.079260915337221
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
            "trajectory_id": "baseline--water-validation-0053",
            "native_frame": 63,
            "counts_xyz": [
              14,
              9,
              11
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.007842379676958717,
              "pressure_mean_free_relative_l2": 0.01782590751472229,
              "pressure_absolute_rms": 0.8572848480318463,
              "pressure_absolute_max": 2.079260915337221
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.006607987297926298,
              "pressure_mean_free_relative_l2": 0.019326517469390975,
              "pressure_absolute_rms": 0.3861304495724323,
              "pressure_absolute_max": 0.8899202165976351
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C62/results.json",
          "sha256": "d771e78df8ed34b8826b3738a58fc012532d9c6f5d444e2ab9601e25ea5c977e"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"single-mode-fit","early_weight":0.05,"fitting_case":{"amplitude_reference":"original-train-correction-rms","cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"f8c83b51250237f9f48d4098000b79437cab12750aad2974a4c2ebe59434f80e","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"a0a83d11e22bd65c988a0606e9c16a93313aecbc1be6c3815efb68a408e5cffa","source_witness_sha256":"484061bab3beb6fc4f82a495f7f7fa7a6284f6ca077c473e91e1f4883fbb7595"},"diagnostic_population":"train_only","elapsed_s":76.52764676400693,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.160658734278286e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":17131008,"peak_reserved_bytes":39845888,"tf32":false},"fine_tuning_statistics":{"single_case":{"amplitude_rms_pa":2.8535734640782584,"case":{"amplitude_reference":"original-train-correction-rms","cells_per_metre":16,"counts_xyz":[13,15,16],"mode_zyx":[0,0,1]},"optimizer_examples":2000,"optimizer_network_passes_per_example":4,"trace":[{"measurements":[{"passes":1,"relative_equation_residual":0.019573677828909987},{"passes":2,"relative_equation_residual":0.0005464875548407806},{"passes":4,"relative_equation_residual":5.206602482768572e-07},{"passes":8,"relative_equation_residual":2.6233688957465098e-12}],"update":100},{"measurements":[{"passes":1,"relative_equation_residual":0.017173386538457652},{"passes":2,"relative_equation_residual":0.0006889619054975057},{"passes":4,"relative_equation_residual":4.1197283788467575e-07},{"passes":8,"relative_equation_residual":5.722136379930817e-12}],"update":200},{"measurements":[{"passes":1,"relative_equation_residual":0.010864462209603033},{"passes":2,"relative_equation_residual":0.00014406928838266842},{"passes":4,"relative_equation_residual":4.184064094401222e-08},{"passes":8,"relative_equation_residual":2.1345798444903086e-12}],"update":300},{"measurements":[{"passes":1,"relative_equation_residual":0.005698163222713211},{"passes":2,"relative_equation_residual":8.743576054241218e-05},{"passes":4,"relative_equation_residual":2.0126754995782084e-08},{"passes":8,"relative_equation_residual":2.7956316512024407e-12}],"update":400},{"measurements":[{"passes":1,"relative_equation_residual":0.011196078423521482},{"passes":2,"relative_equation_residual":0.00011705058141853108},{"passes":4,"relative_equation_residual":1.9478961953526902e-08},{"passes":8,"relative_equation_residual":4.795274314290802e-12}],"update":500},{"measurements":[{"passes":1,"relative_equation_residual":0.010438430508775435},{"passes":2,"relative_equation_residual":0.00014494555198744417},{"passes":4,"relative_equation_residual":4.112327995217511e-08},{"passes":8,"relative_equation_residual":1.3911839594850453e-11}],"update":600},{"measurements":[{"passes":1,"relative_equation_residual":0.0020896587505324035},{"passes":2,"relative_equation_residual":2.9028872499366457e-05},{"passes":4,"relative_equation_residual":9.467308183843558e-09},{"passes":8,"relative_equation_residual":5.357253083339778e-12}],"update":700},{"measurements":[{"passes":1,"relative_equation_residual":0.006097152324095726},{"passes":2,"relative_equation_residual":8.415346263222647e-05},{"passes":4,"relative_equation_residual":1.6859873541798513e-08},{"passes":8,"relative_equation_residual":1.5727076394273156e-11}],"update":800},{"measurements":[{"passes":1,"relative_equation_residual":0.012059350628597254},{"passes":2,"relative_equation_residual":0.0001729696055827183},{"passes":4,"relative_equation_residual":5.124593269921195e-08},{"passes":8,"relative_equation_residual":1.3252215179523757e-11}],"update":900},{"measurements":[{"passes":1,"relative_equation_residual":0.009669218730392436},{"passes":2,"relative_equation_residual":7.364091985357137e-05},{"passes":4,"relative_equation_residual":3.0036633728005355e-08},{"passes":8,"relative_equation_residual":5.077686223760789e-11}],"update":1000},{"measurements":[{"passes":1,"relative_equation_residual":0.0016741804351167139},{"passes":2,"relative_equation_residual":8.277846447939469e-06},{"passes":4,"relative_equation_residual":1.3889836437561826e-08},{"passes":8,"relative_equation_residual":1.8015386708295716e-11}],"update":1100},{"measurements":[{"passes":1,"relative_equation_residual":0.0009971461854507264},{"passes":2,"relative_equation_residual":6.3122527578792974e-06},{"passes":4,"relative_equation_residual":1.9037344037400694e-08},{"passes":8,"relative_equation_residual":3.857634420153071e-11}],"update":1200},{"measurements":[{"passes":1,"relative_equation_residual":0.0010487155319437841},{"passes":2,"relative_equation_residual":5.831579635757898e-06},{"passes":4,"relative_equation_residual":1.066123967680692e-08},{"passes":8,"relative_equation_residual":9.465704965388113e-12}],"update":1300},{"measurements":[{"passes":1,"relative_equation_residual":0.00027725319472918697},{"passes":2,"relative_equation_residual":3.836339475141101e-06},{"passes":4,"relative_equation_residual":1.1153152593224651e-08},{"passes":8,"relative_equation_residual":1.0649779251307395e-11}],"update":1400},{"measurements":[{"passes":1,"relative_equation_residual":0.0007424697476290204},{"passes":2,"relative_equation_residual":4.125320278486334e-06},{"passes":4,"relative_equation_residual":7.513027805473533e-09},{"passes":8,"relative_equation_residual":2.9571310657237132e-12}],"update":1500},{"measurements":[{"passes":1,"relative_equation_residual":0.0002152249069535803},{"passes":2,"relative_equation_residual":2.949674454812962e-06},{"passes":4,"relative_equation_residual":1.5037649784267835e-08},{"passes":8,"relative_equation_residual":2.4437276310375058e-11}],"update":1600},{"measurements":[{"passes":1,"relative_equation_residual":0.0001993209436597891},{"passes":2,"relative_equation_residual":2.668960821378176e-06},{"passes":4,"relative_equation_residual":1.0510261177715667e-08},{"passes":8,"relative_equation_residual":8.421645538553483e-12}],"update":1700},{"measurements":[{"passes":1,"relative_equation_residual":0.0001448122935337105},{"passes":2,"relative_equation_residual":3.1107196354024854e-06},{"passes":4,"relative_equation_residual":1.141522965686639e-08},{"passes":8,"relative_equation_residual":2.534911887483589e-12}],"update":1800},{"measurements":[{"passes":1,"relative_equation_residual":0.00010916362128980439},{"passes":2,"relative_equation_residual":2.7157229147360657e-06},{"passes":4,"relative_equation_residual":1.3263607257474492e-08},{"passes":8,"relative_equation_residual":1.9830757057376002e-11}],"update":1900},{"measurements":[{"passes":1,"relative_equation_residual":0.00013131579055926504},{"passes":2,"relative_equation_residual":2.5039011578979914e-06},{"passes":4,"relative_equation_residual":1.0085840581232199e-08},{"passes":8,"relative_equation_residual":1.3869408067836577e-11}],"update":2000}]},"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-2000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5"},"history.json":{"bytes":10992,"role":"training_diagnostics","sha256":"4d18791f5eb44c52c3fa172d7392ec4d1ced45a8f933e0c1f0a8034314999711"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"2448cb14f26f6f974c122e6121c573ac570d3d85ec405668971e7f23581936b7"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":477,"role":"result_summary","sha256":"7fe8a56c82e3a683568ce4c51d781b5fadb43e7deb06edf9446eb8667238dfb5"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"ae6f3aa3d1692bfc53f3d989278447cf84d47ce242bf180fdeaf86f27fa88aa5"}},"per_case_inference_s":[0.020144709007581696,0.0038472579908557236,0.004939517006278038,0.003862516983645037,0.0034015240089502186,0.0032019870122894645,0.004049848997965455,0.004306961986003444,0.008823276002658531,0.004872330988291651,0.00582998400204815,0.005303326004650444],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"2732c62c068d152be94762cdf5929904fde03bf5f072a624c9775b54bbadc4bc","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"a5ca1869357beeb44ea4656998668c42d7b6ce84632bd3b04c71688985020ff6","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"84c16294304467bc75e4edd342240b8d62e4ddfc8af069c8209240884ac31b6b","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"df984c9ca2d7c33a2c67a56e864d4e5fb1dc5ebb2bc14399ded2c726d1cc4ebb","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"9c8e2b02e4f16ab516cb271817a7e0ae9c703fba091b0a815fe8d46f943f3d98","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"5aad47b186c6808d8b0d85dd9bf1f9c5fd77a3065005a5a321e182c653d8c4c2","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"beb4a8af30906fbbbe0528ec3450a24fbb69e6f03c8215781a14fdfac59cd196","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"09cfe00b314a89f2a51a7dd5d54248f60b32afa67f751423547ecffb88f80901","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"4313b724def6ac191a753fce67ca0e93d8d4175aec6a2ea86d6698de2a67d554","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"04bda69328d0f5619eee6fe11690181410a5f5001b885b27e6ee708baa82eae2","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"40773a26cbb57935bdebb325b0f322d6bddef715ae16c2d5230c76fdee33512e","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"e6680da292efd20a65c6c26a9f6ff5bd3c0abf2c82f6c359185a0b8104570b16","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2343051264,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
