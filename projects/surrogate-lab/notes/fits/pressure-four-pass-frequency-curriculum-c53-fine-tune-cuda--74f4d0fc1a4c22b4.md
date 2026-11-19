# Pressure four-pass · frequency-curriculum · C53 fine-tune CUDA

Retrospective source review · fit-74f4d0fc1a4c22b4

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear, bias-free one-channel F32 multiscale pressure CNN warm-started from exact C53 tensors. F64 sealed cubic-grid equation assembly uses q=(rho/h)r and fixed train-only normalization. Shared neural corrections are gauged and accumulated in F64. The fitting objective unrolls four pressure corrections with final logarithmic equation loss and total earlier-stage weight 0.05; the captured default inference still executes two. These passes do not advance physical simulation time.

**Purpose.** Rebalance only the supplemental spatial-frequency bank toward axis fundamentals while retaining the five physical/manufactured/direction streams.

**Experiment.** 1,500 CUDA Adam updates; model seed 20261019, retained physical-index seed 2026101901, initial rate 1e-05 and schedule [{'update': 1000, 'learning_rate': 1e-06}, {'update': 1350, 'learning_rate': 1e-07}]. Sixteen physical training trajectories/32 frames plus two manufactured and two frozen captured-direction streams remain active. Only the supplemental bank changes: eight grid/density pairs each have 32 spatial patterns at four amplitudes; the 12/12/8 pattern allocation repeats each axis fundamental four times. Its 1,500 actual category draws are 579 fundamental, 542 other low and 379 high. This is fixed spatial-frequency weighting, not a time-varying temporal curriculum. The student records 30,000 root forwards and its frozen eight-correction teacher records 15,400 forwards plus 5,575 cache hits. Physical metadata remains dt=1/60 s and fluid density 1000 kg/m3; grid density means cells per metre, not fluid density. Calibration and consumed diagnostics add no independent optimizer scenes.

**Observed outcome.** Retained final fitting loss 10.203955527938316. Common 128 consumed pressure requests use default TWO-correction inference: worst mean-free pressure relative L2 0.0057310314613825235, equation relative L2 0.0041662695773564927, median case-median complete GPU pressure call 4.3604935000000005 ms. Host input/output, transfers and synchronization are included; setup/scoring are excluded. Fitting-bank 1/2/4/8-correction traces are retained consumed-case equation diagnostics, not independent pressure or full-step benchmarks. No new model execution, fresh four-correction common benchmark or rollout result is added.

**Lesson.** A physical data manifest does not establish physical optimizer exposure. The frequency curriculum is a static supplemental spatial distribution with repeated modes; it does not train a range of physical time intervals. Report calibration data, actual optimizer population, update aggregation, fitting recurrence and measured inference recurrence separately.

**Reproduction.** Retain exact typed native source/config/runtime/input receipts, C53 warm tensors, fine-tuning contract, normalization, final checkpoint and the actual optimizer owner. Recreate manufactured fundamental fields from explicit grid/mode/density definitions and original train-calibrated amplitude; reproduce private balanced cycles and gradient aggregation. For the curriculum preserve its repeated patterns and supplemental category order. Historical publication/restore uses CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, intended four-correction common evaluation, new-Rheon regeneration and long own-history drift remain unqualified. Preserve original receipts and ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/receipt.json",
    "sha256": "74f4d0fc1a4c22b4c1dc4ba6634c0601bed17267808eae1870a4a6d1971fdfa1"
  },
  "history": {
    "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/history.json",
    "sha256": "e9366a86eb47973230032032f739430cb92f45382ae7f419181c968be9cd72ec"
  },
  "sources": [
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/analytical.py",
      "sha256": "4eed23c11dccde2b015de8d35acfa7dae8fd452674c4533673f1261d59a13863"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/config.json",
      "sha256": "7f9b94a2795f88b702001c83a6b69b1abdead960ffa3566edc114eae923e01b1"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/fine-tuning.json",
      "sha256": "b1b9173b48f2d46fea60cf0503f002ae57643fffb928f777c439a7e7669253ee"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/fine_tuning.py",
      "sha256": "376c4f72945704f533c5bd804a40c71560d089fc7bb1f8f822a102654a6ceaaa"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/mode-curriculum.json",
      "sha256": "1cd44b4df76c501ab2d4f3e3a9595c29ff1ec836d72633c3f66bf00be992c502"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/model.py",
      "sha256": "a2ebefe6a8ce12c7e6e1e12fa82b835029f364b50e91284f33b60bfe31b0eaee"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/original_analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/original_fine_tuning.py",
      "sha256": "6e8d5e11744c0622f700bc645226d5dd96dbf831dd9bea754328c97dc70ecc12"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/train.py",
      "sha256": "18b5ae6232e7a00760288d37c015905363fdea4b97e0d1815bc12033e0d69e12"
    },
    {
      "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-project-01/source-witness.json",
    "sha256": "890bdd185f2a188da018055305cb9933eadae9ca47275f068dfd78952dfd78ae"
  },
  "training_population": {
    "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/splits.json",
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
    "item_id": "fit-74f4d0fc1a4c22b4",
    "kind": "training_attempt",
    "name": "Pressure four-pass · frequency-curriculum · C53 fine-tune CUDA",
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
    "receipt_sha256": "74f4d0fc1a4c22b4c1dc4ba6634c0601bed17267808eae1870a4a6d1971fdfa1",
    "receipt_paths": [
      "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "ef4298f2e657491080732e7aaf089895",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure four-pass · frequency-curriculum · C53 fine-tune CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1500,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "7f9b94a2795f88b702001c83a6b69b1abdead960ffa3566edc114eae923e01b1",
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
      "sample_order_sha256": "6fb1c513e2db57f19de256b347099538cf1b84c52de210119f56660c15382c0a",
      "source_witness_sha256": "890bdd185f2a188da018055305cb9933eadae9ca47275f068dfd78952dfd78ae"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1500.npz",
        "sha256": "0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2475646976,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/62"
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
      "receipt": "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-single-mode-fit-v1/frequency-curriculum-study-01/train/receipt.json",
    "sha256": "74f4d0fc1a4c22b4c1dc4ba6634c0601bed17267808eae1870a4a6d1971fdfa1",
    "bytes": 10680
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C63": {
        "name": "frequency-curriculum",
        "weights_sha256": "0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004166269577356493,
          "pressure_mean_free_relative_l2": 0.0057310314613825235,
          "pressure_absolute_rms": 0.22499266670772672,
          "pressure_absolute_max": 0.4848518109184988
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
            "native_frame": 63,
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
              "common_equation_relative_l2": 0.002541047912270369,
              "pressure_mean_free_relative_l2": 0.0057310314613825235,
              "pressure_absolute_rms": 0.22499266670772672,
              "pressure_absolute_max": 0.4848518109184988
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004166269577356493,
              "pressure_mean_free_relative_l2": 0.0055251262698124545,
              "pressure_absolute_rms": 0.1254753525982979,
              "pressure_absolute_max": 0.2693782292913697
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C63/results.json",
          "sha256": "b19350db37cce3b2476cc11663ff5c7872507266836d814ffa7a7c428e657266"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"frequency-curriculum","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","low_mode_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"weight":0.0},"relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.1,"training_passes":4},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[13,15,16]},{"cells_per_metre":16,"counts_xyz":[9,11,11]},{"cells_per_metre":8,"counts_xyz":[15,9,7]},{"cells_per_metre":16,"counts_xyz":[11,5,15]},{"cells_per_metre":8,"counts_xyz":[7,16,5]},{"cells_per_metre":16,"counts_xyz":[11,12,7]},{"cells_per_metre":8,"counts_xyz":[5,13,9]},{"cells_per_metre":16,"counts_xyz":[16,7,11]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101907,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"7f9b94a2795f88b702001c83a6b69b1abdead960ffa3566edc114eae923e01b1","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"6fb1c513e2db57f19de256b347099538cf1b84c52de210119f56660c15382c0a","source_witness_sha256":"890bdd185f2a188da018055305cb9933eadae9ca47275f068dfd78952dfd78ae"},"diagnostic_population":"train_only","elapsed_s":343.4005605010025,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":4.642634934270973e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":897565184,"peak_reserved_bytes":973078528,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":5575,"teacher_cache_peak_bytes":131831024,"teacher_forwards":15400},"heldout_evaluations":0,"mode":"train","optimizer_updates":1500,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-1500.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2"},"history.json":{"bytes":220863,"role":"training_diagnostics","sha256":"e9366a86eb47973230032032f739430cb92f45382ae7f419181c968be9cd72ec"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"31d7ff3ddb48f88461d3a83b8a01a5bd29a91d5517fa2f2a6948afef6c4bd707"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":477,"role":"result_summary","sha256":"1a9cead689af9aa9541f679992045fde4f17af8ec54c0974edfd5b461dc911a9"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"0e63e88de534e09a6b30e61e31eec6470791f606dd00f652f8c31023735bf2f2"}},"per_case_inference_s":[0.025021823006682098,0.005198059981921688,0.006371447001583874,0.005012030014768243,0.004460536001715809,0.004180523013928905,0.005299250013194978,0.005492301977938041,0.011247514019487426,0.006194015993969515,0.0074645180138759315,0.006739499018294737],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"9d21180cb77bedbc1a41efebfb53f687a0265f07716b9fb2ba5184be23c4ebe3","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"28d0a8d92f93fcd6b89c8bbfceaee86ba8f143238b0dea02a143e182698f323a","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"526a053503b5e0507688a19b1d2a77f238d4e3cc1cd9c10454a8cd6b93560c09","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"816eef4c44d356bfdd5c64828e60bc1c2f097647155f398313845f26633ed6f5","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"2b8bacacf1735de0048b062ef0c6a276044a20782dd2c5631be2393b4fc06c26","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"9ed9cc62fdb09428c7fae469a761cab022210e8a0d8e907adfe5775b81fc86a7","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"349832bc13cfd9b4ddfb064c6541e8abbe87a00d8487375a963f4a7267febcf5","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"386abc60743a7927a108c982357ddb7532fc48e0922e1025fe372d0ec48ea93d","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"9db645431d1d8957238be016dd3a14bda916496bd34065ab63a59eb6bc3b418e","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"56937c7d37eaa74161bdb71677afb64c9dc317c0d8ce51fac3ff202b9c81499a","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"73295492aa8a8537dc51a5bd6aded6c649f7b2595fe5725c0e65e23ecdc8b094","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"d8a351ac4bfef09ab499ed9cf93ea7f39c7476dcb7d6cb58c12ea55418c89a1a","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2475646976,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
