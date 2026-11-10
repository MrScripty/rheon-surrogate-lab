# water-current003 CUDA

Retrospective source review · fit-c629219cd2e83d6d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 370,016-parameter linear one-channel F32 multiscale pressure CNN with 19 bias-free tensors. F64 residual is scaled by rho/h and train-only RMS; two shared-weight internal corrections recompute the F64 equation residual and accumulate cell-zero-gauged pressure. Training combines final physical loss, nonfinal equation contraction, first/final low-spatial-mode feedback and fixed captured residual-direction streams. The extra current-policy bank affects fitting only; it adds no deployed pressure pass and no physical simulation output.

**Purpose.** Test late captured directions from the frozen current-policy bank at weight 0.03, retaining the matched C53 architecture, initialization and physical/manufactured fitting streams.

**Experiment.** 5,000 CUDA Adam updates, batch one, initial rate 0.001 with drops to 0.0001 at 3000 and 0.00001 at 4500. Model seed 20261015; physical sample seed 2026102501. Sixteen physical fitting trajectories/32 frames cover small cubic-cell grids at 8/16 cells per metre and dt=1/60 s. Primary and supplemental manufactured banks plus the earlier captured-direction stream remain active. The additional current-policy stream has 144 frozen cases, 15 balanced groups, 5,000 proposals and weight 0.03, sampled at its late correction stage. All five streams together record 50,000 neural root forwards. The bank is captured before fitting and is not regenerated online from each updated network.

**Observed outcome.** Retained final training loss 0.0012204573077904081. On the common 128 consumed pressure requests, worst mean-free pressure relative L2 0.0067340121660916007, equation relative L2 0.0045794348229344518; median of case-median complete GPU pressure calls 4.5737692499999998 ms. Timings include host input/output, validation/packing, transfers, two corrections and synchronization, excluding setup/scoring. These measurements retain the already-recorded host-metadata-overlap flag; its timing effect was not measured. Native history validation is train-only and the wrapper adds one consumed validation probe. No new model execution or drift result is added.

**Lesson.** Early/late refer to internal correction directions of the captured source policy, not physical simulation timesteps. A lower auxiliary weight changes optimization, not deployed work. Current-policy rehearsal is a fixed offline bank, not ongoing on-policy trajectory collection. Compare both pressure and equation tails on the same cases and keep inherited fitting labels, consumed probes and generalization claims distinct.

**Reproduction.** Retain the exact native evidence recipe, all witnessed sources/config/runtime, fixed physical/primary/supplemental/direction/current input blobs, train-only normalization, initial tensors and all private sample orders. Preserve the selected early/late payload and frozen generating checkpoint. The historical native wrapper/restore performs CPU neural forwards; qualify a GPU-only publication/replay route before rerunning. Clean retraining, new-Rheon regeneration and own-history drift remain unqualified; original receipts/configurations and historical Trial ancestry remain exact.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-current-feedback-weight-v1/current003-study-01/train/receipt.json",
    "sha256": "c629219cd2e83d6d2d09fb50c4874ba26dfe8bbf97895e7b5e2f4294e10887f8"
  },
  "history": {
    "path": "runs/pressure-current-feedback-weight-v1/current003-study-01/train/history.json",
    "sha256": "5cb7f9a45a5ebdfb26343d41a865bed5fedba929deea3d8ec26a5faf171075e3"
  },
  "sources": [
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/config.json",
      "sha256": "83b37db907360f1564a5285d063828475b3e7deed62827b1257ef209f38128c1"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/custody-validator.py",
      "sha256": "247c573d6a9bfdc4e189b3a940650c91102cb40273751225f0fef6dd79999e6f"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/train.py",
      "sha256": "152d714442cca48a62afbcd0f2e0c2d43000d2a17710400b50ec1788a5bb6fa7"
    },
    {
      "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-current-feedback-weight-v1/current003-project-01/source-witness.json",
    "sha256": "e97e86921cd76a045e37560872298b50353c71b7d2c7f05c2f04289b81fe202e"
  },
  "training_population": {
    "path": "runs/pressure-current-feedback-weight-v1/current003-study-01/train/splits.json",
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
    "item_id": "fit-c629219cd2e83d6d",
    "kind": "training_attempt",
    "name": "water-current003 CUDA",
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
    "receipt_sha256": "c629219cd2e83d6d2d09fb50c4874ba26dfe8bbf97895e7b5e2f4294e10887f8",
    "receipt_paths": [
      "runs/pressure-current-feedback-weight-v1/current003-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "598ef9774bd547fba9eddb72f0b6114d",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "water-current003 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "83b37db907360f1564a5285d063828475b3e7deed62827b1257ef209f38128c1",
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
        "splits": {
          "bytes": 256361,
          "sha256": "bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423",
      "source_witness_sha256": "e97e86921cd76a045e37560872298b50353c71b7d2c7f05c2f04289b81fe202e"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "2d0bc355844b5319c04bd28a9a1b49e73fbf9d3e50c110105267d34b806399a9",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "2d0bc355844b5319c04bd28a9a1b49e73fbf9d3e50c110105267d34b806399a9",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2627145728,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/53"
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
      "receipt": "runs/pressure-current-feedback-weight-v1/current003-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-current-feedback-weight-v1/current003-study-01/train/receipt.json",
    "sha256": "c629219cd2e83d6d2d09fb50c4874ba26dfe8bbf97895e7b5e2f4294e10887f8",
    "bytes": 9971
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C54": {
        "name": "current003",
        "weights_sha256": "2d0bc355844b5319c04bd28a9a1b49e73fbf9d3e50c110105267d34b806399a9",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.004579434822934452,
          "pressure_mean_free_relative_l2": 0.006734012166091601,
          "pressure_absolute_rms": 0.38953066661539537,
          "pressure_absolute_max": 0.8659294639792476
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
              "common_equation_relative_l2": 0.0033520453027050287,
              "pressure_mean_free_relative_l2": 0.006199087737883127,
              "pressure_absolute_rms": 0.38953066661539537,
              "pressure_absolute_max": 0.8659294639792476
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.004579434822934452,
              "pressure_mean_free_relative_l2": 0.006734012166091601,
              "pressure_absolute_rms": 0.09836225940704961,
              "pressure_absolute_max": 0.3187728118049221
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C54/results.json",
          "sha256": "34f377176cc5a0e555eeb5e0dbb855e2d6e2595a627ad664baa5a31ae522a40b"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.03},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"83b37db907360f1564a5285d063828475b3e7deed62827b1257ef209f38128c1","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"e97e86921cd76a045e37560872298b50353c71b7d2c7f05c2f04289b81fe202e"},"diagnostic_population":"train_only","elapsed_s":547.5013804219925,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.1938204124817275e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":870009344,"peak_reserved_bytes":920649728,"tf32":false},"heldout_evaluations":0,"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"21924973935cbad8da3ead87240f44c33b4235fb978b00805390e4738abeb713"},"checkpoint-5000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2d0bc355844b5319c04bd28a9a1b49e73fbf9d3e50c110105267d34b806399a9"},"history.json":{"bytes":216237,"role":"training_diagnostics","sha256":"5cb7f9a45a5ebdfb26343d41a865bed5fedba929deea3d8ec26a5faf171075e3"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"a0b97215de24d6d239abedcddc80abb5e8eba5be386b78900a8bea1cb0eff6e1"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":470,"role":"result_summary","sha256":"bc368fed16714aa8bb01cffd853eb35c63f5e2b8ee8deb097971f5b266242443"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"2d0bc355844b5319c04bd28a9a1b49e73fbf9d3e50c110105267d34b806399a9"}},"per_case_inference_s":[0.024089749000268057,0.0047941289958544075,0.005956668988801539,0.004644706990802661,0.004099648998817429,0.0038901270017959177,0.00493139200261794,0.005170052012545057,0.01063241199881304,0.0058297659998061135,0.006974999996600673,0.006221465999260545],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"72bb18aab62408a0e0242f9d113262e76d86e3fb79f10633b50a5060a1d41ed0","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"5132db29281db1b3d0785569f6286bcbf7ad11c15b3ca22f1eed99ee64999576","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"56f37f680b40654d39b53fc8758d595fcf1f5e0da8936c16b98e6fb77de7f5b7","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"60fc566f8bf05b22d8dc77b7b4cd2c324f2573faf7be789d8b52e2f15f5ad865","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"e047833b9585fcdec443dfd6048c61b6d63c84e635033250decf26dcef9bbd69","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"b7f35815018d0ba103fbbf92f27bfaebdcd4dda6ebcd42a0a7b8edaf8fa8a68d","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"bb26e7d5bbabef6bee91bbdf7acb8eab9e795d6839343195dca3fb0c2afc69f6","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"96ad400580f56e535ad7ad76eac3eaad6141d159065d14e2379a230844ae15af","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"1cad8af74e62f8f252f10808e53d4e57c7ead30a0704c2475cd43f3728769870","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"176a2b08a45bd496d5bd9ffa87e7f0cfef90d8c1f67d7423f41964334def1c01","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"0b9833f70a5f34077cb49b57367530876b5b502d4c26745f8aca004d44fb5274","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"83721e2f4f4da6074f5e049921d8c9ca360ac7474151edc5d48bb467ac0ba588","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2627145728,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
