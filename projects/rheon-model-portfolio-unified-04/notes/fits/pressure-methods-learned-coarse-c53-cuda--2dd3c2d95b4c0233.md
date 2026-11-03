# Pressure methods · learned-coarse · C53 CUDA

Retrospective source review · fit-2dd3c2d95b4c0233

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** The retained 370,016-parameter fine CNN gains 63 sigmoid-bounded positive cosine-mode gains, excluding DC. Modes 0..3 per axis use F64 transform/reconstruction; the cap is max-axis-count squared/2. Initial gain is one at a 16-cubed grid. Before every fine pass, the coarse proposal is accepted only if its actual fine-equation residual norm does not increase. No analytic inverse eigenvalues supply the learned proposal.  F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.

**Purpose.** Test a learned global low-mode correction without an analytic coarse inverse.

**Experiment.** 1,600 CUDA Adam updates, batch one, initial fine rate 1e-05, schedule [{'update': 1201, 'learning_rate': 1e-06}, {'update': 1501, 'learning_rate': 1e-07}], model seed 2026101803, index seed 2026101804. 20 physical training trajectories/40 requests; grid-density counts {'8.0': 10, '16.0': 10}. Actual tier exposure {'32': 534, '64': 533, '16': 533} and sampled voxel count 157,852,608; 3,200 fine-network root forwards. Executed added coarse/head learning-rate multiplier 1. All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.

**Observed outcome.** Retained final fitting loss 12.909954679290081. Common 128 consumed requests: worst mean-free pressure relative L2 0.0064126848366434404, equation relative L2 0.0049418269210486504, median case-median complete GPU pressure call 5.8958297500000008 ms. The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.

**Lesson.** The acceptance gate controls the coarse proposal only; the following fine correction can still worsen the result. The captured optimizer really uses coarse/head rate multiplier 1; retain that executed policy rather than transferring later metadata discrepancies to this fit. Shape features describe voxel counts, not an explicit cell-density input. Learned coefficients, inference stages and scene coverage remain separate quantities.

**Reproduction.** Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v1/learned-coarse-study-01/train/receipt.json",
    "sha256": "2dd3c2d95b4c0233c19aab99267db672d07d52820013e8f6bb4ee1aa4c2f6caa"
  },
  "history": {
    "path": "runs/pressure-methods-v1/learned-coarse-study-01/train/history.json",
    "sha256": "06f08f1e108b67e7a3d76485d138ad5637de7c9d0648f0c390f8062ab6bb3159"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/config.json",
      "sha256": "7e1038a0eba8783e347bd89cc798d29d1dabe977e4dbdbcf94dfd8119b29cc4e"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/fine-tuning.json",
      "sha256": "c6bed937a3ec3c3a6eb6c1021e2caa3ef92999f014e69f6db3da1d2a532e2f7b"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/hybrid-method.json",
      "sha256": "f0642f8a44b51c3379a0620b4cb7fd5ca33a3c6129db02dea3601ffd7c756497"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/mixed_order.py",
      "sha256": "3df84663563940223c95d0367b10c08b8cd53b36961bbd31656f39399d612f3c"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/physical_residual.py",
      "sha256": "41837f919c363dc0e6465ed7812cce9e65726d55dc3c1236c15de7c446d9de79"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v1/learned-coarse-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v1/learned-coarse-project-01/source-witness.json",
    "sha256": "9d6817d3a8c128aa09be22ed00e653698552a02044e166e4c132590e0e11c1d9"
  },
  "training_population": {
    "path": "runs/pressure-methods-v1/learned-coarse-study-01/train/splits.json",
    "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"
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
      "bytes": 19487086,
      "path": "runs/pressure-methods-v1/domain-training-view-01/dataset-00.npz",
      "sha256": "c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971",
      "slot": "dataset"
    },
    {
      "bytes": 19488006,
      "path": "runs/pressure-methods-v1/domain-training-view-01/dataset-01.npz",
      "sha256": "f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5",
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
      "bytes": 1485962,
      "path": "runs/pressure-methods-v1/learned-coarse-initialization-01/weights.npz",
      "sha256": "971dbdbf04bf3a623f9bea36b5ed30233d149b29e40c297d3d37d6de9b4f1b06",
      "slot": "initial_weights"
    },
    {
      "bytes": 33281,
      "path": "runs/pressure-methods-v1/domain-training-view-01/manifest.json",
      "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-2dd3c2d95b4c0233",
    "kind": "training_attempt",
    "name": "Pressure methods · learned-coarse · C53 CUDA",
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
      "learned_coarse_revision": {
        "basis": "Orthonormal separable Neumann cosine modes 0..3, excluding DC",
        "bound": "max_axis_cells^2/2",
        "coefficients": "Positive learned sigmoid gains; no analytical inverse eigenvalues",
        "gate": "Accept learned coarse proposal only when its actual fine-equation residual norm does not increase",
        "initialization": "Gain 1 on 16^3; all original neural tensors preserved",
        "limitation": "Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation",
        "placement": "Learned coarse proposal before every fine neural pass",
        "trainable_parameters": 63
      },
      "restrictions": "uniform-density cubic sealed Neumann operator; no independently specified geometry masks",
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 262144
    },
    "receipt_sha256": "2dd3c2d95b4c0233c19aab99267db672d07d52820013e8f6bb4ee1aa4c2f6caa",
    "receipt_paths": [
      "runs/pressure-methods-v1/learned-coarse-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "4bbbc3e481154c2fb7c5ca997e68a731",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · learned-coarse · C53 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1600,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "7e1038a0eba8783e347bd89cc798d29d1dabe977e4dbdbcf94dfd8119b29cc4e",
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
          "bytes": 19487086,
          "sha256": "c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971"
        },
        "dataset_extra": {
          "bytes": 19488006,
          "sha256": "f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5"
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
          "bytes": 1485962,
          "sha256": "971dbdbf04bf3a623f9bea36b5ed30233d149b29e40c297d3d37d6de9b4f1b06"
        },
        "splits": {
          "bytes": 33281,
          "sha256": "39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "8b70793af829b8a0ee81380fc59aa19901da271e9de5da1c1954a575e87de9f1",
      "source_witness_sha256": "9d6817d3a8c128aa09be22ed00e653698552a02044e166e4c132590e0e11c1d9"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "ef45d2fd96233c4b72844826b3459467964cbc4b8d9f5645acbdaf27cdf43687",
        "bytes": 1485962,
        "stage": "initial"
      },
      {
        "name": "checkpoint-1600.npz",
        "sha256": "e08f8b990bdc200d7e6e78d0bf3037b7bd1b82f4bd53ba0f598a0b6ad9b969df",
        "bytes": 1485962,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "e08f8b990bdc200d7e6e78d0bf3037b7bd1b82f4bd53ba0f598a0b6ad9b969df",
        "bytes": 1485962,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2818039808,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/83"
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
      "receipt": "runs/pressure-methods-v1/learned-coarse-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v1/learned-coarse-study-01/train/receipt.json",
    "sha256": "2dd3c2d95b4c0233c19aab99267db672d07d52820013e8f6bb4ee1aa4c2f6caa",
    "bytes": 11379
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C85": {
        "name": "learned-coarse",
        "weights_sha256": "e08f8b990bdc200d7e6e78d0bf3037b7bd1b82f4bd53ba0f598a0b6ad9b969df",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00494182692104865,
          "pressure_mean_free_relative_l2": 0.00641268483664344,
          "pressure_absolute_rms": 0.4706207510772618,
          "pressure_absolute_max": 1.2337661014147476
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
            "trajectory_id": "baseline--water-validation-0044",
            "native_frame": 63,
            "counts_xyz": [
              14,
              13,
              9
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0034061751375557476,
              "pressure_mean_free_relative_l2": 0.00641268483664344,
              "pressure_absolute_rms": 0.4706207510772618,
              "pressure_absolute_max": 1.2337661014147476
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00494182692104865,
              "pressure_mean_free_relative_l2": 0.0062627419550734835,
              "pressure_absolute_rms": 0.13128803127158462,
              "pressure_absolute_max": 0.3152945833804033
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C85/results.json",
          "sha256": "16f5a6bf578a6122fb3e7af374a76e169ecd703f10a8418be033d32cae55f959"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"learned-coarse","early_weight":0.05,"initial_weights_sha256":"971dbdbf04bf3a623f9bea36b5ed30233d149b29e40c297d3d37d6de9b4f1b06","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"learned_coarse_revision":{"basis":"Orthonormal separable Neumann cosine modes 0..3, excluding DC","bound":"max_axis_cells^2/2","coefficients":"Positive learned sigmoid gains; no analytical inverse eigenvalues","gate":"Accept learned coarse proposal only when its actual fine-equation residual norm does not increase","initialization":"Gain 1 on 16^3; all original neural tensors preserved","limitation":"Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation","placement":"Learned coarse proposal before every fine neural pass","trainable_parameters":63},"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":0,"native_frame":15,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":1,"native_frame":63,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":1,"native_frame":63,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.0625,"counts_xyz":[32,32,32],"frame":0,"native_frame":15,"parent":17,"split":"train","trajectory_id":"large-train-32-cpm16"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":63,"parent":19,"split":"train","trajectory_id":"large-train-64-cpm16"}],"closure":{"experiment_sha256":"7e1038a0eba8783e347bd89cc798d29d1dabe977e4dbdbcf94dfd8119b29cc4e","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":19487086,"sha256":"c12c4d88d0a35979fd5fd1045a295239b2fac0bfae0bbf95410439f0331e9971"},"dataset_extra":{"bytes":19488006,"sha256":"f2dbef26b98a6d2d044626ebca0248efd251970342b1abece67a081aa5b15ef5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485962,"sha256":"971dbdbf04bf3a623f9bea36b5ed30233d149b29e40c297d3d37d6de9b4f1b06"},"splits":{"bytes":33281,"sha256":"39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"8b70793af829b8a0ee81380fc59aa19901da271e9de5da1c1954a575e87de9f1","source_witness_sha256":"9d6817d3a8c128aa09be22ed00e653698552a02044e166e4c132590e0e11c1d9"},"diagnostic_population":"train_only","elapsed_s":78.7767991849978,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":1.2784664985332483e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":351447552,"peak_reserved_bytes":434110464,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":1600,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485962,"role":"checkpoint","sha256":"ef45d2fd96233c4b72844826b3459467964cbc4b8d9f5645acbdaf27cdf43687"},"checkpoint-1600.npz":{"bytes":1485962,"role":"checkpoint","sha256":"e08f8b990bdc200d7e6e78d0bf3037b7bd1b82f4bd53ba0f598a0b6ad9b969df"},"history.json":{"bytes":10801,"role":"training_diagnostics","sha256":"06f08f1e108b67e7a3d76485d138ad5637de7c9d0648f0c390f8062ab6bb3159"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":2472878,"role":"predictions","sha256":"dbca7ed9a6ad440544d65ceb024004eaaa6a93b63ad6e75a99dc9b0635f45814"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":33281,"role":"training_splits","sha256":"39f803cf02f6dab67af9189298bb151a235fb9b51172a542208fefc97bf128d1"},"summary.json":{"bytes":520,"role":"result_summary","sha256":"816d765ed6c09a72d5eb76cf2da1d61f59d8b536249418739298655c0d635ac4"},"weights.npz":{"bytes":1485962,"role":"checkpoint","sha256":"e08f8b990bdc200d7e6e78d0bf3037b7bd1b82f4bd53ba0f598a0b6ad9b969df"}},"per_case_inference_s":[0.033243685989873484,0.007126649987185374,0.0074061919876839966,0.006332535995170474,0.0060002450190950185,0.007319212978472933,0.014439356978982687,0.008463992999168113,0.010063470981549472,0.009191368997562677,0.1335397560033016,0.8217685529962182],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"eb4c72ebcdfbaf04643c2a1810b0b80bc6dd87c3ffa7cb990140ca300fd91e93","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"40a6e199a9e941ebf8f7a69e7c4529c3be51f31515732954dcbb6213cd2852b1","shape":[6,10,16]},"case_002":{"bytes":6336,"dtype":"<f8","sha256":"1198f64212c978f804e887385e3e1f9baa91aba96d035b8b6e68f532b19b5535","shape":[11,6,12]},"case_003":{"bytes":5280,"dtype":"<f8","sha256":"83cb0e718e03c46800edb13246a27bfe9491c1bf1d6acd8051030cfc666c115a","shape":[6,10,11]},"case_004":{"bytes":2880,"dtype":"<f8","sha256":"8a26fc763ae090319bc2c4a1ceb96e1a671f96f01548cd1fcf9ee14cf8d1a8d8","shape":[9,5,8]},"case_005":{"bytes":6912,"dtype":"<f8","sha256":"504b020897e7bbbad520605a39868bf61e8060d422cd3545e4b5025561a1f95c","shape":[9,8,12]},"case_006":{"bytes":20480,"dtype":"<f8","sha256":"330b05f1f031b6fb273c44e6ce684171aacf133b782c617c04bc629461e88654","shape":[16,16,10]},"case_007":{"bytes":7488,"dtype":"<f8","sha256":"23b084b2790ca9aa415c102c33d8d0f452ca518ff5c95737af0608dc8d5f4a91","shape":[12,13,6]},"case_008":{"bytes":9000,"dtype":"<f8","sha256":"919d9d0907a812063fc40ae6b79fe4f933f6463141d27594d648d529f8741ce8","shape":[15,15,5]},"case_009":{"bytes":11760,"dtype":"<f8","sha256":"2b84ec5675cb731a5bf9ee6e748984128fce1fa42a624cafae4766d2ad68726a","shape":[7,15,14]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"423c5644d106a0a995ae1350a08aaa39e182e5baa5d2477e8934b7499d8a12d5","shape":[32,32,32]},"case_011":{"bytes":2097152,"dtype":"<f8","sha256":"bbef0585d5ab875844228fc93a80e3024f021a0928aeb7901882b5ab358c3ed3","shape":[64,64,64]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2818039808,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
