# Pressure methods · v2-num6 · C53 · seed 2026101825 CUDA

Retrospective source review · fit-1a7a25ae8421aa4e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A hybrid pressure request: an F64 Galerkin solve on at most 6 aggregate centres per axis precedes each learned fine correction. Clamped trilinear interpolation defines P; transpose restriction and the fixed-gauge Cholesky solve use P-transpose L P. The fine CNN retains 370,016 learned coefficients. Restriction, numerical solve, prolongation, residual assembly and neural correction all belong to the configured pressure call. F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.

**Purpose.** Test a 6-node-per-axis numerical coarse stage during both fitting and deployment, with the retained C53 fine CNN. This is a matched seed-repeat arm, with model seed 2026101825 and index seed 2026101825.

**Experiment.** 2,400 CUDA Adam updates, batch one, initial fine rate 1e-05, schedule [{'update': 1801, 'learning_rate': 1e-06}, {'update': 2251, 'learning_rate': 1e-07}], model seed 2026101825, index seed 2026101825. 26 physical training trajectories/52 requests; grid-density counts {'8.0': 13, '16.0': 13}. Actual tier exposure {'16': 1600, '32': 400, '64': 400} and sampled voxel count 51,744,487; 4,800 fine-network root forwards. Executed added coarse/head learning-rate multiplier 1. All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.

**Observed outcome.** Retained final fitting loss 14.061761778820935. Common 128 consumed requests: worst mean-free pressure relative L2 0.0001677926651906902, equation relative L2 0.00015173336885974261, median case-median complete GPU pressure call 6.3511170000000003 ms. The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.

**Lesson.** A parameter count describes only the learned part of this hybrid. Numerical work and geometry preparation require separate accounting. C94/C95 isolate four/six-node stages under matched retained exposure; comparisons to earlier C83 also change data and training budget. A repeated seed is evidence about sensitivity within this protocol, not an independent physical holdout.

**Reproduction.** Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-num6-s25-study-01/train/receipt.json",
    "sha256": "1a7a25ae8421aa4e7b4e958d1f8b8e9dd74fa04fd759fbd6dcde71604eaf0d43"
  },
  "history": {
    "path": "runs/pressure-methods-v2/v2-num6-s25-study-01/train/history.json",
    "sha256": "4edb3a0b2a9531a3553508fc18f0ab9a3c20c8dcf7c4dcdfa491b8c8f8bc9362"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/coarse.py",
      "sha256": "a8e2af6c72f2b89ef2d07e382423ae652ba4b8a7d82fb53ddb26481ba4b681ea"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/config.json",
      "sha256": "2f3974cc8c4ff94bc3efac9fd6dcacaf4bbfb3103a277a3cc926da576a1af642"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/data.py",
      "sha256": "cf01d003cdf525e78cb3ec2b5e004186fa28c65ecadd63ffea969d85009674f9"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/fine-tuning.json",
      "sha256": "222303e0bb149185a2d648c7507a44ff7f1f0b39bc17b2c9e5a2695817ab4f9f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/hybrid-method.json",
      "sha256": "ab70994aa4938351d480f0d11e6d3734cf16b9e4f75ccbcfc30d9c56abe91f89"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/methods_view.py",
      "sha256": "3418389fb087ae3036549b49338bc7377291d497fa8636c0978f2f382131294f"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/mixed_order.py",
      "sha256": "fdbbfa8d48001aa8fb6e8d530c0bceca3de6d99caa44b945af77a81ba5dd6de7"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/model.py",
      "sha256": "eb889562bf4bb8b41d1183582515aaa45a8fb439190f8d9ab617d36a05b79072"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/physical_residual.py",
      "sha256": "8327721a87e6bbfc1cf0e5b92d253f4c6044a82d11adcaa19b0df0b12ff3e139"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/policy.py",
      "sha256": "83d2f46900b6a8018a02006520c33c6b2b8e2ed5e46dd2fc92337a570b6e36b4"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/portable_policy.py",
      "sha256": "b890bf302e911c024ce54082c06043700f940d2fd812ffdcdc0fbb3d785a2d39"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/train.py",
      "sha256": "5e02e78d3eca1cb7f9eeb04ed0372dc71c8f11da5f26b7cadf629510a929b81b"
    },
    {
      "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v2/v2-num6-s25-project-01/source-witness.json",
    "sha256": "39dede3143563e2c35cec75d5cd961de80f7f20591b55fb330da7a14fc760bfb"
  },
  "training_population": {
    "path": "runs/pressure-methods-v2/v2-num6-s25-study-01/train/splits.json",
    "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"
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
      "bytes": 24736178,
      "path": "runs/pressure-methods-v2/shape-training-view-01/dataset-00.npz",
      "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7",
      "slot": "dataset"
    },
    {
      "bytes": 24737374,
      "path": "runs/pressure-methods-v2/shape-training-view-01/dataset-01.npz",
      "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb",
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
      "bytes": 42433,
      "path": "runs/pressure-methods-v2/shape-training-view-01/manifest.json",
      "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-1a7a25ae8421aa4e",
    "kind": "training_attempt",
    "name": "Pressure methods · v2-num6 · C53 · seed 2026101825 CUDA",
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
      "max_cells": 262144
    },
    "receipt_sha256": "1a7a25ae8421aa4e7b4e958d1f8b8e9dd74fa04fd759fbd6dcde71604eaf0d43",
    "receipt_paths": [
      "runs/pressure-methods-v2/v2-num6-s25-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b459756235ac4497a4e27c50f2b689c1",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods · v2-num6 · C53 · seed 2026101825 CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "2f3974cc8c4ff94bc3efac9fd6dcacaf4bbfb3103a277a3cc926da576a1af642",
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
          "bytes": 24736178,
          "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"
        },
        "dataset_extra": {
          "bytes": 24737374,
          "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"
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
          "bytes": 42433,
          "sha256": "31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "35dead24d641ef4a94c2ef682ab14cccf1e10e0afc82d6f366b8755f642cc320",
      "source_witness_sha256": "39dede3143563e2c35cec75d5cd961de80f7f20591b55fb330da7a14fc760bfb"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954",
        "bytes": 1485452,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222",
        "bytes": 1485452,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222",
        "bytes": 1485452,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2847657984,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/96"
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
      "receipt": "runs/pressure-methods-v2/v2-num6-s25-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v2/v2-num6-s25-study-01/train/receipt.json",
    "sha256": "1a7a25ae8421aa4e7b4e958d1f8b8e9dd74fa04fd759fbd6dcde71604eaf0d43",
    "bytes": 11255
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C98": {
        "name": "v2-num6-s25",
        "weights_sha256": "17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.00015173336885974261,
          "pressure_mean_free_relative_l2": 0.0001677926651906902,
          "pressure_absolute_rms": 0.0025607428487307566,
          "pressure_absolute_max": 0.011750723734133224
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0001",
            "native_frame": 63,
            "counts_xyz": [
              16,
              10,
              6
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00010989597561758482,
              "pressure_mean_free_relative_l2": 0.00012014270311300233,
              "pressure_absolute_rms": 0.0025607428487307566,
              "pressure_absolute_max": 0.011750723734133224
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00015173336885974261,
              "pressure_mean_free_relative_l2": 0.0001677926651906902,
              "pressure_absolute_rms": 0.0008151137068679703,
              "pressure_absolute_max": 0.0024113704044946044
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C98/results.json",
          "sha256": "8f18230c325061d3365e425311d680855ca0176dd531afe6b419efca09d9e790"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"v2-num6","early_weight":0.05,"initial_weights_sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":262144,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Same expanded-domain tier sampler and fixed optimizer-update budget","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"numerical_training_method":{"coarse_nodes_per_axis":6,"cost":"Count geometry setup, restriction/transform, solve/selected inverse, prolongation, residual and neural inference","kind":"numerical-coarse","numerical_operator":"Same sealed fine equation; no reference labels","operator":"Galerkin Ac=P^T L P, float64 Cholesky with fixed coarse gauge","placement":"Numerical stage before each neural pass","transfer":"Clamped trilinear interpolation at coarse aggregate centers; conservative transpose restriction"},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":1,"native_frame":63,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":0,"native_frame":15,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":0,"native_frame":15,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.0625,"counts_xyz":[10,5,7],"frame":1,"native_frame":63,"parent":11,"split":"train","trajectory_id":"baseline--water-train-0011"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.125,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":20,"split":"train","trajectory_id":"v2-train-00-cpm8"},{"cell_size_m":0.125,"counts_xyz":[61,39,27],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"v2-train-01-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[48,13,9],"frame":1,"native_frame":63,"parent":25,"split":"train","trajectory_id":"v2-train-02-cpm16"}],"closure":{"experiment_sha256":"2f3974cc8c4ff94bc3efac9fd6dcacaf4bbfb3103a277a3cc926da576a1af642","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":24736178,"sha256":"8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"},"dataset_extra":{"bytes":24737374,"sha256":"7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"initial_weights":{"bytes":1485452,"sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"splits":{"bytes":42433,"sha256":"31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"35dead24d641ef4a94c2ef682ab14cccf1e10e0afc82d6f366b8755f642cc320","source_witness_sha256":"39dede3143563e2c35cec75d5cd961de80f7f20591b55fb330da7a14fc760bfb"},"diagnostic_population":"train_only","elapsed_s":91.846207124996,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":5.519824242128379e-07,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":369428480,"peak_reserved_bytes":455081984,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1485452,"role":"checkpoint","sha256":"3ac3c9a3a64c436f653947085aa34968b7699c2e0fd6264390ac257e460de954"},"checkpoint-2400.npz":{"bytes":1485452,"role":"checkpoint","sha256":"17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222"},"history.json":{"bytes":12266,"role":"training_diagnostics","sha256":"4edb3a0b2a9531a3553508fc18f0ab9a3c20c8dcf7c4dcdfa491b8c8f8bc9362"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":2837102,"role":"predictions","sha256":"d394c864b2cd428a2eed57ae120ba72669ff4c1169a08147b4117f26697d85d8"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":42433,"role":"training_splits","sha256":"31e0d1c2a777b773eb64826681ad1256013ad2fdfa4435a8185f6ef6aed09bf6"},"summary.json":{"bytes":586,"role":"result_summary","sha256":"acac87de2385056f265ba3bd237764b386f9ab49ae2e2cdcfb8c7b97ce20cc4e"},"weights.npz":{"bytes":1485452,"role":"checkpoint","sha256":"17bfb2d03e2e411a8d9a6eae57166315aca7dcb88cbc146c2c49639b82f53222"}},"per_case_inference_s":[0.030929716012906283,0.009439256973564625,0.007947994978167117,0.006802877993322909,0.008517117996234447,0.006207836995599791,0.01050364799448289,0.00955358098144643,0.7712914790026844,0.07949668800574727,0.32988790899980813,0.024287423002533615],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"bdb80269ffe0541a7b9157602355f665d6952887f80d79a46f7c9ea71eccecab","shape":[16,16,16]},"case_001":{"bytes":8736,"dtype":"<f8","sha256":"ba4c99cbd64492b05cf92e504ac6756c9fee0d415b6e209e1153240a3d11f741","shape":[13,7,12]},"case_002":{"bytes":7200,"dtype":"<f8","sha256":"23ee0c533454b18bd1ba7c31e3da43f6c94ee61753c013f0916995886df475e8","shape":[6,15,10]},"case_003":{"bytes":2880,"dtype":"<f8","sha256":"f2c677d82d994162c68f7db2a1f70aa50d826c2e9231e7f76561a05af52b8ae4","shape":[9,5,8]},"case_004":{"bytes":6048,"dtype":"<f8","sha256":"8eb81fe439681c4099ce47406c462ce5ebef7d98450325f52428af6d505242a7","shape":[9,14,6]},"case_005":{"bytes":2800,"dtype":"<f8","sha256":"8a927b5bf0eecdfbb5f204ff8a3921999f94fe4ad4dca4582b6d18ff9553213e","shape":[7,5,10]},"case_006":{"bytes":9000,"dtype":"<f8","sha256":"d014b2658a9efe7f6959af93b3fd26fbfed363151cd968c87dfb5936f97aa1bf","shape":[15,15,5]},"case_007":{"bytes":11760,"dtype":"<f8","sha256":"22a19f4f9f8be4898ddf3e6305eb94ff074d314f15c8668d25be064fc059fa49","shape":[7,15,14]},"case_008":{"bytes":2097152,"dtype":"<f8","sha256":"15a27d58a0c97fb317d2718112d2fa4532afc148572bc3747fb33a023f97ba6c","shape":[64,64,64]},"case_009":{"bytes":96968,"dtype":"<f8","sha256":"523e2fc0686b376cc32644fef03f774fd9729b4e2bd3a3c2521e0dc9d19fae2c","shape":[17,23,31]},"case_010":{"bytes":513864,"dtype":"<f8","sha256":"c143599b5d1c748428f52ce659c6c9be14b80ed147ca16b7d4842974686c1e4c","shape":[27,39,61]},"case_011":{"bytes":44928,"dtype":"<f8","sha256":"7bff6237e5c3ca9369dc3d6910cd60f6462a387b9b73fab7450f9f4c069d2c86","shape":[9,13,48]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2847657984,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
