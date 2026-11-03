# Pressure methods v3 · v3-replay CUDA

Retrospective source review · fit-a5e80227146675b1

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 371,214-parameter F32 pressure proposal has 370,016 fine CNN coefficients at widths 8/16/24/32, bottleneck 48, 63 learned positive low-mode gains and a 1,135-parameter 3→16 tanh→63 grid-shape head. The fine path is linear and bias-free, with pooling, nearest upsampling, skips and global-mean broadcast. Its input is q=(rho/h)r normalized by fixed original C53 scales; dt is validated scalar metadata, not an independent neural channel. At each correction the learned cosine coarse proposal is gated by its actual F64 fine-equation residual, followed by a neural correction accumulated in F64. Deployment uses two ordinary internal pressure corrections. These are iterations within one request and do not advance velocity or tracer.

**Purpose.** Continue the C93 shape-conditioned full-width pressure proposal with 1,600 preserved small-grid draws and 800 larger-grid draws, testing physical replay rather than inherited manufactured feedback.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one. Fine rate 1e-05; executed optimizer multipliers [1.0, 1000.0]. Every update uses one recorded physical request: 1,600 preserved small-grid samples plus 800 round-robin size-tier samples. There are 26 train-only parents and 52 selected frames, fixed dt=1/60 s and fluid density 1,000 kg/m³. Actual parent grid resolutions: {'8.0': 13, '16.0': 13}; sample tier exposures: {'16': 1600, '32': 400, '64': 400}. Actual fine-network work is 4,800 root forwards / 117,130,762 input voxels. Inherited manufactured, direction and current-policy banks are disabled despite retained config labels. Final/intermediate physical equation feedback remains, with early log-residual weight 0.05 and target 1e-6. The old eight-pass teacher has weight zero. 

**Observed outcome.** Final retained weighted training loss 14.634657926909274. On the common 128 original consumed requests, worst mean-free pressure relative L2 is 0.003081923685320427, equation relative L2 0.0009231457724317273. Retained complete GPU pressure-call median of case medians is 6.0891045000000004 ms, with CPU input/output, validation/packing, transfers, all solver work and synchronization included. Setup and scoring are separate. The native training diagnostic named validation is train-only, and the wrapper probe is already consumed. The common test covers small 8/16-cells-per-metre grids, not the larger training domains. No fresh forward, latency, trajectory or held-out result is added by this review.

**Lesson.** Replay uses physical requests and equal update count, not equal voxel work. Retained labels for analytical/direction/current feedback are inherited metadata: the actual trainer disables those sample banks.

**Reproduction.** Use the exact captured native evidence recipe, all witnessed sources/configuration, runtime, fixed normalization, typed input packet, initial parameter inventory and final-update checkpoint. Preserve the private small/large RNG streams and physical exposure budget, disabled auxiliary banks and executed optimizer groups. Retain channel-slice mappings or exact warm-parent bytes, and distinguish teacher work from student work. Original CPU publication/restore performs neural inference and must be replaced by a qualified GPU-only route before replay. Native recipe rebuilding, clean retraining, new-Rheon regeneration and own-history drift remain unqualified; historical Trial ancestry is unchanged.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-replay-study-03/train/receipt.json",
    "sha256": "a5e80227146675b1fdc1da4f45456e2e90917807cda7f1d34e689dead4c0c8d8"
  },
  "history": {
    "path": "runs/pressure-methods-v3/v3-replay-study-03/train/history.json",
    "sha256": "f17bd368a3f0efde53c40471932a7f0332535b6ac8c10e0aa4ec5553899beb7a"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/config.json",
      "sha256": "0c514c00ed07bfd94102bd4fa97a123d105cdebe65dc44702fbc784fa4e16742"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/data.py",
      "sha256": "09231fe0119461205b74578dab9ce77cbcbea7c537900941432e97f52e51f16d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/fine-tuning.json",
      "sha256": "b8ddfe8101b5d5f9521b9eed46278070a2aa7fe0b1591ba672c00236938f895f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/fine_tuning.py",
      "sha256": "1b49995a3bf55a6d24ba6d5d00c88a8506cb4e6aeca8ca2b17a3c809ef8e223b"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/head_optimizer.py",
      "sha256": "b96a4d5196e514ef8a13561415184df8f7ac1540474782b933a5928b5b8f98a6"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/methods_view.py",
      "sha256": "cebef234faa7131d46668c745a6ddac2064e31484fd6e9fb809e25b66cb28ceb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/model.py",
      "sha256": "63425e0a06f96ccd35476c94fabccb9bdfbd3691c706d98f2fa362ea00b3b4d9"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/train.py",
      "sha256": "6f17f8cd0d640e02330903e16f03280893153c6b58b4cf09735576a9e24b80a0"
    },
    {
      "path": "runs/pressure-methods-v3/v3-replay-project-03/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v3/v3-replay-project-03/source-witness.json",
    "sha256": "14238bf67badc16e971cf049cb4917f395943dc071508c4366a649915f80147f"
  },
  "training_population": {
    "path": "runs/pressure-methods-v3/v3-replay-study-03/train/splits.json",
    "sha256": "e3a56c441024126270ca3903541d8e13a46ca6f94491f9c974435e8878c40ca4"
  },
  "training_inputs": [
    {
      "bytes": 24736178,
      "path": "runs/pressure-methods-v3/replay-training-view-01/dataset-00.npz",
      "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7",
      "slot": "dataset"
    },
    {
      "bytes": 24737374,
      "path": "runs/pressure-methods-v3/replay-training-view-01/dataset-01.npz",
      "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb",
      "slot": "dataset_extra"
    },
    {
      "bytes": 1491574,
      "path": "runs/pressure-methods-v3/head-continuation-initialization-01/weights.npz",
      "sha256": "2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57",
      "slot": "initial_weights"
    },
    {
      "bytes": 42434,
      "path": "runs/pressure-methods-v3/replay-training-view-01/manifest.json",
      "sha256": "e3a56c441024126270ca3903541d8e13a46ca6f94491f9c974435e8878c40ca4",
      "slot": "splits"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a5e80227146675b1",
    "kind": "training_attempt",
    "name": "Pressure methods v3 · v3-replay CUDA",
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
      "max_cells": 2097152
    },
    "receipt_sha256": "a5e80227146675b1fdc1da4f45456e2e90917807cda7f1d34e689dead4c0c8d8",
    "receipt_paths": [
      "runs/pressure-methods-v3/v3-replay-study-03/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "7d21449534554f76aec83a6ff73fceb7",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v3 · v3-replay CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "0c514c00ed07bfd94102bd4fa97a123d105cdebe65dc44702fbc784fa4e16742",
      "inputs": {
        "dataset": {
          "bytes": 24736178,
          "sha256": "8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"
        },
        "dataset_extra": {
          "bytes": 24737374,
          "sha256": "7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"
        },
        "initial_weights": {
          "bytes": 1491574,
          "sha256": "2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57"
        },
        "splits": {
          "bytes": 42434,
          "sha256": "e3a56c441024126270ca3903541d8e13a46ca6f94491f9c974435e8878c40ca4"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "1e3104b3a0a6cab7ee8bd878534fc5f7ac9123e3a941988c39a340a283cf7f06",
      "source_witness_sha256": "14238bf67badc16e971cf049cb4917f395943dc071508c4366a649915f80147f"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57",
        "bytes": 1491574,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "f945e4e812e9db127191a66a254d2a9be8761501b8e32d5b8d0e5b1647ea0730",
        "bytes": 1491574,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "f945e4e812e9db127191a66a254d2a9be8761501b8e32d5b8d0e5b1647ea0730",
        "bytes": 1491574,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2797871104,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/98"
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
      "receipt": "runs/pressure-methods-v3/v3-replay-study-03/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v3/v3-replay-study-03/train/receipt.json",
    "sha256": "a5e80227146675b1fdc1da4f45456e2e90917807cda7f1d34e689dead4c0c8d8",
    "bytes": 11260
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C100": {
        "name": "v3-replay",
        "weights_sha256": "f945e4e812e9db127191a66a254d2a9be8761501b8e32d5b8d0e5b1647ea0730",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.0009231457724317273,
          "pressure_mean_free_relative_l2": 0.003081923685320427,
          "pressure_absolute_rms": 0.042341284496805506,
          "pressure_absolute_max": 0.07144873207378183
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
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0004522100962450042,
              "pressure_mean_free_relative_l2": 0.0030443402528283446,
              "pressure_absolute_rms": 0.042341284496805506,
              "pressure_absolute_max": 0.07144873207378183
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.0009231457724317273,
              "pressure_mean_free_relative_l2": 0.003081923685320427,
              "pressure_absolute_rms": 0.014755731110207063,
              "pressure_absolute_max": 0.02799171481013012
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C100/results.json",
          "sha256": "adbb698afcb3245799cfceaa82ec15d1f9ab7837729a064655f98f75e968de2c"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"fine_tuning":{"arm":"v3-replay","early_weight":0.05,"initial_weights_sha256":"2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57","relative_equation_target":1e-06,"schema":"rheon-fine-tuning-objective-v1","teacher_cache_bytes":268435456,"teacher_passes":8,"teacher_weight":0.0,"training_passes":2},"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"learned_coarse_revision":{"basis":"Orthonormal separable Neumann cosine modes 0..3, excluding DC","bound":"max_axis_cells^2/2","coefficients":"Positive learned sigmoid gains; no analytical inverse eigenvalues","gate":"Accept learned coarse proposal only when its actual fine-equation residual norm does not increase","initialization":"Gain 1 on 16^3; all original neural tensors preserved","limitation":"Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation","placement":"Learned coarse proposal before every fine neural pass","trainable_parameters":63},"max_cells":2097152,"method_comparison":{"normalization":"fixed original C53 train-only scales","sampling":"Balanced spatial tiers; private RNG; fixed update budget.","supervision":"Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer."},"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","shape_informed_coarse":{"added_parameters":1135,"features":"log(Nz/16), log(Ny/16), log(Nx/16)","fine_initialization":"Exact completed v2-headshape CPU restore; matched continuation control","head_learning_rate_multiplier":1000,"network":"3 -> 16 tanh -> 63 additive coarse logits; final layer initially zero","operator":"Existing sealed constant-density equation"},"supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":1,"native_frame":63,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":0,"native_frame":15,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":0,"native_frame":15,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.0625,"counts_xyz":[10,5,7],"frame":1,"native_frame":63,"parent":11,"split":"train","trajectory_id":"baseline--water-train-0011"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.125,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":20,"split":"train","trajectory_id":"v2-train-00-cpm8"},{"cell_size_m":0.125,"counts_xyz":[61,39,27],"frame":0,"native_frame":15,"parent":22,"split":"train","trajectory_id":"v2-train-01-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[48,13,9],"frame":1,"native_frame":63,"parent":25,"split":"train","trajectory_id":"v2-train-02-cpm16"}],"closure":{"experiment_sha256":"0c514c00ed07bfd94102bd4fa97a123d105cdebe65dc44702fbc784fa4e16742","inputs":{"dataset":{"bytes":24736178,"sha256":"8b7386e3831f0bc7cf9154cbb4303902347c26bc3e74b5308ebed9de0618fdd7"},"dataset_extra":{"bytes":24737374,"sha256":"7e0c4d5e8b3fac166303456b30dfda37461be16b4208ab4f8f0c4e6d836898bb"},"initial_weights":{"bytes":1491574,"sha256":"2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57"},"splits":{"bytes":42434,"sha256":"e3a56c441024126270ca3903541d8e13a46ca6f94491f9c974435e8878c40ca4"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"1e3104b3a0a6cab7ee8bd878534fc5f7ac9123e3a941988c39a340a283cf7f06","source_witness_sha256":"14238bf67badc16e971cf049cb4917f395943dc071508c4366a649915f80147f"},"diagnostic_population":"train_only","elapsed_s":90.92115666001337,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.0844977370870765e-06,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":362002432,"peak_reserved_bytes":446693376,"tf32":false},"fine_tuning_statistics":{"teacher_cache_hits":0,"teacher_cache_peak_bytes":0,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1491574,"role":"checkpoint","sha256":"2356d78e43fd20691522a5e1bba83b4992e2930061164b6c81964a623d16de57"},"checkpoint-2400.npz":{"bytes":1491574,"role":"checkpoint","sha256":"f945e4e812e9db127191a66a254d2a9be8761501b8e32d5b8d0e5b1647ea0730"},"history.json":{"bytes":15364,"role":"training_diagnostics","sha256":"f17bd368a3f0efde53c40471932a7f0332535b6ac8c10e0aa4ec5553899beb7a"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":2837102,"role":"predictions","sha256":"5ea4c5472242d4dadac33c2d9a274934309430d774842f6d1749833737a80bbd"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":42434,"role":"training_splits","sha256":"e3a56c441024126270ca3903541d8e13a46ca6f94491f9c974435e8878c40ca4"},"summary.json":{"bytes":584,"role":"result_summary","sha256":"89cc9845804e5a2cfc070e0ee6e9f7ba191f9021af059945ac3aaebf97d40030"},"weights.npz":{"bytes":1491574,"role":"checkpoint","sha256":"f945e4e812e9db127191a66a254d2a9be8761501b8e32d5b8d0e5b1647ea0730"}},"per_case_inference_s":[0.030790180986514315,0.008631447999505326,0.0076518229907378554,0.005732228019041941,0.007371218001935631,0.005162335000932217,0.009548293019179255,0.00872892199549824,0.8816024049883708,0.0891122920147609,0.31681387501885183,0.022657590016024187],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"d3d5f0ccd4221dfcaab3fc0934da1a69f52caf4fe74d5351c9c3fd6a77bee51b","shape":[16,16,16]},"case_001":{"bytes":8736,"dtype":"<f8","sha256":"5395856211f9bb787358a3c8373a86d898da7073413c65738c6b2413984a12a9","shape":[13,7,12]},"case_002":{"bytes":7200,"dtype":"<f8","sha256":"5f1408d0aed6f3448b3298662eb35c8f3c0ab0f0667a1a1a0912b7fea708a528","shape":[6,15,10]},"case_003":{"bytes":2880,"dtype":"<f8","sha256":"bfbc89c57e73c8e20b260aa218bc8ee9bbad1aea096221fe4620d5fbf746c6ef","shape":[9,5,8]},"case_004":{"bytes":6048,"dtype":"<f8","sha256":"2219b24016117ab485bc8beefa9d2cd2ef6f598fd01be4bae435ea4b54fa90fb","shape":[9,14,6]},"case_005":{"bytes":2800,"dtype":"<f8","sha256":"fc8700e9b491d08497733cfc3a58e6ae4ba6021a1714bae8056ca10cd6e1a5f3","shape":[7,5,10]},"case_006":{"bytes":9000,"dtype":"<f8","sha256":"0a7692d4be86a2987f677e1123766954d7aa64c2f4f57fffaf37e11074256b31","shape":[15,15,5]},"case_007":{"bytes":11760,"dtype":"<f8","sha256":"03eb9fcfb40fc31db5bedb19703dd619e5e685dd739434addb5b68180fe21b36","shape":[7,15,14]},"case_008":{"bytes":2097152,"dtype":"<f8","sha256":"6cdbcb67cd6f319a54750d99c938566c011e3ae30e952dbea049a775ca26d8ba","shape":[64,64,64]},"case_009":{"bytes":96968,"dtype":"<f8","sha256":"375bf81001fb7670a73e983ffb75f1f867e49a147d6d860861e352f557b9884b","shape":[17,23,31]},"case_010":{"bytes":513864,"dtype":"<f8","sha256":"46d674a9b9c467adbd5f738529ffe5ccf8a88f8e2129466390bdaa6a4e5560a4","shape":[27,39,61]},"case_011":{"bytes":44928,"dtype":"<f8","sha256":"bcf72d0030893003cfeea80533835ee6616fc32a6f4e21299b1d54a4c271bf35","shape":[9,13,48]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2797871104,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
