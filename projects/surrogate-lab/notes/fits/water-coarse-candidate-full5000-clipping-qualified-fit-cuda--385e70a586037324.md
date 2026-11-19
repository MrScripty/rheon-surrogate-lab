# water-coarse-candidate full5000 clipping-qualified fit CUDA

Retrospective source review · fit-385e70a586037324

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 374,112-parameter linear one-channel F32 pressure CNN: the 370,016-parameter base plus a learned bias-free 64-by-64 spatial coarse matrix. Fine-cell overlap averages restrict to 4-by-4-by-4 bins, dense learned mixing couples the bins globally, clamped cell-centre interpolation reconstructs the full grid, and the mean-free addition scales by (longest cell count/16)^2. Two shared-weight residual corrections use F64 physical assembly and gauging. The coarse branch is learned inference, not a numerical inverse solver; neither correction represents another physical timestep.

**Purpose.** Test whether a small trainable global coarse branch reduces long-wavelength pressure error while preserving the original base network, zero coarse initialization and the matched five fitting streams. The completed full-length fit follows a qualified overflow-only gradient-norm repair; it must be distinguished from capped diagnostic attempts and the later failed simulation qualification.

**Experiment.** 5,000 CUDA Adam updates, batch one, initial rate 0.001 with drops at 3000/4500 to 0.0001/0.00001. Model seed 20261015 and physical sample seed 2026102501. Sixteen physical trajectories/32 frames at 8/16 cells per metre and dt=1/60 s are supplemented with two manufactured spatial banks and two frozen captured-direction streams; all five total 50,000 neural root calls. The current-policy bank uses the late stage at weight 0.1. All 19 initial base tensors match C53; the extra 4,096 coefficients begin at exact zero. Ordinary finite gradient clipping retains its original path; only overflow in a reduction of finite F32 gradients uses scaled F32 norm arithmetic. This is an explicit optimizer change, not a wider model dtype.

**Observed outcome.** Retained final training loss 0.0015872570694780686. Common 128 consumed pressure requests: worst mean-free pressure relative L2 0.015527806633378076, equation relative L2 0.0098092482557465815, median case-median complete GPU pressure call 5.3654272499999998 ms. The pressure and equation tails are worse than C53 on this population. These are pressure-request timings with host input/output and two corrections, not full simulation-step latency; the original possible host-metadata-overlap flag remains. Historical simulation qualification failed: nine windows passed, three native candidate rejections and one independent certificate failure occurred; the 74-case campaign did not finish and the candidate was not promoted. No new execution or drift measurement is added.

**Lesson.** Successful optimization and a learned coarse branch do not establish a usable solver. Preserve completed fit weights, pressure-only errors and failed end-to-end qualification as separate outcomes. The coarse matrix provides learned global coupling rather than an exact conservation or inverse-operator guarantee. Comparisons must disclose the clipping repair and frozen feedback banks; this experiment does not isolate architecture alone from every training change.

**Reproduction.** Retain the actual full_fit.py typed entrypoint, its gradient reference and exact clipping/observer sources, all original code/config/runtime/input bindings, 16-parent split, normalization, private draw orders and both checkpoint inventories. Preserve historical failure dispositions and original receipts. The historical evidence publication/restore performs CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, fresh native solver qualification, new-Rheon regeneration and long own-history drift remain pending; no Trial ancestry is invented.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-coarse-full-fit-v1/study-01/train/receipt.json",
    "sha256": "385e70a58603732475ca524a0cb810cd827f5325f1366a353590d291d289a987"
  },
  "history": {
    "path": "runs/pressure-coarse-full-fit-v1/study-01/train/history.json",
    "sha256": "06e5c1f2b7222be224445893dd9cdd1ba2945b081cfe145af2cc3ab9783277f7"
  },
  "sources": [
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/base_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/baseline_train.py",
      "sha256": "152d714442cca48a62afbcd0f2e0c2d43000d2a17710400b50ec1788a5bb6fa7"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/capture_policy.py",
      "sha256": "c14e314920abb444a50b7b92a02801f4aa483625f82dfec2b12c55bd5fcdfc7f"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/coarse.py",
      "sha256": "007624389a77559b82b9985c5d5a28dde3d9a85021973bef81ebe0ec9c0010f5"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/config.json",
      "sha256": "ecb15f30318eb45f2d3099a02a24ad1e1ce2c437b94b5ccf6d23d54e521a4929"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/custody-validator.py",
      "sha256": "247c573d6a9bfdc4e189b3a940650c91102cb40273751225f0fef6dd79999e6f"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/full_fit.py",
      "sha256": "0c326285555490ad9adcc22f1e8f52eb33ea554ebd83439882e7ad15cc7df316"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/gradient-reference.json",
      "sha256": "a6c7d435040ac417cd4fc3f8fb032e9f8563a2bc86190c4a131fe1b7fd414f07"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/gradient_screen.py",
      "sha256": "ffddc200f0907097092e2d10bd208052a10944469f9b4ce0df8dcdbac39f6974"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/model.py",
      "sha256": "ea63e73008ae48038ee86ff2422d131cfaef5bb7c34d650b8415e7ea47b18b13"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/physical_residual.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/policy.py",
      "sha256": "c3e679239c58332e034064fb8825768fec4fb616f01a22f9c0c4eb1995ce29f2"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/portable_policy.py",
      "sha256": "c9270abbbb05794b68c53d83a14f90d4c16e9d0fea1e28ce3791b8018f862995"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/stability_screen.py",
      "sha256": "205922290a15fa455895de7c516a8895cc17e061ec6eef7677d4c05922ac2688"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/train.py",
      "sha256": "534a53c509d5c1461ac203446396816206652288e7f81c883dd31c65f67595c3"
    },
    {
      "path": "runs/pressure-coarse-full-fit-v1/project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-coarse-full-fit-v1/project-01/source-witness.json",
    "sha256": "2226e49bc095e23ba4e0b24b164530e8a6258c462ecd5bb30e7b2ff138bf5dca"
  },
  "training_population": {
    "path": "runs/pressure-coarse-full-fit-v1/study-01/train/splits.json",
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
    "item_id": "fit-385e70a586037324",
    "kind": "training_attempt",
    "name": "water-coarse-candidate full5000 clipping-qualified fit CUDA",
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
    "receipt_sha256": "385e70a58603732475ca524a0cb810cd827f5325f1366a353590d291d289a987",
    "receipt_paths": [
      "runs/pressure-coarse-full-fit-v1/study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "c01e1741f48a474d9a2a3252b3971827",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "water-coarse-candidate full5000 clipping-qualified fit CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 5000,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "ecb15f30318eb45f2d3099a02a24ad1e1ce2c437b94b5ccf6d23d54e521a4929",
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
      "source_witness_sha256": "2226e49bc095e23ba4e0b24b164530e8a6258c462ecd5bb30e7b2ff138bf5dca"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "6021ebf7385556cedf9d168d74a70c0027c825f27e3e08b969bf534128434723",
        "bytes": 1502110,
        "stage": "initial"
      },
      {
        "name": "checkpoint-5000.npz",
        "sha256": "c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade",
        "bytes": 1502110,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade",
        "bytes": 1502110,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 2664886272,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/55"
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
      "receipt": "runs/pressure-coarse-full-fit-v1/study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-coarse-full-fit-v1/study-01/train/receipt.json",
    "sha256": "385e70a58603732475ca524a0cb810cd827f5325f1366a353590d291d289a987",
    "bytes": 10353
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C56": {
        "name": "full-coarse",
        "weights_sha256": "c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.009809248255746581,
          "pressure_mean_free_relative_l2": 0.015527806633378076,
          "pressure_absolute_rms": 0.4086254479591726,
          "pressure_absolute_max": 1.1776379259100016
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0050",
            "native_frame": 63,
            "counts_xyz": [
              5,
              6,
              5
            ],
            "cell_size_m": 0.125
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0022",
            "native_frame": 15,
            "counts_xyz": [
              16,
              6,
              5
            ],
            "cell_size_m": 0.125
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.009809248255746581,
              "pressure_mean_free_relative_l2": 0.015527806633378076,
              "pressure_absolute_rms": 0.4086254479591726,
              "pressure_absolute_max": 1.1776379259100016
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.00734814998584625,
              "pressure_mean_free_relative_l2": 0.013707985251143232,
              "pressure_absolute_rms": 0.23914627272051142,
              "pressure_absolute_max": 0.5998973976185681
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-01/C56/results.json",
          "sha256": "8172156f682e9afec7e0a13b21f703dc13c24ff2dd821fd0d88ea3caec136432"
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
{"architecture":{"activation":"silu","analytical_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,16,16]},{"cells_per_metre":16,"counts_xyz":[24,24,24]},{"cells_per_metre":16,"counts_xyz":[32,32,32]},{"cells_per_metre":16,"counts_xyz":[40,40,40]},{"cells_per_metre":16,"counts_xyz":[24,16,12]},{"cells_per_metre":8,"counts_xyz":[17,13,9]},{"cells_per_metre":8,"counts_xyz":[33,17,9]},{"cells_per_metre":8,"counts_xyz":[9,25,13]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101507,"weight":0.1},"block_normalization":{"name":"Identity"},"bottleneck_width":48,"channels":["dimensionless_laplacian_rhs"],"conditioning":"physical-residual","convolutions_per_block":2,"coordinates":"implicit grid indices; no coordinate feature maps","correction":{"activation":"linear","steps":2},"counts_supported":"positive per-axis integers within cell budget","current_policy_training_feedback":{"manifest_sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e","seed":2026102801,"stage":"late","weight":0.1},"decoder_widths":[32,24,16,8],"direction_training_feedback":{"manifest_sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc","seed":2026102701,"stage":"late","weight":0.1},"dtype":"float32","encoder_widths":[8,16,24,32],"first_correction_pressure_feedback_weight":0.0,"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"retain original first-convolution residual column and all surviving seeded tensors","input_channels":1,"intermediate_feedback":{"relative_equation_weight":0.1,"stages":"all nonfinal corrections; real and analytical train samples"},"kernel":[3,3,3],"max_cells":65536,"output_scale":"previous-pressure-rms","padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","preprocessing":"q=(rho/h)*residual; q / train-only q RMS","pressure_dtype":"float64","reconstruction":"float32 raw times selected train-only RMS, cell-zero gauge, add supplied base","restrictions":"uniform-density cubic sealed Neumann operator; no independently specified geometry masks","schema":"rheon-physical-residual-pressure-v1","spatial_coarse_correction":{"bins_zyx":[4,4,4],"inference_inverse_solver":false,"initialization":"zeros; original network RNG stream unchanged","mixing":"dense bias-free 64-to-64 trainable matrix","prolongation":"linear cell-centre interpolation with clamped edges","restriction":"fine-cell overlap averages","scale":"(longest cell count / 16)^2","schema":"learned-spatial-coarse-v1"},"supplemental_training_feedback":{"amplitude_reference":"train-correction-cell-rms","cache_max_bytes":805306368,"grid_capacity":16,"grids":[{"cells_per_metre":16,"counts_xyz":[16,23,25]},{"cells_per_metre":16,"counts_xyz":[23,25,16]},{"cells_per_metre":16,"counts_xyz":[25,16,23]},{"cells_per_metre":16,"counts_xyz":[10,7,25]},{"cells_per_metre":16,"counts_xyz":[7,25,10]},{"cells_per_metre":16,"counts_xyz":[25,10,7]}],"patterns":32,"sampling":"uniform-balanced-grid","seed":2026101607,"source":"grids","weight":0.1},"training_band_feedback":{"max_mode":3,"signal_floor_fraction":0.1,"stages":"first-and-final","weight":0.03},"training_objective":"canonical-intermediate-contraction","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[16,10,6],"frame":1,"native_frame":63,"parent":1,"split":"train","trajectory_id":"baseline--water-train-0001"},{"cell_size_m":0.125,"counts_xyz":[12,7,13],"frame":0,"native_frame":15,"parent":2,"split":"train","trajectory_id":"baseline--water-train-0002"},{"cell_size_m":0.125,"counts_xyz":[10,15,6],"frame":1,"native_frame":63,"parent":4,"split":"train","trajectory_id":"baseline--water-train-0004"},{"cell_size_m":0.0625,"counts_xyz":[11,10,6],"frame":0,"native_frame":15,"parent":5,"split":"train","trajectory_id":"baseline--water-train-0005"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":1,"native_frame":63,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.125,"counts_xyz":[12,8,9],"frame":0,"native_frame":15,"parent":8,"split":"train","trajectory_id":"baseline--water-train-0008"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[10,16,16],"frame":0,"native_frame":15,"parent":10,"split":"train","trajectory_id":"baseline--water-train-0010"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":1,"native_frame":63,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[5,15,15],"frame":0,"native_frame":15,"parent":13,"split":"train","trajectory_id":"baseline--water-train-0013"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"}],"closure":{"experiment_sha256":"ecb15f30318eb45f2d3099a02a24ad1e1ce2c437b94b5ccf6d23d54e521a4929","inputs":{"current_dataset":{"bytes":21428086,"sha256":"8b6252c12d18ffb826d907fc4b0b6761f0eb6958802e6e462883beecab6d1521"},"current_manifest":{"bytes":168912,"sha256":"fbeb3908e670f84f2ca9d0ce5b65ff55c29630f86ece64a00f62a78e23206b7e"},"dataset":{"bytes":610070,"sha256":"89f5f3ee7235c33321ce4ba6bf954e5253f6e1c58cda47508f29be1e7340686e"},"dataset_extra":{"bytes":610806,"sha256":"c0abbcbf4282488002a9b69552e80a3a541c5adc866182d7775aaef9048e5fd5"},"direction_dataset":{"bytes":24476086,"sha256":"4f4f9c948ce5d38dfcb9ecdf424eae2e533e3ee83ad6549dff849178fdadabc6"},"direction_manifest":{"bytes":298490,"sha256":"ac036f90dcccf48cb11dbd6a737c142b5ea4ab1d3d34606ef62ab3cda0ca05fc"},"splits":{"bytes":256361,"sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"3ff66d55f76da2af4cf7f39bf2ee510f2216dea9ee18b16645cd99e2e3d37423","source_witness_sha256":"2226e49bc095e23ba4e0b24b164530e8a6258c462ecd5bb30e7b2ff138bf5dca"},"diagnostic_population":"train_only","elapsed_s":591.0335378650052,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":2.18867075432172e-05,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":870590464,"peak_reserved_bytes":920649728,"tf32":false},"heldout_evaluations":0,"mode":"train","optimizer_updates":5000,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1502110,"role":"checkpoint","sha256":"6021ebf7385556cedf9d168d74a70c0027c825f27e3e08b969bf534128434723"},"checkpoint-5000.npz":{"bytes":1502110,"role":"checkpoint","sha256":"c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade"},"history.json":{"bytes":216305,"role":"training_diagnostics","sha256":"06e5c1f2b7222be224445893dd9cdd1ba2945b081cfe145af2cc3ab9783277f7"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":129230,"role":"predictions","sha256":"c78fd4e57a21fd31291329fee6199813ec0bf7cf314827658979d64a95689b80"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":256361,"role":"training_splits","sha256":"bdcfb5f5d8e4dbd54f797d827e2dd0715fcc2bfb6510697fc51996c3d5a5779e"},"summary.json":{"bytes":473,"role":"result_summary","sha256":"79a4661d24a01ead005acad3de3425c14cf937586e34f59df167057e5f2b6dcb"},"weights.npz":{"bytes":1502110,"role":"checkpoint","sha256":"c38091565d58c2a897c2dd6fa853d223262dcc64a2118ee428ea58e17a3b8ade"}},"per_case_inference_s":[0.02456579200224951,0.005296392002492212,0.006257714005187154,0.0050891290011350065,0.004525437005213462,0.004274997001630254,0.00527971499832347,0.0054463749984279275,0.010604765004245564,0.006079954997403547,0.007206890004454181,0.006568736003828235],"prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"918b7c6f754cb3e5b74929434bd257da1bef724c632dbf8d819933cac381991a","shape":[16,16,16]},"case_001":{"bytes":7680,"dtype":"<f8","sha256":"35edfb86ca3e337862f40e1db721db95f68e02a5fe4811f86932d7669137e41a","shape":[6,10,16]},"case_002":{"bytes":8736,"dtype":"<f8","sha256":"4851a0ca40007ecb4486e0adb345781cce3829ebda42b3b6bcae99d87c57242d","shape":[13,7,12]},"case_003":{"bytes":7200,"dtype":"<f8","sha256":"32bdd7380d40de9fac7c980e6286da1ff37c882c6c1ebffaeacbd0999570aa3f","shape":[6,15,10]},"case_004":{"bytes":5280,"dtype":"<f8","sha256":"d7e626bd1045999666eb263812645803151dd5caba3a645517acbd4984571f3c","shape":[6,10,11]},"case_005":{"bytes":2880,"dtype":"<f8","sha256":"211bab7709b59064cacda186d8bc6d5e9b887c1304c8e0da395512cd6188bfc4","shape":[9,5,8]},"case_006":{"bytes":6912,"dtype":"<f8","sha256":"6925db3289255cf3957cc031694397a5fd981abcc3c97d8bb0cf76e02d54ba55","shape":[9,8,12]},"case_007":{"bytes":6048,"dtype":"<f8","sha256":"35b437783de5fccbcd287d81f27d16a86cc3917f6d0f12a26e58fdc5e5a078aa","shape":[9,14,6]},"case_008":{"bytes":20480,"dtype":"<f8","sha256":"dbe99b58032e23759f40894c37d907c722bb4330674cdcfc8a101bf2baf06d16","shape":[16,16,10]},"case_009":{"bytes":7488,"dtype":"<f8","sha256":"0589d7a8ea092aeb2b0479c64065a1fcfae425848ed7bcaf754adf08c34d3304","shape":[12,13,6]},"case_010":{"bytes":9000,"dtype":"<f8","sha256":"10d974b2406ab36933ff1ed4e69ae72ac1f4b45ce2a668ee05c180fd6b1079e1","shape":[15,15,5]},"case_011":{"bytes":11760,"dtype":"<f8","sha256":"4cf8154f25f472fcdbf7058adeee9b7fa277d71fc138f329314b96679c0fa9ed","shape":[7,15,14]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":2664886272,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
