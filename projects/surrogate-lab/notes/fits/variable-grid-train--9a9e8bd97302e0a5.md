# Variable-grid train

Retrospective source review · fit-9a9e8bd97302e0a5

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** An additive pressure-correction CNN with 371,729 float32 parameters in 38 tensors. Four encoder widths 8/16/24/32, bottleneck 48 and decoder widths 32/24/16/8; two 3x3x3 convolutions per block, affine-free four-group GroupNorm and SiLU, a bottleneck global-mean broadcast and nearest-to-skip upsampling. The network consumes seven inputs: residual, previous pressure, physical XYZ coordinates, rho and dt; spacing also affects the coordinates and remains scalar FV metadata. Its linear 1x1x1 head is scaled to physical pressure, gauged by the first correction cell and added to the previous pressure. FV arithmetic and the pressure output remain float64. All network parameters are optimized. The captured input budget is 65,536 cells. This operation proposes pressure for one request; it does not advance the physical simulation.

**Purpose.** Remove three redundant cubic-spacing network channels, preserving scalar spacing in the FV operator, metre-valued coordinates and the surviving seeded coefficients.

**Experiment.** 200 Adam updates at learning rate 0.001, batch size 2 and seed 20261015, with gradient clipping. Four training trajectories supply 96 pressure frames; two validation trajectories supply 48 frames. The shared 9,296,002-byte dataset covers 8 and 16 cells per metre. Training grids are 8x8x8, 9x7x5, 16x16x16 and 17x13x9; validation grids are 12x8x6 and 24x16x12. The recorded forcing labels are steady and moving, with amplitude 0.5 m/s2 and a 1/60 s timestep. The objective equally weights normalized squared pressure-field and FV equation-residual errors. Train-only scales and the exact PCG64 draw sequence are bound. The recipe supplies a final-update candidate and does not retain an optimizer-resume state.

**Observed outcome.** The original final logged training loss is 0.036771804285947604 at update 200. The existing 128-request GPU comparison gives worst mean-free pressure relative L2 0.29089127566759132 and equation relative L2 0.47508074400920064. The training objective and common field errors are different quantities. This source review adds no inference timing or own-history drift measurement.

**Lesson.** The dataset bytes and independently derived 200-update sample sequence are identical to C03. Removing 648 first-layer coefficients does not materially improve the common worst errors. Record the seed agreement separately from trained ancestry. Split metadata reports a different native-build witness; preserve that identity without claiming a fresh independent capture or larger training population.

**Reproduction.** Use the captured project manifest and training recipe, exact source witness, config, runtime, normalization and two typed data slots. Preserve the initial seed inventory and final weight hash. The population metadata retains native binary identity 67d03c454e8293fd0120243c5834e8f67e4a24f2ba66392fef9a815d7b9117ad. Its six capture specifications and frame hashes describe the original data. Clean training/data regeneration and conversion to the new Rheon have not been executed or qualified. The original CPU-only contract remains historical evidence; future execution requires a qualified GPU route under the current no-CPU-neural-forward policy.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/receipt.json",
    "sha256": "9a9e8bd97302e0a5d16f4c24a9b21a5daeb7708d09481b1b77bb818c77196524"
  },
  "history": {
    "path": "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/history.json",
    "sha256": "b2ea1233a920da96e35b335c913a619e36520c8acbf6813e4751752bb0ca2e0b"
  },
  "sources": [
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/config.json",
      "sha256": "8f34f5e0c9853bfc9a8b8193334919e99835aa72e6f21f1c51354063596de6ac"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/data.py",
      "sha256": "fd2272eaceb8d3c56e1fe60189be0b466348e33fe531c0ea6e477f2f6807c8ff"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/model.py",
      "sha256": "cada0206bdb1180ffdc0dde08f1fcb351455c3164e80e809b73feeeaf4bca9ef"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/policy.py",
      "sha256": "5b3bd3863147e8586c78cb53881f6806f17330a25aa747bdf62bb95b7f1b1abd"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/qualification.json",
      "sha256": "bd875a67e45cb2d71fa188092f90cd9087e5f3647fb4f67d61037cb199191170"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/train.py",
      "sha256": "454b12707ec317ec25ff72050647a79a0b2459bb646302cbe103d7d0403c51e4"
    }
  ],
  "source_witness": {
    "path": "runs/variable-pressure-v2-workspace-1/project/source-witness.json",
    "sha256": "802abe311e2c23009cb176c28307f88f648b7f7b033f77b82e78ac53aa30f0aa"
  },
  "training_population": {
    "path": "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/splits.json",
    "sha256": "6925843cbf536bf46662d96ea2dd4b4bfa8b3567481a1a5d51e2f4a7d6974b57"
  },
  "training_inputs": [
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/inputs/dataset.npz",
      "sha256": "743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f",
      "slot": "dataset",
      "bytes": 9296002
    },
    {
      "path": "runs/variable-pressure-v2-workspace-1/project/inputs/splits.json",
      "sha256": "6925843cbf536bf46662d96ea2dd4b4bfa8b3567481a1a5d51e2f4a7d6974b57",
      "slot": "splits",
      "bytes": 122850
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-9a9e8bd97302e0a5",
    "kind": "training_attempt",
    "name": "Variable-grid train",
    "task": "pressure_proposal",
    "schema": "rheon-variable-pressure-training-v2",
    "method": {
      "schema": "rheon-cubic-groupnorm-model-v3",
      "input_channels": 7,
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
      "counts_supported": "positive per-axis integers within cell budget",
      "max_cells": 65536
    },
    "receipt_sha256": "9a9e8bd97302e0a5d16f4c24a9b21a5daeb7708d09481b1b77bb818c77196524",
    "receipt_paths": [
      "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/receipt.json",
      "runs/variable-pressure-v2-workspace-1/execution/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/89e5e42c4563405d9afb3942907aea14/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-89e5e42c4563405d9afb3942907aea14/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "89e5e42c4563405d9afb3942907aea14",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog",
          "archived_schema8_catalog"
        ],
        "graph_name": "Variable-grid train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 200,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "8f34f5e0c9853bfc9a8b8193334919e99835aa72e6f21f1c51354063596de6ac",
      "inputs": {
        "dataset": {
          "bytes": 9296002,
          "sha256": "743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f"
        },
        "splits": {
          "bytes": 122850,
          "sha256": "6925843cbf536bf46662d96ea2dd4b4bfa8b3567481a1a5d51e2f4a7d6974b57"
        }
      },
      "normalization_sha256": "2ce6c3ab6722269b97b6ae61e978d219c0ee8d3d56c471cba0b3d85188a64da1",
      "runtime_manifest_sha256": "ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7",
      "sample_order_sha256": "8368c1854402320e687cb5cf48f2819981e797a6cd2aac7e795bf8af25029aea",
      "source_witness_sha256": "802abe311e2c23009cb176c28307f88f648b7f7b033f77b82e78ac53aa30f0aa"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c",
        "bytes": 1497594,
        "stage": "initial"
      },
      {
        "name": "checkpoint-020.npz",
        "sha256": "3c5cc9d1b6017f2bb249889c9483d4401a04ee485a4365ff103189edecd706d5",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-040.npz",
        "sha256": "9e9dddcfa3ce3fa7bce918c7f32214130539709601148bbe8bc8645b6fa9744b",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-060.npz",
        "sha256": "7af82d108cff562c390456da3909fd59d51d3d9d02b7348502d79855707dfde2",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-080.npz",
        "sha256": "6d685258146ad2585ff75f7514ad1a7b356781456bde62ad70ce79add2cd157d",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "76b7bad13184a11746bf0de7b3414a3bec09f13b05125f1fcfc742032f02ea9d",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-120.npz",
        "sha256": "9eb7744e9a6811b7ceca44ccd338c13952283b5cf402389e7f51f34b94374f0a",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-140.npz",
        "sha256": "95abad944de5b8baede0dc1d5042a1a297cb2da01bb7dc5b9d29745127d5fad7",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-160.npz",
        "sha256": "91cb3bb5171be6e5743c076be8d7ce3deb6cc6a22d0bfb706fab6200443d671b",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-180.npz",
        "sha256": "6014478a3879bd12193bd844d137b0571d45fbc4f81fa07f1fdb7861fe739f86",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-200.npz",
        "sha256": "9a9593165d428fcb4b4353176d67ca697d75bc1e7c9fc52cf08cf224524b6944",
        "bytes": 1497594,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "9a9593165d428fcb4b4353176d67ca697d75bc1e7c9fc52cf08cf224524b6944",
        "bytes": 1497594,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "interop_threads": 1,
      "numpy": "2.3.5",
      "peak_rss_bytes": 467050496,
      "threads": 1,
      "torch": "2.8.0+cpu"
    },
    "named_entries": [
      "cohorts/3"
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
      "receipt": "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/variable-pressure-v2-workspace-1/execution/stage-00/artifacts/receipt.json",
    "sha256": "9a9e8bd97302e0a5d16f4c24a9b21a5daeb7708d09481b1b77bb818c77196524",
    "bytes": 7622
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C04": {
        "name": "Variable grid · 7 channels",
        "weights_sha256": "9a9593165d428fcb4b4353176d67ca697d75bc1e7c9fc52cf08cf224524b6944",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.47508074400920064,
          "pressure_mean_free_relative_l2": 0.2908912756675913,
          "pressure_absolute_rms": 19.82980320720171,
          "pressure_absolute_max": 73.27425500892735
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
              "common_equation_relative_l2": 0.3723237220284103,
              "pressure_mean_free_relative_l2": 0.20039394172722075,
              "pressure_absolute_rms": 19.82980320720171,
              "pressure_absolute_max": 73.27425500892735
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.47508074400920064,
              "pressure_mean_free_relative_l2": 0.2908912756675913,
              "pressure_absolute_rms": 5.993449177419022,
              "pressure_absolute_max": 15.096569776943294
            }
          }
        },
        "execution_variant": "legacy_CPU_features_CUDA_CNN_CPU_pressure",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C04/results.json",
          "sha256": "2ca06f3f966d6245d72e1447c5b4b5934cb698c335810d20fd087d8924a24bfb"
        },
        "CNN_operation_AST_preserved": true,
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","initialization":"seeded original blocks; first convolution retains columns 0,1,2,3,4,8,9","input_channels":7,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one scalar h per sample; no cell-size network channels","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v3","upsampling":"nearest_to_skip_shape"},"cases":[{"cell_size_m":0.125,"counts_xyz":[8,8,8],"frame":0,"parent":0,"split":"train","trajectory_id":"variable-train-510000"},{"cell_size_m":0.125,"counts_xyz":[8,8,8],"frame":23,"parent":0,"split":"train","trajectory_id":"variable-train-510000"},{"cell_size_m":0.125,"counts_xyz":[9,7,5],"frame":0,"parent":1,"split":"train","trajectory_id":"variable-train-510001"},{"cell_size_m":0.125,"counts_xyz":[9,7,5],"frame":23,"parent":1,"split":"train","trajectory_id":"variable-train-510001"},{"cell_size_m":0.125,"counts_xyz":[12,8,6],"frame":0,"parent":2,"split":"validation","trajectory_id":"variable-validation-510002"},{"cell_size_m":0.125,"counts_xyz":[12,8,6],"frame":23,"parent":2,"split":"validation","trajectory_id":"variable-validation-510002"},{"cell_size_m":0.0625,"counts_xyz":[16,16,16],"frame":0,"parent":3,"split":"train","trajectory_id":"variable-train-510100"},{"cell_size_m":0.0625,"counts_xyz":[16,16,16],"frame":23,"parent":3,"split":"train","trajectory_id":"variable-train-510100"},{"cell_size_m":0.0625,"counts_xyz":[17,13,9],"frame":0,"parent":4,"split":"train","trajectory_id":"variable-train-510101"},{"cell_size_m":0.0625,"counts_xyz":[17,13,9],"frame":23,"parent":4,"split":"train","trajectory_id":"variable-train-510101"},{"cell_size_m":0.0625,"counts_xyz":[24,16,12],"frame":0,"parent":5,"split":"validation","trajectory_id":"variable-validation-510102"},{"cell_size_m":0.0625,"counts_xyz":[24,16,12],"frame":23,"parent":5,"split":"validation","trajectory_id":"variable-validation-510102"}],"closure":{"experiment_sha256":"8f34f5e0c9853bfc9a8b8193334919e99835aa72e6f21f1c51354063596de6ac","inputs":{"dataset":{"bytes":9296002,"sha256":"743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f"},"splits":{"bytes":122850,"sha256":"6925843cbf536bf46662d96ea2dd4b4bfa8b3567481a1a5d51e2f4a7d6974b57"}},"normalization_sha256":"2ce6c3ab6722269b97b6ae61e978d219c0ee8d3d56c471cba0b3d85188a64da1","runtime_manifest_sha256":"ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7","sample_order_sha256":"8368c1854402320e687cb5cf48f2819981e797a6cd2aac7e795bf8af25029aea","source_witness_sha256":"802abe311e2c23009cb176c28307f88f648b7f7b033f77b82e78ac53aa30f0aa"},"elapsed_s":7.8497310819948325,"mode":"train","optimizer_updates":200,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1497594,"role":"checkpoint","sha256":"2e5c789c4583791f7d472c93952ac6fe937a1f32f3d0bf06e81daaf6517f889c"},"checkpoint-020.npz":{"bytes":1497594,"role":"checkpoint","sha256":"3c5cc9d1b6017f2bb249889c9483d4401a04ee485a4365ff103189edecd706d5"},"checkpoint-040.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9e9dddcfa3ce3fa7bce918c7f32214130539709601148bbe8bc8645b6fa9744b"},"checkpoint-060.npz":{"bytes":1497594,"role":"checkpoint","sha256":"7af82d108cff562c390456da3909fd59d51d3d9d02b7348502d79855707dfde2"},"checkpoint-080.npz":{"bytes":1497594,"role":"checkpoint","sha256":"6d685258146ad2585ff75f7514ad1a7b356781456bde62ad70ce79add2cd157d"},"checkpoint-100.npz":{"bytes":1497594,"role":"checkpoint","sha256":"76b7bad13184a11746bf0de7b3414a3bec09f13b05125f1fcfc742032f02ea9d"},"checkpoint-120.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9eb7744e9a6811b7ceca44ccd338c13952283b5cf402389e7f51f34b94374f0a"},"checkpoint-140.npz":{"bytes":1497594,"role":"checkpoint","sha256":"95abad944de5b8baede0dc1d5042a1a297cb2da01bb7dc5b9d29745127d5fad7"},"checkpoint-160.npz":{"bytes":1497594,"role":"checkpoint","sha256":"91cb3bb5171be6e5743c076be8d7ce3deb6cc6a22d0bfb706fab6200443d671b"},"checkpoint-180.npz":{"bytes":1497594,"role":"checkpoint","sha256":"6014478a3879bd12193bd844d137b0571d45fbc4f81fa07f1fdb7861fe739f86"},"checkpoint-200.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9a9593165d428fcb4b4353176d67ca697d75bc1e7c9fc52cf08cf224524b6944"},"history.json":{"bytes":64847,"role":"training_diagnostics","sha256":"b2ea1233a920da96e35b335c913a619e36520c8acbf6813e4751752bb0ca2e0b"},"normalization.json":{"bytes":2230,"role":"preprocessing","sha256":"2ce6c3ab6722269b97b6ae61e978d219c0ee8d3d56c471cba0b3d85188a64da1"},"predictions.npz":{"bytes":196534,"role":"predictions","sha256":"6210884d991ce4a6136342b0264420fd03f3b21679b23b6320d14887a65e4b9d"},"splits.json":{"bytes":122850,"role":"training_splits","sha256":"6925843cbf536bf46662d96ea2dd4b4bfa8b3567481a1a5d51e2f4a7d6974b57"},"summary.json":{"bytes":257,"role":"result_summary","sha256":"64677e84a184aa3676809e37141433da4453a59dd78e80f3285ab057e9dcb828"},"weights.npz":{"bytes":1497594,"role":"checkpoint","sha256":"9a9593165d428fcb4b4353176d67ca697d75bc1e7c9fc52cf08cf224524b6944"}},"per_case_inference_s":[0.002336431003641337,0.001964531998964958,0.0016904910007724538,0.0016079519991762936,0.0018355250067543238,0.0018831280031008646,0.007503143002395518,0.007353879002039321,0.004516897999565117,0.004263825001544319,0.007282728009158745,0.006681470011244528],"prediction_inventory":{"case_000":{"bytes":4096,"dtype":"<f8","sha256":"edcd71f48d510345a423ff8f3f952f38ecb33a354eba5cbd6bbf243f20b8e696","shape":[8,8,8]},"case_001":{"bytes":4096,"dtype":"<f8","sha256":"f4887ae0cd9029d498b6b1d36df6acef744799ac570dba81b092970c1a942999","shape":[8,8,8]},"case_002":{"bytes":2520,"dtype":"<f8","sha256":"92df6358820d90f967546738fd470208dc60f72482be32a7eff1641097eb51be","shape":[5,7,9]},"case_003":{"bytes":2520,"dtype":"<f8","sha256":"129e0eaf55efcc2eab26a3b057a7d8f21a75775da8f2c7f532dbeced70bdfc38","shape":[5,7,9]},"case_004":{"bytes":4608,"dtype":"<f8","sha256":"14660979044d533bee59fde761d528babcfc624ed92c37ec90f5b7d769e02913","shape":[6,8,12]},"case_005":{"bytes":4608,"dtype":"<f8","sha256":"decfa9db24f0f9d15c44da8dbae2b27614e65bdbc480942289bb46665de51e8c","shape":[6,8,12]},"case_006":{"bytes":32768,"dtype":"<f8","sha256":"e9e0aaaf71bddbaeb471391ea5c3a791ee429fa9498bcf91c1244a93b92488a7","shape":[16,16,16]},"case_007":{"bytes":32768,"dtype":"<f8","sha256":"b0ccc019ceab7a601c5e5f1e20393a3f8380abcb398ce71269c86d5b3c800608","shape":[16,16,16]},"case_008":{"bytes":15912,"dtype":"<f8","sha256":"48f71d44800057ca0fc639dc14069c12d701dfc3097d2d1f3badf0d6444fa6c5","shape":[9,13,17]},"case_009":{"bytes":15912,"dtype":"<f8","sha256":"459417b5fe6f83e7385ef9fe12bf5fb02b2a848507928a754ed54e859f715d33","shape":[9,13,17]},"case_010":{"bytes":36864,"dtype":"<f8","sha256":"63b7aa53bdddc4b8b319f1227521bf11731330cad26ff82e6455d16a7632247a","shape":[12,16,24]},"case_011":{"bytes":36864,"dtype":"<f8","sha256":"62962e12f8dcbbc6f2786570f531509a35b00c7337027a9e1b83611afad9b997","shape":[12,16,24]}},"runtime":{"interop_threads":1,"numpy":"2.3.5","peak_rss_bytes":467050496,"threads":1,"torch":"2.8.0+cpu"},"schema":"rheon-variable-pressure-training-v2","status":"complete"}
```
