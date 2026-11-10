# Variable-grid train

Retrospective source review · fit-0a492f0abed283ef

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** An additive pressure-correction CNN with 372,377 float32 parameters in 38 tensors. Four encoder widths 8/16/24/32, bottleneck 48 and decoder widths 32/24/16/8; two 3x3x3 convolutions per block, affine-free four-group GroupNorm and SiLU, a bottleneck global-mean broadcast and nearest-to-skip upsampling. The network consumes ten inputs: residual, previous pressure, physical XYZ coordinates, three cubic-spacing maps, rho and dt. Its linear 1x1x1 head is scaled to physical pressure, gauged by the first correction cell and added to the previous pressure. FV arithmetic and the pressure output remain float64. All network parameters are optimized. The captured input budget is 65,536 cells. This operation proposes pressure for one request; it does not advance the physical simulation.

**Purpose.** Test one pressure-correction CNN across rectangular cubic-cell grids and two physical spacings, using spatial and physical metadata channels.

**Experiment.** 200 Adam updates at learning rate 0.001, batch size 2 and seed 20261015, with gradient clipping. Four training trajectories supply 96 pressure frames; two validation trajectories supply 48 frames. The shared 9,296,002-byte dataset covers 8 and 16 cells per metre. Training grids are 8x8x8, 9x7x5, 16x16x16 and 17x13x9; validation grids are 12x8x6 and 24x16x12. The recorded forcing labels are steady and moving, with amplitude 0.5 m/s2 and a 1/60 s timestep. The objective equally weights normalized squared pressure-field and FV equation-residual errors. Train-only scales and the exact PCG64 draw sequence are bound. The recipe supplies a final-update candidate and does not retain an optimizer-resume state.

**Observed outcome.** The original final logged training loss is 0.032813642949730951 at update 200. The existing 128-request GPU comparison gives worst mean-free pressure relative L2 0.29085258217051368 and equation relative L2 0.47514673942070834. The training objective and common field errors are different quantities. This source review adds no inference timing or own-history drift measurement.

**Lesson.** This establishes an early variable-grid baseline with a small physical scene population and substantial common-request pressure/equation error. Shape support and metadata channels alone do not establish broad accuracy. The timestep channel was trained at one physical timestep, so wider timestep generalization remains unqualified.

**Reproduction.** Use the captured project manifest and training recipe, exact source witness, config, runtime, normalization and two typed data slots. Preserve the initial seed inventory and final weight hash. The population metadata retains native binary identity 5bc09317f6296ed170c68a53cca8915c79f3bc841556d1876d6ef6498ab1d807. Its six capture specifications and frame hashes describe the original data. Clean training/data regeneration and conversion to the new Rheon have not been executed or qualified. The original CPU-only contract remains historical evidence; future execution requires a qualified GPU route under the current no-CPU-neural-forward policy.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/receipt.json",
    "sha256": "0a492f0abed283ef87f591438a618cb6b03e14016fd6762aa1a14e78617ca4f0"
  },
  "history": {
    "path": "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/history.json",
    "sha256": "5e2d478d3a661e3e6fe724e4e900e70cf5c745cb6d9203adfceaa538db1e2ed0"
  },
  "sources": [
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/config.json",
      "sha256": "6611e84d31b033bde115adac06b95cb2c3623650665856d449d4ad5d7d9c5ea7"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/data.py",
      "sha256": "fdc920229558ed439d2b6398a934a7f1ce1d59ca747c2e1a256153e609e96d29"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/frozen_gn.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/model.py",
      "sha256": "97b857d67a58040d2b775d5117aa7a6da390c30bb88a64af274aca1467cca085"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/policy.py",
      "sha256": "1403a8d0d58260e7b5224a35412f5523ba158e30e45e31d750a8b536bd258ceb"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/qualification.json",
      "sha256": "d86c10da3899bc75bec6c84f8c41ea97febc1556394015c7df0a79fea6780533"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/train.py",
      "sha256": "9ba3b8a5bdba2bfae694b2ab2952a4cd77c3307cb1e3b1110cbf4f656463e77a"
    }
  ],
  "source_witness": {
    "path": "runs/variable-pressure-v1-workspace-1/project/source-witness.json",
    "sha256": "e9b0b5d2b54c638a5ad84a4dcf96afa46c2b9d715a626aa199575395427f6916"
  },
  "training_population": {
    "path": "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/splits.json",
    "sha256": "8b471d5ce3e15e2d402a8b52f30318f22244f7a458ee082dbc8776c37d2ec026"
  },
  "training_inputs": [
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/inputs/dataset.npz",
      "sha256": "743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f",
      "slot": "dataset",
      "bytes": 9296002
    },
    {
      "path": "runs/variable-pressure-v1-workspace-1/project/inputs/splits.json",
      "sha256": "8b471d5ce3e15e2d402a8b52f30318f22244f7a458ee082dbc8776c37d2ec026",
      "slot": "splits",
      "bytes": 122923
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-0a492f0abed283ef",
    "kind": "training_attempt",
    "name": "Variable-grid train",
    "task": "pressure_proposal",
    "schema": "rheon-variable-pressure-training-v1",
    "method": {
      "schema": "rheon-cubic-groupnorm-model-v2",
      "input_channels": 10,
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
    "receipt_sha256": "0a492f0abed283ef87f591438a618cb6b03e14016fd6762aa1a14e78617ca4f0",
    "receipt_paths": [
      "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/receipt.json",
      "runs/variable-pressure-v1-workspace-1/execution/stage-00/artifacts/receipt.json",
      "runs/variable-pressure-v1-workspace-1/execution/redownloaded-stage-00/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-01/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-02/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-rehearsal-03/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/f94e2a345cb14cdf8672365dbe8d615f/receipt.json",
      "runs/dependency-migration-v1/completed-current-backup-04/catalog/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-02/original-archive/catalog/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-03/original-archive/catalog/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/domain-stage-04/original-archive/catalog/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json",
      "runs/dependency-migration-v1/domain-original-stage-01/original-archive/catalog/workspace-runs/.external-f94e2a345cb14cdf8672365dbe8d615f/output/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "f94e2a345cb14cdf8672365dbe8d615f",
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
      "experiment_sha256": "6611e84d31b033bde115adac06b95cb2c3623650665856d449d4ad5d7d9c5ea7",
      "inputs": {
        "dataset": {
          "bytes": 9296002,
          "sha256": "743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f"
        },
        "splits": {
          "bytes": 122923,
          "sha256": "8b471d5ce3e15e2d402a8b52f30318f22244f7a458ee082dbc8776c37d2ec026"
        }
      },
      "normalization_sha256": "dbd4d3925aed958f186dcdd1912b25367174c49546fe546847b2ad12016372b3",
      "runtime_manifest_sha256": "ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7",
      "sample_order_sha256": "8368c1854402320e687cb5cf48f2819981e797a6cd2aac7e795bf8af25029aea",
      "source_witness_sha256": "e9b0b5d2b54c638a5ad84a4dcf96afa46c2b9d715a626aa199575395427f6916"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "44bca839dafcc0c44495de65b10eaa1fe126b029f04014229019d932416427f8",
        "bytes": 1500186,
        "stage": "initial"
      },
      {
        "name": "checkpoint-020.npz",
        "sha256": "0a22fc65bb3241a38d1b9e572855cc5b533332d1719eab4cc0757c60272e168a",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-040.npz",
        "sha256": "c3d04e8dcca598646e0564a061efaf8bd9368b739a94a9482559bbc79635313a",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-060.npz",
        "sha256": "804369db83cd9cc9e1a36899f6af99a63bddd4220bd28499f6209dc3ff3b5558",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-080.npz",
        "sha256": "deb32802da7ac71ac8a075176b3d163363b38071bd029b77c4f04876fcd0ec90",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "00a204e8c3ad286d91f79baf2dcfd3db9892ca53280ab368a4ba0834cdd3f287",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-120.npz",
        "sha256": "d62bba5d8c178509e91fc30a8f39711b1127e1a4975b15a8f7ca7d3622cf9b3c",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-140.npz",
        "sha256": "82e6b706ce8dc3e40a9087a16d45cc950b537426b424eb6fe464fdff63500716",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-160.npz",
        "sha256": "04337d1c69100cbe6e0fa7901bde6d63912c19d38ff2e94639c5a3fc40ac5e72",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-180.npz",
        "sha256": "31a703da87dbaea71db263a6b3e83ab51bb7f358a72c7c91282e4857f7d49df3",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-200.npz",
        "sha256": "eec1afd3da7dbe1e15b43c6b8d7cdffa17db1431307156254f1fefee440cc99f",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "eec1afd3da7dbe1e15b43c6b8d7cdffa17db1431307156254f1fefee440cc99f",
        "bytes": 1500186,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "interop_threads": 1,
      "numpy": "2.3.5",
      "peak_rss_bytes": 466759680,
      "threads": 1,
      "torch": "2.8.0+cpu"
    },
    "named_entries": [
      "cohorts/2"
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
      "receipt": "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/variable-pressure-v1-workspace-1/execution/reopened-stage-00/receipt.json",
    "sha256": "0a492f0abed283ef87f591438a618cb6b03e14016fd6762aa1a14e78617ca4f0",
    "bytes": 7662
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C03": {
        "name": "Variable grid · 10 channels",
        "weights_sha256": "eec1afd3da7dbe1e15b43c6b8d7cdffa17db1431307156254f1fefee440cc99f",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.47514673942070834,
          "pressure_mean_free_relative_l2": 0.2908525821705137,
          "pressure_absolute_rms": 19.830196821886627,
          "pressure_absolute_max": 73.27429845970872
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
              "common_equation_relative_l2": 0.3723240305527547,
              "pressure_mean_free_relative_l2": 0.20042359650757968,
              "pressure_absolute_rms": 19.830196821886627,
              "pressure_absolute_max": 73.27429845970872
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.47514673942070834,
              "pressure_mean_free_relative_l2": 0.2908525821705137,
              "pressure_absolute_rms": 5.993390343558238,
              "pressure_absolute_max": 15.096134465122182
            }
          }
        },
        "execution_variant": "legacy_CPU_features_CUDA_CNN_CPU_pressure",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C03/results.json",
          "sha256": "034569d9de204fcf623049a395275945a7e275f3f305a17e43079dff8681cdaf"
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
{"architecture":{"activation":"silu","block_normalization":{"affine":false,"epsilon":1e-05,"groups":4,"name":"GroupNorm","position":"each block convolution before SiLU"},"bottleneck_width":48,"convolutions_per_block":2,"coordinates":"cell centres in metres; unit scale 1 m","counts_supported":"positive per-axis integers within cell budget","decoder_widths":[32,24,16,8],"dtype":"float32","encoder_widths":[8,16,24,32],"gauge":"subtract_first_correction_cell","global_path":"add_spatial_mean_broadcast","head":"linear_1x1x1","input_channels":10,"kernel":[3,3,3],"max_cells":65536,"padding":"replicate","physical_cells":"one shared spacing on X/Y/Z","pool":"per-axis average, kernel=min(2,count), ceil, exclude padding","pressure_dtype":"float64","schema":"rheon-cubic-groupnorm-model-v2","upsampling":"nearest_to_skip_shape"},"cases":[{"counts_xyz":[8,8,8],"frame":0,"parent":0,"spacing_m":[0.125,0.125,0.125],"split":"train","trajectory_id":"variable-train-510000"},{"counts_xyz":[8,8,8],"frame":23,"parent":0,"spacing_m":[0.125,0.125,0.125],"split":"train","trajectory_id":"variable-train-510000"},{"counts_xyz":[9,7,5],"frame":0,"parent":1,"spacing_m":[0.125,0.125,0.125],"split":"train","trajectory_id":"variable-train-510001"},{"counts_xyz":[9,7,5],"frame":23,"parent":1,"spacing_m":[0.125,0.125,0.125],"split":"train","trajectory_id":"variable-train-510001"},{"counts_xyz":[12,8,6],"frame":0,"parent":2,"spacing_m":[0.125,0.125,0.125],"split":"validation","trajectory_id":"variable-validation-510002"},{"counts_xyz":[12,8,6],"frame":23,"parent":2,"spacing_m":[0.125,0.125,0.125],"split":"validation","trajectory_id":"variable-validation-510002"},{"counts_xyz":[16,16,16],"frame":0,"parent":3,"spacing_m":[0.0625,0.0625,0.0625],"split":"train","trajectory_id":"variable-train-510100"},{"counts_xyz":[16,16,16],"frame":23,"parent":3,"spacing_m":[0.0625,0.0625,0.0625],"split":"train","trajectory_id":"variable-train-510100"},{"counts_xyz":[17,13,9],"frame":0,"parent":4,"spacing_m":[0.0625,0.0625,0.0625],"split":"train","trajectory_id":"variable-train-510101"},{"counts_xyz":[17,13,9],"frame":23,"parent":4,"spacing_m":[0.0625,0.0625,0.0625],"split":"train","trajectory_id":"variable-train-510101"},{"counts_xyz":[24,16,12],"frame":0,"parent":5,"spacing_m":[0.0625,0.0625,0.0625],"split":"validation","trajectory_id":"variable-validation-510102"},{"counts_xyz":[24,16,12],"frame":23,"parent":5,"spacing_m":[0.0625,0.0625,0.0625],"split":"validation","trajectory_id":"variable-validation-510102"}],"closure":{"experiment_sha256":"6611e84d31b033bde115adac06b95cb2c3623650665856d449d4ad5d7d9c5ea7","inputs":{"dataset":{"bytes":9296002,"sha256":"743cf8184e71041f0bf1d09dae9ae6fafa0cd5fcee10fc2301f7d263bcc3fb0f"},"splits":{"bytes":122923,"sha256":"8b471d5ce3e15e2d402a8b52f30318f22244f7a458ee082dbc8776c37d2ec026"}},"normalization_sha256":"dbd4d3925aed958f186dcdd1912b25367174c49546fe546847b2ad12016372b3","runtime_manifest_sha256":"ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7","sample_order_sha256":"8368c1854402320e687cb5cf48f2819981e797a6cd2aac7e795bf8af25029aea","source_witness_sha256":"e9b0b5d2b54c638a5ad84a4dcf96afa46c2b9d715a626aa199575395427f6916"},"elapsed_s":7.986910029998398,"mode":"train","optimizer_updates":200,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1500186,"role":"checkpoint","sha256":"44bca839dafcc0c44495de65b10eaa1fe126b029f04014229019d932416427f8"},"checkpoint-020.npz":{"bytes":1500186,"role":"checkpoint","sha256":"0a22fc65bb3241a38d1b9e572855cc5b533332d1719eab4cc0757c60272e168a"},"checkpoint-040.npz":{"bytes":1500186,"role":"checkpoint","sha256":"c3d04e8dcca598646e0564a061efaf8bd9368b739a94a9482559bbc79635313a"},"checkpoint-060.npz":{"bytes":1500186,"role":"checkpoint","sha256":"804369db83cd9cc9e1a36899f6af99a63bddd4220bd28499f6209dc3ff3b5558"},"checkpoint-080.npz":{"bytes":1500186,"role":"checkpoint","sha256":"deb32802da7ac71ac8a075176b3d163363b38071bd029b77c4f04876fcd0ec90"},"checkpoint-100.npz":{"bytes":1500186,"role":"checkpoint","sha256":"00a204e8c3ad286d91f79baf2dcfd3db9892ca53280ab368a4ba0834cdd3f287"},"checkpoint-120.npz":{"bytes":1500186,"role":"checkpoint","sha256":"d62bba5d8c178509e91fc30a8f39711b1127e1a4975b15a8f7ca7d3622cf9b3c"},"checkpoint-140.npz":{"bytes":1500186,"role":"checkpoint","sha256":"82e6b706ce8dc3e40a9087a16d45cc950b537426b424eb6fe464fdff63500716"},"checkpoint-160.npz":{"bytes":1500186,"role":"checkpoint","sha256":"04337d1c69100cbe6e0fa7901bde6d63912c19d38ff2e94639c5a3fc40ac5e72"},"checkpoint-180.npz":{"bytes":1500186,"role":"checkpoint","sha256":"31a703da87dbaea71db263a6b3e83ab51bb7f358a72c7c91282e4857f7d49df3"},"checkpoint-200.npz":{"bytes":1500186,"role":"checkpoint","sha256":"eec1afd3da7dbe1e15b43c6b8d7cdffa17db1431307156254f1fefee440cc99f"},"history.json":{"bytes":64862,"role":"training_diagnostics","sha256":"5e2d478d3a661e3e6fe724e4e900e70cf5c745cb6d9203adfceaa538db1e2ed0"},"normalization.json":{"bytes":2510,"role":"preprocessing","sha256":"dbd4d3925aed958f186dcdd1912b25367174c49546fe546847b2ad12016372b3"},"predictions.npz":{"bytes":196534,"role":"predictions","sha256":"7977b7e2ec95982b7532a996490c43a4c85954eb49c47271a3286e121f9b1fe4"},"splits.json":{"bytes":122923,"role":"training_splits","sha256":"8b471d5ce3e15e2d402a8b52f30318f22244f7a458ee082dbc8776c37d2ec026"},"summary.json":{"bytes":259,"role":"result_summary","sha256":"c60fd32ff34c6a0f0edcff20fb4bec28b14ea3e5d119c1fbe0918a92bf35f003"},"weights.npz":{"bytes":1500186,"role":"checkpoint","sha256":"eec1afd3da7dbe1e15b43c6b8d7cdffa17db1431307156254f1fefee440cc99f"}},"per_case_inference_s":[0.001976925996132195,0.0017898119986057281,0.0015220380009850487,0.0014220980083337054,0.001608673992450349,0.0018125300121027976,0.0070678479969501495,0.006711314999847673,0.004186373989796266,0.004109793997486122,0.006955832010135055,0.0068265180016169325],"prediction_inventory":{"case_000":{"bytes":4096,"dtype":"<f8","sha256":"42915ec4d575e9e91afb37eab7ba7c5435ae7089f7552cbe96fd0b6e49c2936f","shape":[8,8,8]},"case_001":{"bytes":4096,"dtype":"<f8","sha256":"80391c2ce2d7dbd3ff7509d73a2755ca9711fdaa84bbcaae33dacc66a9697f0a","shape":[8,8,8]},"case_002":{"bytes":2520,"dtype":"<f8","sha256":"d13a9580a4103c5b2b62762887f95b2eb21e68d9b6653631e8c448a66f4503d5","shape":[5,7,9]},"case_003":{"bytes":2520,"dtype":"<f8","sha256":"7b3ba003f501898e63c116edde1ec06320de53f079a35defd1bc301abbe38403","shape":[5,7,9]},"case_004":{"bytes":4608,"dtype":"<f8","sha256":"3d0e59771b6b50826c613057f91a20c2f93c2afa09a8fc9e56aa552eb4974e25","shape":[6,8,12]},"case_005":{"bytes":4608,"dtype":"<f8","sha256":"b39f761b78d6621569a3597234fabbd92a34f9205300ae0484b7b7657871ddeb","shape":[6,8,12]},"case_006":{"bytes":32768,"dtype":"<f8","sha256":"badeed9b684f47688d81df8abf851b8962fa90f2e03ed3013be2e18002b77019","shape":[16,16,16]},"case_007":{"bytes":32768,"dtype":"<f8","sha256":"f15fc6e1848d15a21ed7350c256fb94c7f49b3c68145d5c7b0ee58830e6187bb","shape":[16,16,16]},"case_008":{"bytes":15912,"dtype":"<f8","sha256":"f90a0f28db98d57c102cd667015f2a0acd839d67829c4225834ecd06bb86996e","shape":[9,13,17]},"case_009":{"bytes":15912,"dtype":"<f8","sha256":"7390423fcd5587f0cd367092c4c60629322999a47b5b012f755ac3082bb45140","shape":[9,13,17]},"case_010":{"bytes":36864,"dtype":"<f8","sha256":"8889d1ea4b8e939f0e342bb43ace9a0031ff1f3c271b632eab090979c0cd903f","shape":[12,16,24]},"case_011":{"bytes":36864,"dtype":"<f8","sha256":"09536404aaab6530c861c6f9ddfc79aea86ac4f52fdab1068541ba64e717454c","shape":[12,16,24]}},"runtime":{"interop_threads":1,"numpy":"2.3.5","peak_rss_bytes":466759680,"threads":1,"torch":"2.8.0+cpu"},"schema":"rheon-variable-pressure-training-v1","status":"complete"}
```
