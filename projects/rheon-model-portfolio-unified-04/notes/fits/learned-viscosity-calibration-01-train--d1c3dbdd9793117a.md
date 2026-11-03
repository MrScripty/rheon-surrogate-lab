# Learned viscosity calibration 01 train

Retrospective source review · fit-d1c3dbdd9793117a

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A single bias-free 3×3×3 convolution maps three velocity channels to three dimensionless viscous-stiffness channels with 243 learned coefficients. Targets represent the native negative symmetric-strain stiffness on free face degrees of freedom. Physical viscosity/timestep/spacing scaling and state advancement belong to the composing generator.

**Purpose.** Calibrate the local native viscosity operator as a learned projection and freeze those coefficients in later complete generators, separating operator calibration from temporal fitting.

**Experiment.** Use 80 manufactured velocity patterns on the padded 13³ lattice: 64 fitting patterns and 16 development patterns, with a native-qualified teacher and explicit valid-output mask. No temporal simulation scenes or reserved scenes enter training. Zero coefficients are optimized for 1,000 CUDA SGD updates in F64, batch eight distinct fitting patterns per update, rate 0.25. PCG64 sampling uses seed 2026102062; its full order is independently reproduced. Training then publishes F32 coefficients. All source/recipe/teacher/output identities and complete initial/final tensors verify. The later generator asset has identical tensor bytes even though its NPZ archive encoding differs.

**Observed outcome.** Original calibration passes its declared diagnostic thresholds. Recorded worst development relative error is 5.7499849008635439e-16 in the training F64 execution and 3.8777215641808174e-18 in the published F32 execution. These are retained original component diagnostics, not new forwards or complete-state accuracy. No new GPU timing or trajectory drift is measured here.

**Lesson.** A tiny operator calibration error on manufactured patterns does not certify general fluid-scene accuracy or accumulated drift. Keep training F64 and published F32 arithmetic separate. A later frozen composition inherits the coefficient values and calibration cost, not a full-state training result. Equal tensors in differently encoded NPZ archives are a value-equivalence observation, not identical artifact bytes.

**Reproduction.** Use the exact captured recipe/config/model/trainer/runtime, native-qualified teacher archive/manifest and its masks/split, zero initialization, private PCG64 order and F64 SGD followed by F32 publication. Preserve the original native build and qualification provenance for old-reference reproduction. The old trainer evaluates the published model and final probe on CPU; qualify GPU-only publication/restore before rerunning. Clean retraining, new-Rheon teacher regeneration and complete-generator validation remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-pair-01/train/receipt.json",
    "sha256": "d1c3dbdd9793117a7901d1588ab0078131fc5c5afda0c6a613c0441543e2e501"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-pair-01/train/history.json",
    "sha256": "62e9446c022b11483d6bb0783884b66bd0b9c41da1e4bca380cffb121ab8598f"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/viscosity_calibration_model.py",
      "sha256": "32f0b82f0d653f86f4c519564f4a1f19bb9324b9d00410dda063c88aeb7f5c62"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/viscosity_calibration_train.py",
      "sha256": "48395497fd5b939565da4c0020d4e760bf94e92502971c1872159ff029e85cca"
    }
  ],
  "training_inputs": [
    {
      "bytes": 1233001,
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-teacher-01/teacher.npz",
      "sha256": "ee4da23a5cd68d578d013ef5ce5a40aa040536c045752faa918e07c6bce32bbb",
      "slot": "teacher"
    },
    {
      "bytes": 3402,
      "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-teacher-01/manifest.json",
      "sha256": "072681c9dc87b03a01a769e2559a08eee4694766fab06a5ceaf6af33227ec99a",
      "slot": "teacher_manifest"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-d1c3dbdd9793117a",
    "kind": "training_attempt",
    "name": "Learned viscosity calibration 01 train",
    "task": "viscosity_component",
    "schema": "rheon-viscosity-calibration-fit-v1",
    "method": {},
    "receipt_sha256": "d1c3dbdd9793117a7901d1588ab0078131fc5c5afda0c6a613c0441543e2e501",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/viscosity-calibration-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "256e9efaab374dc09d75faf12e081a43",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Learned viscosity calibration 01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1000,
    "parameter_fits": null,
    "declared_parameter_count": 243,
    "training_closure": {
      "config_sha256": "cea083cb2b127955408e15ee49a24e8e006e9d58343d650430c3fc98cc4d9efb",
      "inputs": {
        "teacher": {
          "bytes": 1233001,
          "sha256": "ee4da23a5cd68d578d013ef5ce5a40aa040536c045752faa918e07c6bce32bbb"
        },
        "teacher_manifest": {
          "bytes": 3402,
          "sha256": "072681c9dc87b03a01a769e2559a08eee4694766fab06a5ceaf6af33227ec99a"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "viscosity_calibration_model.py": "32f0b82f0d653f86f4c519564f4a1f19bb9324b9d00410dda063c88aeb7f5c62",
        "viscosity_calibration_train.py": "48395497fd5b939565da4c0020d4e760bf94e92502971c1872159ff029e85cca"
      }
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd",
        "bytes": 1260,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "f523403c0ff1ed6108130b7a9ba83675283254810bcce75ef30a401bee6465fa",
        "bytes": 1260,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [],
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
      "receipt": "runs/full-ai-next-priorities-v1/viscosity-calibration-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "viscosity_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/viscosity-calibration-pair-01/train/receipt.json",
    "sha256": "d1c3dbdd9793117a7901d1588ab0078131fc5c5afda0c6a613c0441543e2e501",
    "bytes": 2909
  },
  "bound_training_configuration": {
    "config_sha256": "cea083cb2b127955408e15ee49a24e8e006e9d58343d650430c3fc98cc4d9efb",
    "graph_id": "37f7826ea39c408b9d0bd068172f3de5",
    "graph_document_sha256": "0eedb5f221c3ee29c7de16d16a9627b12c9826f2ff5eb6a3f1115a30f8230370",
    "recipe_path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/train-recipe.json",
    "recipe_sha256": "4c506c550a938080e6087d699b33e6fba29b469ce6770735a886c0d7e98c99c1",
    "config_path": "runs/full-ai-next-priorities-v1/viscosity-calibration-project-01/config.json",
    "configuration": {
      "schema": "rheon-viscosity-calibration-v1",
      "updates": 1000,
      "batch_size": 8,
      "learning_rate": 0.25,
      "seed": 2026102061,
      "sampling_seed": 2026102062,
      "wall_seconds": 1200,
      "double_tolerance": 1e-08,
      "published_tolerance": 1e-06,
      "identifier": "viscosity-calibration-01",
      "registration_sha256": "702e7ab154f37656e0c09810e878881825f7ee87f9a7901933805313a434476c"
    }
  },
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {},
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
{
  "schema": "rheon-viscosity-calibration-fit-v1",
  "status": "complete",
  "mode": "train",
  "optimizer_updates": 1000,
  "parameters": 243,
  "trainable_parameters": 243,
  "task": "learned_viscosity_coefficients",
  "closure": {
    "config_sha256": "cea083cb2b127955408e15ee49a24e8e006e9d58343d650430c3fc98cc4d9efb",
    "inputs": {
      "teacher": {
        "bytes": 1233001,
        "sha256": "ee4da23a5cd68d578d013ef5ce5a40aa040536c045752faa918e07c6bce32bbb"
      },
      "teacher_manifest": {
        "bytes": 3402,
        "sha256": "072681c9dc87b03a01a769e2559a08eee4694766fab06a5ceaf6af33227ec99a"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "viscosity_calibration_model.py": "32f0b82f0d653f86f4c519564f4a1f19bb9324b9d00410dda063c88aeb7f5c62",
      "viscosity_calibration_train.py": "48395497fd5b939565da4c0020d4e760bf94e92502971c1872159ff029e85cca"
    }
  },
  "initial_inventory": {
    "projection.weight": {
      "shape": [
        3,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "96c5023576733f88726277165f7fa44936426956bcbcca126d7f1169216e1a4b"
    }
  },
  "final_inventory": {
    "projection.weight": {
      "shape": [
        3,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "cc4836fbce947f9956aeacf004ed185042f0fc3fc213b657945a9eef0d7a35d6"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "tf32": false,
    "amp": false,
    "training_dtype": "float64",
    "published_dtype": "float32",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "peak_cuda_allocated_bytes": 11250688
  },
  "outputs": {
    "probe.npy": {
      "bytes": 52856,
      "sha256": "baed44518b24602b6deb0edd0ade843ebbb9cda1da736331afd553f08674c65a"
    },
    "weights.npz": {
      "bytes": 1260,
      "sha256": "f7c76bf1e24ce44d7d4b863140445c30f36a547526168f61917d7505b16953dd"
    },
    "initial-weights.npz": {
      "bytes": 1260,
      "sha256": "f523403c0ff1ed6108130b7a9ba83675283254810bcce75ef30a401bee6465fa"
    },
    "history.json": {
      "bytes": 2438,
      "sha256": "62e9446c022b11483d6bb0783884b66bd0b9c41da1e4bca380cffb121ab8598f"
    },
    "normalization.json": {
      "bytes": 117,
      "sha256": "bb08aafa81f4f0a6839e5918f7230d8793cd9c367b7ca59a9831e15b745b0a51"
    }
  },
  "originating_training_run": null,
  "full_cpu_probe_bitwise_verified": false,
  "elapsed_s": 5.066749070014339,
  "numerical_pressure_solves": 0,
  "native_sampling_at_candidate_inference": false,
  "full_state_forecast": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-viscosity-calibration-v1",
  "updates": 1000,
  "batch_size": 8,
  "learning_rate": 0.25,
  "seed": 2026102061,
  "sampling_seed": 2026102062,
  "wall_seconds": 1200,
  "double_tolerance": 1e-08,
  "published_tolerance": 1e-06,
  "identifier": "viscosity-calibration-01",
  "registration_sha256": "702e7ab154f37656e0c09810e878881825f7ee87f9a7901933805313a434476c"
}

```
