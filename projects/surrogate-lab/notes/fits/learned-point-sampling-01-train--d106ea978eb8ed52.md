# Learned point sampling 01 train

Retrospective source review · fit-d106ea978eb8ed52

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A single bias-free 3×3×3 convolution maps three velocity channels to 12 sampled point-velocity channels with 972 learned coefficients. The channels encode four destination kinds (three velocity-face types and tracer-cell points), each with xyz components. It learns negative sampled velocity; it does not itself backtrace a particle, advance a carrier, solve pressure or generate a simulation state.

**Purpose.** Replace a repeated local point-velocity sampling operation with a learned linear projection whose coefficients can be reused in displacement-based generators.

**Experiment.** Use 80 manufactured velocity patterns on the padded 13³ lattice: 64 fitting patterns and 16 development patterns, with a native-qualified teacher and explicit valid-output mask. No temporal simulation scenes or reserved scenes enter training. Zero coefficients are optimized for 1,000 CUDA SGD updates in F64, batch eight distinct fitting patterns per update, rate 0.5. PCG64 sampling uses seed 2026102042; its full order is independently reproduced. Training then publishes F32 coefficients. All source/recipe/teacher/output identities and complete initial/final tensors verify. The later generator asset has identical tensor bytes even though its NPZ archive encoding differs.

**Observed outcome.** Original calibration passes its declared diagnostic thresholds. Recorded worst development relative error is 7.6578329608257531e-16 in the training F64 execution and 2.4925103320814926e-08 in the published F32 execution. These are retained original component diagnostics, not new forwards or complete-state accuracy. No new GPU timing or trajectory drift is measured here.

**Lesson.** A tiny operator calibration error on manufactured patterns does not certify general fluid-scene accuracy or accumulated drift. Keep training F64 and published F32 arithmetic separate. A later frozen composition inherits the coefficient values and calibration cost, not a full-state training result. Equal tensors in differently encoded NPZ archives are a value-equivalence observation, not identical artifact bytes.

**Reproduction.** Use the exact captured recipe/config/model/trainer/runtime, native-qualified teacher archive/manifest and its masks/split, zero initialization, private PCG64 order and F64 SGD followed by F32 publication. Preserve the original native build and qualification provenance for old-reference reproduction. The old trainer evaluates the published model and final probe on CPU; qualify GPU-only publication/restore before rerunning. Clean retraining, new-Rheon teacher regeneration and complete-generator validation remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/point-sampling-pair-01/train/receipt.json",
    "sha256": "d106ea978eb8ed52a3ea29f30527cee82a2b7328ba1a3e33a3b1d2242bdd78ca"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/point-sampling-pair-01/train/history.json",
    "sha256": "df9ba0bcff1a75123571cbdce35e4ad4585f621187caab36aff41440ccf12019"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/point_sampling_model.py",
      "sha256": "c0456f7f1f19ca2a55cc0bd89734fa893bbcd055cfc9fb561c772aacbb6bf2f3"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/point_sampling_train.py",
      "sha256": "60a4beab25a1760d37dd9076108e5f59660f9022a5e1cafe660d8021546315e4"
    }
  ],
  "training_inputs": [
    {
      "bytes": 9834127,
      "path": "runs/full-ai-next-priorities-v1/point-teacher-01/teacher.npz",
      "sha256": "a10807822eb806b9e14e129c166e0b4350348afd5fca0b0dee27d2a37555d790",
      "slot": "teacher"
    },
    {
      "bytes": 5295,
      "path": "runs/full-ai-next-priorities-v1/point-teacher-01/manifest.json",
      "sha256": "e368e81cb751f93d6066c29e01928ebc4a9c2b9e3ce723c24d6ad1d7eca7b023",
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
    "item_id": "fit-d106ea978eb8ed52",
    "kind": "training_attempt",
    "name": "Learned point sampling 01 train",
    "task": "transport_component",
    "schema": "rheon-point-sampling-fit-v1",
    "method": {},
    "receipt_sha256": "d106ea978eb8ed52a3ea29f30527cee82a2b7328ba1a3e33a3b1d2242bdd78ca",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/point-sampling-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "b71b064b725f4937ae2e12a1e59bce8c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Learned point sampling 01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1000,
    "parameter_fits": null,
    "declared_parameter_count": 972,
    "training_closure": {
      "config_sha256": "362ec3d175d1c3162173e64046e914066c1d6ec128bab161505f29013003ce6a",
      "inputs": {
        "teacher": {
          "bytes": 9834127,
          "sha256": "a10807822eb806b9e14e129c166e0b4350348afd5fca0b0dee27d2a37555d790"
        },
        "teacher_manifest": {
          "bytes": 5295,
          "sha256": "e368e81cb751f93d6066c29e01928ebc4a9c2b9e3ce723c24d6ad1d7eca7b023"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "point_sampling_model.py": "c0456f7f1f19ca2a55cc0bd89734fa893bbcd055cfc9fb561c772aacbb6bf2f3",
        "point_sampling_train.py": "60a4beab25a1760d37dd9076108e5f59660f9022a5e1cafe660d8021546315e4"
      }
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635",
        "bytes": 4176,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "698a3f03a0ebe8ce04a7cc4436c743df4bffcec2c916094128aa544cb3a50701",
        "bytes": 4176,
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
      "receipt": "runs/full-ai-next-priorities-v1/point-sampling-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "transport_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/point-sampling-pair-01/train/receipt.json",
    "sha256": "d106ea978eb8ed52a3ea29f30527cee82a2b7328ba1a3e33a3b1d2242bdd78ca",
    "bytes": 2883
  },
  "bound_training_configuration": {
    "config_sha256": "362ec3d175d1c3162173e64046e914066c1d6ec128bab161505f29013003ce6a",
    "graph_id": "b292b6fe4f504956b5b13b684eea36b5",
    "graph_document_sha256": "cdcae689234cce5b41907f119f81e7ae016d786e443b7aba6d110dfd90562d81",
    "recipe_path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/train-recipe.json",
    "recipe_sha256": "9178586c60d7116d886775b329cb7f80becf7cfa8b67aca10fa6d089ab2fa942",
    "config_path": "runs/full-ai-next-priorities-v1/point-sampling-project-01/config.json",
    "configuration": {
      "schema": "rheon-point-sampling-calibration-v1",
      "updates": 1000,
      "batch_size": 8,
      "learning_rate": 0.5,
      "seed": 2026102041,
      "sampling_seed": 2026102042,
      "wall_seconds": 1500,
      "double_tolerance": 1e-07,
      "published_tolerance": 1e-06,
      "identifier": "point-sampling-01",
      "registration_sha256": "43c24eb7bee48fa377125210f3d1fd8e792b831bb2bd46dcfd8e8e3fd390696f"
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
  "schema": "rheon-point-sampling-fit-v1",
  "status": "complete",
  "mode": "train",
  "optimizer_updates": 1000,
  "parameters": 972,
  "trainable_parameters": 972,
  "task": "learned_point_sampling",
  "closure": {
    "config_sha256": "362ec3d175d1c3162173e64046e914066c1d6ec128bab161505f29013003ce6a",
    "inputs": {
      "teacher": {
        "bytes": 9834127,
        "sha256": "a10807822eb806b9e14e129c166e0b4350348afd5fca0b0dee27d2a37555d790"
      },
      "teacher_manifest": {
        "bytes": 5295,
        "sha256": "e368e81cb751f93d6066c29e01928ebc4a9c2b9e3ce723c24d6ad1d7eca7b023"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "point_sampling_model.py": "c0456f7f1f19ca2a55cc0bd89734fa893bbcd055cfc9fb561c772aacbb6bf2f3",
      "point_sampling_train.py": "60a4beab25a1760d37dd9076108e5f59660f9022a5e1cafe660d8021546315e4"
    }
  },
  "initial_inventory": {
    "projection.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "d60d6cd6fbd560cb387c6294aeab1e088bc4dbf4b85b8341702120cfd5fcde32"
    }
  },
  "final_inventory": {
    "projection.weight": {
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "db727a29917fae433efcc201f7771f7220b82090c4944346f3551e723cc9d3b1"
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
    "peak_cuda_allocated_bytes": 31443456
  },
  "outputs": {
    "probe.npy": {
      "bytes": 211040,
      "sha256": "d6cae1e6e44232080cff46ea0340070ae35ddf66be8f5a6c728231c991651b4c"
    },
    "weights.npz": {
      "bytes": 4176,
      "sha256": "88446b9820c746a458f3fc5da008e1de16822c63c5aaed130088962b9bfc9635"
    },
    "initial-weights.npz": {
      "bytes": 4176,
      "sha256": "698a3f03a0ebe8ce04a7cc4436c743df4bffcec2c916094128aa544cb3a50701"
    },
    "history.json": {
      "bytes": 2437,
      "sha256": "df9ba0bcff1a75123571cbdce35e4ad4585f621187caab36aff41440ccf12019"
    },
    "normalization.json": {
      "bytes": 127,
      "sha256": "50cc8aa7011be683851a7fb00ebd233e7d2aa948f5dbfde12d6dcb2712e29451"
    }
  },
  "originating_training_run": null,
  "full_cpu_probe_bitwise_verified": false,
  "elapsed_s": 4.898952161020134,
  "numerical_pressure_solves": 0,
  "native_sampling_at_candidate_inference": false,
  "full_state_forecast": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-point-sampling-calibration-v1",
  "updates": 1000,
  "batch_size": 8,
  "learning_rate": 0.5,
  "seed": 2026102041,
  "sampling_seed": 2026102042,
  "wall_seconds": 1500,
  "double_tolerance": 1e-07,
  "published_tolerance": 1e-06,
  "identifier": "point-sampling-01",
  "registration_sha256": "43c24eb7bee48fa377125210f3d1fd8e792b831bb2bd46dcfd8e8e3fd390696f"
}

```
