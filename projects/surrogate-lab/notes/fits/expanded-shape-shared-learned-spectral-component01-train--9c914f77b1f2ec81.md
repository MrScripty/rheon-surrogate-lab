# Expanded shape-shared learned spectral component01 train

Retrospective source review · fit-9c914f77b1f2ec81

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** Four learned float64 coefficients map the fixed features [1, cos(pi*x/nx), cos(pi*y/ny), cos(pi*z/nz)] to a shape-shared modal response. Pressure decoding applies fixed cosine transforms, divides by that learned response, removes the constant mode and fixes the pressure gauge. This is a learned spectral response, rather than the C107 pressure CNN. Generic transform/feature operations are prescribed; response coefficients come from training.

**Purpose.** Replace shape-specific learned modal tables with one compact response shared across grid shapes. Keep this pressure component distinct from a complete fluid-state generator and from its implementation/precision variants.

**Experiment.** 160 CUDA LBFGS step calls over all148 training scenes, with an equal-scene native-response residual objective. The history records12 parameter-changing steps and328 closure evaluations. Development/reserved examples were excluded by the captured training-view contract.

**Observed outcome.** The final recorded training objective is 2.9623043795351706e-18. It measures the component objective on its training view; it is not full-state pressure-field error, equation residual or rollout drift. The typed CPU restore copied the originating CUDA probe without a CPU forward. Later complete generators use these weights, but their accuracy/timing belongs to each complete execution contract.

**Lesson.** A small learned component can exploit a fixed modal representation. Its low training objective does not establish arbitrary geometry/boundary support or make reduced internal time integration equally accurate. Compare complete outputs and actual advances under the same reference schedule before attributing a generator improvement to this fit.

**Reproduction.** The exact original receipt, training configuration and source/data hashes remain embedded below. Retain the selected and initial weights plus typed restore records. A copied GPU probe is not a CPU inference test, and retained hashes are not a successful clean retraining. Dataset regeneration and execution through the new Tuldok owner still require verification.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/shared-spectral-pair-01/train/receipt.json",
    "sha256": "9c914f77b1f2ec811e269ea9d50157e882883540ef19c9806b130a00f4db7beb"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/shared-spectral-pair-01/train/history.json",
    "sha256": "98548a1e63853e19e6410e01e488794cd122b467025be57d563fccab5d28c030"
  },
  "sources": [
    {
      "path": "experiments/full-ai-data-diversity-v1/shared_spectral_model_v1.py",
      "sha256": "b3ce3854538cbfab75ff17a938c72374e243eea51b0ee23be5550ddc0ef88a23",
      "binding": "/closure/sources/shared_spectral_model_v1.py"
    },
    {
      "path": "experiments/full-ai-data-diversity-v1/train_shared_spectral_v1.py",
      "sha256": "db04a1904dd8543d5ce1d9d6b022e94505049d4a72c22afe59e22a31aa3d0d9e",
      "binding": "/closure/sources/train_shared_spectral_v1.py"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-9c914f77b1f2ec81",
    "kind": "training_attempt",
    "name": "Expanded shape-shared learned spectral component01 train",
    "task": "pressure_component",
    "schema": "rheon-shared-spectral-managed-result-v1",
    "method": {},
    "receipt_sha256": "9c914f77b1f2ec811e269ea9d50157e882883540ef19c9806b130a00f4db7beb",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/shared-spectral-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a529f0706382449391c0a0a60c3b4c43",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Expanded shape-shared learned spectral component01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 160,
    "parameter_fits": null,
    "declared_parameter_count": 4,
    "training_closure": {
      "config_sha256": "ae11b806639c3e01c0049fdeb93707d0a92fc8a920a8a17b114721e207a5c28a",
      "inputs": {
        "manifest": {
          "bytes": 1268195,
          "sha256": "3c79350355a3b4d7d0a7f501d982944b9362b2d3269f43e14798f57c70c049d9"
        },
        "normalization": {
          "bytes": 688,
          "sha256": "d2dba4c0de661230cf8e6c3e22d0f153cd4809616927563dd23d0c78286e74cf"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 10844745,
          "sha256": "eda950d28075625125f0842bc35defb185dd923bd32568ea3b1e2d725d42d830"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "shared_spectral_model_v1.py": "b3ce3854538cbfab75ff17a938c72374e243eea51b0ee23be5550ddc0ef88a23",
        "train_shared_spectral_v1.py": "db04a1904dd8543d5ce1d9d6b022e94505049d4a72c22afe59e22a31aa3d0d9e"
      },
      "role": "shape_shared_spectral_pressure_component"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "1570880b28ed771baf1577a189ddf2951262391a98795550cd4b2e407a7bdc87",
        "bytes": 310,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "76e108023ee0c163c80755201d68fd578c13f729bc34bc4338f9e4448430b698",
        "bytes": 310,
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
      "receipt": "runs/full-ai-data-diversity-v1/shared-spectral-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/shared-spectral-pair-01/train/receipt.json",
    "sha256": "9c914f77b1f2ec811e269ea9d50157e882883540ef19c9806b130a00f4db7beb",
    "bytes": 3276
  },
  "bound_training_configuration": {
    "config_sha256": "ae11b806639c3e01c0049fdeb93707d0a92fc8a920a8a17b114721e207a5c28a",
    "graph_id": "8ef0434774a2477ca36bdae3de09b361",
    "graph_document_sha256": "ff76b75b142b8b8a317da79c097e619397480c083988263b2a97069e9522f1a3",
    "recipe_path": "runs/full-ai-data-diversity-v1/shared-spectral-project-01/train-recipe.json",
    "recipe_sha256": "346c1aeebfbb0861b6b20bd36709bea06a6b7aac89bc431027735d04615cb11a",
    "config_path": "runs/full-ai-data-diversity-v1/shared-spectral-project-01/config.json",
    "configuration": {
      "identifier": "expanded-shared-spectral-01",
      "role": "shape_shared_spectral_pressure_component",
      "objective": "equal_scene_native_response_residual_v1",
      "seed": 2026102071,
      "optimizer_calls": 160,
      "updates": 160,
      "wall_seconds": 600,
      "training_scenes": 148,
      "inference_device": "cuda",
      "cpu_restore": "Exact typed weights only; no CPU forward; originating CUDA probe copied and verified later onCUDA"
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
  "status": "complete",
  "schema": "rheon-shared-spectral-managed-result-v1",
  "mode": "train",
  "role": "shape_shared_spectral_pressure_component",
  "optimizer_updates": 160,
  "parameters": 4,
  "trainable_parameters": 4,
  "closure": {
    "config_sha256": "ae11b806639c3e01c0049fdeb93707d0a92fc8a920a8a17b114721e207a5c28a",
    "inputs": {
      "manifest": {
        "bytes": 1268195,
        "sha256": "3c79350355a3b4d7d0a7f501d982944b9362b2d3269f43e14798f57c70c049d9"
      },
      "normalization": {
        "bytes": 688,
        "sha256": "d2dba4c0de661230cf8e6c3e22d0f153cd4809616927563dd23d0c78286e74cf"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 10844745,
        "sha256": "eda950d28075625125f0842bc35defb185dd923bd32568ea3b1e2d725d42d830"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "shared_spectral_model_v1.py": "b3ce3854538cbfab75ff17a938c72374e243eea51b0ee23be5550ddc0ef88a23",
      "train_shared_spectral_v1.py": "db04a1904dd8543d5ce1d9d6b022e94505049d4a72c22afe59e22a31aa3d0d9e"
    },
    "role": "shape_shared_spectral_pressure_component"
  },
  "initial_inventory": {
    "coefficients": {
      "dtype": "<f8",
      "shape": [
        4
      ],
      "sha256": "b0f20f633b6b924a4a2f9e7d18aea7debc1ee04cf4ef03930c3f110f27ced0af"
    }
  },
  "final_inventory": {
    "coefficients": {
      "dtype": "<f8",
      "shape": [
        4
      ],
      "sha256": "f5ce81cdd1a773d1e2e6da2bcf4a3f98a2ec181060ef4a6b4c121e200f3a1bd0"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 99975168,
    "all_learning_and_forward_on_cuda": true
  },
  "outputs": {
    "probe.npy": {
      "bytes": 4283280,
      "sha256": "d3dc2f98cf57c4339cca1e0bb857c0fe87b0b388bf18db047b1fcd34d5d6814a"
    },
    "weights.npz": {
      "bytes": 310,
      "sha256": "1570880b28ed771baf1577a189ddf2951262391a98795550cd4b2e407a7bdc87"
    },
    "initial-weights.npz": {
      "bytes": 310,
      "sha256": "76e108023ee0c163c80755201d68fd578c13f729bc34bc4338f9e4448430b698"
    },
    "history.json": {
      "bytes": 6844,
      "sha256": "98548a1e63853e19e6410e01e488794cd122b467025be57d563fccab5d28c030"
    },
    "normalization.json": {
      "bytes": 688,
      "sha256": "d2dba4c0de661230cf8e6c3e22d0f153cd4809616927563dd23d0c78286e74cf"
    }
  },
  "originating_training_run": null,
  "typed_cpu_weight_restore_verified": false,
  "full_cpu_probe_bitwise_verified": false,
  "gpu_probe_copied_without_cpu_forward": false,
  "probe_scope": "Learned response on every535394native training feature row, generated onCUDA; CPU restore copies the bound GPU probe without a forward pass.",
  "cpu_model_forward_executed": false,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.172161797992885
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "expanded-shared-spectral-01",
  "role": "shape_shared_spectral_pressure_component",
  "objective": "equal_scene_native_response_residual_v1",
  "seed": 2026102071,
  "optimizer_calls": 160,
  "updates": 160,
  "wall_seconds": 600,
  "training_scenes": 148,
  "inference_device": "cuda",
  "cpu_restore": "Exact typed weights only; no CPU forward; originating CUDA probe copied and verified later onCUDA"
}

```
