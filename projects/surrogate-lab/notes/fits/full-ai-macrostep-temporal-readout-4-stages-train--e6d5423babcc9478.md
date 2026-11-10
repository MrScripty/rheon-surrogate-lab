# Full-AI macrostep temporal readout 4 stages train

Retrospective source review · fit-e6d5423babcc9478

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 12-parameter float64 affine temporal readout with three coefficient groups: pressure, shared vx/vy/vz, and tracer. Each group combines differences between the current/intermediate states and the last of 4 frozen internal advances. Zero coefficients initially return the final backbone state. Pressure is gauge-fixed; velocity/tracer cast to float32. The frozen backbone supplies all spatial transport, forcing, viscosity and pressure work. This fit does not train that backbone.

**Purpose.** Replace eight fine internal advances with 4 larger uniform internal advances for each 1/60 s physical output, then learn a small global temporal combination to correct the endpoint. This is distinct from predicting four future physical frames from one distant endpoint.

**Experiment.** CUDA double LBFGS with 160 outer optimizer calls, max_iter 1 per call, strong-Wolfe search, history 20, and 398 actual closure evaluations; all 160 calls change parameters. Optimize equally averaged scene/window/four-metric squared residuals from full-grid QR factors.148 train scenes at starts 32/96 contribute 296 windows, but only 29 N8 scenes/58windows depend on these coefficients;119 N1 scenes/238windows are fixed controls. Active grids span eight shapes through 64 cubed and 8/10/16/25/50/100 cells per metre. Final optimizer state is saved; no held-out model selection is performed.

**Observed outcome.** First logged objective after optimizer call 1 is 6.7715114181321024e-06; final call 160 is 3.3163564067785536e-07. This is a mean squared training objective, not worst pressure error. Original complete-generator GPU evaluation records 1,018 output rows: all 920 N1 bypass outputs pass and all 98 N8 outputs fail the diagnostic 1e-5 gate. N8 worst equation/pressure/velocity/tracer relative errors are {'equation': 0.011137948941449339, 'pressure': 0.0035705867966021603, 'velocity': 0.0004819809965560842, 'tracer': 0.001190681859851122}. No new replay, independent confirmation, 600-output trajectory or qualified latency is established by this review.

**Lesson.** A very small global affine correction can lower the compressed training loss without repairing reduced-integration field errors. Passing unchanged N1 outputs cannot establish accuracy on N8 cases. Three/four internal stages are not three/four future physical outputs: the runner repeats the endpoint computation sequentially for every K output, feeding its own generated state forward. Training factors describe F64 affine fields before final F32 casts, so deployed fields require separate scoring.

**Reproduction.** Use the exact captured four-source trainer/configuration, static manifest/normalization/runtime/design shard and initial/final typed NPZ artifacts bound here. prepare_macrostep_loss_view_v1.py constructs full-grid QR loss factors from native training fields and the frozen managed backbone; no reference pressure solver runs during deployed prediction. macrostep_student_v1.py restores the learned readout then rounds every coefficient to 2^-40 before GPU evaluation. Small design/output artifacts, selected generation/execution sources and restore parity were freshly checked. The 592 native field archives and complete 1,698-member transitive generation closure were not reread. Clean retraining, new-Rheon regeneration and fresh complete-generator latency remain unqualified; CPU restore copies the original GPU probe without a model forward.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-pair-01/train/receipt.json",
    "sha256": "e6d5423babcc9478bd389fc654e5be0f3137f6161a7dc458f2d2dbaba55e3a24"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-pair-01/train/history.json",
    "sha256": "a87d2f2a0c93adbefeca0bfc1b4b8851ed3a9e650a9bdf4aa5d369b6f6fcd027"
  },
  "sources": [
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/macrostep_readout_v1.py",
      "sha256": "0a62c5c436cc6e4525c0004acdee2b36fac34ca9acbd288c35a4b90dfcb0e2fc"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/train_macrostep_readout_v1.py",
      "sha256": "8692e21c9f79080adb08c6cb6fe083d27b18dbe1f4e6e4de07fe0c926c63d6c5"
    }
  ],
  "training_inputs": [
    {
      "slot": "manifest",
      "path": "runs/full-ai-data-diversity-v1/macrostep-loss-view-01/manifest.json",
      "sha256": "9eac9364ba296075cbb39ff5a5ab073592e9e82797f624eade9574cf38f78cc6",
      "bytes": 390851
    },
    {
      "slot": "normalization",
      "path": "runs/full-ai-data-diversity-v1/macrostep-loss-view-01/normalization.json",
      "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14",
      "bytes": 644
    },
    {
      "slot": "runtime_manifest",
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "bytes": 7989288
    },
    {
      "slot": "part_00",
      "path": "runs/full-ai-data-diversity-v1/macrostep-loss-view-01/readout-s4.npz",
      "sha256": "97587f009b437dfe64cc4e2d14bac0d8887844b1f0385927d2a6c846732bf264",
      "bytes": 34715
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-e6d5423babcc9478",
    "kind": "training_attempt",
    "name": "Full-AI macrostep temporal readout 4 stages train",
    "task": "full_state_readout_component",
    "schema": "rheon-macrostep-readout-managed-result-v1",
    "method": {},
    "receipt_sha256": "e6d5423babcc9478bd389fc654e5be0f3137f6161a7dc458f2d2dbaba55e3a24",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/macrostep-readout-s4-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "72beaf5d2c994bfdbf1b236b8893c52a",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full-AI macrostep temporal readout 4 stages train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 160,
    "parameter_fits": null,
    "declared_parameter_count": 12,
    "training_closure": {
      "config_sha256": "5258b69d38e3b04a29b6d44b5ac36f3f112f6b941071fbbe448cf5847839002f",
      "inputs": {
        "manifest": {
          "bytes": 390851,
          "sha256": "9eac9364ba296075cbb39ff5a5ab073592e9e82797f624eade9574cf38f78cc6"
        },
        "normalization": {
          "bytes": 644,
          "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 34715,
          "sha256": "97587f009b437dfe64cc4e2d14bac0d8887844b1f0385927d2a6c846732bf264"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "macrostep_readout_v1.py": "0a62c5c436cc6e4525c0004acdee2b36fac34ca9acbd288c35a4b90dfcb0e2fc",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "train_macrostep_readout_v1.py": "8692e21c9f79080adb08c6cb6fe083d27b18dbe1f4e6e4de07fe0c926c63d6c5"
      },
      "role": "macrostep_temporal_readout_component"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "9ab98f9b2c036193f957a5bd091f2a42f6459870d9628a74b9f0e8bffd236638",
        "bytes": 382,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "e0971dc66cf863221984f277886a7ca1e2a406231b7652f29cbda4ff55f0d051",
        "bytes": 382,
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
      "receipt": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_readout_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-pair-01/train/receipt.json",
    "sha256": "e6d5423babcc9478bd389fc654e5be0f3137f6161a7dc458f2d2dbaba55e3a24",
    "bytes": 3379
  },
  "bound_training_configuration": {
    "config_sha256": "5258b69d38e3b04a29b6d44b5ac36f3f112f6b941071fbbe448cf5847839002f",
    "graph_id": "eca0063df685425abda86d00fbb7e724",
    "graph_document_sha256": "6c2a6e01bd943b1931559a2401cdfdfb821f35f5b1c87e4b4a42761771c10130",
    "recipe_path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/train-recipe.json",
    "recipe_sha256": "ffb94b6bf031f416f22178a886896a0874e6e7ed4cfe4cde3e3a6b40277f5186",
    "config_path": "runs/full-ai-data-diversity-v1/macrostep-readout-s4-project-01/config.json",
    "configuration": {
      "identifier": "macrostep-readout-s4-01",
      "role": "macrostep_temporal_readout_component",
      "stages": 4,
      "optimizer_calls": 160,
      "updates": 160,
      "objective": "equal_scene_window_four_metric_QR_residual_v1",
      "seed": 2026102126,
      "wall_seconds": 600,
      "uniform_deployment_quantization": 9.094947017729282e-13
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
  "schema": "rheon-macrostep-readout-managed-result-v1",
  "mode": "train",
  "role": "macrostep_temporal_readout_component",
  "optimizer_updates": 160,
  "parameters": 12,
  "trainable_parameters": 12,
  "closure": {
    "config_sha256": "5258b69d38e3b04a29b6d44b5ac36f3f112f6b941071fbbe448cf5847839002f",
    "inputs": {
      "manifest": {
        "bytes": 390851,
        "sha256": "9eac9364ba296075cbb39ff5a5ab073592e9e82797f624eade9574cf38f78cc6"
      },
      "normalization": {
        "bytes": 644,
        "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 34715,
        "sha256": "97587f009b437dfe64cc4e2d14bac0d8887844b1f0385927d2a6c846732bf264"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "macrostep_readout_v1.py": "0a62c5c436cc6e4525c0004acdee2b36fac34ca9acbd288c35a4b90dfcb0e2fc",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "train_macrostep_readout_v1.py": "8692e21c9f79080adb08c6cb6fe083d27b18dbe1f4e6e4de07fe0c926c63d6c5"
    },
    "role": "macrostep_temporal_readout_component"
  },
  "initial_inventory": {
    "residual_weights": {
      "dtype": "<f8",
      "shape": [
        3,
        4
      ],
      "sha256": "2ea9ab9198d1638007400cd2c3bef1cc745b864b76011a0e1bc52180ac6452d4"
    }
  },
  "final_inventory": {
    "residual_weights": {
      "dtype": "<f8",
      "shape": [
        3,
        4
      ],
      "sha256": "b031247bd74b6f55e4a5cdd8b888395b3dae7550c323e8a07b772c7a9b9e5815"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 67587072,
    "all_learning_and_forward_on_cuda": true
  },
  "outputs": {
    "probe.npy": {
      "bytes": 47488,
      "sha256": "e2ba55cf4da271fce415878a168cc5c31652b2fd55577222b60c0439d8172762"
    },
    "weights.npz": {
      "bytes": 382,
      "sha256": "9ab98f9b2c036193f957a5bd091f2a42f6459870d9628a74b9f0e8bffd236638"
    },
    "initial-weights.npz": {
      "bytes": 382,
      "sha256": "e0971dc66cf863221984f277886a7ca1e2a406231b7652f29cbda4ff55f0d051"
    },
    "history.json": {
      "bytes": 6852,
      "sha256": "a87d2f2a0c93adbefeca0bfc1b4b8851ed3a9e650a9bdf4aa5d369b6f6fcd027"
    },
    "normalization.json": {
      "bytes": 644,
      "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
    }
  },
  "originating_training_run": null,
  "typed_cpu_weight_restore_verified": false,
  "full_cpu_probe_bitwise_verified": false,
  "gpu_probe_copied_without_cpu_forward": false,
  "probe_scope": "Supervised compact residual vectors for all148scenes/two windows/four metrics, generated onCUDA. No full-state inference or accuracy claim. CPU restore copies the bound GPU probe without a forward pass.",
  "cpu_model_forward_executed": false,
  "causal_full_state_generator": false,
  "student_stages": 4,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.457237324968446
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "macrostep-readout-s4-01",
  "role": "macrostep_temporal_readout_component",
  "stages": 4,
  "optimizer_calls": 160,
  "updates": 160,
  "objective": "equal_scene_window_four_metric_QR_residual_v1",
  "seed": 2026102126,
  "wall_seconds": 600,
  "uniform_deployment_quantization": 9.094947017729282e-13
}

```
