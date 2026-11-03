# Full-AI macrostep temporal readout 3 stages train

Retrospective source review · fit-6eadd58ed412cffc

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 9-parameter float64 affine temporal readout with three coefficient groups: pressure, shared vx/vy/vz, and tracer. Each group combines differences between the current/intermediate states and the last of 3 frozen internal advances. Zero coefficients initially return the final backbone state. Pressure is gauge-fixed; velocity/tracer cast to float32. The frozen backbone supplies all spatial transport, forcing, viscosity and pressure work. This fit does not train that backbone.

**Purpose.** Replace eight fine internal advances with 3 larger uniform internal advances for each 1/60 s physical output, then learn a small global temporal combination to correct the endpoint. This is distinct from predicting four future physical frames from one distant endpoint.

**Experiment.** CUDA double LBFGS with 160 outer optimizer calls, max_iter 1 per call, strong-Wolfe search, history 20, and 413 actual closure evaluations; all 160 calls change parameters. Optimize equally averaged scene/window/four-metric squared residuals from full-grid QR factors.148 train scenes at starts 32/96 contribute 296 windows, but only 29 N8 scenes/58windows depend on these coefficients;119 N1 scenes/238windows are fixed controls. Active grids span eight shapes through 64 cubed and 8/10/16/25/50/100 cells per metre. Final optimizer state is saved; no held-out model selection is performed.

**Observed outcome.** First logged objective after optimizer call 1 is 2.1019028577875506e-05; final call 160 is 9.2549343084186759e-07. This is a mean squared training objective, not worst pressure error. Original complete-generator GPU evaluation records 1,018 output rows: all 920 N1 bypass outputs pass and all 98 N8 outputs fail the diagnostic 1e-5 gate. N8 worst equation/pressure/velocity/tracer relative errors are {'equation': 0.01865609298538165, 'pressure': 0.005945252550651944, 'velocity': 0.0004398758954446882, 'tracer': 0.0019449209596585978}. No new replay, independent confirmation, 600-output trajectory or qualified latency is established by this review.

**Lesson.** A very small global affine correction can lower the compressed training loss without repairing reduced-integration field errors. Passing unchanged N1 outputs cannot establish accuracy on N8 cases. Three/four internal stages are not three/four future physical outputs: the runner repeats the endpoint computation sequentially for every K output, feeding its own generated state forward. Training factors describe F64 affine fields before final F32 casts, so deployed fields require separate scoring.

**Reproduction.** Use the exact captured four-source trainer/configuration, static manifest/normalization/runtime/design shard and initial/final typed NPZ artifacts bound here. prepare_macrostep_loss_view_v1.py constructs full-grid QR loss factors from native training fields and the frozen managed backbone; no reference pressure solver runs during deployed prediction. macrostep_student_v1.py restores the learned readout then rounds every coefficient to 2^-40 before GPU evaluation. Small design/output artifacts, selected generation/execution sources and restore parity were freshly checked. The 592 native field archives and complete 1,698-member transitive generation closure were not reread. Clean retraining, new-Rheon regeneration and fresh complete-generator latency remain unqualified; CPU restore copies the original GPU probe without a model forward.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-pair-01/train/receipt.json",
    "sha256": "6eadd58ed412cffcd43a61ef537ccadb6d730ba9f16346bd4ec16321ca37857e"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-pair-01/train/history.json",
    "sha256": "ee46134a384ef3d38295e7941faa3ff90ca48cd039a6cc6c9a1d6f7a3b51f8d0"
  },
  "sources": [
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/macrostep_readout_v1.py",
      "sha256": "0a62c5c436cc6e4525c0004acdee2b36fac34ca9acbd288c35a4b90dfcb0e2fc"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/train_macrostep_readout_v1.py",
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
      "path": "runs/full-ai-data-diversity-v1/macrostep-loss-view-01/readout-s3.npz",
      "sha256": "5ae0f62067a0337ae407e95b46e3d5c9e7978864d1c292bd71f72904d435d288",
      "bytes": 24655
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-6eadd58ed412cffc",
    "kind": "training_attempt",
    "name": "Full-AI macrostep temporal readout 3 stages train",
    "task": "full_state_readout_component",
    "schema": "rheon-macrostep-readout-managed-result-v1",
    "method": {},
    "receipt_sha256": "6eadd58ed412cffcd43a61ef537ccadb6d730ba9f16346bd4ec16321ca37857e",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/macrostep-readout-s3-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "9a121f8a3c774ae295c762669047e0eb",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full-AI macrostep temporal readout 3 stages train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 160,
    "parameter_fits": null,
    "declared_parameter_count": 9,
    "training_closure": {
      "config_sha256": "113e58c3b3382e4db444e4bf9cd7aa2725b79351a43fc70aa82d1fd005948a79",
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
          "bytes": 24655,
          "sha256": "5ae0f62067a0337ae407e95b46e3d5c9e7978864d1c292bd71f72904d435d288"
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
        "sha256": "d93db9d921ffbc3be64bdb1f3c2c73a5387226dc2e546f11100c17573bc0e5b4",
        "bytes": 358,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "0b183948d6ee6699fde37aeab34cce581b65f0c45c1f014dde1e199f9b9aa42e",
        "bytes": 358,
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
      "receipt": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_readout_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-pair-01/train/receipt.json",
    "sha256": "6eadd58ed412cffcd43a61ef537ccadb6d730ba9f16346bd4ec16321ca37857e",
    "bytes": 3377
  },
  "bound_training_configuration": {
    "config_sha256": "113e58c3b3382e4db444e4bf9cd7aa2725b79351a43fc70aa82d1fd005948a79",
    "graph_id": "c4ff43d169b74dceac5137491ff368f1",
    "graph_document_sha256": "715b647e65e006bf0c05b9c90f5a372ddbe2eec9738318c8d324090836e175f0",
    "recipe_path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/train-recipe.json",
    "recipe_sha256": "934e4af34a8452c18dded72834eba35a1010ff0d7bc3108257240648f200c235",
    "config_path": "runs/full-ai-data-diversity-v1/macrostep-readout-s3-project-01/config.json",
    "configuration": {
      "identifier": "macrostep-readout-s3-01",
      "role": "macrostep_temporal_readout_component",
      "stages": 3,
      "optimizer_calls": 160,
      "updates": 160,
      "objective": "equal_scene_window_four_metric_QR_residual_v1",
      "seed": 2026102125,
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
  "parameters": 9,
  "trainable_parameters": 9,
  "closure": {
    "config_sha256": "113e58c3b3382e4db444e4bf9cd7aa2725b79351a43fc70aa82d1fd005948a79",
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
        "bytes": 24655,
        "sha256": "5ae0f62067a0337ae407e95b46e3d5c9e7978864d1c292bd71f72904d435d288"
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
        3
      ],
      "sha256": "834a709ba2534ebe3ee1397fd4f7bd288b2acc1d20a08d6c862dcd99b6f04400"
    }
  },
  "final_inventory": {
    "residual_weights": {
      "dtype": "<f8",
      "shape": [
        3,
        3
      ],
      "sha256": "3eed4d4edc0565665c2be3ea9df5c87c2c125f98c6d4689aba5b127506e6f480"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 67468800,
    "all_learning_and_forward_on_cuda": true
  },
  "outputs": {
    "probe.npy": {
      "bytes": 38016,
      "sha256": "a6217d93b2810f5964609e21d0bc5672c174d2f78667fc48e3ce4d1276c81e38"
    },
    "weights.npz": {
      "bytes": 358,
      "sha256": "d93db9d921ffbc3be64bdb1f3c2c73a5387226dc2e546f11100c17573bc0e5b4"
    },
    "initial-weights.npz": {
      "bytes": 358,
      "sha256": "0b183948d6ee6699fde37aeab34cce581b65f0c45c1f014dde1e199f9b9aa42e"
    },
    "history.json": {
      "bytes": 6841,
      "sha256": "ee46134a384ef3d38295e7941faa3ff90ca48cd039a6cc6c9a1d6f7a3b51f8d0"
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
  "student_stages": 3,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.332444833009504
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "macrostep-readout-s3-01",
  "role": "macrostep_temporal_readout_component",
  "stages": 3,
  "optimizer_calls": 160,
  "updates": 160,
  "objective": "equal_scene_window_four_metric_QR_residual_v1",
  "seed": 2026102125,
  "wall_seconds": 600,
  "uniform_deployment_quantization": 9.094947017729282e-13
}

```
