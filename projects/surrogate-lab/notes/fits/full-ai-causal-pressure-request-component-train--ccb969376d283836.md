# Full AI causal pressure request component train

Retrospective source review · fit-ccb969376d283836

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 4,385-parameter F32 causal one-step pressure-request predictor. The factored branch uses an 81-coefficient learned linear flux on current velocity plus known force, a velocity-only nonlinear branch with width 16, and physical h/dt/rho unit conditioning. The calibrated flux is frozen during the 10,000-update nonlinear fit. Final output is a mean-free future q request, not pressure or a complete simulation state.

**Purpose.** Separate a native-calibrated linear request operator from learned velocity-dependent nonlinear terms, testing whether the structured decomposition improves causal request prediction.

**Experiment.** Use 16 original consumed fitting trajectories with 129 states each, at 8/16 cells per metre and four shapes including 32³ and rectangles. All 64 archived velocity/request field hashes are independently decoded and checked. Training selects initial step 0 with 25% probability, steps 1–8 with 25%, and steps 9–120 with 50%; full private PCG64 order and 10,000 main root calls verify. Targets are separately stored q at step+1. An additional 1,000 full-batch F64 SGD updates calibrate flux on 64 manufactured patterns with 16 development patterns, then copy the learned coefficients to F32 and freeze them. The main F32 Adam fit has 4,304 trainable coefficients. Both main fits use rate 0.003 with drops at 6001/9001, gradient cap 1 and final-update selection. The offline objective applies a fixed analytical pressure inverse to predicted-request errors for supervision; it is excluded from inference. All native training timesteps are 1/60 s despite explicit scalar conditioning.

**Observed outcome.** Original training completes 11000 total declared optimizer updates; final logged main objective 0.0005032768240198493. Complete initial/final tensors, source/recipe/input/output identities and sample/work counts verify. The later coupled generator inherits the exact calibrated flux tensor values; it does not inherit the nonlinear request coefficients. No new request/pressure GPU benchmark or full-state/own-history drift result is measured in this source review.

**Lesson.** Explicit dt input does not establish unseen-timestep generalization when training uses one cadence. A request predictor needs a separately identified pressure component and state update before it becomes a simulator. The factored design changes width, architecture, calibration data and precision as well as freezing flux, so a gain cannot be assigned to a single change. The analytical inverse is a training loss tool; distinguish that from candidate inference and disclose it rather than claiming the whole training method has no inverse.

**Reproduction.** Preserve exact captured model, objective, data codec, config, typed recipe/runtime, original F32 velocity/q archive and normalization, private biased-start sampling stream and main Adam schedule. The factored fit also needs its linear teacher, 1,000-update F64 calibration stage and the calibrated frozen flux before main fitting; initial-weights and post-calibration checkpoint-000 represent different states. Original final publication/restore probes execute CPU neural inference and need a GPU-only variant before rerunning. Clean retraining, new-Rheon generation, comparable complete compositions and600-output drift remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-factored-request-pair-02/train/receipt.json",
    "sha256": "ccb969376d2838366aae760b964b106dad3dbb0c24b57301e200643e9febfea4"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-factored-request-pair-02/train/history.json",
    "sha256": "59bf1152453ef8c73844027652a5f829a6e35e202fdf385d77da96b901d23ece"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/flux_calibration.py",
      "sha256": "db400104a711406a864bc4ef3322dba572f4aeb1376e8220494921659833cdc4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/managed_fit_artifacts.py",
      "sha256": "99cdd1d6069e65fae0c7ce7de8a32a928ea0397cf53817ef2ec477a3f98a7da0"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/pressure_request_data.py",
      "sha256": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/pressure_request_model.py",
      "sha256": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/pressure_request_objective.py",
      "sha256": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/pressure_request_train.py",
      "sha256": "c6752bbbfad1716a46d271135c4d7a2bb52a119fb9fba05c2cf4820ec91febd9"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    }
  ],
  "training_inputs": [
    {
      "bytes": 2163874,
      "path": "runs/full-ai-next-priorities-v1/flux-teacher-bank-01/teacher.npz",
      "sha256": "9e278a4bd3c5e075d6c82cd432ffd2be381c0aba738dfb0b90ebeb86d6f6a759",
      "slot": "linear_teacher"
    },
    {
      "bytes": 32769,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/manifest.json",
      "sha256": "0fb750c39884ff3662da54646c5319e5b78d253ec680ace78df1d54bc73a518d",
      "slot": "manifest"
    },
    {
      "bytes": 410,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/normalization.json",
      "sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-00.bin",
      "sha256": "682c60058a6b9b5dce99f3da276b459092302ad08ca1a2c58fa2319765f2d4ba",
      "slot": "part_00"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-01.bin",
      "sha256": "471eb5112344457e312033e269cfa3dc775d2725996dac711b2553f75e39f10d",
      "slot": "part_01"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-02.bin",
      "sha256": "8ac8ca7e051a725963315fd46b6ef0fc941471469db788e850c47210a877ac91",
      "slot": "part_02"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-03.bin",
      "sha256": "6b50f03eb2c3560e189bcbcc944377c49a1fc3dae3df538ac0c5c090eeac8933",
      "slot": "part_03"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-04.bin",
      "sha256": "3090a8efff0173e41a1ba377f935617c29f08327a8415971d7c87cb9bb19d5ee",
      "slot": "part_04"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-05.bin",
      "sha256": "1b6f41893d133ff430f575ae8c16502a76ff2cd8b2d0827a64e5eef12d751361",
      "slot": "part_05"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-06.bin",
      "sha256": "b99e6da465263a9eed015b4e16f85e309349fe26d04d722f7c0dd524ec564e99",
      "slot": "part_06"
    },
    {
      "bytes": 25165824,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-07.bin",
      "sha256": "00e80f1a9ae6fbf5b5eb0da51df8ed16d5b7784a9dbb36127581c3c96b99e5ae",
      "slot": "part_07"
    },
    {
      "bytes": 12666652,
      "path": "runs/full-ai-next-priorities-v1/temporal-request-training-view-01/part-08.bin",
      "sha256": "882d829d8d22d104bf6b6aa214bf0bd87fb1b4d7f53753b232b486ac622e6c94",
      "slot": "part_08"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-ccb969376d283836",
    "kind": "training_attempt",
    "name": "Full AI causal pressure request component train",
    "task": "pressure_request_component",
    "schema": "rheon-causal-request-training-result-v1",
    "method": {},
    "receipt_sha256": "ccb969376d2838366aae760b964b106dad3dbb0c24b57301e200643e9febfea4",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-factored-request-pair-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "1b8e01644d0b4d60b05f2e8211b378cc",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI causal pressure request component train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 11000,
    "parameter_fits": null,
    "declared_parameter_count": 4385,
    "training_closure": {
      "config_sha256": "f00cf4a1a8cc90c85e762cb02dbd3f12193eeb246ed5f2993fecf3601aa647ca",
      "inputs": {
        "linear_teacher": {
          "bytes": 2163874,
          "sha256": "9e278a4bd3c5e075d6c82cd432ffd2be381c0aba738dfb0b90ebeb86d6f6a759"
        },
        "manifest": {
          "bytes": 32769,
          "sha256": "0fb750c39884ff3662da54646c5319e5b78d253ec680ace78df1d54bc73a518d"
        },
        "normalization": {
          "bytes": 410,
          "sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 25165824,
          "sha256": "682c60058a6b9b5dce99f3da276b459092302ad08ca1a2c58fa2319765f2d4ba"
        },
        "part_01": {
          "bytes": 25165824,
          "sha256": "471eb5112344457e312033e269cfa3dc775d2725996dac711b2553f75e39f10d"
        },
        "part_02": {
          "bytes": 25165824,
          "sha256": "8ac8ca7e051a725963315fd46b6ef0fc941471469db788e850c47210a877ac91"
        },
        "part_03": {
          "bytes": 25165824,
          "sha256": "6b50f03eb2c3560e189bcbcc944377c49a1fc3dae3df538ac0c5c090eeac8933"
        },
        "part_04": {
          "bytes": 25165824,
          "sha256": "3090a8efff0173e41a1ba377f935617c29f08327a8415971d7c87cb9bb19d5ee"
        },
        "part_05": {
          "bytes": 25165824,
          "sha256": "1b6f41893d133ff430f575ae8c16502a76ff2cd8b2d0827a64e5eef12d751361"
        },
        "part_06": {
          "bytes": 25165824,
          "sha256": "b99e6da465263a9eed015b4e16f85e309349fe26d04d722f7c0dd524ec564e99"
        },
        "part_07": {
          "bytes": 25165824,
          "sha256": "00e80f1a9ae6fbf5b5eb0da51df8ed16d5b7784a9dbb36127581c3c96b99e5ae"
        },
        "part_08": {
          "bytes": 12666652,
          "sha256": "882d829d8d22d104bf6b6aa214bf0bd87fb1b4d7f53753b232b486ac622e6c94"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "flux_calibration.py": "db400104a711406a864bc4ef3322dba572f4aeb1376e8220494921659833cdc4",
        "managed_fit_artifacts.py": "99cdd1d6069e65fae0c7ce7de8a32a928ea0397cf53817ef2ec477a3f98a7da0",
        "pressure_request_data.py": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc",
        "pressure_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
        "pressure_request_objective.py": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110",
        "pressure_request_train.py": "c6752bbbfad1716a46d271135c4d7a2bb52a119fb9fba05c2cf4820ec91febd9",
        "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
        "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
        "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
      },
      "horizons": 1,
      "normalization_sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503",
      "task": "causal_pressure_request_component"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "b76d90f41bdfb825daf45b60f91fe095b47fc45a6e67a6690ba590034241f800",
        "bytes": 18832,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "f9539bcff66e25b76c350219f0410ec3729e8e4506ac5f442b7cf6d3422d99e8",
        "bytes": 18832,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
        "bytes": 18832,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "c6dd2c2b7d0e24bf6034569952dd84a118d7a85ba112bbd7cdea1006f0e98357",
        "bytes": 18832,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd",
        "bytes": 18832,
        "stage": "retained_snapshot"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "causal-components/1"
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-factored-request-pair-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_request_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-factored-request-pair-02/train/receipt.json",
    "sha256": "ccb969376d2838366aae760b964b106dad3dbb0c24b57301e200643e9febfea4",
    "bytes": 7282
  },
  "bound_training_configuration": {
    "config_sha256": "f00cf4a1a8cc90c85e762cb02dbd3f12193eeb246ed5f2993fecf3601aa647ca",
    "graph_id": "529239bd89b646caa2e07596c2946975",
    "graph_document_sha256": "87f3ac24851647985061ab619f542d2e00ed5112f05a2f54080443e3f73a1476",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/train-recipe.json",
    "recipe_sha256": "76db5cedf698216ce478b29c86f4dbf4290e0ec59e8c741d4ff300b5786407b1",
    "config_path": "runs/full-ai-next-priorities-v1/b1-factored-request-project-02/config.json",
    "configuration": {
      "schema": "rheon-factored-request-fit-v1",
      "identifier": "b1-factored-request",
      "horizons": 1,
      "updates": 11000,
      "main_updates": 10000,
      "linear_pretrain_updates": 1000,
      "learning_rate": 0.003,
      "schedule": [
        {
          "update": 6001,
          "learning_rate": 0.001
        },
        {
          "update": 9001,
          "learning_rate": 0.0003
        }
      ],
      "seed": 2026101994,
      "sampling_seed": 2026101995,
      "gradient_norm_cap": 1.0,
      "wall_seconds": 1800,
      "component_only": true,
      "future_request_input": false,
      "current_pressure_input": false,
      "physical_context": {
        "dt_s": 0.016666666666666666,
        "rho_kg_m3": 1000.0,
        "mu_pa_s": 0.001,
        "boundaries": "closed regular grid"
      },
      "registration_sha256": "455c7c1a7674490ce6bbf04ff8e748c449e1734980b7cc9534914ddf052c5c39",
      "selection": "Final update10000; checkpoints0/2000/10000retained; no development checkpoint reselection",
      "analytical_inverse": "Offline objective only; no inverse in model inference"
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
  "schema": "rheon-causal-request-training-result-v1",
  "status": "complete",
  "mode": "train",
  "optimizer_updates": 11000,
  "parameters": 4385,
  "horizons": 1,
  "closure": {
    "config_sha256": "f00cf4a1a8cc90c85e762cb02dbd3f12193eeb246ed5f2993fecf3601aa647ca",
    "inputs": {
      "linear_teacher": {
        "bytes": 2163874,
        "sha256": "9e278a4bd3c5e075d6c82cd432ffd2be381c0aba738dfb0b90ebeb86d6f6a759"
      },
      "manifest": {
        "bytes": 32769,
        "sha256": "0fb750c39884ff3662da54646c5319e5b78d253ec680ace78df1d54bc73a518d"
      },
      "normalization": {
        "bytes": 410,
        "sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 25165824,
        "sha256": "682c60058a6b9b5dce99f3da276b459092302ad08ca1a2c58fa2319765f2d4ba"
      },
      "part_01": {
        "bytes": 25165824,
        "sha256": "471eb5112344457e312033e269cfa3dc775d2725996dac711b2553f75e39f10d"
      },
      "part_02": {
        "bytes": 25165824,
        "sha256": "8ac8ca7e051a725963315fd46b6ef0fc941471469db788e850c47210a877ac91"
      },
      "part_03": {
        "bytes": 25165824,
        "sha256": "6b50f03eb2c3560e189bcbcc944377c49a1fc3dae3df538ac0c5c090eeac8933"
      },
      "part_04": {
        "bytes": 25165824,
        "sha256": "3090a8efff0173e41a1ba377f935617c29f08327a8415971d7c87cb9bb19d5ee"
      },
      "part_05": {
        "bytes": 25165824,
        "sha256": "1b6f41893d133ff430f575ae8c16502a76ff2cd8b2d0827a64e5eef12d751361"
      },
      "part_06": {
        "bytes": 25165824,
        "sha256": "b99e6da465263a9eed015b4e16f85e309349fe26d04d722f7c0dd524ec564e99"
      },
      "part_07": {
        "bytes": 25165824,
        "sha256": "00e80f1a9ae6fbf5b5eb0da51df8ed16d5b7784a9dbb36127581c3c96b99e5ae"
      },
      "part_08": {
        "bytes": 12666652,
        "sha256": "882d829d8d22d104bf6b6aa214bf0bd87fb1b4d7f53753b232b486ac622e6c94"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "flux_calibration.py": "db400104a711406a864bc4ef3322dba572f4aeb1376e8220494921659833cdc4",
      "managed_fit_artifacts.py": "99cdd1d6069e65fae0c7ce7de8a32a928ea0397cf53817ef2ec477a3f98a7da0",
      "pressure_request_data.py": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc",
      "pressure_request_model.py": "39577e2b6b275850f61f2e440e646ab435def01da1d16b5301fb5329e1fec6b5",
      "pressure_request_objective.py": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110",
      "pressure_request_train.py": "c6752bbbfad1716a46d271135c4d7a2bb52a119fb9fba05c2cf4820ec91febd9",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    "horizons": 1,
    "normalization_sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503",
    "task": "causal_pressure_request_component"
  },
  "main_optimizer_updates": 10000,
  "linear_optimizer_updates": 1000,
  "initial_inventory": {
    "flux.weight": {
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7119f314d3837dac54bf984118de657f6f1e6dfd2a2b2a20510454f762dddf11"
    },
    "left.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "465807a2c3070c86326c5514df507b457d05c278f1606a92cc885b369ddd8bdd"
    },
    "right.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "9d9a28c6535657dc686777a138c72c39f12083038ecfda7171288d5820e09166"
    },
    "mix.weight": {
      "shape": [
        16,
        80,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "e9811db85b970c3dbdbdaef34a63926f96f27645293db5e35f4c5228eb5d0695"
    },
    "head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "7636c9f8da9a82f9a9fcd804a25795a9c900e1c7e19f7442a7b113c8ccd64eb2"
    }
  },
  "final_inventory": {
    "flux.weight": {
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "dacf17b5d8b84199c50513a57cecd3233b6049fe116850333618e930e5911041"
    },
    "left.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "bda0101f0e2163356ecafe087bad39780c023b5a68be5191f816236f3addd63d"
    },
    "right.weight": {
      "shape": [
        16,
        3,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "aa95dd917b7dc7fd4334d63050a550b1a9381df854ef373fada156ec206eb959"
    },
    "mix.weight": {
      "shape": [
        16,
        80,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "bfb4c67335751a70a0114474413e70bfc5ab75d7f41c63bfa13703409453ca52"
    },
    "head.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "390bfffbc96e0ca68a67a6d585116525dca3ad742185c7ec2606ffd2cda0cb1d"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "peak_allocated_bytes": 732988928,
    "peak_reserved_bytes": 761266176,
    "optimizer": "Adam on CUDA",
    "all_optimizer_moments_cuda": true
  },
  "outputs": {
    "checkpoint-000.npz": {
      "bytes": 18832,
      "sha256": "b76d90f41bdfb825daf45b60f91fe095b47fc45a6e67a6690ba590034241f800"
    },
    "probe.npy": {
      "bytes": 464204,
      "sha256": "cb0bb4fca55dbb60743580470bfa8474a5ab231c64ca8517f5a60175079d7cee"
    },
    "checkpoint-2000.npz": {
      "bytes": 18832,
      "sha256": "f9539bcff66e25b76c350219f0410ec3729e8e4506ac5f442b7cf6d3422d99e8"
    },
    "weights.npz": {
      "bytes": 18832,
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    "initial-weights.npz": {
      "bytes": 18832,
      "sha256": "c6dd2c2b7d0e24bf6034569952dd84a118d7a85ba112bbd7cdea1006f0e98357"
    },
    "checkpoint-10000.npz": {
      "bytes": 18832,
      "sha256": "624dce8d87a2ebc1d82a814c7b19e24cdcce99979374aa0f30b8fca73539b2cd"
    },
    "history.json": {
      "bytes": 272790,
      "sha256": "59bf1152453ef8c73844027652a5f829a6e35e202fdf385d77da96b901d23ece"
    },
    "normalization.json": {
      "bytes": 410,
      "sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 80.00887225102633,
  "component_only": true,
  "full_state_temporal_candidate": false,
  "candidate_numerical_pressure_solve": false,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-factored-request-fit-v1",
  "identifier": "b1-factored-request",
  "horizons": 1,
  "updates": 11000,
  "main_updates": 10000,
  "linear_pretrain_updates": 1000,
  "learning_rate": 0.003,
  "schedule": [
    {
      "update": 6001,
      "learning_rate": 0.001
    },
    {
      "update": 9001,
      "learning_rate": 0.0003
    }
  ],
  "seed": 2026101994,
  "sampling_seed": 2026101995,
  "gradient_norm_cap": 1.0,
  "wall_seconds": 1800,
  "component_only": true,
  "future_request_input": false,
  "current_pressure_input": false,
  "physical_context": {
    "dt_s": 0.016666666666666666,
    "rho_kg_m3": 1000.0,
    "mu_pa_s": 0.001,
    "boundaries": "closed regular grid"
  },
  "registration_sha256": "455c7c1a7674490ce6bbf04ff8e748c449e1734980b7cc9534914ddf052c5c39",
  "selection": "Final update10000; checkpoints0/2000/10000retained; no development checkpoint reselection",
  "analytical_inverse": "Offline objective only; no inverse in model inference"
}

```
