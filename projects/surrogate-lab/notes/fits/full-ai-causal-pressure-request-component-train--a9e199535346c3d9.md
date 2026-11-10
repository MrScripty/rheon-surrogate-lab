# Full AI causal pressure request component train

Retrospective source review · fit-a9e199535346c3d9

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 3,936-parameter F32 causal one-step pressure-request predictor. A learned 3³ linear branch, two width-eight local projections with quadratic products, a learned quadratic readout and a 5³ viscous branch consume current velocity and known force. Per-example h, dt, rho and mu condition physical unit factors. It outputs the mean-free next q request; current pressure and future reference requests are not inputs.

**Purpose.** Learn a future pressure request causally from current velocity and prescribed controls, so later pressure experts can be driven without receiving a future reference request.

**Experiment.** Use 16 original consumed fitting trajectories with 129 states each, at 8/16 cells per metre and four shapes including 32³ and rectangles. All 64 archived velocity/request field hashes are independently decoded and checked. Training selects initial step 0 with 25% probability, steps 1–8 with 25%, and steps 9–120 with 50%; full private PCG64 order and 10,000 main root calls verify. Targets are separately stored q at step+1. All 3,936 coefficients train with F32 Adam. Both main fits use rate 0.003 with drops at 6001/9001, gradient cap 1 and final-update selection. The offline objective applies a fixed analytical pressure inverse to predicted-request errors for supervision; it is excluded from inference. All native training timesteps are 1/60 s despite explicit scalar conditioning.

**Observed outcome.** Original training completes 10000 total declared optimizer updates; final logged main objective 1.7000806331634521. Complete initial/final tensors, source/recipe/input/output identities and sample/work counts verify. No new request/pressure GPU benchmark or full-state/own-history drift result is measured in this source review.

**Lesson.** Explicit dt input does not establish unseen-timestep generalization when training uses one cadence. A request predictor needs a separately identified pressure component and state update before it becomes a simulator. The factored design changes width, architecture, calibration data and precision as well as freezing flux, so a gain cannot be assigned to a single change. The analytical inverse is a training loss tool; distinguish that from candidate inference and disclose it rather than claiming the whole training method has no inverse.

**Reproduction.** Preserve exact captured model, objective, data codec, config, typed recipe/runtime, original F32 velocity/q archive and normalization, private biased-start sampling stream and main Adam schedule. The factored fit also needs its linear teacher, 1,000-update F64 calibration stage and the calibrated frozen flux before main fitting; initial-weights and post-calibration checkpoint-000 represent different states. Original final publication/restore probes execute CPU neural inference and need a GPU-only variant before rerunning. Clean retraining, new-Rheon generation, comparable complete compositions and600-output drift remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-causal-request-pair-01/train/receipt.json",
    "sha256": "a9e199535346c3d949c069ba61fbb4408033afc25e3d48aabef1501efa69c713"
  },
  "history": {
    "path": "runs/full-ai-next-priorities-v1/b1-causal-request-pair-01/train/history.json",
    "sha256": "5e58dd9a83863f3e721d1636346c65c8d433f70b421ee98335a3d2033c7ae0ac"
  },
  "sources": [
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/pressure_request_data.py",
      "sha256": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/pressure_request_model.py",
      "sha256": "82558df5d01a5a608f770c7b26cc7577697bf4f7de629311bc6cfb2e7d0807c1"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/pressure_request_objective.py",
      "sha256": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/pressure_request_train.py",
      "sha256": "c4259a1ae76c3615cf7f6007e182301f55e6fdea164a98aeb014267cdea66778"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/temporal_codec.py",
      "sha256": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/temporal_data.py",
      "sha256": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b"
    },
    {
      "path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/temporal_features_v2.py",
      "sha256": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    }
  ],
  "training_inputs": [
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
    "item_id": "fit-a9e199535346c3d9",
    "kind": "training_attempt",
    "name": "Full AI causal pressure request component train",
    "task": "pressure_request_component",
    "schema": "rheon-causal-request-training-result-v1",
    "method": {},
    "receipt_sha256": "a9e199535346c3d949c069ba61fbb4408033afc25e3d48aabef1501efa69c713",
    "receipt_paths": [
      "runs/full-ai-next-priorities-v1/b1-causal-request-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "e4e56be8d1d5481796d2ce9273724175",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI causal pressure request component train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 10000,
    "parameter_fits": null,
    "declared_parameter_count": 3936,
    "training_closure": {
      "config_sha256": "901ee02790bcdd4430f14bc0396005b65d201a884e2bd72cfd7e9f5c774b1e7e",
      "inputs": {
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
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "pressure_request_data.py": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc",
        "pressure_request_model.py": "82558df5d01a5a608f770c7b26cc7577697bf4f7de629311bc6cfb2e7d0807c1",
        "pressure_request_objective.py": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110",
        "pressure_request_train.py": "c4259a1ae76c3615cf7f6007e182301f55e6fdea164a98aeb014267cdea66778",
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
        "sha256": "aa9db2c373fc190b03dbecd6fe7cf1777582eabba8ae7e8cb7954c39c8f7d403",
        "bytes": 17058,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2000.npz",
        "sha256": "c097f26ee4ab54f70d1103c0cd4696aadf686ca0ee33af8322c43d00c2553943",
        "bytes": 17058,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "8533e0e609ac25c7ab1a9379c068125b6010d6e77eea2b3446650f659052658d",
        "bytes": 17058,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "aa9db2c373fc190b03dbecd6fe7cf1777582eabba8ae7e8cb7954c39c8f7d403",
        "bytes": 17058,
        "stage": "initial"
      },
      {
        "name": "checkpoint-10000.npz",
        "sha256": "8533e0e609ac25c7ab1a9379c068125b6010d6e77eea2b3446650f659052658d",
        "bytes": 17058,
        "stage": "retained_snapshot"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "causal-components/0"
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
      "receipt": "runs/full-ai-next-priorities-v1/b1-causal-request-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_request_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-next-priorities-v1/b1-causal-request-pair-01/train/receipt.json",
    "sha256": "a9e199535346c3d949c069ba61fbb4408033afc25e3d48aabef1501efa69c713",
    "bytes": 6982
  },
  "bound_training_configuration": {
    "config_sha256": "901ee02790bcdd4430f14bc0396005b65d201a884e2bd72cfd7e9f5c774b1e7e",
    "graph_id": "9930164524b44320ac2028cce3070470",
    "graph_document_sha256": "8d00a8cf21ce70ae0e80ba25de3a37f6d107938a448499b2472d343eb5513037",
    "recipe_path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/train-recipe.json",
    "recipe_sha256": "0bfe168df359d05fb1d0e7082c9fb8dff6f8f6854a2e1b81ea769fa07e8fffe1",
    "config_path": "runs/full-ai-next-priorities-v1/b1-causal-request-project-01/config.json",
    "configuration": {
      "schema": "rheon-causal-request-fit-v1",
      "identifier": "b1-causal-request",
      "horizons": 1,
      "updates": 10000,
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
      "registration_sha256": "aced7a26cfe84968f267005de1da10fc3b96e9560745a11c46f8956999a7db85",
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
  "optimizer_updates": 10000,
  "parameters": 3936,
  "horizons": 1,
  "closure": {
    "config_sha256": "901ee02790bcdd4430f14bc0396005b65d201a884e2bd72cfd7e9f5c774b1e7e",
    "inputs": {
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
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "pressure_request_data.py": "4128be0c71682d77440160f526d4e5228845a0b9c70c7675602d3c50c343e3cc",
      "pressure_request_model.py": "82558df5d01a5a608f770c7b26cc7577697bf4f7de629311bc6cfb2e7d0807c1",
      "pressure_request_objective.py": "1426b0cba7eb9eb9cc61148c5e33c62ecc3066f03f5526260876a3ef0796a110",
      "pressure_request_train.py": "c4259a1ae76c3615cf7f6007e182301f55e6fdea164a98aeb014267cdea66778",
      "temporal_codec.py": "64a17a42559497a04c75b5a4ca7988e899a5cb8690e28dd348b4fee0ac1cf6c4",
      "temporal_data.py": "7f215a720f4b03ed3e15c43bfa73e0cbe5d99b572d549b8dc33cf9fe27b1036b",
      "temporal_features_v2.py": "d64da548837aa7c9ed34b2991316a234879df82d48ef377abcea571c32b4280a"
    },
    "horizons": 1,
    "normalization_sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503",
    "task": "causal_pressure_request_component"
  },
  "initial_inventory": {
    "linear.weight": {
      "shape": [
        1,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "f4bd841308415de6ed2727462cd66a7333ac8155b4e8e95de0220355189c785c"
    },
    "left.weight": {
      "shape": [
        8,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "35c32cb6c6aafe0b66e02827ba938eca337780bd0e8369dcfcc278be1fd89639"
    },
    "right.weight": {
      "shape": [
        8,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "caa3761e47f1199628dd03651340aa53bbc0db1dcf498a37c63b629d95aaa61d"
    },
    "quadratic.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "b3765410a2d8113e29c1341f4548c168058025d8b26aba089bb18b4c12c31a22"
    },
    "viscous.weight": {
      "shape": [
        1,
        6,
        5,
        5,
        5
      ],
      "dtype": "<f4",
      "sha256": "c81ca5eda5947c7826ad046fdbdc2a25a846b835a6c34c237cc8b3afbe9ec6cc"
    }
  },
  "final_inventory": {
    "linear.weight": {
      "shape": [
        1,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "ce2e464317ae8ac6e9a1cc0d7957dae0f33412709f90e5ade2a78e6d21d38ced"
    },
    "left.weight": {
      "shape": [
        8,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "2b1ee968d5deaf34b80443c31cbb2478532155d2814ec0ca2077c9a23ca1b7cc"
    },
    "right.weight": {
      "shape": [
        8,
        6,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "a73ead1007de7cba450a08e937f6bfca44c1c4eb308ce75214c2cf087b7d35b6"
    },
    "quadratic.weight": {
      "shape": [
        1,
        16,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "13f6cdffe44ef22925deb298e7f2904ee2ef8d818db687cce919287c69a882a1"
    },
    "viscous.weight": {
      "shape": [
        1,
        6,
        5,
        5,
        5
      ],
      "dtype": "<f4",
      "sha256": "a00757412f7e08b0f582afe24283d462eee87df16de5223fe79fa8654b7b28da"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float32",
    "peak_allocated_bytes": 715398656,
    "peak_reserved_bytes": 740294656,
    "optimizer": "Adam on CUDA",
    "all_optimizer_moments_cuda": true
  },
  "outputs": {
    "checkpoint-000.npz": {
      "bytes": 17058,
      "sha256": "aa9db2c373fc190b03dbecd6fe7cf1777582eabba8ae7e8cb7954c39c8f7d403"
    },
    "probe.npy": {
      "bytes": 464204,
      "sha256": "30ae8e5d2ba3809823c56195a8c6f1407222d8dd3c34c08fe89ccb972cb4ee30"
    },
    "checkpoint-2000.npz": {
      "bytes": 17058,
      "sha256": "c097f26ee4ab54f70d1103c0cd4696aadf686ca0ee33af8322c43d00c2553943"
    },
    "weights.npz": {
      "bytes": 17058,
      "sha256": "8533e0e609ac25c7ab1a9379c068125b6010d6e77eea2b3446650f659052658d"
    },
    "initial-weights.npz": {
      "bytes": 17058,
      "sha256": "aa9db2c373fc190b03dbecd6fe7cf1777582eabba8ae7e8cb7954c39c8f7d403"
    },
    "checkpoint-10000.npz": {
      "bytes": 17058,
      "sha256": "8533e0e609ac25c7ab1a9379c068125b6010d6e77eea2b3446650f659052658d"
    },
    "history.json": {
      "bytes": 270017,
      "sha256": "5e58dd9a83863f3e721d1636346c65c8d433f70b421ee98335a3d2033c7ae0ac"
    },
    "normalization.json": {
      "bytes": 410,
      "sha256": "0d0e32e6777a2e0d8a264c7f32cd5da5423ae7faab8c5e102dff08e50e3ed503"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "elapsed_s": 87.96769740595482,
  "component_only": true,
  "full_state_temporal_candidate": false,
  "candidate_numerical_pressure_solve": false,
  "scientific_acceptance_or_promotion": false
}

```

## Exact hash-bound training configuration

```json
{
  "schema": "rheon-causal-request-fit-v1",
  "identifier": "b1-causal-request",
  "horizons": 1,
  "updates": 10000,
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
  "registration_sha256": "aced7a26cfe84968f267005de1da10fc3b96e9560745a11c46f8956999a7db85",
  "selection": "Final update10000; checkpoints0/2000/10000retained; no development checkpoint reselection",
  "analytical_inverse": "Offline objective only; no inverse in model inference"
}

```
