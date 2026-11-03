# Full-AI direct residual supervision compact CUDA fit train

Retrospective source review · fit-fd5dc9866f07dccc

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A10,876-parameter float32 CUDA correction CNN:42 cell-lattice input channels, two width8 clamped3×3×3 convolutions with SiLU, three staggered transposed-convolution velocity heads and one tracer head. Output heads initialize to zero. The quiescence gate uses current/coarse-change/known-force channels. Pressure correction is supplied through the frozen learned coupled projection; this subnet does not directly emit pressure. Inputs include spacing100h, group duration480dt, group width and ordered known forces; future fields are labels only.

**Purpose.** Continue the exact selected S3 weights but directly supervise the four missing correction heads, while tracking the coupled physical objective separately.

**Experiment.** CUDA full-batch LBFGS, 24 maximum iterations and 60 maximum closures. Cache the same teacher-derived inputs/targets on GPU; select the best complete closure including the exact warm start. Recorded internal iteration counters are not minibatch update counts. Objective is per-head output-RMS-normalized correction MSE; physical metrics are diagnostics, not the selection objective. All148 training scenes are counted, but only29N8 scenes provide parameter-dependent gradients:9 starts×3 groups=783 local samples. The119N1 scenes contribute1,071 fixed control windows with zero parameter derivative. Active grids span eight shapes through64³, at8/10/16/25/50/100cells per metre; the complete fixed/control population includes19shapes through128³.

**Observed outcome.** Selected training objective 0.74515826956031694; selected closure 51 after 24 recorded LBFGS iterations / 51 closure evaluations. Its separately recorded physical objective is 6.3435674501696377e-05; the direct-delta objective cannot be compared numerically with S1/S3 physical objectives. The retained complete-generator replay passes all 920 N1 outputs and fails 1151 of 1200 N8 outputs at the diagnostic1e-5 gate. These are original consumed cases, not independent validation or qualified new latency.

**Lesson.** Optimization improvements on local teacher-state corrections do not establish accurate own-history reduced integration. The N8 runner advances groups2/2/3/1 with three correction calls per physical1/60s output; K1–K4 outputs still advance sequentially. It does not predict a distant endpoint with cheap temporal intermediate frames. N1 bypasses this correction and preserves the original branch. Keep effective gradient scenes, fixed controls, loss choice, trained checkpoint and complete execution variant separate.

**Reproduction.** Use the exact captured trainer/configuration/static inputs and retained selected/initial inventories shown here and in the original facts below. S1/S2 share initialization; S3 starts from selected S1, S4 from selected S3. The physical-state view is prepared by prepare_internal_residual_view_v1.py; compact views regenerate teacher labels with the captured teacher-code archive and learned assets. Static inputs and checkpoint tensors were freshly checked; the9.77GB full /0.671GB compact archives were not reread, labels were not decoded and clean retraining/new-Rheon regeneration remain unqualified. CPU restore receipts copy the CUDA probe without a CPU model forward.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-pair-01/train/receipt.json",
    "sha256": "fd5dc9866f07dcccbb8877253fb7cf02e42e7078786af60fa2d8801cdf6f1ef3"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-pair-01/train/history.json",
    "sha256": "088e370276561c5a8333c7325202cdb69382cc82b04487cc909bb389ba674d4e"
  },
  "sources": [
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/capture_memory_telemetry_v1.py",
      "sha256": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/compact_internal_training_view_v2.py",
      "sha256": "91dd922158a1e2b41347d6920a147ce7a01b09da050cd28266fc66796e0703cf"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/internal_residual_cached_view_v1.py",
      "sha256": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/internal_residual_dataset_v1.py",
      "sha256": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/internal_residual_loss_v1.py",
      "sha256": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/internal_residual_network_v1.py",
      "sha256": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/internal_residual_network_v2.py",
      "sha256": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/managed_input_stream_v1.py",
      "sha256": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/teacher-code.zip",
      "sha256": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/train_internal_residual_v4.py",
      "sha256": "896718840ea8cd470430bb917ee05838ba6b9354d4b3695be8982264b120ba98"
    }
  ],
  "training_inputs": [
    {
      "slot": "manifest",
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-data-01/manifest.json",
      "sha256": "1cec5c9255b9e40e8fb9391acd90604f5b7d09056a4ea05cbfbed0c2103a4768",
      "bytes": 4625289
    },
    {
      "slot": "normalization",
      "path": "runs/full-ai-data-diversity-v1/internal-residual-view-01/network-normalization.json",
      "sha256": "2e3cb8d34ff8846a28d2b6155eaafbf5a732cd0aa28bf7526539d2a319bdbc21",
      "bytes": 1521
    },
    {
      "slot": "optimization_normalization",
      "path": "runs/full-ai-data-diversity-v1/internal-residual-view-01/optimization-normalization.json",
      "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11",
      "bytes": 918
    },
    {
      "slot": "runtime_manifest",
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "bytes": 7989288
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-fd5dc9866f07dccc",
    "kind": "training_attempt",
    "name": "Full-AI direct residual supervision compact CUDA fit train",
    "task": "full_state_correction_component",
    "schema": "rheon-internal-residual-managed-v4",
    "method": {},
    "receipt_sha256": "fd5dc9866f07dcccbb8877253fb7cf02e42e7078786af60fa2d8801cdf6f1ef3",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/internal-residual-s4-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "73fee50ec91e496283484a26b69c87fc",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full-AI direct residual supervision compact CUDA fit train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 24,
    "parameter_fits": null,
    "declared_parameter_count": 10876,
    "training_closure": {
      "config_sha256": "06d7feb415494202e2a0c0e03c77895415cc2d64ebf75d0800b541b10164f6a0",
      "inputs": {
        "manifest": {
          "bytes": 4625289,
          "sha256": "1cec5c9255b9e40e8fb9391acd90604f5b7d09056a4ea05cbfbed0c2103a4768"
        },
        "normalization": {
          "bytes": 1521,
          "sha256": "2e3cb8d34ff8846a28d2b6155eaafbf5a732cd0aa28bf7526539d2a319bdbc21"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 671002882,
          "sha256": "9a2708e60365e8efaf74112fea72089f1e7a523aa104021c4c90c0feece66a36"
        },
        "optimization_normalization": {
          "bytes": 918,
          "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
        }
      },
      "role": "internal_group_nonlinear_residual_component",
      "sources": {
        "capture_memory_telemetry_v1.py": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204",
        "compact_internal_training_view_v2.py": "91dd922158a1e2b41347d6920a147ce7a01b09da050cd28266fc66796e0703cf",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "internal_residual_cached_view_v1.py": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8",
        "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
        "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
        "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
        "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
        "teacher-code.zip": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc",
        "train_internal_residual_v4.py": "896718840ea8cd470430bb917ee05838ba6b9354d4b3695be8982264b120ba98"
      }
    },
    "checkpoints": [
      {
        "name": "last-evaluated-weights.npz",
        "sha256": "2af8d84d4517471f890584393e96efa83e1e11296de73d361c38f81e5068e70e",
        "bytes": 47324,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "2af8d84d4517471f890584393e96efa83e1e11296de73d361c38f81e5068e70e",
        "bytes": 47324,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2",
        "bytes": 47324,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "residual-fits/3"
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
      "receipt": "runs/full-ai-data-diversity-v1/internal-residual-s4-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_correction_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s4-pair-01/train/receipt.json",
    "sha256": "fd5dc9866f07dcccbb8877253fb7cf02e42e7078786af60fa2d8801cdf6f1ef3",
    "bytes": 10526
  },
  "bound_training_configuration": {
    "config_sha256": "06d7feb415494202e2a0c0e03c77895415cc2d64ebf75d0800b541b10164f6a0",
    "graph_id": "48d9c3d104054ebcbe461e6f6346c15a",
    "graph_document_sha256": "c423c863f115724053648d3a9c9729ae2198712f00c10a98a0f821e28c15a2dc",
    "recipe_path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/train-recipe.json",
    "recipe_sha256": "4666d72e021d746cda7d93f1d537bfe7936692deff387e03fd487b08abaf384d",
    "config_path": "runs/full-ai-data-diversity-v1/internal-residual-s4-project-01/config.json",
    "configuration": {
      "identifier": "internal-residual-s4-direct-delta-LBFGS",
      "role": "internal_group_nonlinear_residual_component",
      "seed": 1729,
      "optimizer": "full_batch_LBFGS",
      "maximum_optimizer_iterations": 24,
      "maximum_closure_evaluations": 60,
      "initial_step_learning_rate": 1.0,
      "subsequent_step_learning_rate": 1.0,
      "selection": "minimum_full148scene_direct_delta_objective_over_initial_and_complete_closures_earlier_tie",
      "qualified_initial_objective": 0.7665071603885028,
      "qualified_initial_inventory": {
        "input_rms": {
          "shape": [
            1,
            42,
            1,
            1,
            1
          ],
          "dtype": "<f4",
          "sha256": "03607579d061e0a3afb7037cc5e7bbf69985070dc0e4e3aefa0f29451a69108d"
        },
        "output_rms": {
          "shape": [
            4
          ],
          "dtype": "<f4",
          "sha256": "f6dcc9570fba7f8d765ace2909fe76a00d84a802e25b8432ac2d523d5ecd0d87"
        },
        "conv1.weight": {
          "shape": [
            8,
            42,
            3,
            3,
            3
          ],
          "dtype": "<f4",
          "sha256": "270298ab5a33aefa09e0463d5828d9d017de53c80dda340b757dcfbb84b8fd85"
        },
        "conv1.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "7b3dfd86627f558c210b56cc6f458634643e8f30023b2e62c1c509f9fbec670c"
        },
        "conv2.weight": {
          "shape": [
            8,
            8,
            3,
            3,
            3
          ],
          "dtype": "<f4",
          "sha256": "61a2a0174ef4d64ec6fae53e3cf179b66b3c25c4f6a49e308006a912840e215f"
        },
        "conv2.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "6b4e6759f3b856c1f1fb3095db3e9d97dcde9748c8a3632e61c3a2582fab316b"
        },
        "velocity.0.weight": {
          "shape": [
            8,
            1,
            1,
            1,
            2
          ],
          "dtype": "<f4",
          "sha256": "ddb564f517fc95076dfe0de99795a982743c7f8ae419f3d5ad33c4ddff2d22cf"
        },
        "velocity.0.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "bf0a2d788588181c6fbf94324ab97d41cbfa908ff4013f0f7734a4513fc9607d"
        },
        "velocity.1.weight": {
          "shape": [
            8,
            1,
            1,
            2,
            1
          ],
          "dtype": "<f4",
          "sha256": "bf0b7dfc52fe77a5831da48d43788f40091bf1ee2503880bd6bbf00081ecc535"
        },
        "velocity.1.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "356c75ddc004ee90e9f56efc0eb62b3a451173cfff5e625a05646c84879c20a2"
        },
        "velocity.2.weight": {
          "shape": [
            8,
            1,
            2,
            1,
            1
          ],
          "dtype": "<f4",
          "sha256": "b5be0eaf2defe41df7c3d8bee731865e16eba1a462e362be95454281fe56fffb"
        },
        "velocity.2.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "5cd606ead5f045b362651d20493041509020b77b7f7d0ec18425e2f9ea5708bb"
        },
        "tracer.weight": {
          "shape": [
            1,
            8,
            1,
            1,
            1
          ],
          "dtype": "<f4",
          "sha256": "e050a74b3abbb0682d0f55c9d16234887b8ed0fc0bec591a5f348aa11d6217ca"
        },
        "tracer.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "c5f48bd23b5f2bb9697c26b31548360b643a148c8e80f234e6a7d7e683e30517"
        }
      },
      "qualified_initial_gradient_norms": {
        "conv1.weight": 0.014344844964333943,
        "conv1.bias": 8.793608542694665e-05,
        "conv2.weight": 0.027295263879977428,
        "conv2.bias": 0.0001023488575278503,
        "velocity.0.weight": 0.01564564570092267,
        "velocity.0.bias": 0.007132446393370628,
        "velocity.1.weight": 0.13045994696089047,
        "velocity.1.bias": 0.003620981937274337,
        "velocity.2.weight": 0.09200242459700894,
        "velocity.2.bias": 0.005964244250208139,
        "tracer.weight": 0.0027318245620267466,
        "tracer.bias": 0.0003040946612600237
      },
      "expected_cached_tensor_bytes": 13765858200,
      "wall_seconds": 1800,
      "maximum_peak_rss_bytes": 4294967296,
      "maximum_gpu_allocated_bytes": 17179869184,
      "maximum_gpu_reserved_bytes": 21474836480,
      "scientific_intervention": "S3 exact warm start; directly supervise all four missing corrections with frozen per-head output RMS. Physical losses tracked separately; native gates unchanged; no architecture change.",
      "native_accuracy_claim": false,
      "latency_claim": false,
      "full_generator_target_pass": false,
      "objective_kind": "direct_normalized_delta_MSE",
      "all148training_scenes_in_objective": true,
      "active_correction_scenes": 29,
      "fixed_exact_bypass_control_scenes": 119
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
  "schema": "rheon-internal-residual-managed-v4",
  "mode": "train",
  "role": "internal_group_nonlinear_residual_component",
  "parameters": 10876,
  "optimizer": "full_batch_LBFGS",
  "objective_kind": "direct_normalized_delta_MSE",
  "optimizer_updates": 24,
  "closure_evaluations": 51,
  "closure": {
    "config_sha256": "06d7feb415494202e2a0c0e03c77895415cc2d64ebf75d0800b541b10164f6a0",
    "inputs": {
      "manifest": {
        "bytes": 4625289,
        "sha256": "1cec5c9255b9e40e8fb9391acd90604f5b7d09056a4ea05cbfbed0c2103a4768"
      },
      "normalization": {
        "bytes": 1521,
        "sha256": "2e3cb8d34ff8846a28d2b6155eaafbf5a732cd0aa28bf7526539d2a319bdbc21"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 671002882,
        "sha256": "9a2708e60365e8efaf74112fea72089f1e7a523aa104021c4c90c0feece66a36"
      },
      "optimization_normalization": {
        "bytes": 918,
        "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
      }
    },
    "role": "internal_group_nonlinear_residual_component",
    "sources": {
      "capture_memory_telemetry_v1.py": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204",
      "compact_internal_training_view_v2.py": "91dd922158a1e2b41347d6920a147ce7a01b09da050cd28266fc66796e0703cf",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "internal_residual_cached_view_v1.py": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8",
      "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
      "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
      "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
      "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
      "teacher-code.zip": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc",
      "train_internal_residual_v4.py": "896718840ea8cd470430bb917ee05838ba6b9354d4b3695be8982264b120ba98"
    }
  },
  "initial_inventory": {
    "input_rms": {
      "shape": [
        1,
        42,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "03607579d061e0a3afb7037cc5e7bbf69985070dc0e4e3aefa0f29451a69108d"
    },
    "output_rms": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "f6dcc9570fba7f8d765ace2909fe76a00d84a802e25b8432ac2d523d5ecd0d87"
    },
    "conv1.weight": {
      "shape": [
        8,
        42,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "270298ab5a33aefa09e0463d5828d9d017de53c80dda340b757dcfbb84b8fd85"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "7b3dfd86627f558c210b56cc6f458634643e8f30023b2e62c1c509f9fbec670c"
    },
    "conv2.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "61a2a0174ef4d64ec6fae53e3cf179b66b3c25c4f6a49e308006a912840e215f"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "6b4e6759f3b856c1f1fb3095db3e9d97dcde9748c8a3632e61c3a2582fab316b"
    },
    "velocity.0.weight": {
      "shape": [
        8,
        1,
        1,
        1,
        2
      ],
      "dtype": "<f4",
      "sha256": "ddb564f517fc95076dfe0de99795a982743c7f8ae419f3d5ad33c4ddff2d22cf"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "bf0a2d788588181c6fbf94324ab97d41cbfa908ff4013f0f7734a4513fc9607d"
    },
    "velocity.1.weight": {
      "shape": [
        8,
        1,
        1,
        2,
        1
      ],
      "dtype": "<f4",
      "sha256": "bf0b7dfc52fe77a5831da48d43788f40091bf1ee2503880bd6bbf00081ecc535"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "356c75ddc004ee90e9f56efc0eb62b3a451173cfff5e625a05646c84879c20a2"
    },
    "velocity.2.weight": {
      "shape": [
        8,
        1,
        2,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b5be0eaf2defe41df7c3d8bee731865e16eba1a462e362be95454281fe56fffb"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "5cd606ead5f045b362651d20493041509020b77b7f7d0ec18425e2f9ea5708bb"
    },
    "tracer.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "e050a74b3abbb0682d0f55c9d16234887b8ed0fc0bec591a5f348aa11d6217ca"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "c5f48bd23b5f2bb9697c26b31548360b643a148c8e80f234e6a7d7e683e30517"
    }
  },
  "final_inventory": {
    "input_rms": {
      "shape": [
        1,
        42,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "03607579d061e0a3afb7037cc5e7bbf69985070dc0e4e3aefa0f29451a69108d"
    },
    "output_rms": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "f6dcc9570fba7f8d765ace2909fe76a00d84a802e25b8432ac2d523d5ecd0d87"
    },
    "conv1.weight": {
      "shape": [
        8,
        42,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "0be0d37e0ac06cc20dc8d683e247710d689e24a27a51e353b6c8e16b53eec803"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "7d9f60a5e767f9d8db6ff983c97790e98a721836f9303550cb316438d471b0b3"
    },
    "conv2.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "468afd13b4c776e701068be4c2bf7a160f7d20b800e4e4133d16fab0aeb9e936"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "3510542ba9b9e3103f98b13689484ef9e03ff6ddfb624b05ebe8bfb305f17698"
    },
    "velocity.0.weight": {
      "shape": [
        8,
        1,
        1,
        1,
        2
      ],
      "dtype": "<f4",
      "sha256": "b7f3bc162fdab0e430da8a7c24eac0c06dcef947d01e781d954a18d0ca059ac5"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "c7706b6e08d9639cef51d07b9b8df9f0f1933490703e70d9afba1868f148de55"
    },
    "velocity.1.weight": {
      "shape": [
        8,
        1,
        1,
        2,
        1
      ],
      "dtype": "<f4",
      "sha256": "24d8efbaff901fb677e1abe436ef1ba80aa0080e1da67c472902945286c3ca82"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "7ec05a22a7f9de335f4439b71989cefaca6f2f2aa22bd223b9d7b0d8bf1d2661"
    },
    "velocity.2.weight": {
      "shape": [
        8,
        1,
        2,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "196b637f889a2a0e0631d0aedbfbebffe4565b5ac0040e025045d923b60bc9d0"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "83e6cf383a0f445af742bd9627dda121250451a261e19dbffa432302e3eec91f"
    },
    "tracer.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "1a5c8db16c9cb27c4dbd9915db072cd6d22c8a81c579a75bdc5c296d2989d531"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "0da6468c35b0f3b266924c86edd94352ec22b60d964ba7aedbb799d1dcd1f55f"
    }
  },
  "execution": {
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "all_learning_and_forward_on_cuda": true,
    "tf32": false,
    "amp": false,
    "cached_tensor_bytes": 13765858200,
    "memory_counters": {
      "peak_since_exec_bytes": 2212163584,
      "current_rss_bytes": 2207236096,
      "getrusage_peak_including_preexec_bytes": 2212163584
    },
    "peak_cuda_allocated_bytes": 14121407488,
    "peak_cuda_reserved_bytes": 14543749120,
    "ambient_repository_reads_denied": []
  },
  "outputs": {
    "last-evaluated-weights.npz": {
      "bytes": 47324,
      "sha256": "2af8d84d4517471f890584393e96efa83e1e11296de73d361c38f81e5068e70e"
    },
    "probe.npy": {
      "bytes": 630400,
      "sha256": "fda5205aad05b496e489283ad07fb00a6c8f6e2ef92f4d18def3c3b5d6d88360"
    },
    "weights.npz": {
      "bytes": 47324,
      "sha256": "2af8d84d4517471f890584393e96efa83e1e11296de73d361c38f81e5068e70e"
    },
    "initial-weights.npz": {
      "bytes": 47324,
      "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2"
    },
    "optimization-normalization.json": {
      "bytes": 918,
      "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
    },
    "history.json": {
      "bytes": 63473,
      "sha256": "088e370276561c5a8333c7325202cdb69382cc82b04487cc909bb389ba674d4e"
    },
    "normalization.json": {
      "bytes": 1521,
      "sha256": "2e3cb8d34ff8846a28d2b6155eaafbf5a732cd0aa28bf7526539d2a319bdbc21"
    }
  },
  "originating_training_run": null,
  "typed_cpu_weight_restore_verified": false,
  "gpu_probe_copied_without_cpu_forward": false,
  "full_cpu_probe_bitwise_verified": false,
  "cpu_model_forward_executed": false,
  "selected_evaluation": 51,
  "selected_full148scene_objective": 0.7451582695603169,
  "probe": {
    "index": 0,
    "field_order": [
      "pressure",
      "vx",
      "vy",
      "vz",
      "tracer"
    ],
    "shapes": {
      "pressure": [
        24,
        40,
        16
      ],
      "vx": [
        24,
        40,
        17
      ],
      "vy": [
        24,
        41,
        16
      ],
      "vz": [
        25,
        40,
        16
      ],
      "tracer": [
        24,
        40,
        16
      ]
    },
    "elements": 78784,
    "dtype": "<f8"
  },
  "training_scenes": 148,
  "expanded_reserved52_references_opened": 0,
  "native_accuracy_claim": false,
  "latency_claim": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_seconds": 371.976731673989
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "internal-residual-s4-direct-delta-LBFGS",
  "role": "internal_group_nonlinear_residual_component",
  "seed": 1729,
  "optimizer": "full_batch_LBFGS",
  "maximum_optimizer_iterations": 24,
  "maximum_closure_evaluations": 60,
  "initial_step_learning_rate": 1.0,
  "subsequent_step_learning_rate": 1.0,
  "selection": "minimum_full148scene_direct_delta_objective_over_initial_and_complete_closures_earlier_tie",
  "qualified_initial_objective": 0.7665071603885028,
  "qualified_initial_inventory": {
    "input_rms": {
      "shape": [
        1,
        42,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "03607579d061e0a3afb7037cc5e7bbf69985070dc0e4e3aefa0f29451a69108d"
    },
    "output_rms": {
      "shape": [
        4
      ],
      "dtype": "<f4",
      "sha256": "f6dcc9570fba7f8d765ace2909fe76a00d84a802e25b8432ac2d523d5ecd0d87"
    },
    "conv1.weight": {
      "shape": [
        8,
        42,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "270298ab5a33aefa09e0463d5828d9d017de53c80dda340b757dcfbb84b8fd85"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "7b3dfd86627f558c210b56cc6f458634643e8f30023b2e62c1c509f9fbec670c"
    },
    "conv2.weight": {
      "shape": [
        8,
        8,
        3,
        3,
        3
      ],
      "dtype": "<f4",
      "sha256": "61a2a0174ef4d64ec6fae53e3cf179b66b3c25c4f6a49e308006a912840e215f"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "6b4e6759f3b856c1f1fb3095db3e9d97dcde9748c8a3632e61c3a2582fab316b"
    },
    "velocity.0.weight": {
      "shape": [
        8,
        1,
        1,
        1,
        2
      ],
      "dtype": "<f4",
      "sha256": "ddb564f517fc95076dfe0de99795a982743c7f8ae419f3d5ad33c4ddff2d22cf"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "bf0a2d788588181c6fbf94324ab97d41cbfa908ff4013f0f7734a4513fc9607d"
    },
    "velocity.1.weight": {
      "shape": [
        8,
        1,
        1,
        2,
        1
      ],
      "dtype": "<f4",
      "sha256": "bf0b7dfc52fe77a5831da48d43788f40091bf1ee2503880bd6bbf00081ecc535"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "356c75ddc004ee90e9f56efc0eb62b3a451173cfff5e625a05646c84879c20a2"
    },
    "velocity.2.weight": {
      "shape": [
        8,
        1,
        2,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "b5be0eaf2defe41df7c3d8bee731865e16eba1a462e362be95454281fe56fffb"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "5cd606ead5f045b362651d20493041509020b77b7f7d0ec18425e2f9ea5708bb"
    },
    "tracer.weight": {
      "shape": [
        1,
        8,
        1,
        1,
        1
      ],
      "dtype": "<f4",
      "sha256": "e050a74b3abbb0682d0f55c9d16234887b8ed0fc0bec591a5f348aa11d6217ca"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "c5f48bd23b5f2bb9697c26b31548360b643a148c8e80f234e6a7d7e683e30517"
    }
  },
  "qualified_initial_gradient_norms": {
    "conv1.weight": 0.014344844964333943,
    "conv1.bias": 8.793608542694665e-05,
    "conv2.weight": 0.027295263879977428,
    "conv2.bias": 0.0001023488575278503,
    "velocity.0.weight": 0.01564564570092267,
    "velocity.0.bias": 0.007132446393370628,
    "velocity.1.weight": 0.13045994696089047,
    "velocity.1.bias": 0.003620981937274337,
    "velocity.2.weight": 0.09200242459700894,
    "velocity.2.bias": 0.005964244250208139,
    "tracer.weight": 0.0027318245620267466,
    "tracer.bias": 0.0003040946612600237
  },
  "expected_cached_tensor_bytes": 13765858200,
  "wall_seconds": 1800,
  "maximum_peak_rss_bytes": 4294967296,
  "maximum_gpu_allocated_bytes": 17179869184,
  "maximum_gpu_reserved_bytes": 21474836480,
  "scientific_intervention": "S3 exact warm start; directly supervise all four missing corrections with frozen per-head output RMS. Physical losses tracked separately; native gates unchanged; no architecture change.",
  "native_accuracy_claim": false,
  "latency_claim": false,
  "full_generator_target_pass": false,
  "objective_kind": "direct_normalized_delta_MSE",
  "all148training_scenes_in_objective": true,
  "active_correction_scenes": 29,
  "fixed_exact_bypass_control_scenes": 119
}

```
