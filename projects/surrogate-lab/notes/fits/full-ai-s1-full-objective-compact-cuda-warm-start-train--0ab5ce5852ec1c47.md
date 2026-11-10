# Full-AI S1 full-objective compact CUDA warm start train

Retrospective source review · fit-0ab5ce5852ec1c47

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A10,876-parameter float32 CUDA correction CNN:42 cell-lattice input channels, two width8 clamped3×3×3 convolutions with SiLU, three staggered transposed-convolution velocity heads and one tracer head. Output heads initialize to zero. The quiescence gate uses current/coarse-change/known-force channels. Pressure correction is supplied through the frozen learned coupled projection; this subnet does not directly emit pressure. Inputs include spacing100h, group duration480dt, group width and ordered known forces; future fields are labels only.

**Purpose.** Continue the exact selected S1 weights using deterministic full-objective quasi-Newton optimization instead of single-sample AdamW; preserve the same correction network and physical objective.

**Experiment.** CUDA full-batch LBFGS, 8 maximum iterations and 18 maximum closures. Cache the same teacher-derived inputs/targets on GPU; select the best complete closure including the exact warm start. Recorded internal iteration counters are not minibatch update counts. All148 training scenes are counted, but only29N8 scenes provide parameter-dependent gradients:9 starts×3 groups=783 local samples. The119N1 scenes contribute1,071 fixed control windows with zero parameter derivative. Active grids span eight shapes through64³, at8/10/16/25/50/100cells per metre; the complete fixed/control population includes19shapes through128³.

**Observed outcome.** Selected training objective 1.9016923678013046e-05; selected closure 18 after 7 recorded LBFGS iterations / 18 closure evaluations. The retained complete-generator replay passes all 920 N1 outputs and fails 1151 of 1200 N8 outputs at the diagnostic1e-5 gate. These are original consumed cases, not independent validation or qualified new latency.

**Lesson.** Optimization improvements on local teacher-state corrections do not establish accurate own-history reduced integration. The N8 runner advances groups2/2/3/1 with three correction calls per physical1/60s output; K1–K4 outputs still advance sequentially. It does not predict a distant endpoint with cheap temporal intermediate frames. N1 bypasses this correction and preserves the original branch. Keep effective gradient scenes, fixed controls, loss choice, trained checkpoint and complete execution variant separate.

**Reproduction.** Use the exact captured trainer/configuration/static inputs and retained selected/initial inventories shown here and in the original facts below. S1/S2 share initialization; S3 starts from selected S1, S4 from selected S3. The physical-state view is prepared by prepare_internal_residual_view_v1.py; compact views regenerate teacher labels with the captured teacher-code archive and learned assets. Static inputs and checkpoint tensors were freshly checked; the9.77GB full /0.671GB compact archives were not reread, labels were not decoded and clean retraining/new-Rheon regeneration remain unqualified. CPU restore receipts copy the CUDA probe without a CPU model forward.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-pair-01/train/receipt.json",
    "sha256": "0ab5ce5852ec1c4748a4c2998a70418a70496bc57a25b031d0a15033178d7f65"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-pair-01/train/history.json",
    "sha256": "dcd34334672dc9ecc4b77b070f22d93bfd8983524638c4d2fa0868d35253d473"
  },
  "sources": [
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/capture_memory_telemetry_v1.py",
      "sha256": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/compact_internal_training_view_v1.py",
      "sha256": "98cd5e65667541f67c8120e9ed0c8de1122323db2dd4c0076357fdfa9786fb68"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/internal_residual_cached_view_v1.py",
      "sha256": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/internal_residual_dataset_v1.py",
      "sha256": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/internal_residual_loss_v1.py",
      "sha256": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/internal_residual_network_v1.py",
      "sha256": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/internal_residual_network_v2.py",
      "sha256": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/managed_input_stream_v1.py",
      "sha256": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/teacher-code.zip",
      "sha256": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/train_internal_residual_v3.py",
      "sha256": "42ab54bc5da9a93b554923b5836493c488fbe9f96e9a779bfed23101ca61de3a"
    }
  ],
  "training_inputs": [
    {
      "slot": "manifest",
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-data-01/manifest.json",
      "sha256": "807526eb4e9d6ffcc1b65d70a58293846bb1156139218a5ad619332d983175b3",
      "bytes": 4624750
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
    "item_id": "fit-0ab5ce5852ec1c47",
    "kind": "training_attempt",
    "name": "Full-AI S1 full-objective compact CUDA warm start train",
    "task": "full_state_correction_component",
    "schema": "rheon-internal-residual-managed-v3",
    "method": {},
    "receipt_sha256": "0ab5ce5852ec1c4748a4c2998a70418a70496bc57a25b031d0a15033178d7f65",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/internal-residual-s3-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "694cb1b50123400190be2e1ddb3d6029",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full-AI S1 full-objective compact CUDA warm start train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 7,
    "parameter_fits": null,
    "declared_parameter_count": 10876,
    "training_closure": {
      "config_sha256": "a9b87388c47d3b7e8ed272b6e18d7a64a382ddb545f6c42eb9917aedfce8c1c0",
      "inputs": {
        "manifest": {
          "bytes": 4624750,
          "sha256": "807526eb4e9d6ffcc1b65d70a58293846bb1156139218a5ad619332d983175b3"
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
        "compact_internal_training_view_v1.py": "98cd5e65667541f67c8120e9ed0c8de1122323db2dd4c0076357fdfa9786fb68",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "internal_residual_cached_view_v1.py": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8",
        "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
        "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
        "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
        "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
        "teacher-code.zip": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc",
        "train_internal_residual_v3.py": "42ab54bc5da9a93b554923b5836493c488fbe9f96e9a779bfed23101ca61de3a"
      }
    },
    "checkpoints": [
      {
        "name": "last-evaluated-weights.npz",
        "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2",
        "bytes": 47324,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2",
        "bytes": 47324,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "0717cab7f65627831c474db95fba2335b13101b6ea3ec30e3ef0549e36eedba0",
        "bytes": 47324,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "residual-fits/2"
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
      "receipt": "runs/full-ai-data-diversity-v1/internal-residual-s3-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_correction_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s3-pair-01/train/receipt.json",
    "sha256": "0ab5ce5852ec1c4748a4c2998a70418a70496bc57a25b031d0a15033178d7f65",
    "bytes": 10478
  },
  "bound_training_configuration": {
    "config_sha256": "a9b87388c47d3b7e8ed272b6e18d7a64a382ddb545f6c42eb9917aedfce8c1c0",
    "graph_id": "a13aa929cfaa46c4aeb78d7b110c6e75",
    "graph_document_sha256": "6d468c7b524ebbed9d871c10fc8d4fc64375f9dc4da1328e73305afd66c41211",
    "recipe_path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/train-recipe.json",
    "recipe_sha256": "3d4b08285f9a0f4f726daad54076eed9769d736f54d7c8eeb52520319b78ec92",
    "config_path": "runs/full-ai-data-diversity-v1/internal-residual-s3-project-02/config.json",
    "configuration": {
      "identifier": "internal-residual-s3-full-LBFGS",
      "role": "internal_group_nonlinear_residual_component",
      "seed": 1729,
      "optimizer": "full_batch_LBFGS",
      "maximum_optimizer_iterations": 8,
      "maximum_closure_evaluations": 18,
      "initial_step_learning_rate": 50.0,
      "subsequent_step_learning_rate": 1.0,
      "selection": "minimum_full148scene_objective_over_initial_and_complete_closures_earlier_tie",
      "qualified_initial_objective": 1.9069764311845483e-05,
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
          "sha256": "4990e54147940fc85685116f9daf024448c0c6d49f0c67127a04fc5232e970d4"
        },
        "conv1.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "c0164443b52dc2ea7943c091a71c86908dd20d21fa6a358b51c1b6d5bfc16e8b"
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
          "sha256": "5a332823e383b89e2cac963fc0fcb5f24d0c512b479a3547925334e697ab0f39"
        },
        "conv2.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "6cae470ef1ff26f9bd6ac87548f1a82fce9a3b33a8065bbef76f0a80d1e7d367"
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
          "sha256": "3807d2fe81b600d3189925e622654e99b415c7d4c73bf34c28c9485348c3ef00"
        },
        "velocity.0.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "dd4f5b9c0f5a7cca5c36777fc8d366ae121a2579e9961fa7bbc091d8270f6cff"
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
          "sha256": "f6975ea322ae7532c9465939e3d47f3d11795d7d6e580702deea3d8654361e8f"
        },
        "velocity.1.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "05aa576101ad83c038777ee25b1cc03ca025dccf18f09c7948b973c258ccd31d"
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
          "sha256": "0fde1d232f537f6349fddcc1a2f26c9199fb5d620a36709944c10b28d20d0fb3"
        },
        "velocity.2.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "f89fdc2cb9746b9fd8b976c3c4c8d6df6e8bd2ce3c37acd3968096f03952fd4b"
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
          "sha256": "fc2267ee58b80ee367d84447166c30edd90e8ea4343e8d9bc6d079bdc02bbb80"
        },
        "tracer.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "9b9711d72dcacee2e3e7141ee2f902e2c6067f820fec72adfa074c64dc6aae12"
        }
      },
      "qualified_initial_gradient_norms": {
        "conv1.weight": 1.1427520884771442e-07,
        "conv1.bias": 8.331031871396027e-09,
        "conv2.weight": 2.4826715456768083e-07,
        "conv2.bias": 3.6745625716993237e-09,
        "velocity.0.weight": 2.464731686352843e-07,
        "velocity.0.bias": 2.093575858452823e-05,
        "velocity.1.weight": 5.992363061004961e-07,
        "velocity.1.bias": 2.5937119062291458e-05,
        "velocity.2.weight": 2.0901905578384438e-07,
        "velocity.2.bias": 2.11969072552165e-05,
        "tracer.weight": 3.9434248817401046e-11,
        "tracer.bias": 5.1852853608092e-12
      },
      "expected_cached_tensor_bytes": 12127297320,
      "wall_seconds": 900,
      "maximum_peak_rss_bytes": 4294967296,
      "maximum_gpu_allocated_bytes": 17179869184,
      "maximum_gpu_reserved_bytes": 21474836480,
      "scientific_intervention": "Exact managed-restored S1warm start; deterministic full-objective LBFGS replaces single-sample AdamW. Initial step50matches measured local curvature, then unit quasi-Newton scale.",
      "native_accuracy_claim": false,
      "latency_claim": false,
      "full_generator_target_pass": false
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
  "schema": "rheon-internal-residual-managed-v3",
  "mode": "train",
  "role": "internal_group_nonlinear_residual_component",
  "parameters": 10876,
  "optimizer": "full_batch_LBFGS",
  "optimizer_updates": 7,
  "closure_evaluations": 18,
  "closure": {
    "config_sha256": "a9b87388c47d3b7e8ed272b6e18d7a64a382ddb545f6c42eb9917aedfce8c1c0",
    "inputs": {
      "manifest": {
        "bytes": 4624750,
        "sha256": "807526eb4e9d6ffcc1b65d70a58293846bb1156139218a5ad619332d983175b3"
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
      "compact_internal_training_view_v1.py": "98cd5e65667541f67c8120e9ed0c8de1122323db2dd4c0076357fdfa9786fb68",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "internal_residual_cached_view_v1.py": "b22804da2bcb3118f4fc4767d8fd35b4d83a797beaed7b70ac6b3077b702a6c8",
      "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
      "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
      "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
      "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
      "teacher-code.zip": "f105095b2f86fcb07556263c76b2ae1df429c5e602b5fb51a04bd4102b4d1acc",
      "train_internal_residual_v3.py": "42ab54bc5da9a93b554923b5836493c488fbe9f96e9a779bfed23101ca61de3a"
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
      "sha256": "4990e54147940fc85685116f9daf024448c0c6d49f0c67127a04fc5232e970d4"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "c0164443b52dc2ea7943c091a71c86908dd20d21fa6a358b51c1b6d5bfc16e8b"
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
      "sha256": "5a332823e383b89e2cac963fc0fcb5f24d0c512b479a3547925334e697ab0f39"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "6cae470ef1ff26f9bd6ac87548f1a82fce9a3b33a8065bbef76f0a80d1e7d367"
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
      "sha256": "3807d2fe81b600d3189925e622654e99b415c7d4c73bf34c28c9485348c3ef00"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "dd4f5b9c0f5a7cca5c36777fc8d366ae121a2579e9961fa7bbc091d8270f6cff"
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
      "sha256": "f6975ea322ae7532c9465939e3d47f3d11795d7d6e580702deea3d8654361e8f"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "05aa576101ad83c038777ee25b1cc03ca025dccf18f09c7948b973c258ccd31d"
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
      "sha256": "0fde1d232f537f6349fddcc1a2f26c9199fb5d620a36709944c10b28d20d0fb3"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "f89fdc2cb9746b9fd8b976c3c4c8d6df6e8bd2ce3c37acd3968096f03952fd4b"
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
      "sha256": "fc2267ee58b80ee367d84447166c30edd90e8ea4343e8d9bc6d079bdc02bbb80"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "9b9711d72dcacee2e3e7141ee2f902e2c6067f820fec72adfa074c64dc6aae12"
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
  "execution": {
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "all_learning_and_forward_on_cuda": true,
    "tf32": false,
    "amp": false,
    "cached_tensor_bytes": 12127297320,
    "memory_counters": {
      "peak_since_exec_bytes": 2380492800,
      "current_rss_bytes": 2380492800,
      "getrusage_peak_including_preexec_bytes": 2380230656
    },
    "peak_cuda_allocated_bytes": 12482437120,
    "peak_cuda_reserved_bytes": 12918456320,
    "ambient_repository_reads_denied": []
  },
  "outputs": {
    "last-evaluated-weights.npz": {
      "bytes": 47324,
      "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2"
    },
    "probe.npy": {
      "bytes": 630400,
      "sha256": "f8f0a0a1f99d8615021ebb85c2136ca2a7820415b253973772a252fc6a15e978"
    },
    "weights.npz": {
      "bytes": 47324,
      "sha256": "1ec83ae15776b2fed2b6f0ae125b203afc02ac9de39b1ff27451825edacdbcb2"
    },
    "initial-weights.npz": {
      "bytes": 47324,
      "sha256": "0717cab7f65627831c474db95fba2335b13101b6ea3ec30e3ef0549e36eedba0"
    },
    "optimization-normalization.json": {
      "bytes": 918,
      "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
    },
    "history.json": {
      "bytes": 20211,
      "sha256": "dcd34334672dc9ecc4b77b070f22d93bfd8983524638c4d2fa0868d35253d473"
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
  "selected_evaluation": 18,
  "selected_full148scene_objective": 1.9016923678013046e-05,
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
  "elapsed_seconds": 208.056053125998
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "internal-residual-s3-full-LBFGS",
  "role": "internal_group_nonlinear_residual_component",
  "seed": 1729,
  "optimizer": "full_batch_LBFGS",
  "maximum_optimizer_iterations": 8,
  "maximum_closure_evaluations": 18,
  "initial_step_learning_rate": 50.0,
  "subsequent_step_learning_rate": 1.0,
  "selection": "minimum_full148scene_objective_over_initial_and_complete_closures_earlier_tie",
  "qualified_initial_objective": 1.9069764311845483e-05,
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
      "sha256": "4990e54147940fc85685116f9daf024448c0c6d49f0c67127a04fc5232e970d4"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "c0164443b52dc2ea7943c091a71c86908dd20d21fa6a358b51c1b6d5bfc16e8b"
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
      "sha256": "5a332823e383b89e2cac963fc0fcb5f24d0c512b479a3547925334e697ab0f39"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "6cae470ef1ff26f9bd6ac87548f1a82fce9a3b33a8065bbef76f0a80d1e7d367"
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
      "sha256": "3807d2fe81b600d3189925e622654e99b415c7d4c73bf34c28c9485348c3ef00"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "dd4f5b9c0f5a7cca5c36777fc8d366ae121a2579e9961fa7bbc091d8270f6cff"
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
      "sha256": "f6975ea322ae7532c9465939e3d47f3d11795d7d6e580702deea3d8654361e8f"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "05aa576101ad83c038777ee25b1cc03ca025dccf18f09c7948b973c258ccd31d"
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
      "sha256": "0fde1d232f537f6349fddcc1a2f26c9199fb5d620a36709944c10b28d20d0fb3"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "f89fdc2cb9746b9fd8b976c3c4c8d6df6e8bd2ce3c37acd3968096f03952fd4b"
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
      "sha256": "fc2267ee58b80ee367d84447166c30edd90e8ea4343e8d9bc6d079bdc02bbb80"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "9b9711d72dcacee2e3e7141ee2f902e2c6067f820fec72adfa074c64dc6aae12"
    }
  },
  "qualified_initial_gradient_norms": {
    "conv1.weight": 1.1427520884771442e-07,
    "conv1.bias": 8.331031871396027e-09,
    "conv2.weight": 2.4826715456768083e-07,
    "conv2.bias": 3.6745625716993237e-09,
    "velocity.0.weight": 2.464731686352843e-07,
    "velocity.0.bias": 2.093575858452823e-05,
    "velocity.1.weight": 5.992363061004961e-07,
    "velocity.1.bias": 2.5937119062291458e-05,
    "velocity.2.weight": 2.0901905578384438e-07,
    "velocity.2.bias": 2.11969072552165e-05,
    "tracer.weight": 3.9434248817401046e-11,
    "tracer.bias": 5.1852853608092e-12
  },
  "expected_cached_tensor_bytes": 12127297320,
  "wall_seconds": 900,
  "maximum_peak_rss_bytes": 4294967296,
  "maximum_gpu_allocated_bytes": 17179869184,
  "maximum_gpu_reserved_bytes": 21474836480,
  "scientific_intervention": "Exact managed-restored S1warm start; deterministic full-objective LBFGS replaces single-sample AdamW. Initial step50matches measured local curvature, then unit quasi-Newton scale.",
  "native_accuracy_claim": false,
  "latency_claim": false,
  "full_generator_target_pass": false
}

```
