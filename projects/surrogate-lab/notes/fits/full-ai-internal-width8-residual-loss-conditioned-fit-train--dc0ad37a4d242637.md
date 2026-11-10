# Full-AI internal width8 residual loss-conditioned fit train

Retrospective source review · fit-dc0ad37a4d242637

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A10,876-parameter float32 CUDA correction CNN:42 cell-lattice input channels, two width8 clamped3×3×3 convolutions with SiLU, three staggered transposed-convolution velocity heads and one tracer head. Output heads initialize to zero. The quiescence gate uses current/coarse-change/known-force channels. Pressure correction is supplied through the frozen learned coupled projection; this subnet does not directly emit pressure. Inputs include spacing100h, group duration480dt, group width and ordered known forces; future fields are labels only.

**Purpose.** Change only training metric conditioning while preserving S1 architecture, initialization, ordering, scene population and two-epoch optimizer schedule.

**Experiment.** Two CUDA AdamW epochs consume all783 active scene/start/group samples each, cosine learning rate0.003→0.0003, epsilon1e-12, zero weight decay and gradient clip1. Select the minimum complete training objective including initialization; no development/reserved selection. Training metric weights are equation0.3686505980, pressure0.7769568476, velocity13044.41817 and tracer3922.417979; physical acceptance metrics remain unchanged. All148 training scenes are counted, but only29N8 scenes provide parameter-dependent gradients:9 starts×3 groups=783 local samples. The119N1 scenes contribute1,071 fixed control windows with zero parameter derivative. Active grids span eight shapes through64³, at8/10/16/25/50/100cells per metre; the complete fixed/control population includes19shapes through128³.

**Observed outcome.** Selected training objective 1.9138810975032417e-05; selected epoch 0 after 1,566 updates. The selected complete tensor inventory exactly equals initialization; the zero-output heads remain selected despite executing the training updates. The retained complete-generator replay passes all 920 N1 outputs and fails 1147 of 1200 N8 outputs at the diagnostic1e-5 gate. These are original consumed cases, not independent validation or qualified new latency.

**Lesson.** Optimization improvements on local teacher-state corrections do not establish accurate own-history reduced integration. The N8 runner advances groups2/2/3/1 with three correction calls per physical1/60s output; K1–K4 outputs still advance sequentially. It does not predict a distant endpoint with cheap temporal intermediate frames. N1 bypasses this correction and preserves the original branch. Keep effective gradient scenes, fixed controls, loss choice, trained checkpoint and complete execution variant separate.

**Reproduction.** Use the exact captured trainer/configuration/static inputs and retained selected/initial inventories shown here and in the original facts below. S1/S2 share initialization; S3 starts from selected S1, S4 from selected S3. The physical-state view is prepared by prepare_internal_residual_view_v1.py; compact views regenerate teacher labels with the captured teacher-code archive and learned assets. Static inputs and checkpoint tensors were freshly checked; the9.77GB full /0.671GB compact archives were not reread, labels were not decoded and clean retraining/new-Rheon regeneration remain unqualified. CPU restore receipts copy the CUDA probe without a CPU model forward.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/train/receipt.json",
    "sha256": "dc0ad37a4d2426371884797298b6ffde5fc2ec345a55f574c271e0db247d0099"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/train/history.json",
    "sha256": "46daa1ad05dd308d63ec47cb3aef45765f6a65aff700a11bd52f5f5ddd2c8e63"
  },
  "sources": [
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/capture_memory_telemetry_v1.py",
      "sha256": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/internal_residual_dataset_v1.py",
      "sha256": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/internal_residual_loss_v1.py",
      "sha256": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/internal_residual_network_v1.py",
      "sha256": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/internal_residual_network_v2.py",
      "sha256": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/internal_residual_weighted_loss_v1.py",
      "sha256": "25a7d5838c487cb977034771ed689376e9796f2dfdb174fb8f92456801df3828"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/managed_input_stream_v1.py",
      "sha256": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692"
    },
    {
      "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/train_internal_residual_v2.py",
      "sha256": "dcf3636648c90cacf75716af111ba804fb6016f55310d5ffd6408b8ec59a194a"
    }
  ],
  "training_inputs": [
    {
      "slot": "manifest",
      "path": "runs/full-ai-data-diversity-v1/internal-residual-view-01/manifest.json",
      "sha256": "37187b6c684035b40a3cffdc4286959dd7299700768384c53e83f106e92a5b0d",
      "bytes": 4538664
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
    "item_id": "fit-dc0ad37a4d242637",
    "kind": "training_attempt",
    "name": "Full-AI internal width8 residual loss-conditioned fit train",
    "task": "full_state_correction_component",
    "schema": "rheon-internal-residual-managed-v2",
    "method": {},
    "receipt_sha256": "dc0ad37a4d2426371884797298b6ffde5fc2ec345a55f574c271e0db247d0099",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "acb748bc23a54a1d88ae11a0c9848d09",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full-AI internal width8 residual loss-conditioned fit train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 1566,
    "parameter_fits": null,
    "declared_parameter_count": 10876,
    "training_closure": {
      "config_sha256": "39925d59e6424919fe35eccde7e57e778c96e06d91bddc31d9d671f58f63f877",
      "inputs": {
        "manifest": {
          "bytes": 4538664,
          "sha256": "37187b6c684035b40a3cffdc4286959dd7299700768384c53e83f106e92a5b0d"
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
          "bytes": 9769855506,
          "sha256": "a0c78a988ca74c659b5262eca252fc70b2f1de9f342b710f9d679c4e7b40850c"
        },
        "optimization_normalization": {
          "bytes": 918,
          "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
        }
      },
      "sources": {
        "capture_memory_telemetry_v1.py": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
        "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
        "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
        "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
        "internal_residual_weighted_loss_v1.py": "25a7d5838c487cb977034771ed689376e9796f2dfdb174fb8f92456801df3828",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
        "train_internal_residual_v2.py": "dcf3636648c90cacf75716af111ba804fb6016f55310d5ffd6408b8ec59a194a"
      },
      "role": "internal_group_nonlinear_residual_component"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "a6804c7c4ccd17893738061bca3808ce652956d62f6223700567bf59aa4ce66d",
        "bytes": 47324,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "a6804c7c4ccd17893738061bca3808ce652956d62f6223700567bf59aa4ce66d",
        "bytes": 47324,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "residual-fits/1"
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
      "receipt": "runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "full_state_correction_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/internal-residual-s2-pair-01/train/receipt.json",
    "sha256": "dc0ad37a4d2426371884797298b6ffde5fc2ec345a55f574c271e0db247d0099",
    "bytes": 10175
  },
  "bound_training_configuration": {
    "config_sha256": "39925d59e6424919fe35eccde7e57e778c96e06d91bddc31d9d671f58f63f877",
    "graph_id": "7990f214c5904268bef4e44f9e34bd7a",
    "graph_document_sha256": "59191ea0d59fd2b00b7dc7259161b9079f435ca4646b5fda02c0a0e17deae366",
    "recipe_path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/train-recipe.json",
    "recipe_sha256": "a8e2fc6017784e76c7457c2dc9741028aeac05d1d0da5dc085da32835ea1d1df",
    "config_path": "runs/full-ai-data-diversity-v1/internal-residual-s2-project-01/config.json",
    "configuration": {
      "identifier": "internal-residual-width8-s2-loss-conditioned",
      "role": "internal_group_nonlinear_residual_component",
      "seed": 1729,
      "epochs": 2,
      "optimizer_updates": 1566,
      "optimizer": "AdamW",
      "learning_rate": 0.003,
      "minimum_learning_rate": 0.0003,
      "adam_epsilon": 1e-12,
      "weight_decay": 0.0,
      "gradient_clip_norm": 1.0,
      "wall_seconds": 900,
      "maximum_peak_rss_bytes": 4294967296,
      "maximum_gpu_allocated_bytes": 8589934592,
      "selection": "minimum_full148scene_training_objective_earlier_epoch_tie",
      "qualified_initial_objective": 1.9138810975032417e-05,
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
          "sha256": "842af574b0f6c52f98dd3e36c5362f125388329f5ab90749f099b8b4197dc87b"
        },
        "conv1.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "bdeeb9033b3080885fa6048b0d6966e5d896bc2e4b71eb24d8ec587b61a2c4e6"
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
          "sha256": "b38d93285f61e12b76c774dd6f848a46170dec45a275712cb64cd7572ba1b1bf"
        },
        "conv2.bias": {
          "shape": [
            8
          ],
          "dtype": "<f4",
          "sha256": "28bf707bba31b12e08e935c5249573252a4fff62af2cd5add14ded24b759019e"
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
          "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        },
        "velocity.0.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
          "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        },
        "velocity.1.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
          "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        },
        "velocity.2.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
          "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
        },
        "tracer.bias": {
          "shape": [
            1
          ],
          "dtype": "<f4",
          "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
        }
      },
      "evaluation_normalization": {
        "status": "frozen_before_managed_fit",
        "source": "runs/full-ai-next-priorities-v1/temporal-training-view-02/normalization.json",
        "source_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
        "relative_denominator_rms_floor": {
          "pressure": 9.47211357321988e-08,
          "velocity": 6.465589999830133e-11,
          "tracer": 1.2995164216288916e-10
        },
        "degenerate_absolute_rms_tolerance": {
          "pressure": 9.47211357321988e-13,
          "velocity": 6.465589999830133e-16,
          "tracer": 1.2995164216288918e-15
        },
        "maximum_relative_error": 1e-05,
        "pressure_comparison": "Mean-free",
        "equation_rhs_floor": 1e-30
      },
      "native_accuracy_claim": false,
      "latency_claim": false,
      "full_generator_target_pass": false,
      "metric_weights": {
        "equation": 0.3686505980368089,
        "pressure": 0.7769568475978544,
        "velocity": 13044.418167693124,
        "tracer": 3922.417978570662
      },
      "weight_rule": "Training-only initial active mean total loss/4 divided by each metric mean; fixed controls use the same weights.",
      "comparison": "S1 architecture, initialization, training scenes, seed, ordering, optimizer and two epochs unchanged; explicit loss weights are the scientific intervention."
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
  "schema": "rheon-internal-residual-managed-v2",
  "mode": "train",
  "role": "internal_group_nonlinear_residual_component",
  "training_metric_weights": {
    "equation": 0.3686505980368089,
    "pressure": 0.7769568475978544,
    "velocity": 13044.418167693124,
    "tracer": 3922.417978570662
  },
  "parameters": 10876,
  "optimizer_updates": 1566,
  "closure": {
    "config_sha256": "39925d59e6424919fe35eccde7e57e778c96e06d91bddc31d9d671f58f63f877",
    "inputs": {
      "manifest": {
        "bytes": 4538664,
        "sha256": "37187b6c684035b40a3cffdc4286959dd7299700768384c53e83f106e92a5b0d"
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
        "bytes": 9769855506,
        "sha256": "a0c78a988ca74c659b5262eca252fc70b2f1de9f342b710f9d679c4e7b40850c"
      },
      "optimization_normalization": {
        "bytes": 918,
        "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
      }
    },
    "sources": {
      "capture_memory_telemetry_v1.py": "d744fde08087a2f7dec54471c6fe58edaf5dae5b52cf2a4b1532d7b32eb69204",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "internal_residual_dataset_v1.py": "6bee626665dc99fad6c0d692e876572f28dfaf846a7448d6eef8acab0c66f474",
      "internal_residual_loss_v1.py": "b44c6a2f63c2be8a10c910c8f741aafaa023217b940ddfda772edb745a9ceb0b",
      "internal_residual_network_v1.py": "c50c92f43ead3db86f4a0341df57380f66badf9fb763e910a2fd42e6368bcad3",
      "internal_residual_network_v2.py": "48fc8bd8316797949b98fec8a347a2f4e2b880ff00b21ea185275da771f75184",
      "internal_residual_weighted_loss_v1.py": "25a7d5838c487cb977034771ed689376e9796f2dfdb174fb8f92456801df3828",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
      "train_internal_residual_v2.py": "dcf3636648c90cacf75716af111ba804fb6016f55310d5ffd6408b8ec59a194a"
    },
    "role": "internal_group_nonlinear_residual_component"
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
      "sha256": "842af574b0f6c52f98dd3e36c5362f125388329f5ab90749f099b8b4197dc87b"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "bdeeb9033b3080885fa6048b0d6966e5d896bc2e4b71eb24d8ec587b61a2c4e6"
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
      "sha256": "b38d93285f61e12b76c774dd6f848a46170dec45a275712cb64cd7572ba1b1bf"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "28bf707bba31b12e08e935c5249573252a4fff62af2cd5add14ded24b759019e"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "842af574b0f6c52f98dd3e36c5362f125388329f5ab90749f099b8b4197dc87b"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "bdeeb9033b3080885fa6048b0d6966e5d896bc2e4b71eb24d8ec587b61a2c4e6"
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
      "sha256": "b38d93285f61e12b76c774dd6f848a46170dec45a275712cb64cd7572ba1b1bf"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "28bf707bba31b12e08e935c5249573252a4fff62af2cd5add14ded24b759019e"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    }
  },
  "execution": {
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "all_learning_and_forward_on_cuda": true,
    "residual_dtype": "float32",
    "coupled_loss_dtype": "float64",
    "tf32": false,
    "amp": false,
    "memory_counters": {
      "peak_since_exec_bytes": 2636009472,
      "current_rss_bytes": 2526728192,
      "getrusage_peak_including_preexec_bytes": 2636009472
    },
    "peak_cuda_allocated_bytes": 295615488
  },
  "outputs": {
    "probe.npy": {
      "bytes": 630400,
      "sha256": "7686d50e7e1d89278e19dafe612738b3810b86a13d794cd8cd71304538a64324"
    },
    "weights.npz": {
      "bytes": 47324,
      "sha256": "a6804c7c4ccd17893738061bca3808ce652956d62f6223700567bf59aa4ce66d"
    },
    "initial-weights.npz": {
      "bytes": 47324,
      "sha256": "a6804c7c4ccd17893738061bca3808ce652956d62f6223700567bf59aa4ce66d"
    },
    "optimization-normalization.json": {
      "bytes": 918,
      "sha256": "f5a9e68c448bcccbdc9aa583f6266225896656bcfd7605f909921d123a7bed11"
    },
    "history.json": {
      "bytes": 48239,
      "sha256": "46daa1ad05dd308d63ec47cb3aef45765f6a65aff700a11bd52f5f5ddd2c8e63"
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
  "selected_epoch": 0,
  "selected_full148scene_objective": 1.9138810975032417e-05,
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
  "cpu_model_forward_executed": false,
  "training_scenes": 148,
  "expanded_reserved52_references_opened": 0,
  "native_accuracy_claim": false,
  "latency_claim": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_seconds": 330.9034917950048
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "internal-residual-width8-s2-loss-conditioned",
  "role": "internal_group_nonlinear_residual_component",
  "seed": 1729,
  "epochs": 2,
  "optimizer_updates": 1566,
  "optimizer": "AdamW",
  "learning_rate": 0.003,
  "minimum_learning_rate": 0.0003,
  "adam_epsilon": 1e-12,
  "weight_decay": 0.0,
  "gradient_clip_norm": 1.0,
  "wall_seconds": 900,
  "maximum_peak_rss_bytes": 4294967296,
  "maximum_gpu_allocated_bytes": 8589934592,
  "selection": "minimum_full148scene_training_objective_earlier_epoch_tie",
  "qualified_initial_objective": 1.9138810975032417e-05,
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
      "sha256": "842af574b0f6c52f98dd3e36c5362f125388329f5ab90749f099b8b4197dc87b"
    },
    "conv1.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "bdeeb9033b3080885fa6048b0d6966e5d896bc2e4b71eb24d8ec587b61a2c4e6"
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
      "sha256": "b38d93285f61e12b76c774dd6f848a46170dec45a275712cb64cd7572ba1b1bf"
    },
    "conv2.bias": {
      "shape": [
        8
      ],
      "dtype": "<f4",
      "sha256": "28bf707bba31b12e08e935c5249573252a4fff62af2cd5add14ded24b759019e"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.0.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.1.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    },
    "velocity.2.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
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
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "tracer.bias": {
      "shape": [
        1
      ],
      "dtype": "<f4",
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    }
  },
  "evaluation_normalization": {
    "status": "frozen_before_managed_fit",
    "source": "runs/full-ai-next-priorities-v1/temporal-training-view-02/normalization.json",
    "source_sha256": "fb77ae2ee0fd4ef67969866be000561036b8311b6ca46f258ccd1cdd82c0c037",
    "relative_denominator_rms_floor": {
      "pressure": 9.47211357321988e-08,
      "velocity": 6.465589999830133e-11,
      "tracer": 1.2995164216288916e-10
    },
    "degenerate_absolute_rms_tolerance": {
      "pressure": 9.47211357321988e-13,
      "velocity": 6.465589999830133e-16,
      "tracer": 1.2995164216288918e-15
    },
    "maximum_relative_error": 1e-05,
    "pressure_comparison": "Mean-free",
    "equation_rhs_floor": 1e-30
  },
  "native_accuracy_claim": false,
  "latency_claim": false,
  "full_generator_target_pass": false,
  "metric_weights": {
    "equation": 0.3686505980368089,
    "pressure": 0.7769568475978544,
    "velocity": 13044.418167693124,
    "tracer": 3922.417978570662
  },
  "weight_rule": "Training-only initial active mean total loss/4 divided by each metric mean; fixed controls use the same weights.",
  "comparison": "S1 architecture, initialization, training scenes, seed, ordering, optimizer and two epochs unchanged; explicit loss weights are the scientific intervention."
}

```
