# Complete24scene guided residual width4 full-population CUDA fit train

Retrospective source review · fit-10d3a512b41ab2a6

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A width-four F32 correction network with 1,724 fitted coefficients surrounds a frozen learned full-field guide. One shared spatial trunk per block encodes current/predicted endpoint fields, spacing, physical horizon, ordered prescribed controls and within-block event offsets. Endpoint and intermediate heads correct velocity/tracer. Pressure correction is tied to the frozen learned projection of the velocity correction, with one correction-pressure response per physical output. The guide is evaluated without autograd; only the residual network is fitted.

**Purpose.** Fit small missing temporal corrections around a time-aware frozen guide rather than replacing all learned simulation operators or running more fine advances.

**Experiment.** All 24 independent training masters, 240 selected scene/start windows, 2,880 blocks and 7,200 physical outputs enter every complete score. Six grid resolutions (8/10/16/25/50/100 cells per metre), 16 shapes through 64³ and rectangular 128-cell axes, four force profiles and three motion types are present. Targets come from the qualified 240-step/s bank; every window covers 30/60/120 outputs/s crossed with K=1/2/3/4. Scenes, starts and temporal configurations are balanced, with 1/K weights within blocks. Four complete-gradient iterations each evaluate a normalized probe and quadratic candidate: eight recorded parameter trials, four mean-loss-accepted steps, not eight Adam updates. Probe step is 0.001. Future native states/RHS are labels only; guide-derived RMS sets correction units.

**Observed outcome.** Retained initial objective 0.00038572750643147162; selected candidate-4 objective 0.00038570072731590482, worst pressure 0.15726904156304763, equation 0.57289303769673949, velocity 0.11209480681022042, tracer 0.16502286946653341. Mean-best candidate-4 objective is 0.00038570072731590482. Selected, mean-best and last-evaluated checkpoints contain identical tensors; the small objective reduction leaves worst field errors essentially unchanged. These are original training-population scores, not fresh common-benchmark, held-out or latency results. The full-generator target was not met.

**Lesson.** Rank and metric definitions must accompany every checkpoint. Minimizing the maximum across equation/pressure/velocity/tracer can prefer a slightly improved dominant equation error while worsening other fields and average loss. Original scores use frozen RMS denominator floors and native target RHS; they cannot be equated with same-request pressure accuracy of the near-exact F64 anchors. Small corrections still require the complete frozen guide and its cost; fitted residual parameter count is not total generator size.

**Reproduction.** Retain the exact configuration, all five entrypoint source artifacts, captured model-source/asset ZIP, runtime and loss normalization, and the single lossless training archive. All source members and archived tensor states were checked in place; native field framing was checked but fields were not decoded. Reproduce the balanced complete scores, F64 gradient accumulation, F32 parameter trials and distinct worst/mean/last selections. Historical CPU restore constructs and checks typed weights and copies the saved GPU probe without neural inference. The original temporary Project root is absent; rebuilding its typed recipe bindings and a clean GPU-only replay/new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/receipt.json",
    "sha256": "10d3a512b41ab2a6944ff514c3cbd17b2e338aa28e66e171319d92afe1b92758"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/history.json",
    "sha256": "ef7dd447441832bb15a9a12afbf4ceab5fcc867b563e210ec81e0517fde369b5"
  },
  "sources": [
    {
      "path": "experiments/full-ai-data-diversity-v1/portable_shared_runtime_v1.py",
      "sha256": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6"
    },
    {
      "path": "experiments/full-ai-data-diversity-v1/managed_input_stream_v1.py",
      "sha256": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692"
    },
    {
      "path": "experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "experiments/full-ai-data-diversity-v1/train_guided_shared_v1.py",
      "sha256": "05539ca754088b4882bb3036ab19005162fd0551c8e46e8ae8a1c7dfef6f4526"
    }
  ],
  "training_inputs": [
    {
      "bytes": 3760799,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/ff4313531df10ad3f72ebd65aa73dfa65745b315931a86e6abd0ed121a1aceda",
      "sha256": "ff4313531df10ad3f72ebd65aa73dfa65745b315931a86e6abd0ed121a1aceda",
      "slot": "manifest"
    },
    {
      "bytes": 644,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14",
      "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 5827264,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/fce40f6b5cb92f257588b661c6b988772d08ed28083dc5e4d16e94689dba099c",
      "sha256": "fce40f6b5cb92f257588b661c6b988772d08ed28083dc5e4d16e94689dba099c",
      "slot": "model_source"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-10d3a512b41ab2a6",
    "kind": "training_attempt",
    "name": "Complete24scene guided residual width4 full-population CUDA fit train",
    "task": "guided_shared_component",
    "schema": "rheon.guided-shared-managed-fit.v1",
    "method": {},
    "receipt_sha256": "10d3a512b41ab2a6944ff514c3cbd17b2e338aa28e66e171319d92afe1b92758",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "96c3def4d43646849912807914b453ad",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Complete24scene guided residual width4 full-population CUDA fit train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 8,
    "parameter_fits": 1,
    "declared_parameter_count": null,
    "training_closure": {
      "config_sha256": "e7c03c57b0746131f2b0853d4533859f15e17fb9120edefa1a834d1ae6f1a6a3",
      "inputs": {
        "manifest": {
          "bytes": 3760799,
          "sha256": "ff4313531df10ad3f72ebd65aa73dfa65745b315931a86e6abd0ed121a1aceda"
        },
        "normalization": {
          "bytes": 644,
          "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "model_source": {
          "bytes": 5827264,
          "sha256": "fce40f6b5cb92f257588b661c6b988772d08ed28083dc5e4d16e94689dba099c"
        },
        "part_00": {
          "bytes": 2808112843,
          "sha256": "91123b4712cdb785ca7bf99805723be01323a05661fcba4f79790bfd375fa53c"
        }
      },
      "sources": {
        "portable_shared_runtime_v1.py": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "train_guided_shared_v1.py": "05539ca754088b4882bb3036ab19005162fd0551c8e46e8ae8a1c7dfef6f4526"
      }
    },
    "checkpoints": [
      {
        "name": "best-mean-weights.npz",
        "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e",
        "bytes": 15170,
        "stage": "retained_snapshot"
      },
      {
        "name": "last-evaluated-weights.npz",
        "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e",
        "bytes": 15170,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e",
        "bytes": 15170,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "d86420fcebb798588e50a010328de1ae8261e23150cf8a7a6389ddba4b3d86cd",
        "bytes": 15170,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": null,
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
      "receipt": "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "guided_shared_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/receipt.json",
    "sha256": "10d3a512b41ab2a6944ff514c3cbd17b2e338aa28e66e171319d92afe1b92758",
    "bytes": 28711
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
  },
  "bound_training_document": {
    "fit_item_id": "fit-10d3a512b41ab2a6",
    "document_kind": "configuration",
    "document_path": "runs/model-portfolio-migration-v1/guided-config-custody-recovery-01/exact-text-sources/e7c03c57b0746131f2b0853d4533859f15e17fb9120edefa1a834d1ae6f1a6a3.json",
    "expected_sha256": "e7c03c57b0746131f2b0853d4533859f15e17fb9120edefa1a834d1ae6f1a6a3",
    "binding_pointer": "/closure/config_sha256",
    "binding_field": "config_sha256",
    "recovery_report": "guided-config-custody-recovery-01.json",
    "primary_receipt_path": "runs/full-ai-data-diversity-v1/guided-shared-managed-pair-01/train/receipt.json",
    "primary_receipt_sha256": "10d3a512b41ab2a6944ff514c3cbd17b2e338aa28e66e171319d92afe1b92758",
    "document_bytes": 17019,
    "verified_direct_original_receipt_binding": true
  }
}
```

## Exact original receipt bytes

```json
{
  "schema": "rheon.guided-shared-managed-fit.v1",
  "mode": "train",
  "closure": {
    "config_sha256": "e7c03c57b0746131f2b0853d4533859f15e17fb9120edefa1a834d1ae6f1a6a3",
    "inputs": {
      "manifest": {
        "bytes": 3760799,
        "sha256": "ff4313531df10ad3f72ebd65aa73dfa65745b315931a86e6abd0ed121a1aceda"
      },
      "normalization": {
        "bytes": 644,
        "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "model_source": {
        "bytes": 5827264,
        "sha256": "fce40f6b5cb92f257588b661c6b988772d08ed28083dc5e4d16e94689dba099c"
      },
      "part_00": {
        "bytes": 2808112843,
        "sha256": "91123b4712cdb785ca7bf99805723be01323a05661fcba4f79790bfd375fa53c"
      }
    },
    "sources": {
      "portable_shared_runtime_v1.py": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "train_guided_shared_v1.py": "05539ca754088b4882bb3036ab19005162fd0551c8e46e8ae8a1c7dfef6f4526"
    }
  },
  "parameter_fits": 1,
  "optimizer_updates": 8,
  "accepted_steps": 4,
  "checkpoint_inventories": {
    "initial-weights.npz": {
      "force1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          24,
          3
        ],
        "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
      },
      "force1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
      },
      "force2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          3
        ],
        "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
      },
      "force2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
      },
      "event1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          5,
          1
        ],
        "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
      },
      "event1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
      },
      "event2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          1
        ],
        "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
      },
      "event2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
      },
      "mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          66,
          1,
          1,
          1
        ],
        "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
      },
      "local.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          4,
          3,
          3,
          3
        ],
        "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
      },
      "local.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
      },
      "endpoint_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "endpoint_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "endpoint_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "endpoint_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "interior_mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          12,
          1,
          1,
          1
        ],
        "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
      },
      "interior_mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
      },
      "interior_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "interior_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "interior_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      },
      "interior_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
      }
    },
    "weights.npz": {
      "force1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          24,
          3
        ],
        "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
      },
      "force1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
      },
      "force2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          3
        ],
        "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
      },
      "force2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
      },
      "event1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          5,
          1
        ],
        "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
      },
      "event1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
      },
      "event2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          1
        ],
        "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
      },
      "event2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
      },
      "mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          66,
          1,
          1,
          1
        ],
        "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
      },
      "local.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          4,
          3,
          3,
          3
        ],
        "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
      },
      "local.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
      },
      "endpoint_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "3cbf8802fb0cac8edc6e62439c15b98687bc7cb546179716b35f4e852c02fb1f"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "0bb90c990b0f64d8647198dc1ae80686902cdb96eeb29a04fb40a6c150bf808f"
      },
      "endpoint_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "1d8684c38a2c88d617c5c28c9fb00add45214cfaf9371c5abbaebbf80dbf62ad"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "b9aab2a36c888e747da2789945e947be2a34c278c96449d406625509786ff21a"
      },
      "endpoint_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "452c0e9fa00055fddfba512440d86b75ec7d78f2c9a2b88edb8e0a2cbf990eb0"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "fedd2610beb00863a188e935ea5f7bacc0cc809fd05fbae30572fd37d40ab1e2"
      },
      "endpoint_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "05cce95935a7f78438c4aaa11749eb30b455acac0bcc9680d0d21b2979bce5ba"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "64c4afee54fb98dc1ebcd64ab63c668d6344391e856452e220df83ea6201b6e4"
      },
      "interior_mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          12,
          1,
          1,
          1
        ],
        "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
      },
      "interior_mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
      },
      "interior_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "0e4fff8d268600c9b1b47d1881944a3cfb2b0601b6333262200f742c46079aa9"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e18b933f8e196f18f6491f68ff887dbce0dc4d49a9f835879846496e0f385cb7"
      },
      "interior_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "e8d6fb21b0575fc568327dcfab14cfa346b1e1f57445e5895f7626eeb7ddac59"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "4f53e2256296f46ddc229fc9cdab083ec6b49a2865c0c562459bd37662e4c02d"
      },
      "interior_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "73b54bd0f412bd3c991704554746c29c26e5474c6a734e9485e29b174a4ca238"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "2069c48276b75744441b1326534faa9f83423ddbdfe5c602978a335cd3e5f55a"
      },
      "interior_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "f748d035051ef1e22f5d73db3bb31a213b07b054089b1e1b4986bc0b26d3edcb"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e55bdd55a77dfd64bfe101a58ab5c457cba7e477eef77d7b89dcc0251aba2ac7"
      }
    },
    "best-mean-weights.npz": {
      "force1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          24,
          3
        ],
        "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
      },
      "force1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
      },
      "force2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          3
        ],
        "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
      },
      "force2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
      },
      "event1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          5,
          1
        ],
        "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
      },
      "event1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
      },
      "event2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          1
        ],
        "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
      },
      "event2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
      },
      "mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          66,
          1,
          1,
          1
        ],
        "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
      },
      "local.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          4,
          3,
          3,
          3
        ],
        "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
      },
      "local.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
      },
      "endpoint_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "3cbf8802fb0cac8edc6e62439c15b98687bc7cb546179716b35f4e852c02fb1f"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "0bb90c990b0f64d8647198dc1ae80686902cdb96eeb29a04fb40a6c150bf808f"
      },
      "endpoint_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "1d8684c38a2c88d617c5c28c9fb00add45214cfaf9371c5abbaebbf80dbf62ad"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "b9aab2a36c888e747da2789945e947be2a34c278c96449d406625509786ff21a"
      },
      "endpoint_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "452c0e9fa00055fddfba512440d86b75ec7d78f2c9a2b88edb8e0a2cbf990eb0"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "fedd2610beb00863a188e935ea5f7bacc0cc809fd05fbae30572fd37d40ab1e2"
      },
      "endpoint_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "05cce95935a7f78438c4aaa11749eb30b455acac0bcc9680d0d21b2979bce5ba"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "64c4afee54fb98dc1ebcd64ab63c668d6344391e856452e220df83ea6201b6e4"
      },
      "interior_mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          12,
          1,
          1,
          1
        ],
        "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
      },
      "interior_mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
      },
      "interior_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "0e4fff8d268600c9b1b47d1881944a3cfb2b0601b6333262200f742c46079aa9"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e18b933f8e196f18f6491f68ff887dbce0dc4d49a9f835879846496e0f385cb7"
      },
      "interior_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "e8d6fb21b0575fc568327dcfab14cfa346b1e1f57445e5895f7626eeb7ddac59"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "4f53e2256296f46ddc229fc9cdab083ec6b49a2865c0c562459bd37662e4c02d"
      },
      "interior_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "73b54bd0f412bd3c991704554746c29c26e5474c6a734e9485e29b174a4ca238"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "2069c48276b75744441b1326534faa9f83423ddbdfe5c602978a335cd3e5f55a"
      },
      "interior_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "f748d035051ef1e22f5d73db3bb31a213b07b054089b1e1b4986bc0b26d3edcb"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e55bdd55a77dfd64bfe101a58ab5c457cba7e477eef77d7b89dcc0251aba2ac7"
      }
    },
    "last-evaluated-weights.npz": {
      "force1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          24,
          3
        ],
        "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
      },
      "force1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
      },
      "force2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          3
        ],
        "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
      },
      "force2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
      },
      "event1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          5,
          1
        ],
        "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
      },
      "event1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
      },
      "event2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          1
        ],
        "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
      },
      "event2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
      },
      "mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          66,
          1,
          1,
          1
        ],
        "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
      },
      "local.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          4,
          3,
          3,
          3
        ],
        "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
      },
      "local.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
      },
      "endpoint_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "3cbf8802fb0cac8edc6e62439c15b98687bc7cb546179716b35f4e852c02fb1f"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "0bb90c990b0f64d8647198dc1ae80686902cdb96eeb29a04fb40a6c150bf808f"
      },
      "endpoint_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "1d8684c38a2c88d617c5c28c9fb00add45214cfaf9371c5abbaebbf80dbf62ad"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "b9aab2a36c888e747da2789945e947be2a34c278c96449d406625509786ff21a"
      },
      "endpoint_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "452c0e9fa00055fddfba512440d86b75ec7d78f2c9a2b88edb8e0a2cbf990eb0"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "fedd2610beb00863a188e935ea5f7bacc0cc809fd05fbae30572fd37d40ab1e2"
      },
      "endpoint_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "05cce95935a7f78438c4aaa11749eb30b455acac0bcc9680d0d21b2979bce5ba"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "64c4afee54fb98dc1ebcd64ab63c668d6344391e856452e220df83ea6201b6e4"
      },
      "interior_mix.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          12,
          1,
          1,
          1
        ],
        "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
      },
      "interior_mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
      },
      "interior_velocity.0.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          1,
          2
        ],
        "sha256": "0e4fff8d268600c9b1b47d1881944a3cfb2b0601b6333262200f742c46079aa9"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e18b933f8e196f18f6491f68ff887dbce0dc4d49a9f835879846496e0f385cb7"
      },
      "interior_velocity.1.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          1,
          2,
          1
        ],
        "sha256": "e8d6fb21b0575fc568327dcfab14cfa346b1e1f57445e5895f7626eeb7ddac59"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "4f53e2256296f46ddc229fc9cdab083ec6b49a2865c0c562459bd37662e4c02d"
      },
      "interior_velocity.2.weight": {
        "dtype": "<f4",
        "shape": [
          4,
          1,
          2,
          1,
          1
        ],
        "sha256": "73b54bd0f412bd3c991704554746c29c26e5474c6a734e9485e29b174a4ca238"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "2069c48276b75744441b1326534faa9f83423ddbdfe5c602978a335cd3e5f55a"
      },
      "interior_tracer.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "f748d035051ef1e22f5d73db3bb31a213b07b054089b1e1b4986bc0b26d3edcb"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e55bdd55a77dfd64bfe101a58ab5c457cba7e477eef77d7b89dcc0251aba2ac7"
      }
    }
  },
  "cpu_model_forward_executed": false,
  "model_family": "guided_shared_residual_v3",
  "full_generator_target_pass": false,
  "training_master_count": 24,
  "reserved52_references_opened": 0,
  "execution": {
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "peak_gpu_allocated_bytes": 537946624,
    "peak_RSS_bytes": 2678947840,
    "ambient_repository_reads": 0
  },
  "outputs": {
    "best-mean-weights.npz": {
      "bytes": 15170,
      "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e"
    },
    "last-evaluated-weights.npz": {
      "bytes": 15170,
      "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e"
    },
    "probe.npy": {
      "bytes": 16023680,
      "sha256": "154e9c97c51d107bb4bcbb81a1280fce3a6795a65973994491164c49af9fa059"
    },
    "weights.npz": {
      "bytes": 15170,
      "sha256": "5a4473543f6760120004edab4ad114799094cba6d3c9961968ceb83cc9d0163e"
    },
    "initial-weights.npz": {
      "bytes": 15170,
      "sha256": "d86420fcebb798588e50a010328de1ae8261e23150cf8a7a6389ddba4b3d86cd"
    },
    "history.json": {
      "bytes": 1011791,
      "sha256": "ef7dd447441832bb15a9a12afbf4ceab5fcc867b563e210ec81e0517fde369b5"
    },
    "normalization.json": {
      "bytes": 644,
      "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
    }
  },
  "elapsed_seconds": 1379.4283384959563
}

```

## Exact receipt-bound training configuration

Binding in the original receipt: `/closure/config_sha256`.

```json
{
  "identifier": "guided-shared-rms-width4-ray-01",
  "widths": [
    4
  ],
  "optimizer": "full_population_quadratic_ray",
  "maximum_iterations": 4,
  "probe_normalized_step": 0.001,
  "wall_seconds": 1740,
  "maximum_gpu_allocated_bytes": 12884901888,
  "maximum_peak_rss_bytes": 3221225472,
  "maximum_probe_bytes": 25165824,
  "maximum_output_bytes": 67108864,
  "initial_objective": 0.0003857275064314716,
  "initial_model_inventory": {
    "guide.point": {
      "dtype": "<f8",
      "shape": [
        12,
        3,
        3,
        3,
        3
      ],
      "sha256": "0b4cfa65c8615ed3c8ee7e574fe3f3a80247b451abcbc2e1b4c574114bf7403b"
    },
    "guide.viscosity": {
      "dtype": "<f8",
      "shape": [
        3,
        3,
        3,
        3,
        3
      ],
      "sha256": "54a960a9e391d47994a3e5b0266da89a424e3e86883bad1a78642a3321fb90ed"
    },
    "guide.flux": {
      "dtype": "<f8",
      "shape": [
        1,
        3,
        3,
        3,
        3
      ],
      "sha256": "584f700536a19aa575ff2bc36edc791e023856d8ae5dbba194707f6b26f2e6c2"
    },
    "guide.momentum": {
      "dtype": "<f8",
      "shape": [
        3,
        1,
        3,
        3,
        3
      ],
      "sha256": "d5394032a96712c2bc5b5bd84d96d560d3ef7910945b36648fef21e4ab3a907c"
    },
    "guide.mid_gate": {
      "dtype": "<f8",
      "shape": [
        4,
        3
      ],
      "sha256": "4e336572f79ea1c4a492b68ef6652e6b008288e743da3f1f8a2f63f5d9b1767e"
    },
    "guide.mix": {
      "dtype": "<f8",
      "shape": [
        4,
        3,
        3
      ],
      "sha256": "fb80dfe016eefec389493583d22dd003cc2c453ea016b11014ec261db82072be"
    },
    "guide._kernel_point_rows": {
      "dtype": "<i4",
      "shape": [
        13
      ],
      "sha256": "c8926633e50106b835bc9ba8c8bd1d883db234cef184ff8fe4f4691343e3a315"
    },
    "guide._kernel_point_channels": {
      "dtype": "<i4",
      "shape": [
        33
      ],
      "sha256": "b9ee693a2555667a73d79a0621f9d0b9dc7cfa46e3f92ba03f32c2dfcaa585ba"
    },
    "guide._kernel_point_offsets": {
      "dtype": "<i4",
      "shape": [
        33,
        3
      ],
      "sha256": "b2810a1078b472b777f263add53217239aec45f3b1f64897db48b1c993e9c3f0"
    },
    "guide._kernel_viscosity_rows": {
      "dtype": "<i4",
      "shape": [
        4
      ],
      "sha256": "be70140dd2e9dfdbf3169d41f1d836be95a0f7a92377023e699fa8630c5a66a5"
    },
    "guide._kernel_viscosity_channels": {
      "dtype": "<i4",
      "shape": [
        45
      ],
      "sha256": "1bb71e40ddcffe9207801dd167c02bf69e02d36600421e73c2d5c3535e6d697b"
    },
    "guide._kernel_viscosity_offsets": {
      "dtype": "<i4",
      "shape": [
        45,
        3
      ],
      "sha256": "820978c5e04f513b74ad50f02b554d628345a5c7d04f1bb2aa5978bb8817cb8c"
    },
    "guide._kernel_flux_rows": {
      "dtype": "<i4",
      "shape": [
        2
      ],
      "sha256": "20aef0af2bb62267e2f5c157abdc51d1ce6506e42478ae204f1fab5494907df1"
    },
    "guide._kernel_flux_channels": {
      "dtype": "<i4",
      "shape": [
        6
      ],
      "sha256": "72f5c72c6cb4606addb80b36f987f422de2676af9c8c8bc61554c477b3ff81c3"
    },
    "guide._kernel_flux_offsets": {
      "dtype": "<i4",
      "shape": [
        6,
        3
      ],
      "sha256": "a211e520a86857fdfb6ae61cfa753dbb291a13198d700a68d40a78c2804af204"
    },
    "guide._kernel_momentum_rows": {
      "dtype": "<i4",
      "shape": [
        4
      ],
      "sha256": "ce18f5c9b62e24ece371f92f5bbdb067a5a59a86e5d0f3ecfff02e17da6446d2"
    },
    "guide._kernel_momentum_channels": {
      "dtype": "<i4",
      "shape": [
        6
      ],
      "sha256": "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
    },
    "guide._kernel_momentum_offsets": {
      "dtype": "<i4",
      "shape": [
        6,
        3
      ],
      "sha256": "4ac9179282e04af5fd27bd4a5b76506a7579774d50792889578021201941bc0c"
    },
    "guide.shared_pressure.coefficients": {
      "dtype": "<f8",
      "shape": [
        4
      ],
      "sha256": "8e9b88d9d2687e3436fc3ea6e1e7bbeb56a479118e51a535a30096f88fa673d1"
    },
    "guide.shared_temporal.force1.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        24,
        3
      ],
      "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
    },
    "guide.shared_temporal.force1.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
    },
    "guide.shared_temporal.force2.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        8,
        3
      ],
      "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
    },
    "guide.shared_temporal.force2.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
    },
    "guide.shared_temporal.event1.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        5,
        1
      ],
      "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
    },
    "guide.shared_temporal.event1.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
    },
    "guide.shared_temporal.event2.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        8,
        1
      ],
      "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
    },
    "guide.shared_temporal.event2.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
    },
    "guide.shared_temporal.mix.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        66,
        1,
        1,
        1
      ],
      "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
    },
    "guide.shared_temporal.mix.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
    },
    "guide.shared_temporal.local.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
    },
    "guide.shared_temporal.local.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
    },
    "guide.shared_temporal.endpoint_velocity.0.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        1,
        2
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.endpoint_velocity.0.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.endpoint_velocity.1.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        2,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.endpoint_velocity.1.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.endpoint_velocity.2.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        2,
        1,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.endpoint_velocity.2.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.endpoint_tracer.weight": {
      "dtype": "<f4",
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    },
    "guide.shared_temporal.endpoint_tracer.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.interior_mix.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        12,
        1,
        1,
        1
      ],
      "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
    },
    "guide.shared_temporal.interior_mix.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
    },
    "guide.shared_temporal.interior_velocity.0.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        1,
        2
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.interior_velocity.0.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.interior_velocity.1.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        2,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.interior_velocity.1.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.interior_velocity.2.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        2,
        1,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "guide.shared_temporal.interior_velocity.2.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "guide.shared_temporal.interior_tracer.weight": {
      "dtype": "<f4",
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    },
    "guide.shared_temporal.interior_tracer.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.force1.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        24,
        3
      ],
      "sha256": "2a58c45631cfb686e24898c6cf9719f199a9f930915bb434946304d1275235d2"
    },
    "residual.force1.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "e9816a4f426c511bcb8c70cd1c6cf9d16e5afa3d8c8a8891d7f060f5ed57821a"
    },
    "residual.force2.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        8,
        3
      ],
      "sha256": "b4c0e668b32eeaa95bf7583b2239f3b77ed9a41522bc7e81ccc7516eb4e8ea61"
    },
    "residual.force2.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "1b23fcabc6a3c228500864ee6680f22d192a36e21df3f8f89ea8f62bdd178c9e"
    },
    "residual.event1.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        5,
        1
      ],
      "sha256": "d3a742d84523038d717130aa1b9aac2c7be13d0dcdae7d9d7653318fecb464d8"
    },
    "residual.event1.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "a2d5756db0558842a2fd135085264550efeadade3d5a24ec16920fd3f16201ac"
    },
    "residual.event2.weight": {
      "dtype": "<f4",
      "shape": [
        8,
        8,
        1
      ],
      "sha256": "6e5417e92f161dfa32881ae8e3b7446ed4a870033719fd89f6b67526b9f34092"
    },
    "residual.event2.bias": {
      "dtype": "<f4",
      "shape": [
        8
      ],
      "sha256": "a1c14368ad00464573839da49834b3dcd8df2d9a12e6027d002b73b71d892ce4"
    },
    "residual.mix.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        66,
        1,
        1,
        1
      ],
      "sha256": "0b977c06648b8aa758ef4c48593a075772ee08f49086762270b5625eadf5a86f"
    },
    "residual.mix.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "67b5ba07ad40bc563a4fc877fac17e5d12f44d15f8c8f52b63ed03b0a11dcf67"
    },
    "residual.local.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        4,
        3,
        3,
        3
      ],
      "sha256": "a1783f9e829ed82998f2c52ecfef6c48730019e84f3dab3c5455ffcfeac4bd15"
    },
    "residual.local.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "baaf760d6597cec4abb5319440e78b92f03b93f61ceed182fc488b9767ec234e"
    },
    "residual.endpoint_velocity.0.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        1,
        2
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.endpoint_velocity.0.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.endpoint_velocity.1.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        2,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.endpoint_velocity.1.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.endpoint_velocity.2.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        2,
        1,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.endpoint_velocity.2.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.endpoint_tracer.weight": {
      "dtype": "<f4",
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    },
    "residual.endpoint_tracer.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.interior_mix.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        12,
        1,
        1,
        1
      ],
      "sha256": "63eed2cf5334d23a85414d5575fe08894c974c784ad2b2163346f4880a130fb7"
    },
    "residual.interior_mix.bias": {
      "dtype": "<f4",
      "shape": [
        4
      ],
      "sha256": "359010143ddbfe17a6a522541c1210a0d190eff3e5ef934887ba6c8fc3f76b8b"
    },
    "residual.interior_velocity.0.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        1,
        2
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.interior_velocity.0.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.interior_velocity.1.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        1,
        2,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.interior_velocity.1.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.interior_velocity.2.weight": {
      "dtype": "<f4",
      "shape": [
        4,
        1,
        2,
        1,
        1
      ],
      "sha256": "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"
    },
    "residual.interior_velocity.2.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    },
    "residual.interior_tracer.weight": {
      "dtype": "<f4",
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    },
    "residual.interior_tracer.bias": {
      "dtype": "<f4",
      "shape": [
        1
      ],
      "sha256": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    }
  },
  "initial_qualification_sha256": "1fe1fd9b183094b058ca9ba3a09c6238f16a2a3b29a2a7d5fd4ecd539b461192",
  "probe_identifier": "long-v1-continuous-stationary-r0-cpm8",
  "probe_start_fine_step": 0
}

```
