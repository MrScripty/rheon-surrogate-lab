# Complete24scene independent pressure residual width4 full-population CUDA fit train

Retrospective source review · fit-d18b434d52b73791

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A width-four F32 correction network with 1,732 fitted coefficients surrounds a frozen learned full-field guide. One shared spatial trunk per block encodes current/predicted endpoint fields, spacing, physical horizon, ordered prescribed controls and within-block event offsets. Endpoint and intermediate heads correct velocity/tracer. Two additional bias-free four-coefficient heads predict independent pressure-RHS corrections; the restored learned pressure response converts these to reported pressure. Velocity projection pressure remains internal. This adds two learned correction-pressure responses per physical output.

**Purpose.** Decouple reported pressure fitting from the solenoidal velocity correction while preserving the same frozen guide and inherited trunk initialization.

**Experiment.** All 24 independent training masters, 240 selected scene/start windows, 2,880 blocks and 7,200 physical outputs enter every complete score. Six grid resolutions (8/10/16/25/50/100 cells per metre), 16 shapes through 64³ and rectangular 128-cell axes, four force profiles and three motion types are present. Targets come from the qualified 240-step/s bank; every window covers 30/60/120 outputs/s crossed with K=1/2/3/4. Scenes, starts and temporal configurations are balanced, with 1/K weights within blocks. Four complete-gradient iterations each evaluate a normalized probe and quadratic candidate: eight recorded parameter trials, four mean-loss-accepted steps, not eight Adam updates. Probe step is 0.1. Future native states/RHS are labels only; guide-derived RMS sets correction units.

**Observed outcome.** Retained initial objective 0.00038572750643147162; selected probe-3 objective 0.00095713424401046093, worst pressure 0.4422579543105038, equation 0.57274305159197525, velocity 0.1120777892470745, tracer 0.20738194808974975. Mean-best candidate-4 objective is 0.00038571243738395048. Worst-first selection retains probe-3 even though candidate-4 has lower average loss and much smaller worst pressure error. The selected probe was evaluated but was not the mean-loss-accepted continuation state. These are original training-population scores, not fresh common-benchmark, held-out or latency results. The full-generator target was not met.

**Lesson.** Rank and metric definitions must accompany every checkpoint. Minimizing the maximum across equation/pressure/velocity/tracer can prefer a slightly improved dominant equation error while worsening other fields and average loss. Original scores use frozen RMS denominator floors and native target RHS; they cannot be equated with same-request pressure accuracy of the near-exact F64 anchors. Small corrections still require the complete frozen guide and its cost; fitted residual parameter count is not total generator size.

**Reproduction.** Retain the exact configuration, all five entrypoint source artifacts, captured model-source/asset ZIP, runtime and loss normalization, and the single lossless training archive. All source members and archived tensor states were checked in place; native field framing was checked but fields were not decoded. Reproduce the balanced complete scores, F64 gradient accumulation, F32 parameter trials and distinct worst/mean/last selections. Historical CPU restore constructs and checks typed weights and copies the saved GPU probe without neural inference. The original temporary Project root is absent; rebuilding its typed recipe bindings and a clean GPU-only replay/new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/receipt.json",
    "sha256": "d18b434d52b73791706828c74a664e01bfb8e29ccbe40bfdf2e62912c02c3673"
  },
  "history": {
    "path": "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/history.json",
    "sha256": "1b545f6704934adf85abe5b3510b6a7c8a4828574fc70010954fa0858e118764"
  },
  "sources": [
    {
      "path": "experiments/full-ai-data-diversity-v1/portable_shared_runtime_v1.py",
      "sha256": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6"
    },
    {
      "path": "experiments/full-ai-data-diversity-v1/train_guided_pressure_v1.py",
      "sha256": "e37bd30e912e2c3c18e3bd8f531512b748900a2e8dda2d3f85716e34c1a59830"
    },
    {
      "path": "experiments/full-ai-data-diversity-v1/managed_input_stream_v1.py",
      "sha256": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692"
    },
    {
      "path": "experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    }
  ],
  "training_inputs": [
    {
      "bytes": 3765187,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/cb56435e0a975bdfa97eab8258229167d6f0b021745ee99d112f83bf359bf636",
      "sha256": "cb56435e0a975bdfa97eab8258229167d6f0b021745ee99d112f83bf359bf636",
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
      "bytes": 5867565,
      "path": "projects/rheon-model-portfolio-unified-04/artifacts/sha256/b385dcd251796067f8bad5aadb3f0f2307c1cdf9e75616a1a77b44c45c39847e",
      "sha256": "b385dcd251796067f8bad5aadb3f0f2307c1cdf9e75616a1a77b44c45c39847e",
      "slot": "model_source"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-d18b434d52b73791",
    "kind": "training_attempt",
    "name": "Complete24scene independent pressure residual width4 full-population CUDA fit train",
    "task": "guided_pressure_component",
    "schema": "rheon.guided-pressure-managed-fit.v1",
    "method": {},
    "receipt_sha256": "d18b434d52b73791706828c74a664e01bfb8e29ccbe40bfdf2e62912c02c3673",
    "receipt_paths": [
      "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "533ac29a59524c10bb18ace3ff7445d3",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Complete24scene independent pressure residual width4 full-population CUDA fit train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 8,
    "parameter_fits": 1,
    "declared_parameter_count": null,
    "training_closure": {
      "config_sha256": "2dc2cf44e0d8b2730e26f9d082b5b7653ea0ffa623821c1bc7eb09be7deb87d9",
      "inputs": {
        "manifest": {
          "bytes": 3765187,
          "sha256": "cb56435e0a975bdfa97eab8258229167d6f0b021745ee99d112f83bf359bf636"
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
          "bytes": 5867565,
          "sha256": "b385dcd251796067f8bad5aadb3f0f2307c1cdf9e75616a1a77b44c45c39847e"
        },
        "part_00": {
          "bytes": 2808112843,
          "sha256": "91123b4712cdb785ca7bf99805723be01323a05661fcba4f79790bfd375fa53c"
        }
      },
      "sources": {
        "portable_shared_runtime_v1.py": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6",
        "train_guided_pressure_v1.py": "e37bd30e912e2c3c18e3bd8f531512b748900a2e8dda2d3f85716e34c1a59830",
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
      }
    },
    "checkpoints": [
      {
        "name": "best-mean-weights.npz",
        "sha256": "7b9db9a37b401f197be1f9179fcd92df6856f6d27086733311b6ef989c3ec5b3",
        "bytes": 15774,
        "stage": "retained_snapshot"
      },
      {
        "name": "last-evaluated-weights.npz",
        "sha256": "7b9db9a37b401f197be1f9179fcd92df6856f6d27086733311b6ef989c3ec5b3",
        "bytes": 15774,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "1e0e622689fae6b9d23ec25c1cd77f7ffa5a5d60d2a5ce2e78630f763fc6e9e7",
        "bytes": 15774,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "2aebf459a719db5e934b2bbce8451cf50db7be12e1a4ccf00bf6004c1369c8c4",
        "bytes": 15774,
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
      "receipt": "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "guided_pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/receipt.json",
    "sha256": "d18b434d52b73791706828c74a664e01bfb8e29ccbe40bfdf2e62912c02c3673",
    "bytes": 30700
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
    "fit_item_id": "fit-d18b434d52b73791",
    "document_kind": "configuration",
    "document_path": "runs/model-portfolio-migration-v1/guided-config-custody-recovery-01/exact-text-sources/2dc2cf44e0d8b2730e26f9d082b5b7653ea0ffa623821c1bc7eb09be7deb87d9.json",
    "expected_sha256": "2dc2cf44e0d8b2730e26f9d082b5b7653ea0ffa623821c1bc7eb09be7deb87d9",
    "binding_pointer": "/closure/config_sha256",
    "binding_field": "config_sha256",
    "recovery_report": "guided-config-custody-recovery-01.json",
    "primary_receipt_path": "runs/full-ai-data-diversity-v1/guided-pressure-managed-pair-01/train/receipt.json",
    "primary_receipt_sha256": "d18b434d52b73791706828c74a664e01bfb8e29ccbe40bfdf2e62912c02c3673",
    "document_bytes": 17485,
    "verified_direct_original_receipt_binding": true
  }
}
```

## Exact original receipt bytes

```json
{
  "schema": "rheon.guided-pressure-managed-fit.v1",
  "mode": "train",
  "closure": {
    "config_sha256": "2dc2cf44e0d8b2730e26f9d082b5b7653ea0ffa623821c1bc7eb09be7deb87d9",
    "inputs": {
      "manifest": {
        "bytes": 3765187,
        "sha256": "cb56435e0a975bdfa97eab8258229167d6f0b021745ee99d112f83bf359bf636"
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
        "bytes": 5867565,
        "sha256": "b385dcd251796067f8bad5aadb3f0f2307c1cdf9e75616a1a77b44c45c39847e"
      },
      "part_00": {
        "bytes": 2808112843,
        "sha256": "91123b4712cdb785ca7bf99805723be01323a05661fcba4f79790bfd375fa53c"
      }
    },
    "sources": {
      "portable_shared_runtime_v1.py": "df94e25f81af53395e2784cf905ea26804132473f0841c985421a2440daa6be6",
      "train_guided_pressure_v1.py": "e37bd30e912e2c3c18e3bd8f531512b748900a2e8dda2d3f85716e34c1a59830",
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "managed_input_stream_v1.py": "a9f40979c808ba9a7ff6a2bcd988b84b23b55a0c5c35343ef7839de264a65692",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
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
      },
      "endpoint_pressure.weight": {
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
      "interior_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
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
        "sha256": "de06fec7eeabd66d26ccc083fea42b857d4aecb387b9f689a9b78e8762327f57"
      },
      "force1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "599281b3d805e80a7d8712aee90a4c2ba8d169d47673324017c73299c089b17a"
      },
      "force2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          3
        ],
        "sha256": "59295acaa72abecb40533fd04299472532b1e64f679adb77bf47d21862a64377"
      },
      "force2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "aefc071fb2065b7b609058b65efc3e846ba349daa45c05f51af601561f98162b"
      },
      "event1.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          5,
          1
        ],
        "sha256": "83c034e9dcb545cc191deea9244a27781eaf568431df64c22a9ffed03c8d0d8f"
      },
      "event1.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "0c00f09b15590b3d5c69079a2b6584a5da7321e66e1f2d530cd17f5d7e296ea6"
      },
      "event2.weight": {
        "dtype": "<f4",
        "shape": [
          8,
          8,
          1
        ],
        "sha256": "ee964db481d6052efb430c9c2ef756e5f8a0f70776cfd4a76072c29cf24b22be"
      },
      "event2.bias": {
        "dtype": "<f4",
        "shape": [
          8
        ],
        "sha256": "7f0d9449820f2f74c009cb9865b46cee1331080000bc679744e682b1fb6e1b07"
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
        "sha256": "3adbc6dfcb5ae06590fbeef7f774c5b32072bdaaef856aa3f3a66f54b89a8c97"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "c6e253428b5fcb261e0a43d43c94304095fead1b21aebfadd39b3e7adc97653e"
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
        "sha256": "c5056d282f25233d7f98617cb96532a28ef58e4002007266a4de25b2e0f0d820"
      },
      "local.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "5db6b68ec287df9316e70c7ab19497f3b05b69e8605db3aab45c67e509568c57"
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
        "sha256": "ed6cff437b67dd38cd5813d4b0fe9352aedc3b1aa27ca16f6f5879a458df0b27"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "cbd61d89b25ef2c606e790e5d10d0ed527c69aa03faa66d4d7db1934c1365cd4"
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
        "sha256": "8a46e3c336521f39bbdc366e8f6f374bf12a3c6998028928b3c7a30fd0b8af33"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "cf4a5dd8f0513375485d0347c5a912fed8824c9bf624ab26375c8de228c182dd"
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
        "sha256": "9f5ae160645d9f551f6ac4182fb876bb09916c561fed2a481b173b8da1ea7695"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "c4ebff7ba6d8bafdd5dce7e7268576a68d41108ae6ef18e1b197ee820a2d6cd1"
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
        "sha256": "058d851c4381017e2280c58a87415ad9d94d8f5d92d24b6e5b888e4ef8e91950"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "a2fe0ec02166e71fe7ded1d52dd1e5aa6482b598c9b398f7cee6507bbcba45d7"
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
        "sha256": "a68b091b83731a2dd55bcfef848c8a689e06dcefef3574f7f0836b8c2724b528"
      },
      "interior_mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "d85d9fc9fb9ff93bd8ea28e4cb2b1b386a7aa47962866ef75e01aa71779feab4"
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
        "sha256": "8b5b644e2d927c71083ade3ff37274ef48b0c3ee5692f70fda36eadf95416b0b"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "75873fa0ca77c2f904894246600ccc5f8e3c698c4da848ac0b59ecd20ce16840"
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
        "sha256": "6e09c60cb81c1fb386ff7b768b78d9e8ec78103ee39347251e9a0c83c0bfe200"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "72edf866ba08dbd94f6f04bb3a810e7eec96da88d74f01441ed943e4d1a8e26e"
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
        "sha256": "7ecf2c323abe3d7397b3691fb7abff5d5132559614795cf307daae845bab5d76"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "23ee79eeda078c11a020934895fc8f761cf7aa08b238a5ead3133994798ad9e2"
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
        "sha256": "d328a23460502f6c5a083f79de22324ffb4b62589999c18251f9762b82e4f601"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "93b3bbd6ba8fa9ab704e6ddcb7f42fcfffa63cc77b5b58426802f7b44ee306f5"
      },
      "endpoint_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "1ff9e6801830648e459ac0ea67f08df4366f8e32f26a8394f8fa5d0823fffdc3"
      },
      "interior_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "4765cdd66af04035ae5ff9fe5866049cbdb5d3bfc665b48c5950e0692f0d774f"
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
        "sha256": "041dc05da21ddb4f928ec7a9f942dd1b1414a5f124f0b3d874286604f648cd7b"
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
        "sha256": "feb6be20a7145812bd794f9823178e6ef574e2c121c62ee652bdfe2fa00e5951"
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
        "sha256": "16a62befc2043417733564b60067106106e8f25b0dfa7858d61dc17e5b5cb685"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "30777e1a4c626f81e7a60098cf8e4941fa8744a309edf079fcb4b33073e396eb"
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
        "sha256": "aabc4598ae288f1e86bb65b2517dc1f74dff07e8238572e5738471623440bcef"
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
        "sha256": "a42a9e126112fec99ee4f3d3f4c8787b1508a55a7ff9b4b596730f1c4005935e"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "0a6c6590f5616daf8fb57f2088f55015874a70d6c958148831abe90451ec2d7e"
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
        "sha256": "b14b491849fb41250495f4692fa30e44704b2221909f96a363c858bec940210b"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "6a0572c14681390e080d36804c9d886784798a63ca22f600e5a20ede7a475237"
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
        "sha256": "db4234e2e87a3ae3132053a0db76f0ae0d9424987ac5d4a13923cdeae7d2b93c"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "f02fb18b8ef40b1ee098031816635a466f13c420861d7a40c7ac468e5ceeb812"
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
        "sha256": "6797e8f98e8867c489c7021bf7184ae82bedc26eb7c576ba89390b44e7ea03b3"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "299e9b6bec98d236a5eab4db286697e74eb5c44eda8581d4d39bee7f64a7076c"
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
        "sha256": "62c91ffd5921d2726ed48a5a13b2a63c61db30965524e03bdbddf7eb93f2fd23"
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
        "sha256": "6b8001e5bb92c34bf6035d1c65c7a2bf071a580c8d57b8fa24315b16048b5b25"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "4c9cad074b760ab80372a888d79bed65b8fad82ca3f669da06dae9b65d00b35d"
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
        "sha256": "78b27aa16f64d6379ebe5b5f31286ac0f72f5cfd411cc60b2651688e025c1492"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "3b043fdc107b6b089292e67a7054189ea3abd163fd4ebd9fbf14f200ab88d178"
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
        "sha256": "a77cfa7e622d7e47c274c21962976255a4f85d8ed341f67edd4df2e01da8fba5"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "53e002b74c4f4bd611fc9274753e0846511f84da5e917f2ffb06b7e28658ef9d"
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
        "sha256": "b6d353b8addd1d591000b0d1921757b530b7ec6bf9e48f4f74b059ff70529841"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e99abccd36cd604f902d2f4f76284198953a06bc504bc1ff0608c9d65b4dc0aa"
      },
      "endpoint_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "2fb05995de4260f2348ff0058cebbbf0e0644d659f31e2ec40367a6f5a3dcf18"
      },
      "interior_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "71b3992798b9b5f82c6cf7fde47a3ba5237a867a3ed492d3af08d5ad47ee9f98"
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
        "sha256": "041dc05da21ddb4f928ec7a9f942dd1b1414a5f124f0b3d874286604f648cd7b"
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
        "sha256": "feb6be20a7145812bd794f9823178e6ef574e2c121c62ee652bdfe2fa00e5951"
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
        "sha256": "16a62befc2043417733564b60067106106e8f25b0dfa7858d61dc17e5b5cb685"
      },
      "mix.bias": {
        "dtype": "<f4",
        "shape": [
          4
        ],
        "sha256": "30777e1a4c626f81e7a60098cf8e4941fa8744a309edf079fcb4b33073e396eb"
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
        "sha256": "aabc4598ae288f1e86bb65b2517dc1f74dff07e8238572e5738471623440bcef"
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
        "sha256": "a42a9e126112fec99ee4f3d3f4c8787b1508a55a7ff9b4b596730f1c4005935e"
      },
      "endpoint_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "0a6c6590f5616daf8fb57f2088f55015874a70d6c958148831abe90451ec2d7e"
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
        "sha256": "b14b491849fb41250495f4692fa30e44704b2221909f96a363c858bec940210b"
      },
      "endpoint_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "6a0572c14681390e080d36804c9d886784798a63ca22f600e5a20ede7a475237"
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
        "sha256": "db4234e2e87a3ae3132053a0db76f0ae0d9424987ac5d4a13923cdeae7d2b93c"
      },
      "endpoint_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "f02fb18b8ef40b1ee098031816635a466f13c420861d7a40c7ac468e5ceeb812"
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
        "sha256": "6797e8f98e8867c489c7021bf7184ae82bedc26eb7c576ba89390b44e7ea03b3"
      },
      "endpoint_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "299e9b6bec98d236a5eab4db286697e74eb5c44eda8581d4d39bee7f64a7076c"
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
        "sha256": "62c91ffd5921d2726ed48a5a13b2a63c61db30965524e03bdbddf7eb93f2fd23"
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
        "sha256": "6b8001e5bb92c34bf6035d1c65c7a2bf071a580c8d57b8fa24315b16048b5b25"
      },
      "interior_velocity.0.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "4c9cad074b760ab80372a888d79bed65b8fad82ca3f669da06dae9b65d00b35d"
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
        "sha256": "78b27aa16f64d6379ebe5b5f31286ac0f72f5cfd411cc60b2651688e025c1492"
      },
      "interior_velocity.1.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "3b043fdc107b6b089292e67a7054189ea3abd163fd4ebd9fbf14f200ab88d178"
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
        "sha256": "a77cfa7e622d7e47c274c21962976255a4f85d8ed341f67edd4df2e01da8fba5"
      },
      "interior_velocity.2.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "53e002b74c4f4bd611fc9274753e0846511f84da5e917f2ffb06b7e28658ef9d"
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
        "sha256": "b6d353b8addd1d591000b0d1921757b530b7ec6bf9e48f4f74b059ff70529841"
      },
      "interior_tracer.bias": {
        "dtype": "<f4",
        "shape": [
          1
        ],
        "sha256": "e99abccd36cd604f902d2f4f76284198953a06bc504bc1ff0608c9d65b4dc0aa"
      },
      "endpoint_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "2fb05995de4260f2348ff0058cebbbf0e0644d659f31e2ec40367a6f5a3dcf18"
      },
      "interior_pressure.weight": {
        "dtype": "<f4",
        "shape": [
          1,
          4,
          1,
          1,
          1
        ],
        "sha256": "71b3992798b9b5f82c6cf7fde47a3ba5237a867a3ed492d3af08d5ad47ee9f98"
      }
    }
  },
  "cpu_model_forward_executed": false,
  "model_family": "guided_pressure_residual_v1",
  "full_generator_target_pass": false,
  "training_master_count": 24,
  "reserved52_references_opened": 0,
  "execution": {
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "peak_gpu_allocated_bytes": 540835840,
    "peak_RSS_bytes": 2677207040,
    "ambient_repository_reads": 0
  },
  "outputs": {
    "best-mean-weights.npz": {
      "bytes": 15774,
      "sha256": "7b9db9a37b401f197be1f9179fcd92df6856f6d27086733311b6ef989c3ec5b3"
    },
    "last-evaluated-weights.npz": {
      "bytes": 15774,
      "sha256": "7b9db9a37b401f197be1f9179fcd92df6856f6d27086733311b6ef989c3ec5b3"
    },
    "probe.npy": {
      "bytes": 16023680,
      "sha256": "6beaabde722b46b1ba08f5c273cb06f3783e91468281e9f43d02810d7e199988"
    },
    "weights.npz": {
      "bytes": 15774,
      "sha256": "1e0e622689fae6b9d23ec25c1cd77f7ffa5a5d60d2a5ce2e78630f763fc6e9e7"
    },
    "initial-weights.npz": {
      "bytes": 15774,
      "sha256": "2aebf459a719db5e934b2bbce8451cf50db7be12e1a4ccf00bf6004c1369c8c4"
    },
    "history.json": {
      "bytes": 1018925,
      "sha256": "1b545f6704934adf85abe5b3510b6a7c8a4828574fc70010954fa0858e118764"
    },
    "normalization.json": {
      "bytes": 644,
      "sha256": "a2e53c03d51f9ea6a97e3b65fd54b4c4b7a5ee45825926419370fd68ca19ab14"
    }
  },
  "elapsed_seconds": 1318.353444835986
}

```

## Exact receipt-bound training configuration

Binding in the original receipt: `/closure/config_sha256`.

```json
{
  "identifier": "guided-pressure-width4-ray-01",
  "widths": [
    4
  ],
  "optimizer": "full_population_quadratic_ray",
  "maximum_iterations": 4,
  "probe_normalized_step": 0.1,
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
    },
    "residual.endpoint_pressure.weight": {
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
    "residual.interior_pressure.weight": {
      "dtype": "<f4",
      "shape": [
        1,
        4,
        1,
        1,
        1
      ],
      "sha256": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"
    }
  },
  "initial_qualification_sha256": "f2b09ce90f3055f2aee6cf70c1b3328fd27f35b00b8433bf50cebb7a9dbc2b06",
  "probe_identifier": "long-v1-continuous-stationary-r0-cpm8",
  "probe_start_fine_step": 0
}

```
