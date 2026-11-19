# Full AI learned recurrent sampler transport head 01 train

Retrospective source review · fit-11a2f63a94a9168e

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A float64 carrier-conditioned head has12 midpoint gates (4x3) and36 mixing coefficients (4x3x3). It queries a supplied local3x3x3 velocity patch using fixed trilinear interpolation and donor-range clamping, then applies learned component mixing. Carrier phase distinguishes the three staggered velocity carriers and tracer. This is a local transport component; it does not solve pressure or generate a complete state by itself.

**Purpose.** Learn the midpoint/end sampling responses needed by the recurrent sampler using native local supervision, while retaining compact reusable transport parameters.

**Experiment.** 440 CUDA optimizer calls:400 Adam updates and40 LBFGS updates on the16 original training scenes. The captured objective supervises normalized midpoint/end responses using point velocity, local patch, carrier kind and the supplied alpha. Later compositions use a broader pressure training view; that does not mean this sampler was trained on all148 pressure scenes.

**Observed outcome.** The final recorded training objective is 2.9873033110782076e-24. This local supervised loss is not a simulation-step accuracy or600-step drift result. Exact parameter restoration is retained separately from inference. Complete sampler/spectral compositions require their own field, cadence, internal-work and rollout evaluation.

**Lesson.** Component training coverage must be stated separately. Tiny local interpolation loss is insufficient evidence for different integration intervals, force timing, larger displacements or accumulated own-history stability. Four physical outputs in existing generators still involve the recorded sequence of internal advances; this fit does not implement cheap temporal in-between frames.

**Reproduction.** The exact original receipt, training configuration and source/data hashes remain embedded below. Retain the selected and initial weights plus typed restore records. A copied GPU probe is not a CPU inference test, and retained hashes are not a successful clean retraining. Dataset regeneration and execution through the new Tuldok owner still require verification.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/sampler-head-pair-01/train/receipt.json",
    "sha256": "11a2f63a94a9168ec6e4c0ffb27cc655c7ec8c0b6c295e58d6cb7d88b40bf712"
  },
  "history": {
    "path": "runs/full-ai-tight-targets-v1/sampler-head-pair-01/train/history.json",
    "sha256": "9afd51e588f99a7f810988d21731f21dc9aef5032d89725dabb55f9777e08f1b"
  },
  "sources": [
    {
      "path": "experiments/full-ai-tight-targets-v1/learned_sampler_head.py",
      "sha256": "965dd63bb4e9a36a6ef25d4d1d7c1db894e702b7c2cedff60e674c52163643af",
      "binding": "/closure/sources/learned_sampler_head.py"
    },
    {
      "path": "experiments/full-ai-tight-targets-v1/train_sampler_head.py",
      "sha256": "4ea7865a80a43556479487539e5c42fb3eace55539763c47e87e0ca558faf26b",
      "binding": "/closure/sources/train_sampler_head.py"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-11a2f63a94a9168e",
    "kind": "training_attempt",
    "name": "Full AI learned recurrent sampler transport head 01 train",
    "task": "recurrent_transport_component",
    "schema": "rheon-learned-recurrent-sampler-head-result-v1",
    "method": {},
    "receipt_sha256": "11a2f63a94a9168ec6e4c0ffb27cc655c7ec8c0b6c295e58d6cb7d88b40bf712",
    "receipt_paths": [
      "runs/full-ai-tight-targets-v1/sampler-head-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "65a84a734d8f416bbbdee46c6a636e1b",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI learned recurrent sampler transport head 01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 440,
    "parameter_fits": null,
    "declared_parameter_count": 48,
    "training_closure": {
      "config_sha256": "565049bf502ee1ea4a80e9a622e27e3245af5b2ce03765fd0e71c2ef48a7526e",
      "inputs": {
        "manifest": {
          "bytes": 106310,
          "sha256": "4cf56c1b89be8f862076801d74db92d8316ae1fc02ed477bd0b990a98c9aa123"
        },
        "normalization": {
          "bytes": 3271,
          "sha256": "d61d1142f1ec7e08cf11636ed07f9abfb7607fa0f2c2e24a3a1aa7da38e69e78"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 4380816,
          "sha256": "25d0bc8a0dae17991a6893d234394e1112f74fff6fb050e89a1da327bf3a024b"
        },
        "part_01": {
          "bytes": 2994005,
          "sha256": "ea0763b8829da7d4a6e2d0d6e31514bce6258a24c525a018a7256ae126b39ef5"
        },
        "part_02": {
          "bytes": 2467653,
          "sha256": "9aa1d67408a13bc3b5e0a6b3723d5aa90ab10f3c038522cff181c759b4ca2086"
        },
        "part_03": {
          "bytes": 5439298,
          "sha256": "90e34a88d7f40f272402094e1fb4048353d903cab3a7e783829a5e47bc4cbdfd"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_sampler_head.py": "965dd63bb4e9a36a6ef25d4d1d7c1db894e702b7c2cedff60e674c52163643af",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "train_sampler_head.py": "4ea7865a80a43556479487539e5c42fb3eace55539763c47e87e0ca558faf26b"
      },
      "normalization_sha256": "d61d1142f1ec7e08cf11636ed07f9abfb7607fa0f2c2e24a3a1aa7da38e69e78",
      "role": "recurrent_sampler_transport_fit"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "b0fd9dfa81b6b85d2ae4542a8e0b9261001493b254c16ca67129a1d3eb81602f",
        "bytes": 892,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "a3e2b0c0f0cd3e1193745ee7669b2058d6ab6c380cf466fde7f7edba7d3ddd2b",
        "bytes": 892,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "temporal-models/34",
      "temporal-models/35",
      "temporal-models/36",
      "temporal-models/37",
      "temporal-models/38",
      "temporal-executions/18",
      "temporal-executions/19",
      "temporal-executions/20",
      "temporal-executions/21",
      "temporal-executions/22",
      "temporal-executions/23",
      "temporal-executions/24",
      "temporal-executions/25",
      "temporal-executions/26",
      "temporal-executions/27",
      "temporal-executions/28",
      "temporal-executions/29",
      "temporal-executions/30",
      "temporal-executions/31",
      "temporal-executions/32",
      "temporal-executions/33",
      "temporal-executions/34",
      "temporal-executions/35",
      "temporal-executions/36",
      "temporal-executions/37",
      "temporal-executions/38",
      "temporal-executions/39",
      "temporal-executions/40",
      "temporal-executions/41",
      "temporal-executions/42",
      "temporal-executions/43",
      "temporal-executions/44",
      "temporal-executions/45",
      "temporal-executions/46",
      "temporal-executions/47",
      "temporal-executions/48",
      "temporal-executions/49",
      "temporal-executions/50",
      "temporal-executions/51",
      "temporal-executions/52",
      "temporal-executions/53",
      "temporal-executions/54",
      "temporal-executions/55",
      "temporal-executions/56",
      "temporal-executions/57",
      "temporal-executions/58",
      "temporal-executions/59",
      "temporal-executions/60",
      "temporal-executions/61",
      "temporal-executions/62",
      "temporal-executions/63",
      "temporal-executions/64",
      "temporal-executions/65",
      "temporal-executions/66"
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
      "receipt": "runs/full-ai-tight-targets-v1/sampler-head-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "recurrent_transport_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/sampler-head-pair-01/train/receipt.json",
    "sha256": "11a2f63a94a9168ec6e4c0ffb27cc655c7ec8c0b6c295e58d6cb7d88b40bf712",
    "bytes": 3811
  },
  "bound_training_configuration": {
    "config_sha256": "565049bf502ee1ea4a80e9a622e27e3245af5b2ce03765fd0e71c2ef48a7526e",
    "graph_id": "7944cac60a9547c5b97f21e210223e25",
    "graph_document_sha256": "ece157a642248d13a5fd21bc54aac8aeb00276f1859207542565f1ab5a386f0f",
    "recipe_path": "runs/full-ai-tight-targets-v1/sampler-head-project-03/train-recipe.json",
    "recipe_sha256": "2992b763eb65cf9f3269e5fbb9ef0331a738f4f0688a127ba2ef169c8ac9ec7b",
    "config_path": "runs/full-ai-tight-targets-v1/sampler-head-project-03/config.json",
    "configuration": {
      "identifier": "recurrent-sampler-head-01",
      "role": "recurrent_sampler_transport_fit",
      "objective": "native_supervised_sampler_v1",
      "seed": 2026102061,
      "updates": 440,
      "adam_updates": 400,
      "lbfgs_updates": 40,
      "lbfgs_max_iter": 5,
      "lbfgs_max_eval": 8,
      "batch_points": 8192,
      "learning_rate": 0.03,
      "minimum_learning_rate": 1e-05,
      "wall_seconds": 900
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
  "schema": "rheon-learned-recurrent-sampler-head-result-v1",
  "status": "complete",
  "mode": "train",
  "role": "recurrent_sampler_transport_fit",
  "optimizer_updates": 440,
  "parameters": 48,
  "trainable_parameters": 48,
  "closure": {
    "config_sha256": "565049bf502ee1ea4a80e9a622e27e3245af5b2ce03765fd0e71c2ef48a7526e",
    "inputs": {
      "manifest": {
        "bytes": 106310,
        "sha256": "4cf56c1b89be8f862076801d74db92d8316ae1fc02ed477bd0b990a98c9aa123"
      },
      "normalization": {
        "bytes": 3271,
        "sha256": "d61d1142f1ec7e08cf11636ed07f9abfb7607fa0f2c2e24a3a1aa7da38e69e78"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 4380816,
        "sha256": "25d0bc8a0dae17991a6893d234394e1112f74fff6fb050e89a1da327bf3a024b"
      },
      "part_01": {
        "bytes": 2994005,
        "sha256": "ea0763b8829da7d4a6e2d0d6e31514bce6258a24c525a018a7256ae126b39ef5"
      },
      "part_02": {
        "bytes": 2467653,
        "sha256": "9aa1d67408a13bc3b5e0a6b3723d5aa90ab10f3c038522cff181c759b4ca2086"
      },
      "part_03": {
        "bytes": 5439298,
        "sha256": "90e34a88d7f40f272402094e1fb4048353d903cab3a7e783829a5e47bc4cbdfd"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_sampler_head.py": "965dd63bb4e9a36a6ef25d4d1d7c1db894e702b7c2cedff60e674c52163643af",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "train_sampler_head.py": "4ea7865a80a43556479487539e5c42fb3eace55539763c47e87e0ca558faf26b"
    },
    "normalization_sha256": "d61d1142f1ec7e08cf11636ed07f9abfb7607fa0f2c2e24a3a1aa7da38e69e78",
    "role": "recurrent_sampler_transport_fit"
  },
  "initial_inventory": {
    "mid_gate": {
      "shape": [
        4,
        3
      ],
      "dtype": "<f8",
      "sha256": "2ea9ab9198d1638007400cd2c3bef1cc745b864b76011a0e1bc52180ac6452d4"
    },
    "mix": {
      "shape": [
        4,
        3,
        3
      ],
      "dtype": "<f8",
      "sha256": "6ef2300ff7a438e2a0b7896afce676b4561ef201547ec8f1d94b5d617f21bf10"
    }
  },
  "final_inventory": {
    "mid_gate": {
      "shape": [
        4,
        3
      ],
      "dtype": "<f8",
      "sha256": "b15f4e6d6b75f847b0ab5efa59483c9b5ceb445f1ef9e45356ab293eaa46f637"
    },
    "mix": {
      "shape": [
        4,
        3,
        3
      ],
      "dtype": "<f8",
      "sha256": "bdf556a199e0ec08483c8ed3dd7f01bf43631611dec0478a3a4c67410a7cfe47"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 200335360
  },
  "outputs": {
    "probe.npy": {
      "bytes": 3744032,
      "sha256": "2cc27c2527457e258062f8be5c043fc8a71986b0746fdc0bf9dd351c857f50fd"
    },
    "weights.npz": {
      "bytes": 892,
      "sha256": "b0fd9dfa81b6b85d2ae4542a8e0b9261001493b254c16ca67129a1d3eb81602f"
    },
    "initial-weights.npz": {
      "bytes": 892,
      "sha256": "a3e2b0c0f0cd3e1193745ee7669b2058d6ab6c380cf466fde7f7edba7d3ddd2b"
    },
    "history.json": {
      "bytes": 13878,
      "sha256": "9afd51e588f99a7f810988d21731f21dc9aef5032d89725dabb55f9777e08f1b"
    },
    "normalization.json": {
      "bytes": 3271,
      "sha256": "d61d1142f1ec7e08cf11636ed07f9abfb7607fa0f2c2e24a3a1aa7da38e69e78"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 9.134291481983382
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "recurrent-sampler-head-01",
  "role": "recurrent_sampler_transport_fit",
  "objective": "native_supervised_sampler_v1",
  "seed": 2026102061,
  "updates": 440,
  "adam_updates": 400,
  "lbfgs_updates": 40,
  "lbfgs_max_iter": 5,
  "lbfgs_max_eval": 8,
  "batch_points": 8192,
  "learning_rate": 0.03,
  "minimum_learning_rate": 1e-05,
  "wall_seconds": 900
}

```
