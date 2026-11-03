# Full AI mixed modal query refinement 11 train

Retrospective source review · fit-aaf0a360c4c9622b

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only F64 diagonal model with 38,673 stored log-gains across four registered grid shapes. Each mode has its own positive learned gain; four DC entries are stored but masked from prediction, leaving 38,669 active non-DC gains. Fixed full DCT-II transforms supply coordinates. The model multiplies supplied request coefficients by exponentiated gains; it does not advance velocity or tracer. Physical spacing enters the external q=(rho/h)RHS unit conversion, not an independent learned density channel. Unknown grid keys/shapes require a new exported or trained table.

**Purpose.** Excite all modes jointly with additional offline reference-labeled queries to improve modal coverage and separate weak native-mode signal from coefficient precision.

**Experiment.** The same 144 native pairs are retained bitwise, plus 128 synthetic mixed-mode queries, 32 per geometry. These are extra component queries, not 128 new simulations or scenes. Each full-batch update sees all four geometry groups. Initialization is the exact preceding learned checkpoint. 80 CUDA LBFGS outer updates use strong-Wolfe search with 162 recorded closure evaluations. Native coefficient errors are normalized by train-only per-mode pressure RMS with a 1e-12 relative floor; no p/q coefficient target is formed. Original source/recipe/input/output identities and all F64 initial/final tensor bits verify without neural execution.

**Observed outcome.** The original fit completes its declared 80 outer updates. Final logged training objective is 2.2663416527961059e-25; it is specific to this objective and population. All initial/final parameters and inherited checkpoints verify, and the original non-DC label coverage is independently checked. Fresh common GPU pressure accuracy/timing, full-state compositions, independent holdout and long-horizon drift are not measured by this review.

**Lesson.** Joint mode excitation broadens linear component queries but retains only 16 physical scenes and four geometries. Preserve the offline numerical-label provenance; do not call these independent holdout scenes or a complete generator benchmark.

**Reproduction.** Retain the exact typed recipe, trainer/objective/basis/model/configuration/runtime, all four original coefficient shards, normalization and original native-source hashes. Refinements also require the full exact inherited F64 inventory transported in normalization; decoder labels require the declared fixed coordinate encoding, and mixed queries require their offline reference archive and seed. Original final publication and restore probes run CPU neural inference and need GPU-only qualification before rerunning. Do not reuse tiny component loss as a complete-step accuracy claim; clean retraining and new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/mixed-modal-pair-11/train/receipt.json",
    "sha256": "aaf0a360c4c9622b7732dcffa43316181bb3d1e8150599fc9184704f5dd16c58"
  },
  "history": {
    "path": "runs/full-ai-tight-targets-v1/mixed-modal-pair-11/train/history.json",
    "sha256": "69c5bbc73bd3993b0134f93c67f6038ac806493cd21e081b5374b04529b026f8"
  },
  "sources": [
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/learned_modal_pressure.py",
      "sha256": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/modal_basis.py",
      "sha256": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/modal_precision_supervision.py",
      "sha256": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/train_modal_augmented.py",
      "sha256": "56080c341fcae478a05ef0b8556cb8a152d1f41b2a31b77e41862c00d84090b6"
    }
  ],
  "training_inputs": [
    {
      "bytes": 222215,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/manifest.json",
      "sha256": "80ad803ad0bad1bfe3dacd508a89900c92b80e66e0907c8b789918816c0d2637",
      "slot": "manifest"
    },
    {
      "bytes": 1219700,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/normalization.json",
      "sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 4302712,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/g16_16_16.npz",
      "sha256": "8d23da6de756e65bd8d2c9c93ef6bf6030eb2f26599a435e67f9dde54a8ee4fe",
      "slot": "part_00"
    },
    {
      "bytes": 1144626,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/g11_11_9.npz",
      "sha256": "b634da7620ff624b298e25cc72b513572377fbbff3796bba847d3c5278395a9a",
      "slot": "part_01"
    },
    {
      "bytes": 754520,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/g8_6_15.npz",
      "sha256": "5f2252f5743d13b8ba5f48bd64658031d5feea32f003a239860f457d388db9f0",
      "slot": "part_02"
    },
    {
      "bytes": 34418718,
      "path": "runs/full-ai-tight-targets-v1/mixed-modal-view-11/g32_32_32.npz",
      "sha256": "27ddaef5acafc1b6f0f8b6c83e2aa8092ba5e39fd939e30ce8b1a4e8b04679e1",
      "slot": "part_03"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-aaf0a360c4c9622b",
    "kind": "training_attempt",
    "name": "Full AI mixed modal query refinement 11 train",
    "task": "pressure_component",
    "schema": "rheon-learned-modal-pressure-result-v1",
    "method": {},
    "receipt_sha256": "aaf0a360c4c9622b7732dcffa43316181bb3d1e8150599fc9184704f5dd16c58",
    "receipt_paths": [
      "runs/full-ai-tight-targets-v1/mixed-modal-pair-11/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "dda357a46d4c4ebeaeac30bbb65488f3",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI mixed modal query refinement 11 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 80,
    "parameter_fits": null,
    "declared_parameter_count": 38673,
    "training_closure": {
      "config_sha256": "43bdb2ad4ef8f4d894ca796192f0d683535592cc35b18ea0f3bdf0dbfa9f3500",
      "inputs": {
        "manifest": {
          "bytes": 222215,
          "sha256": "80ad803ad0bad1bfe3dacd508a89900c92b80e66e0907c8b789918816c0d2637"
        },
        "normalization": {
          "bytes": 1219700,
          "sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 4302712,
          "sha256": "8d23da6de756e65bd8d2c9c93ef6bf6030eb2f26599a435e67f9dde54a8ee4fe"
        },
        "part_01": {
          "bytes": 1144626,
          "sha256": "b634da7620ff624b298e25cc72b513572377fbbff3796bba847d3c5278395a9a"
        },
        "part_02": {
          "bytes": 754520,
          "sha256": "5f2252f5743d13b8ba5f48bd64658031d5feea32f003a239860f457d388db9f0"
        },
        "part_03": {
          "bytes": 34418718,
          "sha256": "27ddaef5acafc1b6f0f8b6c83e2aa8092ba5e39fd939e30ce8b1a4e8b04679e1"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_modal_pressure.py": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "modal_basis.py": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4",
        "modal_precision_supervision.py": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146",
        "train_modal_augmented.py": "56080c341fcae478a05ef0b8556cb8a152d1f41b2a31b77e41862c00d84090b6"
      },
      "normalization_sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389",
      "role": "pressure_component_precision_refinement"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "54e54af7c078475d4daaa0145a51fe0e3593bfe6017bb2c18b1d42fb56340ed5",
        "bytes": 310480,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "82496540d6e195056f3d120733097cf4c470af69bef1756dc875022a1f3cb0f6",
        "bytes": 310480,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "tight-components/3"
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
      "receipt": "runs/full-ai-tight-targets-v1/mixed-modal-pair-11/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/mixed-modal-pair-11/train/receipt.json",
    "sha256": "aaf0a360c4c9622b7732dcffa43316181bb3d1e8150599fc9184704f5dd16c58",
    "bytes": 4923
  },
  "bound_training_configuration": {
    "config_sha256": "43bdb2ad4ef8f4d894ca796192f0d683535592cc35b18ea0f3bdf0dbfa9f3500",
    "graph_id": "e3ecf35925ac4154abc5b0ad566ab2a7",
    "graph_document_sha256": "9a48273bd20a5346f1f848740b90eea4bba51fae47bc87a58bb443334b4a2b25",
    "recipe_path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/train-recipe.json",
    "recipe_sha256": "dcea828147135813a2bd581eb380f4b36eb6af454e0eacb469c68545dd7938ef",
    "config_path": "runs/full-ai-tight-targets-v1/mixed-modal-project-11b/config.json",
    "configuration": {
      "identifier": "modal-pressure-mixed-queries-11",
      "role": "pressure_component_precision_refinement",
      "objective": "native_mixed_query_weighted_precision_v3",
      "updates": 80,
      "mode_scale_floor": 1e-12,
      "seed": 2026102201,
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
  "schema": "rheon-learned-modal-pressure-result-v1",
  "status": "complete",
  "mode": "train",
  "role": "pressure_component_precision_refinement",
  "optimizer_updates": 80,
  "parameters": 38673,
  "trainable_parameters": 38673,
  "closure": {
    "config_sha256": "43bdb2ad4ef8f4d894ca796192f0d683535592cc35b18ea0f3bdf0dbfa9f3500",
    "inputs": {
      "manifest": {
        "bytes": 222215,
        "sha256": "80ad803ad0bad1bfe3dacd508a89900c92b80e66e0907c8b789918816c0d2637"
      },
      "normalization": {
        "bytes": 1219700,
        "sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 4302712,
        "sha256": "8d23da6de756e65bd8d2c9c93ef6bf6030eb2f26599a435e67f9dde54a8ee4fe"
      },
      "part_01": {
        "bytes": 1144626,
        "sha256": "b634da7620ff624b298e25cc72b513572377fbbff3796bba847d3c5278395a9a"
      },
      "part_02": {
        "bytes": 754520,
        "sha256": "5f2252f5743d13b8ba5f48bd64658031d5feea32f003a239860f457d388db9f0"
      },
      "part_03": {
        "bytes": 34418718,
        "sha256": "27ddaef5acafc1b6f0f8b6c83e2aa8092ba5e39fd939e30ce8b1a4e8b04679e1"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_modal_pressure.py": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "modal_basis.py": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4",
      "modal_precision_supervision.py": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146",
      "train_modal_augmented.py": "56080c341fcae478a05ef0b8556cb8a152d1f41b2a31b77e41862c00d84090b6"
    },
    "normalization_sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389",
    "role": "pressure_component_precision_refinement"
  },
  "initial_inventory": {
    "log_gains.g11_11_9": {
      "shape": [
        11,
        11,
        9
      ],
      "dtype": "<f8",
      "sha256": "dfd9e245b2ac6c292ba031516be2cfa190a11fde2a0b226d46a312d553ae2c99"
    },
    "log_gains.g16_16_16": {
      "shape": [
        16,
        16,
        16
      ],
      "dtype": "<f8",
      "sha256": "5f62957ace84c3b7ddd8ddfc76ba549fd1350033c749abbd18b4411ce3afada0"
    },
    "log_gains.g32_32_32": {
      "shape": [
        32,
        32,
        32
      ],
      "dtype": "<f8",
      "sha256": "6c9b7cb1d5da88f49bc5a757467663845ee2f335dcbbdfba5d40f9d0130623e3"
    },
    "log_gains.g8_6_15": {
      "shape": [
        8,
        6,
        15
      ],
      "dtype": "<f8",
      "sha256": "5e5cd43e871c5d1967b241eba2f6a8731d96287e517bf212d17fd57ae32b6dd3"
    }
  },
  "final_inventory": {
    "log_gains.g11_11_9": {
      "shape": [
        11,
        11,
        9
      ],
      "dtype": "<f8",
      "sha256": "8046191a44168bdba71055a0b563e872fcb3cae728d4fe9087875aafabf8ee94"
    },
    "log_gains.g16_16_16": {
      "shape": [
        16,
        16,
        16
      ],
      "dtype": "<f8",
      "sha256": "fab71f534a69348db942bac7f8fb7a97526ebf71aa38da5010b20ae6128dead1"
    },
    "log_gains.g32_32_32": {
      "shape": [
        32,
        32,
        32
      ],
      "dtype": "<f8",
      "sha256": "d13bfc5c8db120cc59237023a5f1a21055162e98561be8c75d058fd26fb21ec7"
    },
    "log_gains.g8_6_15": {
      "shape": [
        8,
        6,
        15
      ],
      "dtype": "<f8",
      "sha256": "a9ea41abe2b7d89d1e5a3f8d9fcf7f37d005ec5d2645b65793af75b4365b4f84"
    }
  },
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 223250432
  },
  "outputs": {
    "probe.npy": {
      "bytes": 309512,
      "sha256": "8c1c7b5bd8a2fd48ebe4830f9d2e0ac69a2d91f48bb68868015a3e936b13d1eb"
    },
    "weights.npz": {
      "bytes": 310480,
      "sha256": "54e54af7c078475d4daaa0145a51fe0e3593bfe6017bb2c18b1d42fb56340ed5"
    },
    "initial-weights.npz": {
      "bytes": 310480,
      "sha256": "82496540d6e195056f3d120733097cf4c470af69bef1756dc875022a1f3cb0f6"
    },
    "history.json": {
      "bytes": 6390,
      "sha256": "69c5bbc73bd3993b0134f93c67f6038ac806493cd21e081b5374b04529b026f8"
    },
    "normalization.json": {
      "bytes": 1219700,
      "sha256": "68f9d94a08cacd4658d30810989d1d6c40acb214340c2f7968c70f3250286389"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.758704059990123
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "modal-pressure-mixed-queries-11",
  "role": "pressure_component_precision_refinement",
  "objective": "native_mixed_query_weighted_precision_v3",
  "updates": 80,
  "mode_scale_floor": 1e-12,
  "seed": 2026102201,
  "wall_seconds": 900
}

```
