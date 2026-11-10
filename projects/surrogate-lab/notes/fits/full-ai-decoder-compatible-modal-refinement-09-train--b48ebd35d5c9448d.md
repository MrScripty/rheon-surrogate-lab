# Full AI decoder compatible modal refinement 09 train

Retrospective source review · fit-b48ebd35d5c9448d

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only F64 diagonal model with 38,673 stored log-gains across four registered grid shapes. Each mode has its own positive learned gain; four DC entries are stored but masked from prediction, leaving 38,669 active non-DC gains. Fixed full DCT-II transforms supply coordinates. The model multiplies supplied request coefficients by exponentiated gains; it does not advance velocity or tracer. Physical spacing enters the external q=(rho/h)RHS unit conversion, not an independent learned density channel. Unknown grid keys/shapes require a new exported or trained table.

**Purpose.** Train the same learned gains against labels encoded in the fixed decoder’s actual coordinates, addressing numerical coordinate reconstruction rather than changing physical scenes.

**Experiment.** All 144 original pairs from 16 consumed training scenes, nine snapshots per scene at 8/16 cells per metre. Each full-batch update sees all four geometry groups. Initialization is the exact preceding learned checkpoint. 80 CUDA LBFGS outer updates use strong-Wolfe search with 162 recorded closure evaluations. Native coefficient errors are normalized by train-only per-mode pressure RMS with a 1e-12 relative floor; no p/q coefficient target is formed. Original source/recipe/input/output identities and all F64 initial/final tensor bits verify without neural execution.

**Observed outcome.** The original fit completes its declared 80 outer updates. Final logged training objective is 9.0749947002507764e-16; it is specific to this objective and population. All initial/final parameters and inherited checkpoints verify, and the original non-DC label coverage is independently checked. Fresh common GPU pressure accuracy/timing, full-state compositions, independent holdout and long-horizon drift are not measured by this review.

**Lesson.** Label-coordinate reconstruction can affect precision near floating-point limits without introducing new physical data. The fixed basis inverse used offline is distinct from an analytical physical pressure inverse at inference.

**Reproduction.** Retain the exact typed recipe, trainer/objective/basis/model/configuration/runtime, all four original coefficient shards, normalization and original native-source hashes. Refinements also require the full exact inherited F64 inventory transported in normalization; decoder labels require the declared fixed coordinate encoding, and mixed queries require their offline reference archive and seed. Original final publication and restore probes run CPU neural inference and need GPU-only qualification before rerunning. Do not reuse tiny component loss as a complete-step accuracy claim; clean retraining and new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/decoder-modal-pair-09/train/receipt.json",
    "sha256": "b48ebd35d5c9448d5d03330136251fa092ce88220dddee0ee13cfd82a32993f3"
  },
  "history": {
    "path": "runs/full-ai-tight-targets-v1/decoder-modal-pair-09/train/history.json",
    "sha256": "b010333bb21f7523984b34e2ed5e3dce8e32ba41bf30aae2e837347b66f63e1a"
  },
  "sources": [
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/learned_modal_pressure.py",
      "sha256": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/modal_basis.py",
      "sha256": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/modal_precision_supervision.py",
      "sha256": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/train_modal_precision.py",
      "sha256": "e3c603b25b8cd46ec41b6c8ba7fc66779e4d7cda30cf48aa95ce824821f6b86d"
    }
  ],
  "training_inputs": [
    {
      "bytes": 154115,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/manifest.json",
      "sha256": "bb95f8d391d424da8e13b6d35e6fb7377b4a0042314f36391bc7d6c89e957562",
      "slot": "manifest"
    },
    {
      "bytes": 1219567,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/normalization.json",
      "sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 2277512,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/g16_16_16.npz",
      "sha256": "46828cb43365bdfca13285f39c1c1536882b882a336c0e8353cb2e6c797a2513",
      "slot": "part_00"
    },
    {
      "bytes": 605041,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/g11_11_9.npz",
      "sha256": "f9e4b17c46fbb186515888e30e1f66d09e67fc936448a59e503ecaa607d1c9df",
      "slot": "part_01"
    },
    {
      "bytes": 398858,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/g8_6_15.npz",
      "sha256": "353d8127d50e50a1a49034b6161ba49a9dad9b384b497539b6eb680057cb0089",
      "slot": "part_02"
    },
    {
      "bytes": 18267160,
      "path": "runs/full-ai-tight-targets-v1/decoder-modal-view-09/g32_32_32.npz",
      "sha256": "8300c7160d57d6a237ccb98f3d51ad9443d1e31d7892aca7cd77fa27d6f5bd10",
      "slot": "part_03"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-b48ebd35d5c9448d",
    "kind": "training_attempt",
    "name": "Full AI decoder compatible modal refinement 09 train",
    "task": "pressure_component",
    "schema": "rheon-learned-modal-pressure-result-v1",
    "method": {},
    "receipt_sha256": "b48ebd35d5c9448d5d03330136251fa092ce88220dddee0ee13cfd82a32993f3",
    "receipt_paths": [
      "runs/full-ai-tight-targets-v1/decoder-modal-pair-09/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "aab56647e0fa47e1bae79ddca9a1b03b",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI decoder compatible modal refinement 09 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 80,
    "parameter_fits": null,
    "declared_parameter_count": 38673,
    "training_closure": {
      "config_sha256": "482c3a544f48d0f42a645e394c5612bc02c956f4093f8626bf58a195e6136251",
      "inputs": {
        "manifest": {
          "bytes": 154115,
          "sha256": "bb95f8d391d424da8e13b6d35e6fb7377b4a0042314f36391bc7d6c89e957562"
        },
        "normalization": {
          "bytes": 1219567,
          "sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 2277512,
          "sha256": "46828cb43365bdfca13285f39c1c1536882b882a336c0e8353cb2e6c797a2513"
        },
        "part_01": {
          "bytes": 605041,
          "sha256": "f9e4b17c46fbb186515888e30e1f66d09e67fc936448a59e503ecaa607d1c9df"
        },
        "part_02": {
          "bytes": 398858,
          "sha256": "353d8127d50e50a1a49034b6161ba49a9dad9b384b497539b6eb680057cb0089"
        },
        "part_03": {
          "bytes": 18267160,
          "sha256": "8300c7160d57d6a237ccb98f3d51ad9443d1e31d7892aca7cd77fa27d6f5bd10"
        }
      },
      "sources": {
        "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
        "learned_modal_pressure.py": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1",
        "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
        "modal_basis.py": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4",
        "modal_precision_supervision.py": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146",
        "train_modal_precision.py": "e3c603b25b8cd46ec41b6c8ba7fc66779e4d7cda30cf48aa95ce824821f6b86d"
      },
      "normalization_sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e",
      "role": "pressure_component_precision_refinement"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "82496540d6e195056f3d120733097cf4c470af69bef1756dc875022a1f3cb0f6",
        "bytes": 310480,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "55967199eb87a00cee17c0e03afe379dd8e234f2cd8469315d5a52c2580e6924",
        "bytes": 310480,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "tight-components/2"
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
      "receipt": "runs/full-ai-tight-targets-v1/decoder-modal-pair-09/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/decoder-modal-pair-09/train/receipt.json",
    "sha256": "b48ebd35d5c9448d5d03330136251fa092ce88220dddee0ee13cfd82a32993f3",
    "bytes": 4923
  },
  "bound_training_configuration": {
    "config_sha256": "482c3a544f48d0f42a645e394c5612bc02c956f4093f8626bf58a195e6136251",
    "graph_id": "3f80c4286f9a4309996108db995d9d6b",
    "graph_document_sha256": "16f30b68ad55e83ff0573e39d5093af09ef4164aa9d6863a67ae6006992825db",
    "recipe_path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/train-recipe.json",
    "recipe_sha256": "0977b3cee365f897bcda43be0c1308a1dcc8483b4585e39d0409459091968959",
    "config_path": "runs/full-ai-tight-targets-v1/decoder-modal-project-09/config.json",
    "configuration": {
      "identifier": "modal-pressure-decoder-compatible-09",
      "role": "pressure_component_precision_refinement",
      "objective": "native_coefficient_weighted_precision_v2",
      "updates": 80,
      "mode_scale_floor": 1e-12,
      "seed": 2026102113,
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
    "config_sha256": "482c3a544f48d0f42a645e394c5612bc02c956f4093f8626bf58a195e6136251",
    "inputs": {
      "manifest": {
        "bytes": 154115,
        "sha256": "bb95f8d391d424da8e13b6d35e6fb7377b4a0042314f36391bc7d6c89e957562"
      },
      "normalization": {
        "bytes": 1219567,
        "sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 2277512,
        "sha256": "46828cb43365bdfca13285f39c1c1536882b882a336c0e8353cb2e6c797a2513"
      },
      "part_01": {
        "bytes": 605041,
        "sha256": "f9e4b17c46fbb186515888e30e1f66d09e67fc936448a59e503ecaa607d1c9df"
      },
      "part_02": {
        "bytes": 398858,
        "sha256": "353d8127d50e50a1a49034b6161ba49a9dad9b384b497539b6eb680057cb0089"
      },
      "part_03": {
        "bytes": 18267160,
        "sha256": "8300c7160d57d6a237ccb98f3d51ad9443d1e31d7892aca7cd77fa27d6f5bd10"
      }
    },
    "sources": {
      "custody_validator.py": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57",
      "learned_modal_pressure.py": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1",
      "managed_fit_artifacts.py": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a",
      "modal_basis.py": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4",
      "modal_precision_supervision.py": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146",
      "train_modal_precision.py": "e3c603b25b8cd46ec41b6c8ba7fc66779e4d7cda30cf48aa95ce824821f6b86d"
    },
    "normalization_sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e",
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
      "sha256": "16adcf8e7d130122897a8139331ec5b3d9cf7c05482a30e869d149ab09833911"
    },
    "log_gains.g16_16_16": {
      "shape": [
        16,
        16,
        16
      ],
      "dtype": "<f8",
      "sha256": "bcffe54289fb53edb9cbf4f9d34ab15152dd274350a448e4e3a243bf204ea5b7"
    },
    "log_gains.g32_32_32": {
      "shape": [
        32,
        32,
        32
      ],
      "dtype": "<f8",
      "sha256": "84bbb20e43ac5e920ab66ecab1321cefea102bd953fdcf651f6125108c4021e9"
    },
    "log_gains.g8_6_15": {
      "shape": [
        8,
        6,
        15
      ],
      "dtype": "<f8",
      "sha256": "51a9af2eb6fe4f6a75a1b232510992e20aeb2c898a62a1009407f8a710a237b6"
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
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 130756096
  },
  "outputs": {
    "probe.npy": {
      "bytes": 309512,
      "sha256": "71735b9525c704fd96782057cc61ef1cbd1173742371878d49fa7bf79ee871af"
    },
    "weights.npz": {
      "bytes": 310480,
      "sha256": "82496540d6e195056f3d120733097cf4c470af69bef1756dc875022a1f3cb0f6"
    },
    "initial-weights.npz": {
      "bytes": 310480,
      "sha256": "55967199eb87a00cee17c0e03afe379dd8e234f2cd8469315d5a52c2580e6924"
    },
    "history.json": {
      "bytes": 6310,
      "sha256": "b010333bb21f7523984b34e2ed5e3dce8e32ba41bf30aae2e837347b66f63e1a"
    },
    "normalization.json": {
      "bytes": 1219567,
      "sha256": "66563660e306e3341f30dcb8e7a28ba13d8497faec3f79ac15a3b288953a0c1e"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.6898905980051495
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "modal-pressure-decoder-compatible-09",
  "role": "pressure_component_precision_refinement",
  "objective": "native_coefficient_weighted_precision_v2",
  "updates": 80,
  "mode_scale_floor": 1e-12,
  "seed": 2026102113,
  "wall_seconds": 900
}

```
