# Full AI learned modal precision refinement 01 train

Retrospective source review · fit-561e02c747b96415

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only F64 diagonal model with 38,673 stored log-gains across four registered grid shapes. Each mode has its own positive learned gain; four DC entries are stored but masked from prediction, leaving 38,669 active non-DC gains. Fixed full DCT-II transforms supply coordinates. The model multiplies supplied request coefficients by exponentiated gains; it does not advance velocity or tracer. Physical spacing enters the external q=(rho/h)RHS unit conversion, not an independent learned density channel. Unknown grid keys/shapes require a new exported or trained table.

**Purpose.** Refine the completed modal model with per-mode weighted native coefficient least squares to improve numerical precision without prescribing analytical inverse-gain labels.

**Experiment.** All 144 original pairs from 16 consumed training scenes, nine snapshots per scene at 8/16 cells per metre. Each full-batch update sees all four geometry groups. Initialization is the exact preceding learned checkpoint. 80 CUDA LBFGS outer updates use strong-Wolfe search with 164 recorded closure evaluations. Native coefficient errors are normalized by train-only per-mode pressure RMS with a 1e-12 relative floor; no p/q coefficient target is formed. Original source/recipe/input/output identities and all F64 initial/final tensor bits verify without neural execution.

**Observed outcome.** The original fit completes its declared 80 outer updates. Final logged training objective is 9.0276545206777177e-16; it is specific to this objective and population. All initial/final parameters and inherited checkpoints verify, and the original non-DC label coverage is independently checked. Fresh common GPU pressure accuracy/timing, full-state compositions, independent holdout and long-horizon drift are not measured by this review.

**Lesson.** An 80-step LBFGS refinement is not 80 single Adam forwards: line-search closure work matters. A tiny weighted training objective is not pressure relative error or simulation drift.

**Reproduction.** Retain the exact typed recipe, trainer/objective/basis/model/configuration/runtime, all four original coefficient shards, normalization and original native-source hashes. Refinements also require the full exact inherited F64 inventory transported in normalization; decoder labels require the declared fixed coordinate encoding, and mixed queries require their offline reference archive and seed. Original final publication and restore probes run CPU neural inference and need GPU-only qualification before rerunning. Do not reuse tiny component loss as a complete-step accuracy claim; clean retraining and new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/modal-precision-pair-02/train/receipt.json",
    "sha256": "561e02c747b96415c0e12e51b17f5e0321bc7367cbe93acf80347276325584f1"
  },
  "history": {
    "path": "runs/full-ai-tight-targets-v1/modal-precision-pair-02/train/history.json",
    "sha256": "5592a4878250eabf0e842733b017cb138595216acdcbbb295b643b8d869702c1"
  },
  "sources": [
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/learned_modal_pressure.py",
      "sha256": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/modal_basis.py",
      "sha256": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/modal_precision_supervision.py",
      "sha256": "2e33009d2700ca2302c78b4f64bc5255b19c880e827b9f50e5170f032a913146"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/train_modal_precision.py",
      "sha256": "e3c603b25b8cd46ec41b6c8ba7fc66779e4d7cda30cf48aa95ce824821f6b86d"
    }
  ],
  "training_inputs": [
    {
      "bytes": 155025,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/manifest.json",
      "sha256": "9dc579acf98fef02996bfd09631657f184c94d95270dc0bd3086d4707bc64325",
      "slot": "manifest"
    },
    {
      "bytes": 1218858,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/normalization.json",
      "sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81",
      "slot": "normalization"
    },
    {
      "bytes": 7989288,
      "path": "runs/dependency-migration-v1/runtime/current-manifest-v2.json",
      "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "slot": "runtime_manifest"
    },
    {
      "bytes": 2277547,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/g16_16_16.npz",
      "sha256": "bbb6f887d30e9c24ea8681899029052380ce22b3a28046fc63c57a04ca157844",
      "slot": "part_00"
    },
    {
      "bytes": 605099,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/g11_11_9.npz",
      "sha256": "b0331142f5f630d0dd2ad1db1a332c42faf901c69c25f9f88beb4664634493d3",
      "slot": "part_01"
    },
    {
      "bytes": 398854,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/g8_6_15.npz",
      "sha256": "cb2d1ed2925c5ea5a86c86d8b058d7ad6833ba8a84a36ed61d42d8006f15c8c7",
      "slot": "part_02"
    },
    {
      "bytes": 18267578,
      "path": "runs/full-ai-tight-targets-v1/modal-precision-view-01/g32_32_32.npz",
      "sha256": "7300d0edb0d3dd51afae11949a956f04b7951f2ac5e51449d1d5d871ef935e0b",
      "slot": "part_03"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-561e02c747b96415",
    "kind": "training_attempt",
    "name": "Full AI learned modal precision refinement 01 train",
    "task": "pressure_component",
    "schema": "rheon-learned-modal-pressure-result-v1",
    "method": {},
    "receipt_sha256": "561e02c747b96415c0e12e51b17f5e0321bc7367cbe93acf80347276325584f1",
    "receipt_paths": [
      "runs/full-ai-tight-targets-v1/modal-precision-pair-02/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "af6b5518eb834e2fa0bcae791002fa6f",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Full AI learned modal precision refinement 01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 80,
    "parameter_fits": null,
    "declared_parameter_count": 38673,
    "training_closure": {
      "config_sha256": "a71e035e0a10c0db19291895725aece0ca4452f0b179e501e7287d05b0c837b6",
      "inputs": {
        "manifest": {
          "bytes": 155025,
          "sha256": "9dc579acf98fef02996bfd09631657f184c94d95270dc0bd3086d4707bc64325"
        },
        "normalization": {
          "bytes": 1218858,
          "sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81"
        },
        "runtime_manifest": {
          "bytes": 7989288,
          "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
        },
        "part_00": {
          "bytes": 2277547,
          "sha256": "bbb6f887d30e9c24ea8681899029052380ce22b3a28046fc63c57a04ca157844"
        },
        "part_01": {
          "bytes": 605099,
          "sha256": "b0331142f5f630d0dd2ad1db1a332c42faf901c69c25f9f88beb4664634493d3"
        },
        "part_02": {
          "bytes": 398854,
          "sha256": "cb2d1ed2925c5ea5a86c86d8b058d7ad6833ba8a84a36ed61d42d8006f15c8c7"
        },
        "part_03": {
          "bytes": 18267578,
          "sha256": "7300d0edb0d3dd51afae11949a956f04b7951f2ac5e51449d1d5d871ef935e0b"
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
      "normalization_sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81",
      "role": "pressure_component_precision_refinement"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "55967199eb87a00cee17c0e03afe379dd8e234f2cd8469315d5a52c2580e6924",
        "bytes": 310480,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "a4fd4777319e292f71f5552742ee0233ac0fde038e309b38303fc68e6d3f4518",
        "bytes": 310480,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "tight-components/1"
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
      "receipt": "runs/full-ai-tight-targets-v1/modal-precision-pair-02/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/modal-precision-pair-02/train/receipt.json",
    "sha256": "561e02c747b96415c0e12e51b17f5e0321bc7367cbe93acf80347276325584f1",
    "bytes": 4923
  },
  "bound_training_configuration": {
    "config_sha256": "a71e035e0a10c0db19291895725aece0ca4452f0b179e501e7287d05b0c837b6",
    "graph_id": "20424b0947044d6b8957f7aa818bdefa",
    "graph_document_sha256": "cf38b2f8fcdaa9f1a601bfde3a8cfc72d68bb84e29b18dc54495d10458dbc7c9",
    "recipe_path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/train-recipe.json",
    "recipe_sha256": "084f863cfbd0febc0634b99beb9d4a084890bfcbd2b22bb061c1845a81bf180b",
    "config_path": "runs/full-ai-tight-targets-v1/modal-precision-project-02/config.json",
    "configuration": {
      "identifier": "modal-pressure-precision-01",
      "role": "pressure_component_precision_refinement",
      "objective": "native_coefficient_weighted_precision_v2",
      "updates": 80,
      "mode_scale_floor": 1e-12,
      "seed": 2026102083,
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
    "config_sha256": "a71e035e0a10c0db19291895725aece0ca4452f0b179e501e7287d05b0c837b6",
    "inputs": {
      "manifest": {
        "bytes": 155025,
        "sha256": "9dc579acf98fef02996bfd09631657f184c94d95270dc0bd3086d4707bc64325"
      },
      "normalization": {
        "bytes": 1218858,
        "sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81"
      },
      "runtime_manifest": {
        "bytes": 7989288,
        "sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682"
      },
      "part_00": {
        "bytes": 2277547,
        "sha256": "bbb6f887d30e9c24ea8681899029052380ce22b3a28046fc63c57a04ca157844"
      },
      "part_01": {
        "bytes": 605099,
        "sha256": "b0331142f5f630d0dd2ad1db1a332c42faf901c69c25f9f88beb4664634493d3"
      },
      "part_02": {
        "bytes": 398854,
        "sha256": "cb2d1ed2925c5ea5a86c86d8b058d7ad6833ba8a84a36ed61d42d8006f15c8c7"
      },
      "part_03": {
        "bytes": 18267578,
        "sha256": "7300d0edb0d3dd51afae11949a956f04b7951f2ac5e51449d1d5d871ef935e0b"
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
    "normalization_sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81",
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
      "sha256": "4a187abb9c8f29571323b6e4a2ac6b37c0f4b7583420102338409898b8dfbe26"
    },
    "log_gains.g16_16_16": {
      "shape": [
        16,
        16,
        16
      ],
      "dtype": "<f8",
      "sha256": "32d5c29fdc6e113e954e3b0baf756d6ca2e361017af06501ddbd08ef78e6a2a1"
    },
    "log_gains.g32_32_32": {
      "shape": [
        32,
        32,
        32
      ],
      "dtype": "<f8",
      "sha256": "10410d551047fb584d5c225a730b2e4c819629f45ac109966e1ae7b2d4bd840f"
    },
    "log_gains.g8_6_15": {
      "shape": [
        8,
        6,
        15
      ],
      "dtype": "<f8",
      "sha256": "e13820b27aef0766d3c806dc1b0f04b39250d658f887a3195bb4f3ee723ca660"
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
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 131377664
  },
  "outputs": {
    "probe.npy": {
      "bytes": 309512,
      "sha256": "e291d26822770e162a794b563f150f8177b775a1e5a2d54034f7aad9beb4a16a"
    },
    "weights.npz": {
      "bytes": 310480,
      "sha256": "55967199eb87a00cee17c0e03afe379dd8e234f2cd8469315d5a52c2580e6924"
    },
    "initial-weights.npz": {
      "bytes": 310480,
      "sha256": "a4fd4777319e292f71f5552742ee0233ac0fde038e309b38303fc68e6d3f4518"
    },
    "history.json": {
      "bytes": 6311,
      "sha256": "5592a4878250eabf0e842733b017cb138595216acdcbbb295b643b8d869702c1"
    },
    "normalization.json": {
      "bytes": 1218858,
      "sha256": "a05d6696743a5f545c39480b697cdb9e5da248c7527d1791ece7290d0a238f81"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.6314240339561366
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "modal-pressure-precision-01",
  "role": "pressure_component_precision_refinement",
  "objective": "native_coefficient_weighted_precision_v2",
  "updates": 80,
  "mode_scale_floor": 1e-12,
  "seed": 2026102083,
  "wall_seconds": 900
}

```
