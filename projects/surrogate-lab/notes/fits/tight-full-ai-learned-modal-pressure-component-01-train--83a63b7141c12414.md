# Tight full AI learned modal pressure component 01 train

Retrospective source review · fit-83a63b7141c12414

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A pressure-only F64 diagonal model with 38,673 stored log-gains across four registered grid shapes. Each mode has its own positive learned gain; four DC entries are stored but masked from prediction, leaving 38,669 active non-DC gains. Fixed full DCT-II transforms supply coordinates. The model multiplies supplied request coefficients by exponentiated gains; it does not advance velocity or tracer. Physical spacing enters the external q=(rho/h)RHS unit conversion, not an independent learned density channel. Unknown grid keys/shapes require a new exported or trained table.

**Purpose.** Test whether a geometry-specific full modal capacity can fit native pressure requests in F64 from uniform gains, before reducing parameter sharing.

**Experiment.** All 144 original pairs from 16 consumed training scenes, nine snapshots per scene at 8/16 cells per metre. Each full-batch update sees all four geometry groups. Initialization is uniform zero log-gains. 800 CUDA Adam updates use cosine rate decay from 0.1 to 0.001. Same-sign native p/q modes above a 1e-10 relative request floor supervise log-pressure minus log-request magnitudes; no analytic eigenvalue initializer is loaded. Original source/recipe/input/output identities and all F64 initial/final tensor bits verify without neural execution.

**Observed outcome.** The original fit completes its declared 800 outer updates. Final logged training objective is 1.8995179357619394e-13; it is specific to this objective and population. All initial/final parameters and inherited checkpoints verify, and the original non-DC label coverage is independently checked. Fresh common GPU pressure accuracy/timing, full-state compositions, independent holdout and long-horizon drift are not measured by this review.

**Lesson.** This is a finite geometry-specific gain table, not the later compact shared four-coefficient pressure component. Fitting all modes of four shapes does not prove unseen-shape or full-state generalization.

**Reproduction.** Retain the exact typed recipe, trainer/objective/basis/model/configuration/runtime, all four original coefficient shards, normalization and original native-source hashes. Refinements also require the full exact inherited F64 inventory transported in normalization; decoder labels require the declared fixed coordinate encoding, and mixed queries require their offline reference archive and seed. Original final publication and restore probes run CPU neural inference and need GPU-only qualification before rerunning. Do not reuse tiny component loss as a complete-step accuracy claim; clean retraining and new-Rheon regeneration remain unqualified.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/modal-pressure-pair-01/train/receipt.json",
    "sha256": "83a63b7141c124145d532c1cab47ce66ad0dcf7dc2ae8cdf42ad22136aea6983"
  },
  "history": {
    "path": "runs/full-ai-tight-targets-v1/modal-pressure-pair-01/train/history.json",
    "sha256": "1ad1305bcb28bc71ef4d7df8cd41c3c72dc2151449efca85b88932413da07e35"
  },
  "sources": [
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/custody_validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/learned_modal_pressure.py",
      "sha256": "3577291a97798bcfff3e8a4c5b91463f3b4c9e4f810133ce755e26eeb1420bd1"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/managed_fit_artifacts.py",
      "sha256": "8c62c1bfac2eb4eeeab78d726cc2957bddbf73a4687b70652169cd96b206075a"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/modal_basis.py",
      "sha256": "42beb8b6e8b3eb0ebb1722ba64c77691c16a185abecfa72edcadb4f7c5485bc4"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/modal_supervision.py",
      "sha256": "f71673de0e5d512e200b09c791be7f6c2413eff1253f50d93b3ef7ea3898c9b5"
    },
    {
      "path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/train_modal_pressure.py",
      "sha256": "fee48008a01246dbe5ac5d870d51d87f0ffa7c9b307789cc6ced3b84c1a0db48"
    }
  ],
  "training_inputs": [
    {
      "bytes": 153547,
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/manifest.json",
      "sha256": "77dbc37a73eeddeb061859474c5742a9e333b100ce5a535c4a3a19fd01140769",
      "slot": "manifest"
    },
    {
      "bytes": 455,
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/normalization.json",
      "sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214",
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
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/g16_16_16.npz",
      "sha256": "bbb6f887d30e9c24ea8681899029052380ce22b3a28046fc63c57a04ca157844",
      "slot": "part_00"
    },
    {
      "bytes": 605099,
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/g11_11_9.npz",
      "sha256": "b0331142f5f630d0dd2ad1db1a332c42faf901c69c25f9f88beb4664634493d3",
      "slot": "part_01"
    },
    {
      "bytes": 398854,
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/g8_6_15.npz",
      "sha256": "cb2d1ed2925c5ea5a86c86d8b058d7ad6833ba8a84a36ed61d42d8006f15c8c7",
      "slot": "part_02"
    },
    {
      "bytes": 18267578,
      "path": "runs/full-ai-tight-targets-v1/spectral-training-view-01/g32_32_32.npz",
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
    "item_id": "fit-83a63b7141c12414",
    "kind": "training_attempt",
    "name": "Tight full AI learned modal pressure component 01 train",
    "task": "pressure_component",
    "schema": "rheon-learned-modal-pressure-result-v1",
    "method": {},
    "receipt_sha256": "83a63b7141c124145d532c1cab47ce66ad0dcf7dc2ae8cdf42ad22136aea6983",
    "receipt_paths": [
      "runs/full-ai-tight-targets-v1/modal-pressure-pair-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "2b498bb65aae42f5bb834a1d0152c60c",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Tight full AI learned modal pressure component 01 train",
        "state": "completed"
      }
    ],
    "optimizer_updates": 800,
    "parameter_fits": null,
    "declared_parameter_count": 38673,
    "training_closure": {
      "config_sha256": "1c266f6ced168b4ceabda55ea0b71628f344b13fa31661589c773178d6eda78a",
      "inputs": {
        "manifest": {
          "bytes": 153547,
          "sha256": "77dbc37a73eeddeb061859474c5742a9e333b100ce5a535c4a3a19fd01140769"
        },
        "normalization": {
          "bytes": 455,
          "sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214"
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
        "modal_supervision.py": "f71673de0e5d512e200b09c791be7f6c2413eff1253f50d93b3ef7ea3898c9b5",
        "train_modal_pressure.py": "fee48008a01246dbe5ac5d870d51d87f0ffa7c9b307789cc6ced3b84c1a0db48"
      },
      "normalization_sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214",
      "role": "pressure_component_capacity_probe"
    },
    "checkpoints": [
      {
        "name": "weights.npz",
        "sha256": "a4fd4777319e292f71f5552742ee0233ac0fde038e309b38303fc68e6d3f4518",
        "bytes": 310480,
        "stage": "selected"
      },
      {
        "name": "initial-weights.npz",
        "sha256": "6e40fbb1c89c7f091b016da4e887f7feb1ad48d31772add77841d6e9444468a3",
        "bytes": 310480,
        "stage": "initial"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": null,
    "named_entries": [
      "tight-components/0"
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
      "receipt": "runs/full-ai-tight-targets-v1/modal-pressure-pair-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/full-ai-tight-targets-v1/modal-pressure-pair-01/train/receipt.json",
    "sha256": "83a63b7141c124145d532c1cab47ce66ad0dcf7dc2ae8cdf42ad22136aea6983",
    "bytes": 4892
  },
  "bound_training_configuration": {
    "config_sha256": "1c266f6ced168b4ceabda55ea0b71628f344b13fa31661589c773178d6eda78a",
    "graph_id": "6373adad2fd44a7980465983acb4b8d2",
    "graph_document_sha256": "7a050daf1aa1cf72391fa28cc3c0b1967ef05e2f26c56ee33c1d345470236510",
    "recipe_path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/train-recipe.json",
    "recipe_sha256": "99b6d12a02a67161c9943bfc66f085e74aa404db44da303435bd8bdfdc4aa79c",
    "config_path": "runs/full-ai-tight-targets-v1/modal-pressure-project-02/config.json",
    "configuration": {
      "identifier": "modal-pressure-component-01",
      "role": "pressure_component_capacity_probe",
      "objective": "native_supervised_log_gain_v1",
      "seed": 2026102049,
      "updates": 800,
      "learning_rate": 0.1,
      "minimum_learning_rate": 0.001,
      "label_relative_q_floor": 1e-10,
      "wall_seconds": 600,
      "dtype": "float64",
      "sampler": "All 144 original native training pairs in all four geometry batches per update",
      "init": "Uniform non-DC gains one; zero DC gauge; no analytic inverse or fitted initializer",
      "authority": "docs/plans/full-ai-tight-targets-v1/modal-pressure-component-protocol-01.md"
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
  "role": "pressure_component_capacity_probe",
  "optimizer_updates": 800,
  "parameters": 38673,
  "trainable_parameters": 38673,
  "closure": {
    "config_sha256": "1c266f6ced168b4ceabda55ea0b71628f344b13fa31661589c773178d6eda78a",
    "inputs": {
      "manifest": {
        "bytes": 153547,
        "sha256": "77dbc37a73eeddeb061859474c5742a9e333b100ce5a535c4a3a19fd01140769"
      },
      "normalization": {
        "bytes": 455,
        "sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214"
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
      "modal_supervision.py": "f71673de0e5d512e200b09c791be7f6c2413eff1253f50d93b3ef7ea3898c9b5",
      "train_modal_pressure.py": "fee48008a01246dbe5ac5d870d51d87f0ffa7c9b307789cc6ced3b84c1a0db48"
    },
    "normalization_sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214",
    "role": "pressure_component_capacity_probe"
  },
  "initial_inventory": {
    "log_gains.g11_11_9": {
      "shape": [
        11,
        11,
        9
      ],
      "dtype": "<f8",
      "sha256": "766414c979fd8f699b0da20be051526ac1383b437049fd638aadc2a349a5c235"
    },
    "log_gains.g16_16_16": {
      "shape": [
        16,
        16,
        16
      ],
      "dtype": "<f8",
      "sha256": "c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479"
    },
    "log_gains.g32_32_32": {
      "shape": [
        32,
        32,
        32
      ],
      "dtype": "<f8",
      "sha256": "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90"
    },
    "log_gains.g8_6_15": {
      "shape": [
        8,
        6,
        15
      ],
      "dtype": "<f8",
      "sha256": "32ead73abab870ab0c7ba67a2337215e63ae49394d3c22dbf133e7ce1c7a2a0a"
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
  "execution": {
    "device": "cuda",
    "cuda_initialized": true,
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "dtype": "float64",
    "tf32": false,
    "amp": false,
    "peak_cuda_allocated_bytes": 84938240
  },
  "outputs": {
    "probe.npy": {
      "bytes": 309512,
      "sha256": "00c503b4a6d8f3dda2914bf3fb308e42eaacec67cbdf38e9e114e2a2a8dd28fb"
    },
    "weights.npz": {
      "bytes": 310480,
      "sha256": "a4fd4777319e292f71f5552742ee0233ac0fde038e309b38303fc68e6d3f4518"
    },
    "initial-weights.npz": {
      "bytes": 310480,
      "sha256": "6e40fbb1c89c7f091b016da4e887f7feb1ad48d31772add77841d6e9444468a3"
    },
    "history.json": {
      "bytes": 25241,
      "sha256": "1ad1305bcb28bc71ef4d7df8cd41c3c72dc2151449efca85b88932413da07e35"
    },
    "normalization.json": {
      "bytes": 455,
      "sha256": "aaf71cebe36016335cec89e8b3f10115d583466e0d4cadecc7130a102cfb9214"
    }
  },
  "full_cpu_probe_bitwise_verified": false,
  "originating_training_run": null,
  "causal_full_state_generator": false,
  "scientific_acceptance_or_promotion": false,
  "elapsed_s": 4.663176872010808
}

```

## Exact hash-bound training configuration

```json
{
  "identifier": "modal-pressure-component-01",
  "role": "pressure_component_capacity_probe",
  "objective": "native_supervised_log_gain_v1",
  "seed": 2026102049,
  "updates": 800,
  "learning_rate": 0.1,
  "minimum_learning_rate": 0.001,
  "label_relative_q_floor": 1e-10,
  "wall_seconds": 600,
  "dtype": "float64",
  "sampler": "All 144 original native training pairs in all four geometry batches per update",
  "init": "Uniform non-DC gains one; zero DC gauge; no analytic inverse or fitted initializer",
  "authority": "docs/plans/full-ai-tight-targets-v1/modal-pressure-component-protocol-01.md"
}

```
