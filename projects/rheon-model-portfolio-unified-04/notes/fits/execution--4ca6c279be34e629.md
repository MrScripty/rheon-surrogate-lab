# execution

Retrospective source review · fit-4ca6c279be34e629

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 372,377-parameter, 38-tensor F32 multiscale pressure CNN on a fixed 16³ grid. Ten inputs are native residual, previous pressure, physical XYZ coordinates, separate hx/hy/hz spacing maps, fluid density and dt. Four encoder levels, bottleneck/global-mean path and skip decoder use SiLU with non-affine four-group normalization inserted before each block activation. The additive pressure correction is rescaled and cell-zero gauged in F64. The physical operator supports anisotropic spacing with area/(rho*h) face weights. This produces pressure only, not a full simulation state.

**Purpose.** Repeat the same configured training/restore procedure on a separately captured seed cohort, exercising capture/bootstrap reuse while retaining the original architecture and initialization.

**Experiment.** Two hundred historical CPU Adam updates, batch two, rate 0.001, seed 20261014, PCG64 sampling seed 20261015 and clipping. Twelve training trajectories × 24 frames and four validation trajectories × 24 frames use whole-parent splits. All grids have 16 cells per axis; physical extents vary over four shapes, including anisotropic cell spacing. Fluid density is 1.25 kg/m³ and dt=1/60 s throughout. Forcing includes steady 0.5 and moving 1.0 m/s² scenes. Eleven checkpoints at updates 0,20,…,200 are retained; only the final update is the candidate, with no optimizer-resume state. All original input arrays and per-frame pressure/RHS/residual hashes verify in place.

**Observed outcome.** Original final sampled training loss 0.071226833071870832; original validation objective changes from 8.1732476131712222 to 0.080808430266081338. These normalized pressure/equation objectives are not worst pressure relative L2 and cannot be compared directly with the near-exact F64 anchors. No current common GPU benchmark, complete-step time or trajectory drift is available for these two original contracts.

**Lesson.** Fixed grid counts do not imply fixed physical resolution: these scenes change physical extent and separate per-axis spacing, unlike later cubic-cell water models. The fluid density is also different from the later 1,000 kg/m³ convention. Preserve those operator/normalization contracts and original native identity. Matching initialized tensors and sample indices across cohorts do not make their physical field data identical or establish independent generalization.

**Reproduction.** Use the exact project-relative train/restore recipes, witnessed flat source, original runtime, all five typed input slots, train-only normalization and every observational checkpoint. Native capture and residual lineage are retained down to each row. The old model and physical checks explicitly require CPU; qualify a GPU-only execution/physical-operator variant before any new forward under current policy. Clean retraining, new-Rheon regeneration and common pressure adapters remain unqualified. Original receipts/configuration remain unchanged.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/receipt.json",
    "sha256": "4ca6c279be34e6293c8daf966c26c38c29b046dc19140422fad66f3ffdb0857f"
  },
  "history": {
    "path": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/history.json",
    "sha256": "9cf32a6f5f4f49ff8dfb4d5b546f05ee386435c4f1f10c71a5b0f5ed37d8c7f0"
  },
  "sources": [
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/config.json",
      "sha256": "eabe8c5909b747e29a3b544f9666c865f67a4a2e337140144e84b00ac5abfadd"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/contract.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/data.py",
      "sha256": "fafc7ae87e3456c6282daa3256d92302d68aadbe5090772b7b357052a71b3cea"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/gn_model.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/model_profile.py",
      "sha256": "12a86b642943611757890881b44f697784f48a1fc32d0257c955e7d672575096"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/qualification.json",
      "sha256": "b9047bd0a44f52b38deebb90422ead0e213a538724ab6552f0de4d1631836ba4"
    },
    {
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/train.py",
      "sha256": "a6eb87c83b28390af27ff000854f8a4f4011ed0c7d7312f6cf4274a3209fb9b2"
    }
  ],
  "source_witness": {
    "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/source-witness.json",
    "sha256": "bdcef2d54902f3b507f17a806b9bce1ad819afdf3c3d17e964a7a285ab6b2db8"
  },
  "training_population": {
    "path": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/splits.json",
    "sha256": "0bbd2ac6403ef59603d2a83ed5c242089c1d42401253160a286a6800559c409f"
  },
  "training_inputs": [
    {
      "bytes": 1145917,
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/inputs/origin.json",
      "sha256": "0fa6da4be6a700b9e5ff314302d299f40deeb5be114a5cdd073c9035c0b68281",
      "slot": "origin"
    },
    {
      "bytes": 401235,
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/inputs/splits.json",
      "sha256": "0bbd2ac6403ef59603d2a83ed5c242089c1d42401253160a286a6800559c409f",
      "slot": "splits"
    },
    {
      "bytes": 18890920,
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/inputs/train-0.npz",
      "sha256": "2809b91e08b2be41152663b0e970eeea31ed6831e76e3d5b4c1dea93c01601dd",
      "slot": "train0"
    },
    {
      "bytes": 18890920,
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/inputs/train-1.npz",
      "sha256": "6f97789935a82d01a613096bb5b121584009816e98bf64f4f0f285bf2eac973f",
      "slot": "train1"
    },
    {
      "bytes": 12594856,
      "path": "runs/local-pressure-v1-fresh-bootstrap-2/project/inputs/validation.npz",
      "sha256": "26d3c6b28d7602dbb20862d3be530c9479d1f78484db9d03e9f9cf16e3dc043e",
      "slot": "validation"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-4ca6c279be34e629",
    "kind": "training_attempt",
    "name": "execution",
    "task": "pressure_component",
    "schema": "rheon-local-pressure-training-receipt-v1",
    "method": {},
    "receipt_sha256": "4ca6c279be34e6293c8daf966c26c38c29b046dc19140422fad66f3ffdb0857f",
    "receipt_paths": [
      "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/receipt.json",
      "runs/local-pressure-v1-fresh-bootstrap-2/execution/stage-00/artifacts/receipt.json",
      "runs/local-pressure-v1-fresh-bootstrap-2/catalog/workspace-runs/0f923b6401914fae9d9885b6fa06b9b9/receipt.json",
      "runs/local-pressure-v1-fresh-bootstrap-2/catalog/workspace-runs/.external-0f923b6401914fae9d9885b6fa06b9b9/output/receipt.json"
    ],
    "recorded_Run_observations": [],
    "optimizer_updates": 200,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "eabe8c5909b747e29a3b544f9666c865f67a4a2e337140144e84b00ac5abfadd",
      "inputs": {
        "origin": {
          "bytes": 1145917,
          "sha256": "0fa6da4be6a700b9e5ff314302d299f40deeb5be114a5cdd073c9035c0b68281"
        },
        "splits": {
          "bytes": 401235,
          "sha256": "0bbd2ac6403ef59603d2a83ed5c242089c1d42401253160a286a6800559c409f"
        },
        "train0": {
          "bytes": 18890920,
          "sha256": "2809b91e08b2be41152663b0e970eeea31ed6831e76e3d5b4c1dea93c01601dd"
        },
        "train1": {
          "bytes": 18890920,
          "sha256": "6f97789935a82d01a613096bb5b121584009816e98bf64f4f0f285bf2eac973f"
        },
        "validation": {
          "bytes": 12594856,
          "sha256": "26d3c6b28d7602dbb20862d3be530c9479d1f78484db9d03e9f9cf16e3dc043e"
        }
      },
      "normalization_sha256": "21ab548bea68abc5d5d6a6c459a88664aa67c13a8e521e91c07b6eb3d68fd74e",
      "runtime_manifest_sha256": "ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7",
      "sample_order_sha256": "1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9",
      "source_witness_sha256": "bdcef2d54902f3b507f17a806b9bce1ad819afdf3c3d17e964a7a285ab6b2db8"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "1b4b77cf299d798b6b787bbd12486bc8d4301d7534972c66e9dddce41e93f737",
        "bytes": 1500186,
        "stage": "initial"
      },
      {
        "name": "checkpoint-020.npz",
        "sha256": "1202ee989e27c189c81958dc574c975826b5642ee2cbc2184f5fa2350321b756",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-040.npz",
        "sha256": "ba0952627f002c9d4c998c3be4d3e0dccc1c61f1201d9cb8d4be7c8f35b0ff8b",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-060.npz",
        "sha256": "86e6858be9969317ab1012dd93849c02c7c33357b6d98601fc9137190f2acfd1",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-080.npz",
        "sha256": "8a73859446b1859b22e2d77eedf57063f3840414e108faa6f36ef39529b2cff8",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "2fede00fc71bc4e92fadd6949c8f540fcfef0be7a523f4083cb4e7cbc4e9914f",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-120.npz",
        "sha256": "412231f63b57bef7073553035ce81b9f5a6e6beee8cf4c5cb509335f6a077e37",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-140.npz",
        "sha256": "2f679db862badd40347a0b90d865bb300c139600243302b1ac6096ef206fbba1",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-160.npz",
        "sha256": "e08ea5037ebff1d653f6bcfe20c557d9c2247cbda66ba9c2a3c65e4bdde90cc7",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-180.npz",
        "sha256": "160d6d32c23cdb83bb8ee4c17cf0fe0d0a6f19899810864703eed4fc6088e688",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-200.npz",
        "sha256": "e85e58a8682912e8c896d2af8d267c8e6e880a64dfd7965ac41ce1a5e5853665",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "e85e58a8682912e8c896d2af8d267c8e6e880a64dfd7965ac41ce1a5e5853665",
        "bytes": 1500186,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "interop_threads": 1,
      "numpy": "2.3.5",
      "peak_rss_bytes": 572751872,
      "python": "3.12.3",
      "threads": 1,
      "torch": "2.8.0+cpu"
    },
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
      "receipt": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/receipt.json",
    "sha256": "4ca6c279be34e6293c8daf966c26c38c29b046dc19140422fad66f3ffdb0857f",
    "bytes": 8644
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
    "fit_item_id": "fit-4ca6c279be34e629",
    "document_kind": "experiment",
    "document_path": "runs/model-portfolio-migration-v1/early-training-document-recovery-01/fit-4ca6c279be34e629-experiment.json",
    "expected_sha256": "eabe8c5909b747e29a3b544f9666c865f67a4a2e337140144e84b00ac5abfadd",
    "binding_pointer": "/closure/experiment_sha256",
    "binding_field": "experiment_sha256",
    "recovery_report": "early-training-document-recovery-01.json",
    "primary_receipt_path": "runs/local-pressure-v1-fresh-bootstrap-2/execution/reopened-stage-00/receipt.json",
    "primary_receipt_sha256": "4ca6c279be34e6293c8daf966c26c38c29b046dc19140422fad66f3ffdb0857f",
    "document_bytes": 3185,
    "verified_direct_original_receipt_binding": true
  }
}
```

## Exact original receipt bytes

```json
{"candidate_update":200,"cases":[{"case_id":"reuse-train-310000:frame-000","frame":0,"row_index":0,"split":"train","trajectory_id":"reuse-train-310000"},{"case_id":"reuse-train-310000:frame-023","frame":23,"row_index":23,"split":"train","trajectory_id":"reuse-train-310000"},{"case_id":"reuse-train-310001:frame-000","frame":0,"row_index":24,"split":"train","trajectory_id":"reuse-train-310001"},{"case_id":"reuse-train-310001:frame-023","frame":23,"row_index":47,"split":"train","trajectory_id":"reuse-train-310001"},{"case_id":"reuse-train-310002:frame-000","frame":0,"row_index":48,"split":"train","trajectory_id":"reuse-train-310002"},{"case_id":"reuse-train-310002:frame-023","frame":23,"row_index":71,"split":"train","trajectory_id":"reuse-train-310002"},{"case_id":"reuse-train-310003:frame-000","frame":0,"row_index":72,"split":"train","trajectory_id":"reuse-train-310003"},{"case_id":"reuse-train-310003:frame-023","frame":23,"row_index":95,"split":"train","trajectory_id":"reuse-train-310003"},{"case_id":"reuse-train-310004:frame-000","frame":0,"row_index":96,"split":"train","trajectory_id":"reuse-train-310004"},{"case_id":"reuse-train-310004:frame-023","frame":23,"row_index":119,"split":"train","trajectory_id":"reuse-train-310004"},{"case_id":"reuse-train-310005:frame-000","frame":0,"row_index":120,"split":"train","trajectory_id":"reuse-train-310005"},{"case_id":"reuse-train-310005:frame-023","frame":23,"row_index":143,"split":"train","trajectory_id":"reuse-train-310005"},{"case_id":"reuse-train-310006:frame-000","frame":0,"row_index":144,"split":"train","trajectory_id":"reuse-train-310006"},{"case_id":"reuse-train-310006:frame-023","frame":23,"row_index":167,"split":"train","trajectory_id":"reuse-train-310006"},{"case_id":"reuse-train-310007:frame-000","frame":0,"row_index":168,"split":"train","trajectory_id":"reuse-train-310007"},{"case_id":"reuse-train-310007:frame-023","frame":23,"row_index":191,"split":"train","trajectory_id":"reuse-train-310007"},{"case_id":"reuse-train-310008:frame-000","frame":0,"row_index":192,"split":"train","trajectory_id":"reuse-train-310008"},{"case_id":"reuse-train-310008:frame-023","frame":23,"row_index":215,"split":"train","trajectory_id":"reuse-train-310008"},{"case_id":"reuse-train-310009:frame-000","frame":0,"row_index":216,"split":"train","trajectory_id":"reuse-train-310009"},{"case_id":"reuse-train-310009:frame-023","frame":23,"row_index":239,"split":"train","trajectory_id":"reuse-train-310009"},{"case_id":"reuse-train-310010:frame-000","frame":0,"row_index":240,"split":"train","trajectory_id":"reuse-train-310010"},{"case_id":"reuse-train-310010:frame-023","frame":23,"row_index":263,"split":"train","trajectory_id":"reuse-train-310010"},{"case_id":"reuse-train-310011:frame-000","frame":0,"row_index":264,"split":"train","trajectory_id":"reuse-train-310011"},{"case_id":"reuse-train-310011:frame-023","frame":23,"row_index":287,"split":"train","trajectory_id":"reuse-train-310011"},{"case_id":"reuse-validation-310100:frame-000","frame":0,"row_index":288,"split":"validation","trajectory_id":"reuse-validation-310100"},{"case_id":"reuse-validation-310100:frame-023","frame":23,"row_index":311,"split":"validation","trajectory_id":"reuse-validation-310100"},{"case_id":"reuse-validation-310101:frame-000","frame":0,"row_index":312,"split":"validation","trajectory_id":"reuse-validation-310101"},{"case_id":"reuse-validation-310101:frame-023","frame":23,"row_index":335,"split":"validation","trajectory_id":"reuse-validation-310101"},{"case_id":"reuse-validation-310102:frame-000","frame":0,"row_index":336,"split":"validation","trajectory_id":"reuse-validation-310102"},{"case_id":"reuse-validation-310102:frame-023","frame":23,"row_index":359,"split":"validation","trajectory_id":"reuse-validation-310102"},{"case_id":"reuse-validation-310103:frame-000","frame":0,"row_index":360,"split":"validation","trajectory_id":"reuse-validation-310103"},{"case_id":"reuse-validation-310103:frame-023","frame":23,"row_index":383,"split":"validation","trajectory_id":"reuse-validation-310103"}],"closure":{"experiment_sha256":"eabe8c5909b747e29a3b544f9666c865f67a4a2e337140144e84b00ac5abfadd","inputs":{"origin":{"bytes":1145917,"sha256":"0fa6da4be6a700b9e5ff314302d299f40deeb5be114a5cdd073c9035c0b68281"},"splits":{"bytes":401235,"sha256":"0bbd2ac6403ef59603d2a83ed5c242089c1d42401253160a286a6800559c409f"},"train0":{"bytes":18890920,"sha256":"2809b91e08b2be41152663b0e970eeea31ed6831e76e3d5b4c1dea93c01601dd"},"train1":{"bytes":18890920,"sha256":"6f97789935a82d01a613096bb5b121584009816e98bf64f4f0f285bf2eac973f"},"validation":{"bytes":12594856,"sha256":"26d3c6b28d7602dbb20862d3be530c9479d1f78484db9d03e9f9cf16e3dc043e"}},"normalization_sha256":"21ab548bea68abc5d5d6a6c459a88664aa67c13a8e521e91c07b6eb3d68fd74e","runtime_manifest_sha256":"ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7","sample_order_sha256":"1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9","source_witness_sha256":"bdcef2d54902f3b507f17a806b9bce1ad819afdf3c3d17e964a7a285ab6b2db8"},"elapsed_s":7.398188701001345,"mode":"train","optimizer_updates":200,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1500186,"role":"checkpoint","sha256":"1b4b77cf299d798b6b787bbd12486bc8d4301d7534972c66e9dddce41e93f737"},"checkpoint-020.npz":{"bytes":1500186,"role":"checkpoint","sha256":"1202ee989e27c189c81958dc574c975826b5642ee2cbc2184f5fa2350321b756"},"checkpoint-040.npz":{"bytes":1500186,"role":"checkpoint","sha256":"ba0952627f002c9d4c998c3be4d3e0dccc1c61f1201d9cb8d4be7c8f35b0ff8b"},"checkpoint-060.npz":{"bytes":1500186,"role":"checkpoint","sha256":"86e6858be9969317ab1012dd93849c02c7c33357b6d98601fc9137190f2acfd1"},"checkpoint-080.npz":{"bytes":1500186,"role":"checkpoint","sha256":"8a73859446b1859b22e2d77eedf57063f3840414e108faa6f36ef39529b2cff8"},"checkpoint-100.npz":{"bytes":1500186,"role":"checkpoint","sha256":"2fede00fc71bc4e92fadd6949c8f540fcfef0be7a523f4083cb4e7cbc4e9914f"},"checkpoint-120.npz":{"bytes":1500186,"role":"checkpoint","sha256":"412231f63b57bef7073553035ce81b9f5a6e6beee8cf4c5cb509335f6a077e37"},"checkpoint-140.npz":{"bytes":1500186,"role":"checkpoint","sha256":"2f679db862badd40347a0b90d865bb300c139600243302b1ac6096ef206fbba1"},"checkpoint-160.npz":{"bytes":1500186,"role":"checkpoint","sha256":"e08ea5037ebff1d653f6bcfe20c557d9c2247cbda66ba9c2a3c65e4bdde90cc7"},"checkpoint-180.npz":{"bytes":1500186,"role":"checkpoint","sha256":"160d6d32c23cdb83bb8ee4c17cf0fe0d0a6f19899810864703eed4fc6088e688"},"checkpoint-200.npz":{"bytes":1500186,"role":"checkpoint","sha256":"e85e58a8682912e8c896d2af8d267c8e6e880a64dfd7965ac41ce1a5e5853665"},"history.json":{"bytes":64889,"role":"training_diagnostics","sha256":"9cf32a6f5f4f49ff8dfb4d5b546f05ee386435c4f1f10c71a5b0f5ed37d8c7f0"},"normalization.json":{"bytes":2105,"role":"preprocessing","sha256":"21ab548bea68abc5d5d6a6c459a88664aa67c13a8e521e91c07b6eb3d68fd74e"},"predictions.npz":{"bytes":1049356,"role":"predictions","sha256":"c3e1a25081c7ada55021d6d874807d1ab4fcdaedf58e98e8a1c45f4dccd070e9"},"splits.json":{"bytes":401235,"role":"training_splits","sha256":"0bbd2ac6403ef59603d2a83ed5c242089c1d42401253160a286a6800559c409f"},"summary.json":{"bytes":168,"role":"result_summary","sha256":"14c0ce2bc8442fa53b9a8c46ace93ac59d730a17f09ff2cba7b2edc6e10ace66"},"weights.npz":{"bytes":1500186,"role":"checkpoint","sha256":"e85e58a8682912e8c896d2af8d267c8e6e880a64dfd7965ac41ce1a5e5853665"}},"per_case_inference_s":[0.010859693007660098,0.007537458004662767,0.006338135004625656,0.006824096999480389,0.0069398509949678555,0.006850874007795937,0.006694579002214596,0.006709358989610337,0.0068061749916523695,0.006312158991931938,0.005932791013037786,0.005721613997593522,0.005706572002964094,0.00592383899493143,0.0063355960010085255,0.006614918995182961,0.0068362549936864525,0.006744217011146247,0.006672739997156896,0.006899942003656179,0.006933030002983287,0.006437121002818458,0.005434790000435896,0.005904057994484901,0.005891170003451407,0.005758571991464123,0.006241698996745981,0.006819221001933329,0.006669761001830921,0.006763082012184896,0.006733436996000819,0.006581892987014726],"prediction_inventory":{"pressure":{"bytes":1048576,"dtype":"<f8","sha256":"065224d0c9ad7238b79fb6b608eba02a069283b37d70b2e8607e75d46fa633eb","shape":[32,16,16,16]},"row_indices":{"bytes":256,"dtype":"<i8","sha256":"17acc08ccda81f2e47be3a0229106f0b522dc3f941a72015f8a677c314fc2eb7","shape":[32]}},"runtime":{"interop_threads":1,"numpy":"2.3.5","peak_rss_bytes":572751872,"python":"3.12.3","threads":1,"torch":"2.8.0+cpu"},"schema":"rheon-local-pressure-training-receipt-v1","status":"complete"}
```

## Exact receipt-bound training experiment

Binding in the original receipt: `/closure/experiment_sha256`.

```json
{"geometries_extents_m":{"g0":[1.0,1.0,1.0],"g1":[1.5,1.0,0.75],"g2":[0.75,1.5,1.0],"g3":[1.0,0.75,1.5]},"id":"fresh-groupnorm-cpu-reuse-v1","resources":{"max_output_bytes":33554432,"wall_seconds":600},"schema":"rheon-local-pressure-experiment-v1","training":{"batch_size":2,"checkpoint_interval":20,"learning_rate":0.001,"model":"groupnorm-v1","seed":20261014,"updates":200},"trajectories":[{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":310000,"split":"train","trajectory_id":"reuse-train-310000"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":310001,"split":"train","trajectory_id":"reuse-train-310001"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":310002,"split":"train","trajectory_id":"reuse-train-310002"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":310003,"split":"train","trajectory_id":"reuse-train-310003"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g0","seed":310004,"split":"train","trajectory_id":"reuse-train-310004"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g1","seed":310005,"split":"train","trajectory_id":"reuse-train-310005"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g2","seed":310006,"split":"train","trajectory_id":"reuse-train-310006"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g3","seed":310007,"split":"train","trajectory_id":"reuse-train-310007"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":310008,"split":"train","trajectory_id":"reuse-train-310008"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":310009,"split":"train","trajectory_id":"reuse-train-310009"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":310010,"split":"train","trajectory_id":"reuse-train-310010"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":310011,"split":"train","trajectory_id":"reuse-train-310011"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":310100,"split":"validation","trajectory_id":"reuse-validation-310100"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":310101,"split":"validation","trajectory_id":"reuse-validation-310101"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":310102,"split":"validation","trajectory_id":"reuse-validation-310102"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":310103,"split":"validation","trajectory_id":"reuse-validation-310103"}]}
```
