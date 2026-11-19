# execution

Retrospective source review · fit-0b2b6b6a8c42a47a

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** A 372,377-parameter, 38-tensor F32 multiscale pressure CNN on a fixed 16³ grid. Ten inputs are native residual, previous pressure, physical XYZ coordinates, separate hx/hy/hz spacing maps, fluid density and dt. Four encoder levels, bottleneck/global-mean path and skip decoder use SiLU with non-affine four-group normalization inserted before each block activation. The additive pressure correction is rescaled and cell-zero gauged in F64. The physical operator supports anisotropic spacing with area/(rho*h) face weights. This produces pressure only, not a full simulation state.

**Purpose.** Qualify configured training, observational checkpoint capture and exact typed restoration of the original GroupNorm pressure architecture on measured multi-scene native data.

**Experiment.** Two hundred historical CPU Adam updates, batch two, rate 0.001, seed 20261014, PCG64 sampling seed 20261015 and clipping. Twelve training trajectories × 24 frames and four validation trajectories × 24 frames use whole-parent splits. All grids have 16 cells per axis; physical extents vary over four shapes, including anisotropic cell spacing. Fluid density is 1.25 kg/m³ and dt=1/60 s throughout. Forcing includes steady 0.5 and moving 1.0 m/s² scenes. Eleven checkpoints at updates 0,20,…,200 are retained; only the final update is the candidate, with no optimizer-resume state. All original input arrays and per-frame pressure/RHS/residual hashes verify in place.

**Observed outcome.** Original final sampled training loss 0.06461062673457954; original validation objective changes from 12.20351413488746 to 0.077668611070560944. These normalized pressure/equation objectives are not worst pressure relative L2 and cannot be compared directly with the near-exact F64 anchors. No current common GPU benchmark, complete-step time or trajectory drift is available for these two original contracts.

**Lesson.** Fixed grid counts do not imply fixed physical resolution: these scenes change physical extent and separate per-axis spacing, unlike later cubic-cell water models. The fluid density is also different from the later 1,000 kg/m³ convention. Preserve those operator/normalization contracts and original native identity. Matching initialized tensors and sample indices across cohorts do not make their physical field data identical or establish independent generalization.

**Reproduction.** Use the exact project-relative train/restore recipes, witnessed flat source, original runtime, all five typed input slots, train-only normalization and every observational checkpoint. Native capture and residual lineage are retained down to each row. The old model and physical checks explicitly require CPU; qualify a GPU-only execution/physical-operator variant before any new forward under current policy. Clean retraining, new-Rheon regeneration and common pressure adapters remain unqualified. Original receipts/configuration remain unchanged.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/receipt.json",
    "sha256": "0b2b6b6a8c42a47a446b2ee025fc666211556c65b69f03a5217aade1dbb0a8ee"
  },
  "history": {
    "path": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/history.json",
    "sha256": "dbdd4c9d521ebc50da57891fca4fc9233d6098ae8b052a20d04b53eaab477819"
  },
  "sources": [
    {
      "path": "runs/local-pressure-v1-workspace-5/project/config.json",
      "sha256": "534601df73cc0679a42b2ec402160aa6baabf1b81a0ffb73b003e100720b8ced"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/contract.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/data.py",
      "sha256": "fafc7ae87e3456c6282daa3256d92302d68aadbe5090772b7b357052a71b3cea"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/gn_model.py",
      "sha256": "8cb87b8e9a2aef6e500daaa43f068ed2e9d9ec70b311c6c586956ad76b60dc6c"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/model_profile.py",
      "sha256": "12a86b642943611757890881b44f697784f48a1fc32d0257c955e7d672575096"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/original_train.py",
      "sha256": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/qualification.json",
      "sha256": "33bac688aed4debcc66b584436f47176b0530fb28201fc9124cafc1d3af597d6"
    },
    {
      "path": "runs/local-pressure-v1-workspace-5/project/train.py",
      "sha256": "a6eb87c83b28390af27ff000854f8a4f4011ed0c7d7312f6cf4274a3209fb9b2"
    }
  ],
  "source_witness": {
    "path": "runs/local-pressure-v1-workspace-5/project/source-witness.json",
    "sha256": "112fa23301c094051506d6fc400eb2b50142e7ea78b6dbd172832f5260dbc175"
  },
  "training_population": {
    "path": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/splits.json",
    "sha256": "a5d3df2e57853974ab04ceb69c9e4a5ea84346a8bbc6879a640ae411b2b351e7"
  },
  "training_inputs": [
    {
      "bytes": 1145862,
      "path": "runs/local-pressure-v1-workspace-5/project/inputs/origin.json",
      "sha256": "15c11bd8bd360fadc0bdb3c1bc6ab92df5270dc7351ddab080be0f3ba6776316",
      "slot": "origin"
    },
    {
      "bytes": 401212,
      "path": "runs/local-pressure-v1-workspace-5/project/inputs/splits.json",
      "sha256": "a5d3df2e57853974ab04ceb69c9e4a5ea84346a8bbc6879a640ae411b2b351e7",
      "slot": "splits"
    },
    {
      "bytes": 18890920,
      "path": "runs/local-pressure-v1-workspace-5/project/inputs/train-0.npz",
      "sha256": "832260ccacb231597feb6fc03aa652b1f00177fe998350b762df340ec4427573",
      "slot": "train0"
    },
    {
      "bytes": 18890920,
      "path": "runs/local-pressure-v1-workspace-5/project/inputs/train-1.npz",
      "sha256": "b89ef14b260ed11bee67a9d2a726e083d2f69a9c65d5043cf74b810e6de17457",
      "slot": "train1"
    },
    {
      "bytes": 12594856,
      "path": "runs/local-pressure-v1-workspace-5/project/inputs/validation.npz",
      "sha256": "78efa428485b760f9ef74beac58437094a0ba45fe220945f032590ef92e37a04",
      "slot": "validation"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-0b2b6b6a8c42a47a",
    "kind": "training_attempt",
    "name": "execution",
    "task": "pressure_component",
    "schema": "rheon-local-pressure-training-receipt-v1",
    "method": {},
    "receipt_sha256": "0b2b6b6a8c42a47a446b2ee025fc666211556c65b69f03a5217aade1dbb0a8ee",
    "receipt_paths": [
      "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/receipt.json",
      "runs/local-pressure-v1-workspace-5/execution/stage-00/artifacts/receipt.json",
      "runs/local-pressure-v1-workspace-5/catalog/workspace-runs/de66b697512e426691d8172c911639a2/receipt.json",
      "runs/local-pressure-v1-workspace-5/catalog/workspace-runs/.external-de66b697512e426691d8172c911639a2/output/receipt.json"
    ],
    "recorded_Run_observations": [],
    "optimizer_updates": 200,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "534601df73cc0679a42b2ec402160aa6baabf1b81a0ffb73b003e100720b8ced",
      "inputs": {
        "origin": {
          "bytes": 1145862,
          "sha256": "15c11bd8bd360fadc0bdb3c1bc6ab92df5270dc7351ddab080be0f3ba6776316"
        },
        "splits": {
          "bytes": 401212,
          "sha256": "a5d3df2e57853974ab04ceb69c9e4a5ea84346a8bbc6879a640ae411b2b351e7"
        },
        "train0": {
          "bytes": 18890920,
          "sha256": "832260ccacb231597feb6fc03aa652b1f00177fe998350b762df340ec4427573"
        },
        "train1": {
          "bytes": 18890920,
          "sha256": "b89ef14b260ed11bee67a9d2a726e083d2f69a9c65d5043cf74b810e6de17457"
        },
        "validation": {
          "bytes": 12594856,
          "sha256": "78efa428485b760f9ef74beac58437094a0ba45fe220945f032590ef92e37a04"
        }
      },
      "normalization_sha256": "cad3492c111ae948bfd955131ec98d100ca5ed859811edc79874538b605d8f7d",
      "runtime_manifest_sha256": "ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7",
      "sample_order_sha256": "1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9",
      "source_witness_sha256": "112fa23301c094051506d6fc400eb2b50142e7ea78b6dbd172832f5260dbc175"
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
        "sha256": "25f91a53f0be4b2a99dfb91a3f62527f2ce2d9c0a42972ed4f6031befe494e0f",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-040.npz",
        "sha256": "59d0f65fb86cab52afa92df57191a8ae2c2dd85bccf9f9fdbfc3aa6d737fb0b9",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-060.npz",
        "sha256": "cd8729360f6b2cb920a6090ba35a2bf989c940b269650b2be1e961b9af614453",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-080.npz",
        "sha256": "77c3ca6a41a89f1bb898e39dbbe4cd79020372c93bee079480453297f4295626",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-100.npz",
        "sha256": "509e2a6082ac33fdeff38955a4dd84f6bd28fe458659c86873e71234618c70e9",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-120.npz",
        "sha256": "81b67f3fadfee1d57934a8596825d16863a638410bc808bdd355d82dbdb62bbb",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-140.npz",
        "sha256": "03add543b62ba9cb0702b068ca6c0c3759d489391eaa66aada26726b77457b00",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-160.npz",
        "sha256": "8db2bcdbcbb1c80b8f59e5fa6098680409d37a9c59c22d2abe67482c63b6f29a",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-180.npz",
        "sha256": "dff70f8f5383c0e61d7307cc1aaf10e38d111c7b5096fcc47f95436519b3716e",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "checkpoint-200.npz",
        "sha256": "37e0a18c085c10667d00e92e700047d293819ce576f22eda8b6c373e6ec6f71e",
        "bytes": 1500186,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "37e0a18c085c10667d00e92e700047d293819ce576f22eda8b6c373e6ec6f71e",
        "bytes": 1500186,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "interop_threads": 1,
      "numpy": "2.3.5",
      "peak_rss_bytes": 572416000,
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
      "receipt": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_component",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/receipt.json",
    "sha256": "0b2b6b6a8c42a47a446b2ee025fc666211556c65b69f03a5217aade1dbb0a8ee",
    "bytes": 8639
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
    "fit_item_id": "fit-0b2b6b6a8c42a47a",
    "document_kind": "experiment",
    "document_path": "runs/model-portfolio-migration-v1/early-training-document-recovery-01/fit-0b2b6b6a8c42a47a-experiment.json",
    "expected_sha256": "534601df73cc0679a42b2ec402160aa6baabf1b81a0ffb73b003e100720b8ced",
    "binding_pointer": "/closure/experiment_sha256",
    "binding_field": "experiment_sha256",
    "recovery_report": "early-training-document-recovery-01.json",
    "primary_receipt_path": "runs/local-pressure-v1-workspace-5/execution/reopened-stage-00/receipt.json",
    "primary_receipt_sha256": "0b2b6b6a8c42a47a446b2ee025fc666211556c65b69f03a5217aade1dbb0a8ee",
    "document_bytes": 3179,
    "verified_direct_original_receipt_binding": true
  }
}
```

## Exact original receipt bytes

```json
{"candidate_update":200,"cases":[{"case_id":"local-train-210000:frame-000","frame":0,"row_index":0,"split":"train","trajectory_id":"local-train-210000"},{"case_id":"local-train-210000:frame-023","frame":23,"row_index":23,"split":"train","trajectory_id":"local-train-210000"},{"case_id":"local-train-210001:frame-000","frame":0,"row_index":24,"split":"train","trajectory_id":"local-train-210001"},{"case_id":"local-train-210001:frame-023","frame":23,"row_index":47,"split":"train","trajectory_id":"local-train-210001"},{"case_id":"local-train-210002:frame-000","frame":0,"row_index":48,"split":"train","trajectory_id":"local-train-210002"},{"case_id":"local-train-210002:frame-023","frame":23,"row_index":71,"split":"train","trajectory_id":"local-train-210002"},{"case_id":"local-train-210003:frame-000","frame":0,"row_index":72,"split":"train","trajectory_id":"local-train-210003"},{"case_id":"local-train-210003:frame-023","frame":23,"row_index":95,"split":"train","trajectory_id":"local-train-210003"},{"case_id":"local-train-210004:frame-000","frame":0,"row_index":96,"split":"train","trajectory_id":"local-train-210004"},{"case_id":"local-train-210004:frame-023","frame":23,"row_index":119,"split":"train","trajectory_id":"local-train-210004"},{"case_id":"local-train-210005:frame-000","frame":0,"row_index":120,"split":"train","trajectory_id":"local-train-210005"},{"case_id":"local-train-210005:frame-023","frame":23,"row_index":143,"split":"train","trajectory_id":"local-train-210005"},{"case_id":"local-train-210006:frame-000","frame":0,"row_index":144,"split":"train","trajectory_id":"local-train-210006"},{"case_id":"local-train-210006:frame-023","frame":23,"row_index":167,"split":"train","trajectory_id":"local-train-210006"},{"case_id":"local-train-210007:frame-000","frame":0,"row_index":168,"split":"train","trajectory_id":"local-train-210007"},{"case_id":"local-train-210007:frame-023","frame":23,"row_index":191,"split":"train","trajectory_id":"local-train-210007"},{"case_id":"local-train-210008:frame-000","frame":0,"row_index":192,"split":"train","trajectory_id":"local-train-210008"},{"case_id":"local-train-210008:frame-023","frame":23,"row_index":215,"split":"train","trajectory_id":"local-train-210008"},{"case_id":"local-train-210009:frame-000","frame":0,"row_index":216,"split":"train","trajectory_id":"local-train-210009"},{"case_id":"local-train-210009:frame-023","frame":23,"row_index":239,"split":"train","trajectory_id":"local-train-210009"},{"case_id":"local-train-210010:frame-000","frame":0,"row_index":240,"split":"train","trajectory_id":"local-train-210010"},{"case_id":"local-train-210010:frame-023","frame":23,"row_index":263,"split":"train","trajectory_id":"local-train-210010"},{"case_id":"local-train-210011:frame-000","frame":0,"row_index":264,"split":"train","trajectory_id":"local-train-210011"},{"case_id":"local-train-210011:frame-023","frame":23,"row_index":287,"split":"train","trajectory_id":"local-train-210011"},{"case_id":"local-validation-210100:frame-000","frame":0,"row_index":288,"split":"validation","trajectory_id":"local-validation-210100"},{"case_id":"local-validation-210100:frame-023","frame":23,"row_index":311,"split":"validation","trajectory_id":"local-validation-210100"},{"case_id":"local-validation-210101:frame-000","frame":0,"row_index":312,"split":"validation","trajectory_id":"local-validation-210101"},{"case_id":"local-validation-210101:frame-023","frame":23,"row_index":335,"split":"validation","trajectory_id":"local-validation-210101"},{"case_id":"local-validation-210102:frame-000","frame":0,"row_index":336,"split":"validation","trajectory_id":"local-validation-210102"},{"case_id":"local-validation-210102:frame-023","frame":23,"row_index":359,"split":"validation","trajectory_id":"local-validation-210102"},{"case_id":"local-validation-210103:frame-000","frame":0,"row_index":360,"split":"validation","trajectory_id":"local-validation-210103"},{"case_id":"local-validation-210103:frame-023","frame":23,"row_index":383,"split":"validation","trajectory_id":"local-validation-210103"}],"closure":{"experiment_sha256":"534601df73cc0679a42b2ec402160aa6baabf1b81a0ffb73b003e100720b8ced","inputs":{"origin":{"bytes":1145862,"sha256":"15c11bd8bd360fadc0bdb3c1bc6ab92df5270dc7351ddab080be0f3ba6776316"},"splits":{"bytes":401212,"sha256":"a5d3df2e57853974ab04ceb69c9e4a5ea84346a8bbc6879a640ae411b2b351e7"},"train0":{"bytes":18890920,"sha256":"832260ccacb231597feb6fc03aa652b1f00177fe998350b762df340ec4427573"},"train1":{"bytes":18890920,"sha256":"b89ef14b260ed11bee67a9d2a726e083d2f69a9c65d5043cf74b810e6de17457"},"validation":{"bytes":12594856,"sha256":"78efa428485b760f9ef74beac58437094a0ba45fe220945f032590ef92e37a04"}},"normalization_sha256":"cad3492c111ae948bfd955131ec98d100ca5ed859811edc79874538b605d8f7d","runtime_manifest_sha256":"ce8a0a16972e050b2985b9853f61edd85f70748cf6681ca768c9e3794d5964d7","sample_order_sha256":"1851c05067aa385ea56461eb86a676799cb562e12f5a618a732f64e527d989d9","source_witness_sha256":"112fa23301c094051506d6fc400eb2b50142e7ea78b6dbd172832f5260dbc175"},"elapsed_s":7.608093987000757,"mode":"train","optimizer_updates":200,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":1500186,"role":"checkpoint","sha256":"1b4b77cf299d798b6b787bbd12486bc8d4301d7534972c66e9dddce41e93f737"},"checkpoint-020.npz":{"bytes":1500186,"role":"checkpoint","sha256":"25f91a53f0be4b2a99dfb91a3f62527f2ce2d9c0a42972ed4f6031befe494e0f"},"checkpoint-040.npz":{"bytes":1500186,"role":"checkpoint","sha256":"59d0f65fb86cab52afa92df57191a8ae2c2dd85bccf9f9fdbfc3aa6d737fb0b9"},"checkpoint-060.npz":{"bytes":1500186,"role":"checkpoint","sha256":"cd8729360f6b2cb920a6090ba35a2bf989c940b269650b2be1e961b9af614453"},"checkpoint-080.npz":{"bytes":1500186,"role":"checkpoint","sha256":"77c3ca6a41a89f1bb898e39dbbe4cd79020372c93bee079480453297f4295626"},"checkpoint-100.npz":{"bytes":1500186,"role":"checkpoint","sha256":"509e2a6082ac33fdeff38955a4dd84f6bd28fe458659c86873e71234618c70e9"},"checkpoint-120.npz":{"bytes":1500186,"role":"checkpoint","sha256":"81b67f3fadfee1d57934a8596825d16863a638410bc808bdd355d82dbdb62bbb"},"checkpoint-140.npz":{"bytes":1500186,"role":"checkpoint","sha256":"03add543b62ba9cb0702b068ca6c0c3759d489391eaa66aada26726b77457b00"},"checkpoint-160.npz":{"bytes":1500186,"role":"checkpoint","sha256":"8db2bcdbcbb1c80b8f59e5fa6098680409d37a9c59c22d2abe67482c63b6f29a"},"checkpoint-180.npz":{"bytes":1500186,"role":"checkpoint","sha256":"dff70f8f5383c0e61d7307cc1aaf10e38d111c7b5096fcc47f95436519b3716e"},"checkpoint-200.npz":{"bytes":1500186,"role":"checkpoint","sha256":"37e0a18c085c10667d00e92e700047d293819ce576f22eda8b6c373e6ec6f71e"},"history.json":{"bytes":64893,"role":"training_diagnostics","sha256":"dbdd4c9d521ebc50da57891fca4fc9233d6098ae8b052a20d04b53eaab477819"},"normalization.json":{"bytes":2110,"role":"preprocessing","sha256":"cad3492c111ae948bfd955131ec98d100ca5ed859811edc79874538b605d8f7d"},"predictions.npz":{"bytes":1049356,"role":"predictions","sha256":"02bdbfd67112126d45b378218e3fe8674025d7948f66e210148a3ba0a05d6a48"},"splits.json":{"bytes":401212,"role":"training_splits","sha256":"a5d3df2e57853974ab04ceb69c9e4a5ea84346a8bbc6879a640ae411b2b351e7"},"summary.json":{"bytes":168,"role":"result_summary","sha256":"5cf10f11bc182e0c23deb0c0e56906ec7d559578ab3197711510b7736aa2dd6a"},"weights.npz":{"bytes":1500186,"role":"checkpoint","sha256":"37e0a18c085c10667d00e92e700047d293819ce576f22eda8b6c373e6ec6f71e"}},"per_case_inference_s":[0.011114314998849295,0.008665953995659947,0.006727525993483141,0.006331529002636671,0.006410989997675642,0.005779617989901453,0.005890678992727771,0.0057025129935937,0.0056154539925046265,0.005769340990809724,0.006436137002310716,0.006743923993781209,0.006795310007873923,0.006376819001161493,0.006518771013361402,0.006671169001492672,0.006982363003771752,0.007014791000983678,0.006510844003059901,0.006360508996294811,0.006389900998328812,0.006016662999172695,0.0061068679933669046,0.006578856991836801,0.006646656009252183,0.006655132994637825,0.006541515002027154,0.00659789600467775,0.006529757010866888,0.006528263998916373,0.00653672699991148,0.006315867998637259],"prediction_inventory":{"pressure":{"bytes":1048576,"dtype":"<f8","sha256":"b393d6f840ef490bba25134ac71cf7476a01c37010a75b0786a23ff56bcff462","shape":[32,16,16,16]},"row_indices":{"bytes":256,"dtype":"<i8","sha256":"17acc08ccda81f2e47be3a0229106f0b522dc3f941a72015f8a677c314fc2eb7","shape":[32]}},"runtime":{"interop_threads":1,"numpy":"2.3.5","peak_rss_bytes":572416000,"python":"3.12.3","threads":1,"torch":"2.8.0+cpu"},"schema":"rheon-local-pressure-training-receipt-v1","status":"complete"}
```

## Exact receipt-bound training experiment

Binding in the original receipt: `/closure/experiment_sha256`.

```json
{"geometries_extents_m":{"g0":[1.0,1.0,1.0],"g1":[1.5,1.0,0.75],"g2":[0.75,1.5,1.0],"g3":[1.0,0.75,1.5]},"id":"fresh-groupnorm-cpu-v1","resources":{"max_output_bytes":33554432,"wall_seconds":600},"schema":"rheon-local-pressure-experiment-v1","training":{"batch_size":2,"checkpoint_interval":20,"learning_rate":0.001,"model":"groupnorm-v1","seed":20261014,"updates":200},"trajectories":[{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":210000,"split":"train","trajectory_id":"local-train-210000"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":210001,"split":"train","trajectory_id":"local-train-210001"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":210002,"split":"train","trajectory_id":"local-train-210002"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":210003,"split":"train","trajectory_id":"local-train-210003"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g0","seed":210004,"split":"train","trajectory_id":"local-train-210004"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g1","seed":210005,"split":"train","trajectory_id":"local-train-210005"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g2","seed":210006,"split":"train","trajectory_id":"local-train-210006"},{"amplitude_m_s2":1.0,"dt_s":0.016666666666666666,"forcing_regime":"moving","frames":24,"geometry":"g3","seed":210007,"split":"train","trajectory_id":"local-train-210007"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":210008,"split":"train","trajectory_id":"local-train-210008"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":210009,"split":"train","trajectory_id":"local-train-210009"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":210010,"split":"train","trajectory_id":"local-train-210010"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":210011,"split":"train","trajectory_id":"local-train-210011"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g0","seed":210100,"split":"validation","trajectory_id":"local-validation-210100"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g1","seed":210101,"split":"validation","trajectory_id":"local-validation-210101"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g2","seed":210102,"split":"validation","trajectory_id":"local-validation-210102"},{"amplitude_m_s2":0.5,"dt_s":0.016666666666666666,"forcing_regime":"steady","frames":24,"geometry":"g3","seed":210103,"split":"validation","trajectory_id":"local-validation-210103"}]}
```
