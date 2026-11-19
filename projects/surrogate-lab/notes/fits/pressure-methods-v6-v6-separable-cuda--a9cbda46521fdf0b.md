# Pressure methods v6 · v6-separable CUDA

Retrospective source review · fit-a9cbda46521fdf0b

Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.

**Method.** Three bias-free single-axis length-three convolutions, channels 1→1, with 9 learned F32 coefficients and a fixed residual skip q/8. All layers are linear; F32 execution order remains relevant even when a combined mathematical stencil exists. The input is the normalized physical residual q=(rho/h)r; the F64 reconstructed local pressure correction is cell-zero gauged and added to supplied base pressure. The captured predict_pressure function performs this local readout only; it does not run a V-cycle, GCR, advection or a full simulation advance.

**Purpose.** Test a nine-coefficient separable XYZ stencil, three single-axis convolutions, within the one-iteration numerical/learned training composition.

**Experiment.** 2,400 CUDA Adam updates, batch one, clip norm one; fine rate 0.001 with captured late drops. There are 34 parent groups / 68 saved examples, all train-only. The sampler preserves 1,600 small draws and interleaves 800 larger-tier draws; exact parent/frame exposures and voxel work verify. Grids extend through 128³ at 8/16/32 cells per metre, with fixed dt=1/60 s and fluid density 1,000 kg/m³. Training executes one differentiable flexible GCR iteration combining the original frozen F64 CPU V-cycle and a CUDA neural correction, with true-residual recomputation. Its first coarse residual is independent of neural parameters; there is no subsequent learned-dependent V-cycle requiring the later custom adjoint.  At every training iteration, loss weights are pressure 0.25, pressure-gradient 0.25, equation 0.4 and spatial maximum-equation 0.1, plus log1p(relative equation squared/1e-12); iteration losses are averaged. Captured learning work is 2,400 root neural forwards / 612,197,560 input voxels; inherited auxiliary banks and the old teacher are inactive.

**Observed outcome.** Retained final logged training loss 18.956499976139295. The existing common 128-request result has worst mean-free pressure relative L2 0.26165753830688576, equation relative L2 0.18183158018541942 and GPU median of case-median pressure calls 0.36940174999999997 ms. Scope: Local smoother only on the common raw requests. Numerical hierarchy/GCR composition is excluded and requires a separate complete-solver benchmark. Timings include host-owned input/output, validation/packing, transfers and synchronization for the captured readout, with setup/scoring separate. These are retained consumed small-grid cases, not independent large-grid or trajectory results. No new model execution occurs here.

**Lesson.** Nine separable coefficients constrain the learned local operator differently from a 27-tap dense stencil. Similar local-only timing/error does not establish equivalence of complete solver convergence or trajectory drift.

**Reproduction.** Retain the exact typed native recipe, all witnessed source/config/runtime, fixed normalization, complete original input packet, source population and initial/final tensors. Preserve balanced private RNG streams, physical parent/frame counts and final-update selection. Retain native augmentation receipts or manufactured-pattern generator as applicable; post-coarse and differentiable hierarchy training require their exact numerical arithmetic and gauge/adjoint. The historical wrapper/restore runs CPU neural inference and needs a qualified GPU-only route before replay. Clean retraining, new-Rheon regeneration, complete-composition benchmarking and drift remain unqualified; no historical Trial ancestry is invented.

## Exact source bindings for this review

```json
{
  "original_receipt": {
    "path": "runs/pressure-methods-v6/v6-separable-study-01/train/receipt.json",
    "sha256": "a9cbda46521fdf0b566115ec732688cd14fd9e0c958ca130f8b26ed5adfb6642"
  },
  "history": {
    "path": "runs/pressure-methods-v6/v6-separable-study-01/train/history.json",
    "sha256": "ef0a29b53e2208dda15e67969909567130697f20b02bdcfdff4cd94488484b23"
  },
  "sources": [
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/analytical.py",
      "sha256": "1b5aacdb1888c22ebf4ec25227ae3ffc54e60e5bb0d157e9ffcd06ad015878cb"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/band_feedback.py",
      "sha256": "c9ed84b94c54a7d60220d421e816f6a05ed3452e486c7691567b05b3352adb4d"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/baseline_model.py",
      "sha256": "46636a12f14d6882652eb9a72f958b0d2a5e95aec16ad84772ddf8dcc24a4e04"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/coarse.py",
      "sha256": "faf57fd30b7eba837d387c4c4ca53b67fa128d97f4d6d97727e8306a3853e895"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/config.json",
      "sha256": "121993b41b773b79ee0a9a03ac8467ffefc05d8d1f48f5f61360b19cef52110f"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/custody-validator.py",
      "sha256": "c73b96a1eb006f75f98fa66119f4ccb6eb5421123d48bcde48c08138372ebc57"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/data.py",
      "sha256": "ecb49d27bb91054a088a7ea2718e59a00708ef878bcd295305c6a5ed3ea982b7"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/direction_feedback.py",
      "sha256": "e9b9be263483f99a15a51ff4dbbb9fb28f56066f87ff2786003328d75f87281f"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/evidence.py",
      "sha256": "f7149b7ee9b913218bfd6b9f2dcf5e509c2992ea0f3e38b2f07496df20fbd210"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/fine-tuning.json",
      "sha256": "99ab2a8a92f996af763be2093762d461e01856a7acf54ed804b649787edd57c0"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/fine_tuning.py",
      "sha256": "fd9b0c2ca161260f796d98ac52a4c7a00e640750141feb6ad38bcc04801a2138"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/fixed-normalization.json",
      "sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/frozen_gn.py",
      "sha256": "ce280f6ce86aeccc31521b13e4df039780fda8c112c63616457771377903e08d"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/gradient_clipping.py",
      "sha256": "142d005bda439426300dfc0c10c1f5af36181070354e81f83fe9115a5c12b0cd"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/head_optimizer.py",
      "sha256": "e280386a3df5ca474903a4e6c42ec69177348d76b59226897e41c39af0000818"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/historical-reservation.json",
      "sha256": "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/historical_data.py",
      "sha256": "d001fc4f6ba7465aefab99ca0d7427a11026b0e2b68de2fb51bc679c5832a98f"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/historical_physical.py",
      "sha256": "990cb052fac1853a02325b545455c3a3a3bc78945501edf1417ebb583c7e97d5"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/hybrid-method.json",
      "sha256": "7286631ecc547c35be01b10413acc6e3f85649357bac8c308e25019cffccfbe7"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/learned_coarse_model.py",
      "sha256": "248761d431507dae54403ef68bae79004b8c3e23356d91c8cc4b88289ae0adf0"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/methods_view.py",
      "sha256": "d971132a88e966ddbaf70f6f8a0f5ace2c34b1878c9ba51af9adf221100eff22"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/mixed_order.py",
      "sha256": "93d987b0f10101fb5b380e7310aa77eb6f5c7557198f6d43d90f102393e47b04"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/model.py",
      "sha256": "27cb9925301bcd46fca71568d7015c40a02660b55f61a1ef3b8c3019127300b8"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/native_reconstruct.py",
      "sha256": "bf9f9e5712a22ae01a1dcdeffbdd89a9ba177b50f8aa36deb9692e7f15c61335"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/original_data.py",
      "sha256": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/original_model.py",
      "sha256": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/original_train.py",
      "sha256": "886a66d65ac6214fc3c18805252e5d977336caa6353254b253aa8daef4b239a9"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/physical_residual.py",
      "sha256": "6e1392c26880c034d09c72437fefd4c1865d6540f40e245825a6ad6a4e7f4204"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/policy.py",
      "sha256": "1509131b63b42d1d4e3f4bb2ae9f1950454dd86b906d234c8792b212c06724d7"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/portable_model.py",
      "sha256": "64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/portable_policy.py",
      "sha256": "9aafa54db7d8a7ec73def867d75aeda7860fc06600eebfef2b30c9e0fddd5460"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/prepared.py",
      "sha256": "28263bb05c2ff2dc23d1f36d29cf396a4b9fbfb8f50338ec0e3310d920fc9d76"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/qualification.json",
      "sha256": "27aabdb47e3411a5949e5c7dbc48329c2e887b02fbe742bcbc39f10227161c17"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/reference_spectral.py",
      "sha256": "938737f75d504d469bb03647c19dbfdf330b9547205a264b4deac99fb0ecf64f"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/retained_domain_model.py",
      "sha256": "6c3c831f50c8bd5100d3498f86c68e4658a0eb7c47a3a6b6fbf3b5a34df7a231"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/scenes.py",
      "sha256": "d8646842134cb1086ac362239fd175a0eafb673190e55c470f9ce84d89e96383"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/smoke_train.py",
      "sha256": "fbda6a487c1378389955eecfa2a3b9b7bc9747578a38a99e8df837852ecb9b58"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/support.py",
      "sha256": "0609d725af4dc4c50913d4d0683acb1e9086818f134410a61de6651658a09d25"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/test_inference.py",
      "sha256": "1d6cfec0b353b8e4e0c2e1c9f0f9a9d9bfd6c2b7783c3ab948b984b66b3e99a6"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/train.py",
      "sha256": "b5b11637cbe2a3a4742504781689f35581731290b78101a86d7378ec2a1a1f4d"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/training_diagnostic.py",
      "sha256": "1d903e8d6ac92470893b7496be02e1285194de8b2d4aca93387256fc27b208da"
    },
    {
      "path": "runs/pressure-methods-v6/v6-separable-project-01/training_multilevel.py",
      "sha256": "eb8dfc7f963455112eb430701a941c3a9921b9ca4331e02ee1786fa2e3ad5dba"
    }
  ],
  "source_witness": {
    "path": "runs/pressure-methods-v6/v6-separable-project-01/source-witness.json",
    "sha256": "694af2ccaf8b0c9b115144cdc686efceedc5ac6e2af73436d930afd4e37062b1"
  },
  "training_population": {
    "path": "runs/pressure-methods-v6/v6-separable-study-01/train/splits.json",
    "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
  },
  "training_inputs": [
    {
      "bytes": 22711895,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-00.npz",
      "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5",
      "slot": "dataset"
    },
    {
      "bytes": 25135473,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-01.npz",
      "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368",
      "slot": "dataset_extra"
    },
    {
      "bytes": 844,
      "path": "runs/pressure-methods-v6/v6-separable-initialization-01/weights.npz",
      "sha256": "781cdbe643883d306404885e2a3149db68e2a41f2844085e7b0af21d0d6a9498",
      "slot": "initial_weights"
    },
    {
      "bytes": 57361,
      "path": "runs/pressure-methods-v3/derived-training-view-01/manifest.json",
      "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169",
      "slot": "splits"
    },
    {
      "bytes": 22277708,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-02.npz",
      "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb",
      "slot": "view_02"
    },
    {
      "bytes": 23650806,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-03.npz",
      "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8",
      "slot": "view_03"
    },
    {
      "bytes": 22239100,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-04.npz",
      "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc",
      "slot": "view_04"
    },
    {
      "bytes": 24113006,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-05.npz",
      "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda",
      "slot": "view_05"
    },
    {
      "bytes": 19897861,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-06.npz",
      "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080",
      "slot": "view_06"
    },
    {
      "bytes": 22257574,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-07.npz",
      "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9",
      "slot": "view_07"
    },
    {
      "bytes": 18319876,
      "path": "runs/pressure-methods-v3/derived-training-view-01/dataset-08.npz",
      "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30",
      "slot": "view_08"
    }
  ]
}
```

## Original retained fit facts before this review

```json
{
  "fit": {
    "item_id": "fit-a9cbda46521fdf0b",
    "kind": "training_attempt",
    "name": "Pressure methods v6 · v6-separable CUDA",
    "task": "pressure_proposal",
    "schema": "rheon-water-learning-training-v1",
    "method": {
      "schema": "rheon-multilevel-local-smoother-v1",
      "input_channels": 1,
      "activation": "linear",
      "restrictions": "Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training; larger inference requires separate qualification"
    },
    "receipt_sha256": "a9cbda46521fdf0b566115ec732688cd14fd9e0c958ca130f8b26ed5adfb6642",
    "receipt_paths": [
      "runs/pressure-methods-v6/v6-separable-study-01/train/receipt.json"
    ],
    "recorded_Run_observations": [
      {
        "catalog_id": "a7da4a8de53541a2a0e2aad08005bb71",
        "run_id": "a5d8e97367ef4a0280d6c3b6e498ad41",
        "epoch": "current_catalog",
        "observed_in": [
          "current_catalog"
        ],
        "graph_name": "Pressure methods v6 · v6-separable CUDA",
        "state": "completed"
      }
    ],
    "optimizer_updates": 2400,
    "parameter_fits": null,
    "declared_parameter_count": null,
    "training_closure": {
      "experiment_sha256": "121993b41b773b79ee0a9a03ac8467ffefc05d8d1f48f5f61360b19cef52110f",
      "inputs": {
        "dataset": {
          "bytes": 22711895,
          "sha256": "20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"
        },
        "dataset_extra": {
          "bytes": 25135473,
          "sha256": "2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"
        },
        "initial_weights": {
          "bytes": 844,
          "sha256": "781cdbe643883d306404885e2a3149db68e2a41f2844085e7b0af21d0d6a9498"
        },
        "splits": {
          "bytes": 57361,
          "sha256": "8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"
        },
        "view_02": {
          "bytes": 22277708,
          "sha256": "85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"
        },
        "view_03": {
          "bytes": 23650806,
          "sha256": "98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"
        },
        "view_04": {
          "bytes": 22239100,
          "sha256": "1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"
        },
        "view_05": {
          "bytes": 24113006,
          "sha256": "6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"
        },
        "view_06": {
          "bytes": 19897861,
          "sha256": "8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"
        },
        "view_07": {
          "bytes": 22257574,
          "sha256": "4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"
        },
        "view_08": {
          "bytes": 18319876,
          "sha256": "2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"
        }
      },
      "normalization_sha256": "f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1",
      "runtime_manifest_sha256": "b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682",
      "sample_order_sha256": "157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56",
      "source_witness_sha256": "694af2ccaf8b0c9b115144cdc686efceedc5ac6e2af73436d930afd4e37062b1"
    },
    "checkpoints": [
      {
        "name": "checkpoint-000.npz",
        "sha256": "781cdbe643883d306404885e2a3149db68e2a41f2844085e7b0af21d0d6a9498",
        "bytes": 844,
        "stage": "initial"
      },
      {
        "name": "checkpoint-2400.npz",
        "sha256": "4d4c8f1ef2abde0977141fef53e42351d340ab6a711848737e136ecc9ac330be",
        "bytes": 844,
        "stage": "retained_snapshot"
      },
      {
        "name": "weights.npz",
        "sha256": "4d4c8f1ef2abde0977141fef53e42351d340ab6a711848737e136ecc9ac330be",
        "bytes": 844,
        "stage": "selected"
      }
    ],
    "raw_receipt_status": "complete",
    "declared_runtime": {
      "inference_device": "cpu",
      "interop_threads": 1,
      "numpy": "2.5.3",
      "peak_rss_bytes": 3060752384,
      "threads": 1,
      "torch": "2.14.0+cu132"
    },
    "named_entries": [
      "cohorts/115"
    ],
    "purpose": {
      "status": "source_review_required",
      "evidence": "Captured architecture, training configuration and linked named entries."
    },
    "outcome": {
      "status": "retained_training_evidence_only",
      "fresh_common_benchmark": "pending",
      "scientific_promotion": false
    },
    "lesson": {
      "status": "review_required_no_automatic_causal_claim"
    },
    "reproduction": {
      "receipt": "runs/pressure-methods-v6/v6-separable-study-01/train/receipt.json",
      "source_and_data_closure": "Retained hashes; clean reproduction not yet exercised."
    },
    "applicability": {
      "status": "adapter_and_contract_review_required",
      "task": "pressure_proposal",
      "fresh_result": null
    }
  },
  "original_receipt": {
    "path": "runs/pressure-methods-v6/v6-separable-study-01/train/receipt.json",
    "sha256": "a9cbda46521fdf0b566115ec732688cd14fd9e0c958ca130f8b26ed5adfb6642",
    "bytes": 9924
  },
  "bound_training_configuration": null,
  "qualified_common_pressure": {
    "scope": "128 original consumed pressure requests; not a full-state, independent-validation or drift qualification",
    "capture_and_common_score": {
      "path": "docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json",
      "sha256": "3517bc4076e6ea0da9da857db43461e1be0b841804648bb12f40db2e990f1dcd"
    },
    "subjects": {
      "C117": {
        "name": "v6-separable",
        "weights_sha256": "4d4c8f1ef2abde0977141fef53e42351d340ab6a711848737e136ecc9ac330be",
        "cases": 128,
        "worst": {
          "common_equation_relative_l2": 0.18183158018541942,
          "pressure_mean_free_relative_l2": 0.26165753830688576,
          "pressure_absolute_rms": 17.883939074234533,
          "pressure_absolute_max": 59.677809878168986
        },
        "worst_case": {
          "common_equation_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          },
          "pressure_mean_free_relative_l2": {
            "trajectory_id": "baseline--water-validation-0031",
            "native_frame": 15,
            "counts_xyz": [
              14,
              6,
              16
            ],
            "cell_size_m": 0.0625
          }
        },
        "by_cells_per_metre": {
          "8.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.12777170498029267,
              "pressure_mean_free_relative_l2": 0.18136763886057602,
              "pressure_absolute_rms": 17.883939074234533,
              "pressure_absolute_max": 59.677809878168986
            }
          },
          "16.0": {
            "cases": 64,
            "worst": {
              "common_equation_relative_l2": 0.18183158018541942,
              "pressure_mean_free_relative_l2": 0.26165753830688576,
              "pressure_absolute_rms": 5.431181771480055,
              "pressure_absolute_max": 14.52436945447666
            }
          }
        },
        "execution_variant": "captured_CUDA_eager",
        "result": {
          "path": "runs/model-portfolio-migration-v1/pressure-portfolio-GPU-qualification-02/C117/results.json",
          "sha256": "30125a2581395f60e623d933bc4fc49105b2c1dbc82b8b5dd432686966ba6047"
        },
        "CNN_operation_AST_preserved": null,
        "CPU_neural_forward": false
      }
    },
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
{"architecture":{"activation":"linear","bias":false,"boundary":"Replicated one-cell halo at each convolution","channels":["dimensionless_laplacian_rhs"],"coarse_training_cycles":0,"deployment":"Local correction called after an independently validated numerical multilevel cycle","fixed_skip":"q / 8","input_channels":1,"kernels":"separable","parameters":9,"preprocessing":"q=(rho/h)*residual, divided by fixed original train-only q RMS","reconstruction":"Raw correction times the same q RMS; cell-zero gauge; add supplied base","restrictions":"Uniform-density cubic sealed Neumann operator, up to 128 cells per axis for training; larger inference requires separate qualification","schema":"rheon-multilevel-local-smoother-v1","widths":[1,1]},"cases":[{"cell_size_m":0.125,"counts_xyz":[16,16,16],"frame":0,"native_frame":15,"parent":0,"split":"train","trajectory_id":"baseline--water-train-0000"},{"cell_size_m":0.0625,"counts_xyz":[12,6,11],"frame":1,"native_frame":63,"parent":3,"split":"train","trajectory_id":"baseline--water-train-0003"},{"cell_size_m":0.125,"counts_xyz":[8,5,9],"frame":0,"native_frame":15,"parent":6,"split":"train","trajectory_id":"baseline--water-train-0006"},{"cell_size_m":0.0625,"counts_xyz":[6,14,9],"frame":1,"native_frame":63,"parent":9,"split":"train","trajectory_id":"baseline--water-train-0009"},{"cell_size_m":0.125,"counts_xyz":[6,13,12],"frame":0,"native_frame":15,"parent":12,"split":"train","trajectory_id":"baseline--water-train-0012"},{"cell_size_m":0.0625,"counts_xyz":[14,15,7],"frame":1,"native_frame":63,"parent":15,"split":"train","trajectory_id":"baseline--water-train-0015"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":15,"parent":18,"split":"train","trajectory_id":"large-train-64-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[31,23,17],"frame":1,"native_frame":63,"parent":21,"split":"train","trajectory_id":"v2-train-00-cpm16"},{"cell_size_m":0.125,"counts_xyz":[48,13,9],"frame":0,"native_frame":15,"parent":24,"split":"train","trajectory_id":"v2-train-02-cpm8"},{"cell_size_m":0.0625,"counts_xyz":[64,64,64],"frame":1,"native_frame":15,"parent":27,"split":"train","trajectory_id":"v3-train-00-n64-cpm16"},{"cell_size_m":0.125,"counts_xyz":[64,64,64],"frame":0,"native_frame":0,"parent":30,"split":"train","trajectory_id":"v3-train-01-n64-cpm8"},{"cell_size_m":0.03125,"counts_xyz":[128,128,128],"frame":1,"native_frame":15,"parent":33,"split":"train","trajectory_id":"v3-train-01-n128-cpm32"}],"closure":{"experiment_sha256":"121993b41b773b79ee0a9a03ac8467ffefc05d8d1f48f5f61360b19cef52110f","inputs":{"dataset":{"bytes":22711895,"sha256":"20aa821bd0399b2f1057146e770678063164dcd407438c60c2a80b616d1991f5"},"dataset_extra":{"bytes":25135473,"sha256":"2e020e3ac3c056424c4c10a6d38ca90b75d5c234267b0ee741f85801f6a30368"},"initial_weights":{"bytes":844,"sha256":"781cdbe643883d306404885e2a3149db68e2a41f2844085e7b0af21d0d6a9498"},"splits":{"bytes":57361,"sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"view_02":{"bytes":22277708,"sha256":"85e8ec527f5aa2115d16c304a74fb2a1791f510776101c8b33ed4f0386028acb"},"view_03":{"bytes":23650806,"sha256":"98ea9131adcd137d32b2c88bfdee4a67797df8e7a9175d0b3b1514a979de3fa8"},"view_04":{"bytes":22239100,"sha256":"1c8d23206366da05e0fdadb04ef93e1331d7116641bd54007d991b6af4d417cc"},"view_05":{"bytes":24113006,"sha256":"6dcb2730fa2afa4f65d1a72f527ac3414d7ade576183ea55e441a92cd7973eda"},"view_06":{"bytes":19897861,"sha256":"8cdfb4b865349fcd6123a079fb73e5737a332fe443ae99fb7960534210d66080"},"view_07":{"bytes":22257574,"sha256":"4ac480f3523ed959b1204e52b9cd8cf03df9364746348bef753bd2af8ae689f9"},"view_08":{"bytes":18319876,"sha256":"2f4bce03bbd9eb61d863b0c2f29def8b026ed3d69d24baa77b3adc0f99bbed30"}},"normalization_sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1","runtime_manifest_sha256":"b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682","sample_order_sha256":"157178a36ed5bebdd0971c3643a6720aa377962175614cdde7115ddef5cfbf56","source_witness_sha256":"694af2ccaf8b0c9b115144cdc686efceedc5ac6e2af73436d930afd4e37062b1"},"diagnostic_population":"train_only","elapsed_s":285.2509976840229,"execution":{"amp":false,"batches_cuda":true,"cuda_cpu_pressure_max_abs_difference":0.0,"deterministic_algorithms":true,"device":"cuda:0","gpu":"NVIDIA GeForce RTX 5090 Laptop GPU","model_dtype":"float32","parameters_gradients_adam_moments_cuda":true,"peak_allocated_bytes":1148899840,"peak_reserved_bytes":1199570944,"tf32":false},"fine_tuning_statistics":{"coarse_cycles":2400,"loss_calls":2400,"network_forwards":2400,"teacher_forwards":0},"heldout_evaluations":0,"mode":"train","optimizer_updates":2400,"originating_training_run":null,"outputs":{"checkpoint-000.npz":{"bytes":844,"role":"checkpoint","sha256":"781cdbe643883d306404885e2a3149db68e2a41f2844085e7b0af21d0d6a9498"},"checkpoint-2400.npz":{"bytes":844,"role":"checkpoint","sha256":"4d4c8f1ef2abde0977141fef53e42351d340ab6a711848737e136ecc9ac330be"},"history.json":{"bytes":9360,"role":"training_diagnostics","sha256":"ef0a29b53e2208dda15e67969909567130697f20b02bdcfdff4cd94488484b23"},"normalization.json":{"bytes":6752,"role":"preprocessing","sha256":"f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1"},"predictions.npz":{"bytes":1238286,"role":"predictions","sha256":"748c8844e93d0565b7d847b5d7b3f3a8aafbc19234b3ea83586b1dcec891b411"},"profile.json":{"bytes":189,"role":"training_diagnostics","sha256":"bc399290a58d4042c0034fce0726719aa83d6cbb0569c47714b95b7225ee26e5"},"splits.json":{"bytes":57361,"role":"training_splits","sha256":"8ff78b401411739b1a2e2f397a4e490b9444725089c2fb428aff503a37246169"},"summary.json":{"bytes":678,"role":"result_summary","sha256":"0cb9c60197704ec5f19e420c1f31d5a0b8f4c8b3c989f8d2fda5c3ebffa1945b"},"weights.npz":{"bytes":844,"role":"checkpoint","sha256":"4d4c8f1ef2abde0977141fef53e42351d340ab6a711848737e136ecc9ac330be"}},"per_case_inference_s":[0.0003521610051393509,0.00013884296640753746,0.00012523296754807234,0.00011994800297543406,0.0001195790246129036,0.00012249703286215663,0.004626128007657826,0.00030335603514686227,0.0001794869895093143,0.004276441992260516,0.004282617999706417,0.09203436598181725],"prediction_export_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"c4e5445ca95160a6a4ef0c64b763c14e09a8f28473e5abe7d74473519fdd5d72","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"bf33896f4595e962b925a1cbfac0148d10e7ccbe233b3c6aef7ee23cc336c829","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"25181c96ba10c6d89dd7a826f7de406e21ad87c5a64f227af4d28027ea2d173e","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"38a900ebb8956805fe60afa5099eabd1e896797022d9391aacf796e24df246ce","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"a5ec320ffa96faeeaaadf3c98c76780ae6289ab1ed212ccb9896bff3d4dfb132","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"32a4444447d44e00bbb3600109d3b51619594f4f765162bccf2c2dc972dd7f34","shape":[7,15,14]},"case_006":{"bytes":262144,"dtype":"<f8","sha256":"82b6ab4f28f74c18259192c5463a13f8d24d1a842998f449f1a5d6c5ad8e6d4e","shape":[32,32,32]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"f5639e94d8f61bf7b50e96899496a2a410d10c415739b38674d6ac4038ff8ee5","shape":[17,23,31]},"case_008":{"bytes":22464,"dtype":"<f8","sha256":"8c779be1eb7c53b3b8addab27284594aaece1cd667703bcda9a3515fb1667055","shape":[9,13,24]},"case_009":{"bytes":262144,"dtype":"<f8","sha256":"cd313112d1730a78c87d02af9e35decbb8f019282317ffa89f540abfaa9b5a3d","shape":[32,32,32]},"case_010":{"bytes":262144,"dtype":"<f8","sha256":"9a3fd7e952a22cf1e8a15623e52d8eefa89fbf300d1a47c5f19679e696a5dfa2","shape":[32,32,32]},"case_011":{"bytes":262144,"dtype":"<f8","sha256":"288679e2f303302a1158884efbd4408e1c08d9b5fe2af1bf0377e91eabc82bd2","shape":[32,32,32]}},"prediction_export_policy":"NPZ samples at most32cells/axis; full prediction_inventory hashes every full-resolution value and is checked exactly on fresh CPU restore","prediction_inventory":{"case_000":{"bytes":32768,"dtype":"<f8","sha256":"c4e5445ca95160a6a4ef0c64b763c14e09a8f28473e5abe7d74473519fdd5d72","shape":[16,16,16]},"case_001":{"bytes":6336,"dtype":"<f8","sha256":"bf33896f4595e962b925a1cbfac0148d10e7ccbe233b3c6aef7ee23cc336c829","shape":[11,6,12]},"case_002":{"bytes":2880,"dtype":"<f8","sha256":"25181c96ba10c6d89dd7a826f7de406e21ad87c5a64f227af4d28027ea2d173e","shape":[9,5,8]},"case_003":{"bytes":6048,"dtype":"<f8","sha256":"38a900ebb8956805fe60afa5099eabd1e896797022d9391aacf796e24df246ce","shape":[9,14,6]},"case_004":{"bytes":7488,"dtype":"<f8","sha256":"a5ec320ffa96faeeaaadf3c98c76780ae6289ab1ed212ccb9896bff3d4dfb132","shape":[12,13,6]},"case_005":{"bytes":11760,"dtype":"<f8","sha256":"32a4444447d44e00bbb3600109d3b51619594f4f765162bccf2c2dc972dd7f34","shape":[7,15,14]},"case_006":{"bytes":2097152,"dtype":"<f8","sha256":"9e0141b9336a85413c33cfe3afc5a791fe78f29c9c6412000b69722362bcdde9","shape":[64,64,64]},"case_007":{"bytes":96968,"dtype":"<f8","sha256":"f5639e94d8f61bf7b50e96899496a2a410d10c415739b38674d6ac4038ff8ee5","shape":[17,23,31]},"case_008":{"bytes":44928,"dtype":"<f8","sha256":"b9e367c833641ed8d2e1301988535ab48ae85a3d50b16fb3840a6b6f42b45e7a","shape":[9,13,48]},"case_009":{"bytes":2097152,"dtype":"<f8","sha256":"7d64d85511e895bc140a5a3d45e57b5b6e2b0c68bca129fa9e6e4bf1742a6fe7","shape":[64,64,64]},"case_010":{"bytes":2097152,"dtype":"<f8","sha256":"27a08cb7a2743521081c19aacff50507170307e994141dd36d0af865ecb835e9","shape":[64,64,64]},"case_011":{"bytes":16777216,"dtype":"<f8","sha256":"cf70316e79ace20b2b46e7ce9f2d991629908cca3fdd1017cd01b4d5847847b2","shape":[128,128,128]}},"runtime":{"inference_device":"cpu","interop_threads":1,"numpy":"2.5.3","peak_rss_bytes":3060752384,"threads":1,"torch":"2.14.0+cu132"},"schema":"rheon-water-learning-training-v1","scientific_acceptance_or_model_promotion":false,"status":"complete"}
```
