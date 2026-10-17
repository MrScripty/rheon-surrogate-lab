# Lab migration standards review

Status: no open documented Standards violations; one nonblocking judgement call. Initial WIP review used base `d6146a20a083ed73a8b59fb1d54b14b8ac706945` plus the explicit new-file inventory. The [JSON record](lab-standards-review.json) preserves initial source witnesses and exact follow-up hashes.

**Closed documented violation — LAB-STD-1.** The runtime producer emits `all_original_fields_and_asset_bytes_identical`. The current cohort consumer now requires that single representation through `admit_runtime_receipt`, with both it and `full_second_verification` strictly `True`. The unsupported alternate key is rejected. Independent review confirmed the focused fixture calls the real `dependency_runtime.measure` producer with a tiny stdlib owner, then passes its actual receipt into current admission. All eight focused qualification tests passed independently in 0.007 seconds. No numerical arithmetic changed.

The actual executed cohort runner remains archived byte-for-byte as `numerical-qualification-v1/cohort-runner-source-01.py`, SHA256 `3e7a18e587636bf6232878fbdd92a6b44f0f9a76a8c8b73f7680665f18ab1f72`. This closure preserves the initial mismatch and its disposition; accepted scientific receipts retain their actual producer provenance.

**Judgement call — LAB-SMELL-1.** Cargo environment filtering and graph/build/test/clippy mechanics repeat in `local-pressure-v1/native.py:31` and `water-residual-learning-v1/build_rollout.py:32`. A shared mechanism could reduce drift while each caller retains its distinct scientific closure and receipt. This maintenance suggestion is not a blocker.

The initial review otherwise found source, runtime and artifact authority separate; exact original producer archives retained; numerical sources/weights packaged with current custody; train-only CUDA/CPU smoke material identified; and pressure request/reply and advancement delegated to the public Rheon owner. Historical hashes remain comparator/converter inputs.

Applicable MCP guidance was fully routed with no unresolved categories. This follow-up changed review documents only and ran eight stdlib fixture tests; it performed no model, optimizer, GPU, service, database or trajectory work. It closes the Standards finding without asserting independently owned full cohort, timing, catalog or live enablement acceptance. Exact committed review remains a later step.
