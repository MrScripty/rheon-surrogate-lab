# CUDA portability integration ledger

2026-10-14, America/Vancouver. User authorization: implement CUDA training while
CPU-first applies to deployment inference; keep portability separate from model
changes, preserve historical results, use the existing catalog/UI and new branches.

1. Selected canonical plan, inspected standards and clean worktrees. Measured the
   installed Pumas Python 3.14/Torch 2.14+cu132 environment without modifying it.
   Registered 25,765 assets in the existing Tuldok catalog; no private training catalog.
2. Added generic measured Torch runtime owner/bootstrap and recipe profile. Kept
   process custody in its existing owner and legacy cold material closures intact.
3. Added versioned lab CUDA/CPU execution, deterministic equivalent CUDA gradients,
   sparse checkpoints, independent seeds and explicit bounded schedules. Preserved
   physical inputs, model architecture and all old namespaces. Committed Tuldok
   b881ebd and lab 641f499/59397ff before freezing source/material.
4. Passed 14 real CUDA/CPU model checks and 51 runtime/material/custody checks.
   Qualified all six physical grid fixtures without constructing an optimizer.
5. First approved Run bf6d12d3e90e4e728baa20c5ad82aa8f interrupted before launch:
   supervisor compared the legacy host closure. Retained the failed evidence and
   zero-update status. Added real admission-to-launch regression; fixed dispatch
   in separate Tuldok commit d852e21. Rechecked legacy material and custody.
6. New immutable project 0ae8c995d2ec4f0194c6828dfb8aad35 ran CUDA training
   097f84f0493f45d1a52c1180c1c1032b and CPU restore 284b8c39020e41a5a00468181b7d12bf
   in catalog a7da4a8de53541a2a0e2aad08005bb71. Both completed and drained.
   Twenty CUDA updates; zero CPU updates; 12 exact CPU predictions and eight shared
   artifacts byte identical. Every artifact was downloaded again from reopened Runs.
7. Native CPU diagnostics evaluated all 12 proposals: 0 pass model-output accuracy,
   12 conventional fallback solves qualify. Actual previews published to the open
   Rheon egui live-project interface; Tuldok activity/normal Runs remained visible.
8. Reviewed a final registration resource bound: inventory now stops hashing at the
   aggregate budget, rather than checking only after the full inventory. 25 focused
   tests pass, including this regression; qualified child execution is unchanged.
9. Final runtime head d3514c6 loaded in the idle existing server. Reopened and
   hash-verified current CUDA/CPU and historical CPU artifacts through normal HTTP.
10. Updated current project/source/results pointers and scientific proposal's resolved
   prerequisites. Preserved the longer-run proposal as unexecuted. Completed scope
   is CUDA training and CPU inference portability; no model accuracy claim.

Retained resources: normal lab checkout and local branch; clean Tuldok worktree
serving the existing application; prior/current immutable project material and
native/manifest/Run artifacts. Keep these for user review and continued operation.
No branch rewrite, push, merge, original-checkout cleanup or package installation.
