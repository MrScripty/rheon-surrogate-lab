# Issues and dispositions

- E1, scientific interpretation: temporal PSD and spatial low-mode error are
  distinct. Owner diagnostics; fix now by labelling frequency domain and recording
  reference signal power as well as absolute/relative error. No low-amplitude
  noise-floor conclusion from relative error alone.
- E2, capture boundary: existing project assembly rejects unrelated untracked
  user documents. Owner project.py; fix now with tracked-clean source check plus
  exact committed configuration bytes, preserving immutable captured source
  hashes. Never stage unrelated files to satisfy an ambient cleanliness check.

- E3, workflow performance: managed smoke wall time450.7s versus CUDA recipe5.66s
  and CPU restore recipe2.57s. Owner Tuldok managed runtime; defer runtime
  optimization to a separate portability/performance change. This run preserves
  attestation and custody checks. Revisit if setup dominates subsequent experiments;
  retained phase receipts separate setup from actual optimizer/inference time.

- E1 resolved:temporal/spatial frequency, absolute/relative error and input
  rounding are separated in results.md and source-bound scalar reports.
- E2 resolved:real smoke and both full captures/restores succeeded with unrelated
  untracked user files preserved; configuration and source identities retained.
- E4, experiment outcome:comparative pressure accuracy regresses in some groups
  despite all physical qualifications and lower peak work. Owner next model
  protocol; defer refinement pending review. No replacement, final-informed fit
  or threshold change. Revisit with a predeclared matched train-only experiment.
