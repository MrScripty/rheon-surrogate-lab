# Issues and dispositions

- E1, scientific interpretation: temporal PSD and spatial low-mode error are
  distinct. Owner diagnostics; fix now by labelling frequency domain and recording
  reference signal power as well as absolute/relative error. No low-amplitude
  noise-floor conclusion from relative error alone.
- E2, capture boundary: existing project assembly rejects unrelated untracked
  user documents. Owner project.py; fix now with tracked-clean source check plus
  exact committed configuration bytes, preserving immutable captured source
  hashes. Never stage unrelated files to satisfy an ambient cleanliness check.
