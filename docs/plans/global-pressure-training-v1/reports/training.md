# Actual fixed 200-update training

Run `f555a2b618d64b14b8df460ea3e65296` completed and drained in the real local
Tuldok catalog. Independent numerical and standards reviewers cleared exact
material `7f5ba44378ec004e86fea97dc4bae7a8e1bc5c6cc41c2e32f3640dd3f7ad7837`
before the owner issued one use approval. Existing CLI submission used one
persisted request key. The initial relative configuration-path refusal happened
client-side before admission; the same request body/key succeeded with the
required absolute private path. There was one scientific training attempt.

Committed source is `a9345a49cd47378d662930a09875fcc3174c04e7`. Actual captured
Python 3.12.14, CPU Torch 2.8.0 and NumPy 2.3.5 ran exactly 200 Adam updates,
batch 2, unchanged seeds/configuration, one Torch/inter-op thread. Train-only
normalization and fixed sampling match pre-training qualification. The six
published artifacts total 2,637,617 bytes; all were downloaded through the existing
SHA256-verifying CLI. Weights are 1,500,186 bytes, SHA256
`f46aec5e0de59bca9a2a7a701e85c4904782c11b7ac6c113cfd3f7625c8391fc`.
Actual Run wall interval was 23.454708 seconds; recipe work reported 11.507212
seconds and peak process RSS 522,350,592 bytes. The enclosing Run interval includes
runtime/setup/custody work absent from the recipe timer; neither is a solver benchmark.

Validation was observed only at updates 0 and 200, with residual-loss ratios
0.5236181201626715 and 0.5214263489529538. After a transient lower sampled training
loss, late gradient norms were about 1e-8. All 32 final pressure witnesses are
bitwise identical to previous accepted reference pressure: learned correction is
exactly zero on these witnesses. This fixed training result is unhelpful for solver
acceleration. We retained final update 200 and performed no extra training,
checkpoint selection, architecture or tolerance change.

See [actual receipt and artifact inventory](training.json). Raw arrays, captured
runtime, catalog credentials and numeric weights remain in ignored task-local
storage; no public raw dataset or secret is committed. Historical failed CNN and
modal baseline receipts remain unchanged and describe different experiments.
